"""
facet_accumulation.py -- Facet-area DC accumulation (union-pixel, smart aggregation).

This module runs the identical, already-verified `_poa_accum_kernel` (same
Perez + Faiman physics, same decoupled/precomputed algebra -- imported, not
reimplemented) but restricted to the UNION of pixels actually referenced by
any facet, computed once each. It then aggregates the per-pixel result back
to per-facet totals with vectorized group-by operations. No Python-level
per-pixel loop is ever executed on the host side; the only pixel-level loop
is the existing `prange` Numba kernel, now sized to the union set instead
of the full H*W grid.

Handles both facet layouts transparently, because both reduce to the same
"list of (facet_id, pixel_index) pairs" representation before anything else
happens:

  * REGULAR facets  -- e.g. a rasterized polygon -> a compact, mostly
    contiguous block of DSM pixels per facet.
  * SCATTERED facets -- e.g. pixels picked out by a classifier / segmentation
    mask that are not contiguous at all, or facets that overlap.

Two independent chunking axes
------------------------------------------------
This module is the only pipeline that chops BOTH axes at once -- read
accumulation_kernels.py's module docstring first if the names below are
unfamiliar:

  * `pixel_batch_size` -- how many union pixels are fetched from the
    (large, float16) horizon_map at once. Bounds peak RAM of the
    (pixel_batch_size, n_dirs) float32 slice. A caller-supplied constant
    (default 200_000), NOT derived -- unlike hour_chunk below.
  * `hour_chunk`        -- how many hours ONE kernel call processes,
    within a given pixel batch. Derived by `resolve_hour_chunk()` from
    THAT BATCH'S OWN pixel count (`n_px_batch`), freshly, inside the
    pixel-batching loop -- never from the region's total pixel count
    (`n_union`). Getting this wrong (sizing hour_chunk off n_union while
    the kernel only ever sees n_px_batch pixels) under-shoots the
    per-launch pixel-hour target and causes more kernel calls than
    necessary; see accumulation_kernels.py's CALLER CONTRACT note.

Output
------
Per facet: annual + monthly DC yield in kWh/kWp (same intrinsic per-area
unit as the full-grid pipeline -- it's independent of facet size, so a
5-pixel facet and a 5000-pixel facet are directly comparable), plus facet
area in m^2 and pixel count for anyone who wants a total energy figure by
multiplying in a module wattage/kWp.

A sparse full-size raster reconstruction (facet pixels filled, everything
else NaN) is also provided for visualisation / TIFF export, reusing
`encode_tif_b64` from `full_dsm_accumulation.py`.
"""

from __future__ import annotations

import base64
import functools
import inspect
import io
import math
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import rasterio
from shapely import Polygon
from skimage.draw import polygon as sk_polygon
from solar.tiff_reader import Tiff_Meta_Data, read_tiff_file


from .accumulation_kernels import (
    _BYTES_PER_FLOAT32,
    _COS_Z_FLOOR_85DEG,
    _SIN_Z_EPSILON,
    _STC_TEMP_DEGC,
    _TAN_SUN_EL_OVERHEAD_SENTINEL,
    resolve_hour_chunk,
)
from .accumulation_kernels import (
    poa_accum as _poa_accum_kernel,
)
from .constants import (
    FACET_BUFFER_PX,
    NODATA_SENTINEL_FOR_IRRADIANCE,
)
from .solar_positions import (
    _DEFAULT_ALBEDO,
    _ETA_REF,
    _FAIMAN_U0,
    _FAIMAN_U1,
    _GAMMA_PDC_DEFAULT,
    SolarPreResult,
    SolarResult,
    assemble_solar_result,
    compute_solar_positions_pre,
)
from .terrain import TerrainResult, run_terrain_analysis

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Default pixels-per-batch for the horizon-map fetch/cast to float32.
# See module docstring above -- this is the RAM-bounding knob, independent
# of hour_chunk. Not derived from anything; a flat operational constant.
_DEFAULT_PIXEL_BATCH_SIZE = 200_000

# ============================================================
# FACET PIXEL TABLE  (regular or scattered facets -> one representation)
# ============================================================


@dataclass
class FacetPixelTable:
    """
    Flattened (facet_id, pixel) representation of an arbitrary set of DSM
    facets, plus the deduplicated union of pixels that actually need the
    physics kernel run on them.

    Attributes
    ----------
    facet_ids       : (n_entries,) int32   -- facet id for each retained entry
                       (already deduplicated -- a facet lists a given pixel
                       at most once, even if the input had duplicates)
    entry_px        : (n_entries,) int64   -- DSM flat pixel index (row*W+col)
                       for each entry, aligned with facet_ids
    entry_to_union  : (n_entries,) int32   -- index into `union_px` for each
                       entry -- lets kernel output be gathered back onto
                       facet entries with a single fancy-index, no loop
    union_px        : (n_union,)  int64    -- SORTED, UNIQUE DSM pixel
                       indices referenced by any facet. The physics kernel
                       runs on exactly these pixels, exactly once each,
                       regardless of how many facets reference them or how
                       scattered they are in DSM space.
    facet_pixel_count : (n_facets,) int64  -- pixel count per facet (after
                       intra-facet dedup)
    segment_ids     : list[str]  -- SolarRoofSegment id for each facet index,
                       aligned with the 0..n_facets-1 facet ids (lets callers
                       map any per-facet output array back to its segment)
    n_facets        : int
    H, W            : int  -- DSM shape the pixel indices are defined against
    """

    facet_ids: np.ndarray
    entry_px: np.ndarray
    entry_to_union: np.ndarray
    union_px: np.ndarray
    facet_pixel_count: np.ndarray
    segment_ids: List[str]
    n_facets: int
    H: int
    W: int


def encode_tif_b64(
    arr: np.ndarray, profile: Tiff_Meta_Data, units: str = 'kWh_per_kWp'
) -> str:
    """
    Encode a float32 numpy array as a zstd-compressed GeoTIFF base64 string.

    Parameters
    ----------
    arr     : (H, W) or (bands, H, W) float32
    profile : rasterio profile dict for georeferencing
    units   : tag written into TIFF metadata (default: kWh_per_kWp)

    Returns
    -------
    str -- base64-encoded GeoTIFF bytes
    """
    arr = np.asarray(arr, dtype=np.float32)
    bands = 1 if arr.ndim == 2 else arr.shape[0]
    H, W = arr.shape[-2], arr.shape[-1]
    blk = min(512, H, W)
    p = dict(profile)
    p.update(
        dtype='float32',
        count=bands,
        compress='zstd',
        predictor=3,
        zstd_level=3,
        tiled=True,
        blockxsize=blk,
        blockysize=blk,
        driver='GTiff',
    )
    buf = io.BytesIO()
    with rasterio.open(buf, 'w', **p) as dst:
        dst.update_tags(UNITS=units)
        strip = max(1, blk)
        if arr.ndim == 2:
            for row_off in range(0, H, strip):
                row_end = min(row_off + strip, H)
                win = rasterio.windows.Window(0, row_off, W, row_end - row_off)
                dst.write(arr[row_off:row_end, :], 1, window=win)
        else:
            for b in range(bands):
                for row_off in range(0, H, strip):
                    row_end = min(row_off + strip, H)
                    win = rasterio.windows.Window(0, row_off, W, row_end - row_off)
                    dst.write(arr[b, row_off:row_end, :], b + 1, window=win)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def corrected_cell_size(
    nominal_cell_size: float,
    crs,
    latitude: float,
) -> float:
    """
    Return the TRUE ground cell size in metres, applying the Web Mercator
    cos(latitude) scale correction when the CRS is EPSG:3857 (or its
    deprecated alias). All other CRSs (including UTM, which is already
    locally equidistant to <0.1% error over a 100m tile) are returned
    unchanged.

    dx_real = dx_nominal * cos(latitude)

    See module docstring [OPT-2026-07] for why this matters: an uncorrected
    Web Mercator cell size overstates ground distance away from the equator,
    which understates slope and therefore shortens shadows / inflates yield.
    """
    _WEB_MERCATOR_EPSG_CODES = {3857, 900913}
    epsg = _extract_epsg_code(crs)
    if epsg in _WEB_MERCATOR_EPSG_CODES:
        scale = math.cos(math.radians(latitude))
        corrected = nominal_cell_size * scale
        print(
            f'[geodetic] EPSG:{epsg} (Web Mercator) detected -- '
            f'correcting cell_size {nominal_cell_size:.6f} m -> '
            f'{corrected:.6f} m at latitude {latitude:.4f} '
            f'(scale factor cos({latitude:.2f} deg)={scale:.6f})',
            flush=True,
        )
        return corrected
    return nominal_cell_size


def _extract_epsg_code(crs) -> Optional[int]:
    """
    Best-effort EPSG code extraction from whatever `dsm_metadata['crs']`
    turns out to be -- a rasterio CRS object, a dict, an int, or a string
    like 'EPSG:3857'. Returns None if it can't be determined (caller then
    applies no correction, i.e. behaves exactly as before this change).
    """
    if crs is None:
        return None
    try:
        # rasterio.crs.CRS has .to_epsg()
        code = crs.to_epsg()
        if code is not None:
            return int(code)
    except AttributeError:
        pass
    if isinstance(crs, int):
        return crs
    if isinstance(crs, str):
        s = crs.upper().replace('EPSG:', '').strip()
        if s.isdigit():
            return int(s)
    return None


def _facet_pixel_region(
    facets: list, H: int, W: int
) -> Optional[tuple[int, int, int, int]]:
    """
    Bounding rectangle (r0, r1, c0, c1) in DSM pixel space that encloses every
    facet polygon vertex, padded 1 px and clipped to the grid. Used to limit
    the terrain ray-march to the pixels the accumulation actually consumes
    (the union pixels all lie inside the facet polygons, hence inside this
    box). Returns None if no usable vertices -> caller falls back to full grid.

    Coords are [x, y] = [col, row], matching build_facet_pixel_table.
    """
    min_r = min_c = np.inf
    max_r = max_c = -np.inf
    for facet in facets:
        coords = facet.get('concave_hull_coords_simplified')
        if not coords:
            continue
        verts = np.asarray(coords, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
            continue
        min_c = min(min_c, float(verts[:, 0].min()))
        max_c = max(max_c, float(verts[:, 0].max()))
        min_r = min(min_r, float(verts[:, 1].min()))
        max_r = max(max_r, float(verts[:, 1].max()))

    if not np.isfinite(min_r):
        return None  # no usable polygon -> full grid

    r0 = max(0, int(math.floor(min_r)) - 1)
    r1 = min(H, int(math.ceil(max_r)) + 1)
    c0 = max(0, int(math.floor(min_c)) - 1)
    c1 = min(W, int(math.ceil(max_c)) + 1)
    if r1 <= r0 or c1 <= c0:
        return None
    return (r0, r1, c0, c1)


def build_facet_pixel_table(
    H: int, W: int, facets: list
) -> FacetPixelTable:
    """
    Build a FacetPixelTable by rasterizing each SolarRoofSegment's
    `concave_hull_coords` polygon into DSM pixel space.

    A facet's polygon is a list of [x, y] = [col, row] vertices in DSM pixel
    coordinates (same convention as panel `exteriorCoords`). Each facet is
    rasterized with `skimage.draw.polygon` to the set of DSM pixels it covers.
    Whether facets are compact/contiguous or scattered/overlapping is
    irrelevant past this point -- both reduce to the same
    (facet_id, pixel_index) representation the dedup and union steps operate on.

    Returns
    -------
    FacetPixelTable
    """
    if not facets:
        # Nothing to do if the caller gave us an empty facet list.
        raise ValueError('facets is empty -- nothing to accumulate')

    # These lists collect pieces from every facet before we glue them
    # together into single big arrays (one glue step is much faster than
    # appending pixel-by-pixel in Python).
    rows_parts: List[np.ndarray] = []  # row (y) coordinates of pixels, per facet
    cols_parts: List[np.ndarray] = []  # column (x) coordinates of pixels, per facet
    id_parts: List[np.ndarray] = []  # which facet each pixel belongs to
    segment_ids: List[str] = []  # human/DB id for each facet, in order

    for fid, facet in enumerate(facets):
        # Keep track of the facet's real id (falls back to its list index).
        segment_ids.append(str(facet.get('id', fid)))

        coords = facet.get('concave_hull_coords_simplified')
        if not coords:
            continue  # facet without a polygon -- contributes 0 pixels

        verts = np.asarray(coords, dtype=np.float64)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError(
                f'facet {fid}: concave_hull_coords must be (N, 2) [x, y] pairs'
            )
        if verts.shape[0] < 3:
            # A polygon needs at least 3 corners to have any area.
            raise ValueError(f'facet {fid}: polygon needs at least 3 vertices')

        # coords are [x, y] = [col, row]; sk_polygon expects (row, col).
        # rr/cc are the pixel row/col coordinates that fall inside the polygon.
        rr, cc = sk_polygon(verts[:, 1], verts[:, 0], shape=(H, W))
        if rr.size == 0:
            continue  # polygon falls outside DSM bounds -- 0 pixels

        rows_parts.append(rr.astype(np.int64))
        cols_parts.append(cc.astype(np.int64))
        # Fill an array the same length as this facet's pixel list, all set
        # to this facet's id, so later we know which facet each pixel is from.
        id_parts.append(np.full(rr.shape[0], fid, dtype=np.int64))

    if not rows_parts:
        # Every facet rasterized to zero pixels -- nothing left to process.
        raise ValueError('every facet rasterized to 0 pixels -- nothing to accumulate')

    # Glue every facet's pixel list into one big flat array.
    rows_all = np.concatenate(rows_parts)
    cols_all = np.concatenate(cols_parts)
    facet_ids_raw = np.concatenate(id_parts)

    # Convert (row, col) into a single flat pixel index (row * width + col).
    # sk_polygon already clips to shape=(H, W), so pixels are in-bounds.
    entry_px = (rows_all * W + cols_all).astype(np.int64)

    n_facets = len(facets)

    # ── Intra-facet dedup: a facet must not count the same pixel twice ──
    # (common after rasterizing a polygon, or hand-built scattered lists
    # with accidental repeats). Single vectorized unique over an encoded
    # (facet_id, pixel) key -- no Python loop over entries.
    # Encoding trick: combine facet_id and pixel index into one number so
    # that np.unique can dedup on the (facet_id, pixel) PAIR in one pass.
    dedup_key = facet_ids_raw * np.int64(H * W) + entry_px
    # We only need the position of each first occurrence (`first_idx`);
    # the sorted key values themselves aren't used again below.
    _, first_idx = np.unique(dedup_key, return_index=True)
    facet_ids = facet_ids_raw[first_idx].astype(np.int32)
    entry_px = entry_px[first_idx]

    # ── Cross-facet union: the physics kernel needs to run on each ──
    # ── distinct DSM pixel exactly once, no matter how many facets ──
    # ── (or duplicate entries) reference it.                        ──
    # union_px: the sorted list of unique pixel indices used by ANY facet.
    # entry_to_union: for each entry, which position in union_px it maps to.
    union_px, entry_to_union = np.unique(entry_px, return_inverse=True)
    entry_to_union = entry_to_union.astype(np.int32)

    # Count how many (deduped) pixels belong to each facet.
    facet_pixel_count = np.bincount(facet_ids, minlength=n_facets).astype(np.int64)

    return FacetPixelTable(
        facet_ids=facet_ids,
        entry_px=entry_px,
        entry_to_union=entry_to_union,
        union_px=union_px,
        facet_pixel_count=facet_pixel_count,
        segment_ids=segment_ids,
        n_facets=n_facets,
        H=H,
        W=W,
    )


# ============================================================
# FACET ACCUMULATION RESULT
# ============================================================


@dataclass
class FacetAccumResult:
    """
    Output of run_facet_accumulation().

    Per-union-pixel arrays are kept (not just the facet aggregates) so
    callers can reconstruct a sparse raster for visualisation via
    `scatter_to_raster` without re-running the kernel.
    """

    union_px: np.ndarray  # (n_union,) int64 -- DSM flat pixel index
    annual_dc_wh_per_m2_union: np.ndarray  # (n_union,) float32
    monthly_dc_wh_per_m2_union: np.ndarray  # (12, n_union) float32

    facet_annual_dc_kwh_per_kwp: np.ndarray  # (n_facets,) float32 -- mean over facet px
    facet_monthly_dc_kwh_per_kwp: np.ndarray  # (n_facets, 12) float32
    facet_area_m2: np.ndarray  # (n_facets,) float32
    facet_pixel_count: np.ndarray  # (n_facets,) int64


def scatter_to_raster(
    H: int,
    W: int,
    union_px: np.ndarray,
    values: np.ndarray,
    fill: float = np.nan,
) -> np.ndarray:
    """
    Scatter a (n_union,) or (n_union, k) per-pixel array back onto a full
    (H, W) (or (k, H, W)) raster, filling everything not in `union_px` with
    `fill`. Single vectorized scatter -- no per-pixel loop.
    """
    values = np.asarray(values)
    if values.ndim == 1:
        # Single-band case (e.g. one annual-DC value per pixel).
        # Start with an all-"fill" flat raster, then drop the real
        # values in at the positions listed in union_px.
        out_flat = np.full(H * W, fill, dtype=np.float32)
        out_flat[union_px] = values
        return out_flat.reshape(H, W)
    else:
        # Multi-band case (e.g. 12 monthly values per pixel).
        k = values.shape[0]
        out_flat = np.full((k, H * W), fill, dtype=np.float32)
        out_flat[:, union_px] = values
        return out_flat.reshape(k, H, W)


# ============================================================
# FACET ACCUMULATION LOOP
# ============================================================


def run_facet_accumulation(
    terrain: TerrainResult,
    solar: SolarResult,
    facet_table: FacetPixelTable,
    pixel_batch_size: int = _DEFAULT_PIXEL_BATCH_SIZE,
    t0: Optional[float] = None,
) -> FacetAccumResult:
    """
    Run the shared `_poa_accum_kernel` over the union of facet pixels only,
    then aggregate to per-facet annual/monthly DC yield.

    Parameters
    ----------
    terrain           : TerrainResult for the SAME DSM the facet pixel coords
                        were defined against (facet px coords are in DSM
                        space).
    solar             : SolarResult (same TMY-derived per-hour arrays used by
                        the full-grid pipeline).
    facet_table       : FacetPixelTable from build_facet_pixel_table().
    pixel_batch_size  : union PIXELS processed per horizon-map fetch. Bounds
                        peak RAM of the (pixel_batch_size, n_dirs) float32
                        horizon-map slice -- this is the facet-space analog
                        of `strip_rows` in the full-grid pipeline, except
                        chunking is over an arbitrary (sorted) 1-D pixel
                        index array rather than raster rows, since facet
                        pixels need not be contiguous. Independent of
                        hour_chunk below -- see module docstring.
    t0                : pipeline start time for log timestamps.

    Note
    ----
    hour_chunk (HOURS per kernel call, within one pixel batch) is derived
    fresh INSIDE the pixel-batch loop below, from that batch's own pixel
    count (`n_px_batch`) via `resolve_hour_chunk` -- never from the
    region's total pixel count. A single small facet (one batch, few
    thousand px) gets a large hour_chunk (often the whole year in one
    launch); a big multi-facet merge that spans several pixel_batch_size
    batches gets hour_chunk computed separately, correctly, for EACH batch.

    Returns
    -------
    FacetAccumResult
    """
    if t0 is None:
        # Start our own stopwatch if the caller didn't give us one.
        t0 = time.time()

    union_px = facet_table.union_px
    n_union = union_px.shape[0]  # how many distinct pixels we must compute
    n_facets = facet_table.n_facets
    n_dirs = terrain.n_dirs  # number of horizon-map compass directions
    daylight = solar.daylight_hours  # only daylight hours produce any DC, skip the rest
    n_daylight = len(daylight)

    print(
        f'[{time.time() - t0:5.1f}s] Facets: {n_facets}  '
        f'union pixels: {n_union} (of {terrain.H * terrain.W} DSM px, '
        f'{100.0 * n_union / (terrain.H * terrain.W):.2f}%)',
        flush=True,
    )

    # ── Precompute solar arrays once, identical algebra to the full-grid ──
    # ── pipeline (imported constants, not re-derived) -- purely temporal ──
    # Everything below is a per-hour (time-only) value, computed once for
    # all daylight hours, then reused for every pixel -- this is what makes
    # the "decoupled" Perez/Faiman algebra fast.
    F1_all = solar.F1[daylight].astype(np.float32)  # Perez circumsolar coefficient
    F2_all = solar.F2[daylight].astype(
        np.float32
    )  # Perez horizon-brightening coefficient
    blk_all = solar.beam_blk[daylight].astype(
        np.float32
    )  # far-field (terrain) beam-blocking flag
    dni_all = solar.dni[daylight]  # direct normal irradiance
    dhi_all = solar.dhi[daylight]  # diffuse horizontal irradiance
    ghi_all = solar.ghi[daylight]  # global horizontal irradiance
    temp_all = solar.temp_air[daylight]  # air temperature
    wind_all = solar.wind_speed[daylight]  # wind speed (cools the panel)
    month_all = solar.months[daylight]  # 0-based month index per hour

    cos_z_all = np.cos(solar.zen_rad[daylight]).astype(
        np.float32
    )  # cos(solar zenith angle)
    sin_z_all = np.sin(solar.zen_rad[daylight]).astype(
        np.float32
    )  # sin(solar zenith angle)
    az_all = solar.az_rad[daylight].astype(np.float32)  # solar azimuth angle (radians)

    # Convert azimuth (radians) into a discrete "compass bucket" index,
    # used to look up the right slice of the horizon map.
    az_to_bucket_scale = np.float32(n_dirs) / np.float32(2.0 * math.pi)
    sun_ew_all = (sin_z_all * np.cos(az_all)).astype(
        np.float32
    )  # east-west component of sun vector
    sun_ns_all = (sin_z_all * np.sin(az_all)).astype(
        np.float32
    )  # north-south component of sun vector

    # Clamp cos(zenith) so it never gets too close to zero (which would
    # blow up the circumsolar term C2 below, since it divides by this).
    cos_z_clamped_all = np.maximum(cos_z_all, _COS_Z_FLOOR_85DEG).astype(np.float32)

    # tan(sun elevation) is used for the near-field shadow test. When the
    # sun is basically straight up (sin_z ~ 0), use a huge sentinel value
    # so the shadow test always says "not shadowed" (no meaningful shadow
    # angle when the sun is directly overhead).
    tan_sun_el_all = np.where(
        sin_z_all > _SIN_Z_EPSILON,
        cos_z_all / np.maximum(sin_z_all, _SIN_Z_EPSILON),
        _TAN_SUN_EL_OVERHEAD_SENTINEL,
    ).astype(np.float32)

    # Which compass-direction "bucket" of the horizon map applies to each hour.
    az_bucket_all = ((az_all * az_to_bucket_scale).astype(np.int32) % n_dirs).astype(
        np.int32
    )

    # Faiman thermal model denominator (wind-cooling term), floored so we
    # never divide by something too close to zero.
    faiman_denom_all = np.maximum(
        (np.float32(_FAIMAN_U0) + np.float32(_FAIMAN_U1) * wind_all),
        np.float32(1e-3),
    ).astype(np.float32)

    # Pre-factored Perez diffuse-sky coefficients (see accumulation_kernels
    # docstring for the algebra) -- computed once per hour, shared by
    # every pixel.
    C1_all = (dhi_all * (1.0 - F1_all)).astype(np.float32)  # isotropic sky term
    C2_all = (dhi_all * F1_all / cos_z_clamped_all).astype(
        np.float32
    )  # circumsolar term
    C3_all = (dhi_all * F2_all).astype(np.float32)  # horizon-brightening term

    # Pre-factored Faiman thermal-correction coefficients, same idea.
    Phi1_all = (
        np.float32(1.0) + np.float32(_GAMMA_PDC_DEFAULT) * (temp_all - _STC_TEMP_DEGC)
    ).astype(np.float32)
    Phi2_all = (
        np.float32(_GAMMA_PDC_DEFAULT) * np.float32(1.0 - _ETA_REF) / faiman_denom_all
    ).astype(np.float32)

    # Ground-reflected irradiance term (GHI * albedo), also purely temporal.
    ghi_albedo_all = (ghi_all * np.float32(_DEFAULT_ALBEDO)).astype(np.float32)

    # ── Per-pixel orientation projections, gathered ONCE for the union ──
    # ── set via a single fancy-index each -- no per-pixel Python loop. ──
    # These describe each pixel's roof slope/orientation -- purely spatial,
    # does not change hour to hour.
    cos_slope_full = np.cos(terrain.slope).astype(np.float32).ravel()
    sin_slope_full = np.sin(terrain.slope).astype(np.float32).ravel()
    cos_aspect_full = np.cos(terrain.aspect).astype(np.float32).ravel()
    sin_aspect_full = np.sin(terrain.aspect).astype(np.float32).ravel()
    svf_full = terrain.svf.astype(np.float32).ravel()  # sky-view factor (0-1)

    # Pull out just the values for pixels we actually need (the union set).
    cos_slope_u = cos_slope_full[union_px]
    sin_slope_u = sin_slope_full[union_px]
    proj_ew_u = (sin_slope_u * cos_aspect_full[union_px]).astype(
        np.float32
    )  # east-west surface-normal projection
    proj_ns_u = (sin_slope_u * sin_aspect_full[union_px]).astype(
        np.float32
    )  # north-south surface-normal projection
    svf_u = svf_full[union_px]

    # ── Output accumulators, sized to the union set only ────────────────
    annual_dc_wh_per_m2_union = np.zeros(n_union, dtype=np.float32)
    monthly_dc_wh_per_m2_union = np.zeros((12, n_union), dtype=np.float32)

    # Work through the union pixels in batches so the horizon-map slice
    # for a batch (pixel_batch_size x n_dirs floats) stays a bounded,
    # predictable size in memory instead of loading everything at once.
    pixel_batch_size = max(1, min(pixel_batch_size, n_union))
    n_px_batches = math.ceil(n_union / pixel_batch_size)
    hor_mb_batch = pixel_batch_size * n_dirs * _BYTES_PER_FLOAT32 // (1024 * 1024)
    print(
        f'[{time.time() - t0:5.1f}s] Pixel batches: {n_px_batches} x {pixel_batch_size} px  '
        f'horizon float32 ~{hor_mb_batch} MB/batch (cast per-batch, not upfront)',
        flush=True,
    )

    print(
        f'[{time.time() - t0:5.1f}s] Phase: facet POA+DC accumulation ...', flush=True
    )

    # ── GPU data-residency session (temporal series uploaded once) ────────
    from . import gpu_kernels, perf

    _gpu_sess = (
        gpu_kernels.PoaGpuSession(
            dni_all,
            cos_z_all,
            blk_all,
            month_all,
            sun_ew_all,
            sun_ns_all,
            tan_sun_el_all,
            az_bucket_all,
            C1_all,
            C2_all,
            C3_all,
            Phi1_all,
            Phi2_all,
            ghi_albedo_all,
        )
        if gpu_kernels.GPU_AVAILABLE
        else None
    )

    # Outer loop: one batch of pixels at a time.
    for px_start in range(0, n_union, pixel_batch_size):
        px_end = min(px_start + pixel_batch_size, n_union)
        n_px_batch = px_end - px_start

        # Slice out just this batch's pixels and their static data.
        batch_union_px = union_px[px_start:px_end]
        hor_batch = terrain.horizon_map[batch_union_px].astype(np.float32)

        cs_batch = cos_slope_u[px_start:px_end]
        ss_batch = sin_slope_u[px_start:px_end]
        pew_batch = proj_ew_u[px_start:px_end]
        pns_batch = proj_ns_u[px_start:px_end]
        sv_batch = svf_u[px_start:px_end]

        # Accumulators just for this pixel batch (filled in by the kernel,
        # then copied into the full-size union arrays below).
        ann_batch = np.zeros(n_px_batch, dtype=np.float32)
        mon_batch = np.zeros((12, n_px_batch), dtype=np.float32)

        # Hours-per-kernel-call, sized to THIS BATCH's own pixel count
        # (n_px_batch) -- NOT n_union. This is the fix for the bug where
        # hour_chunk used to be computed once, before this loop, off the
        # region's total pixel count: with pixel_batch_size capping every
        # batch at (at most) pixel_batch_size pixels, sizing off n_union
        # instead would under-shoot the per-launch pixel-hour target
        # whenever n_union > pixel_batch_size, causing more kernel calls
        # than necessary. Computed fresh, per batch, here.
        hour_chunk = resolve_hour_chunk(n_px_batch, n_daylight)
        print(
            f'  batch {px_start // pixel_batch_size + 1}/{n_px_batches}: '
            f'hour_chunk={hour_chunk} (auto, n_px_batch={n_px_batch}, '
            f'n_daylight={n_daylight})',
            flush=True,
        )

        # Time this batch's kernel work only (see full_dsm_accumulation for the
        # same tic/toc pattern -- avoids re-indenting the GPU/CPU branches).
        _t_batch = perf.tic()

        if _gpu_sess is not None:
            # GPU: temporal series already resident; upload batch statics once,
            # loop chunks on-device, copy accumulators back once.
            ann_batch, mon_batch = _gpu_sess.run_batch(
                hor_batch,
                n_dirs,
                cs_batch,
                ss_batch,
                pew_batch,
                pns_batch,
                sv_batch,
                hour_chunk,
            )
        else:
            # Inner loop: process this pixel batch in chunks of hours, so the
            # Numba kernel is called on manageable time-series slices.
            for t_start in range(0, n_daylight, hour_chunk):
                t_end = min(t_start + hour_chunk, n_daylight)
                nc = t_end - t_start
                sl = slice(t_start, t_end)

                # Run the shared physics kernel (same one full_dsm_accumulation
                # uses) over this hour-slice x pixel-batch combination. Results
                # are accumulated in-place into ann_batch / mon_batch.
                _poa_accum_kernel(
                    nc,
                    dni_all[sl],
                    cos_z_all[sl],
                    blk_all[sl],
                    month_all[sl],
                    sun_ew_all[sl],
                    sun_ns_all[sl],
                    tan_sun_el_all[sl],
                    az_bucket_all[sl],
                    C1_all[sl],
                    C2_all[sl],
                    C3_all[sl],
                    Phi1_all[sl],
                    Phi2_all[sl],
                    ghi_albedo_all[sl],
                    hor_batch,
                    n_dirs,
                    cs_batch,
                    ss_batch,
                    pew_batch,
                    pns_batch,
                    sv_batch,
                    ann_batch,
                    mon_batch,
                )

        perf.toc('POA+DC accumulation (facets)', _t_batch)

        # Copy this batch's finished results into the full union-sized arrays.
        annual_dc_wh_per_m2_union[px_start:px_end] = ann_batch
        monthly_dc_wh_per_m2_union[:, px_start:px_end] = mon_batch

        print(
            f'  batch {px_start // pixel_batch_size + 1}/{n_px_batches} done ({n_px_batch} px)',
            flush=True,
        )

    # ============================================================
    # FACET AGGREGATION -- vectorized group-by, no per-pixel loop
    # ============================================================
    facet_ids = facet_table.facet_ids
    entry_to_union = facet_table.entry_to_union
    facet_pixel_count = facet_table.facet_pixel_count

    # Gather per-entry values with a single fancy index each (entries may
    # revisit the same union pixel many times across facets/overlaps --
    # that's fine, it's O(n_entries), the expensive physics was already
    # done exactly once per unique pixel above).
    annual_per_entry = annual_dc_wh_per_m2_union[entry_to_union]

    # Sum each facet's per-entry annual values (bincount = "sum grouped by
    # facet id"), then divide by the facet's pixel count to get a mean.
    facet_annual_sum = np.bincount(
        facet_ids, weights=annual_per_entry, minlength=n_facets
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        facet_annual_mean_wh_per_m2 = np.where(
            facet_pixel_count > 0,
            facet_annual_sum / np.maximum(facet_pixel_count, 1),
            0.0,
        )
    # Convert Wh/m^2 -> kWh/kWp (same intrinsic per-area unit as the
    # full-grid pipeline).
    facet_annual_dc_kwh_per_kwp = (
        facet_annual_mean_wh_per_m2 / np.float32(1000.0)
    ).astype(np.float32)

    # Same idea, but once per calendar month (12 fixed iterations, so this
    # loop's cost does not grow with pixel or facet count).
    facet_monthly_dc_kwh_per_kwp = np.zeros((n_facets, 12), dtype=np.float32)
    for m in range(12):  # fixed 12 iterations -- independent of px/facet count
        monthly_per_entry_m = monthly_dc_wh_per_m2_union[m][entry_to_union]
        facet_month_sum = np.bincount(
            facet_ids, weights=monthly_per_entry_m, minlength=n_facets
        )
        with np.errstate(invalid='ignore', divide='ignore'):
            facet_month_mean = np.where(
                facet_pixel_count > 0,
                facet_month_sum / np.maximum(facet_pixel_count, 1),
                0.0,
            )
        facet_monthly_dc_kwh_per_kwp[:, m] = facet_month_mean / np.float32(1000.0)

    # Physical area of each facet = pixel count * area of one pixel.
    facet_area_m2 = (facet_pixel_count * (terrain.cell_size**2)).astype(np.float32)

    print(
        f'[{time.time() - t0:5.1f}s] Facet annual DC mean='
        f'{facet_annual_dc_kwh_per_kwp.mean():.1f} kWh/kWp  '
        f'max={facet_annual_dc_kwh_per_kwp.max():.1f} kWh/kWp',
        flush=True,
    )

    return FacetAccumResult(
        union_px=union_px,
        annual_dc_wh_per_m2_union=annual_dc_wh_per_m2_union,
        monthly_dc_wh_per_m2_union=monthly_dc_wh_per_m2_union,
        facet_annual_dc_kwh_per_kwp=facet_annual_dc_kwh_per_kwp,
        facet_monthly_dc_kwh_per_kwp=facet_monthly_dc_kwh_per_kwp,
        facet_area_m2=facet_area_m2,
        facet_pixel_count=facet_pixel_count,
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================


def run_facet_dsm(
    dsm_raw: np.ndarray,
    cell_size: float,
    profile: Tiff_Meta_Data,
    latitude: float,
    longitude: float,
    tmy: pd.DataFrame,
    facets: list,
    nodata: Optional[float] = None,
    pixel_batch_size: int = _DEFAULT_PIXEL_BATCH_SIZE,
    visualize: bool = False,
) -> dict:
    """
    Complete facet pipeline: terrain -> solar -> facet accumulation
    (-> optional sparse TIFFs for visualisation).
    """
    t0 = time.time()
    _terrain: dict = {}
    _solar: dict = {}

    def _terrain_worker() -> None:
        # Runs the (slow) terrain/horizon ray-march analysis.
        _terrain['result'] = run_terrain_analysis(
            dsm_raw,
            cell_size,
            latitude,
            longitude,
            nodata=nodata,
            t0=t0,
        )

    def _solar_worker() -> None:
        # Runs the (fast) solar-position precompute.
        _solar['pre'] = compute_solar_positions_pre(tmy, latitude, longitude)

    # Terrain and solar-position setup don't depend on each other, so run
    # them at the same time on two threads instead of one after another.
    from . import perf

    print(f'\n{"--" * 27}\n  Terrain || Solar positions\n{"--" * 27}', flush=True)
    # Wall time of the terrain||solar phase as a whole. The ray-march inside it
    # is timed separately by terrain.svf_and_horizon, so this minus that is the
    # backend-independent part (DSM read, slope/aspect, solar precompute).
    _t_ts = perf.tic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_terrain = pool.submit(_terrain_worker)
        fut_solar = pool.submit(_solar_worker)
        futures_wait([fut_terrain, fut_solar])
    perf.toc('terrain || solar phase (wall)', _t_ts)
    fut_terrain.result()  # re-raise any exception from the terrain thread
    fut_solar.result()  # re-raise any exception from the solar thread

    terrain: TerrainResult = _terrain['result']
    pre: SolarPreResult = _solar['pre']

    return _run_facet_from_terrain_and_pre(
        terrain=terrain,
        pre=pre,
        profile=profile,
        facets=facets,
        pixel_batch_size=pixel_batch_size,
        visualize=visualize,
        t0=t0,
    )


def run_facet_from_terrain(
    terrain: TerrainResult,
    profile: Tiff_Meta_Data,
    latitude: float,
    longitude: float,
    tmy: pd.DataFrame,
    facets: list,
    pixel_batch_size: int = _DEFAULT_PIXEL_BATCH_SIZE,
    visualize: bool = False,
) -> dict:
    """
    Facet pipeline over a PREBUILT TerrainResult -- skips run_terrain_analysis
    entirely. Only solar positions (cheap, per-request) are computed, then the
    facet pixel table and union-pixel accumulation. This is the fast path taken
    when terrain has been cached for the DSM (see terrain_store.py) and a caller
    just wants facet yields for a (possibly new) set of facets.
    """
    t0 = time.time()
    print(
        f'\n{"--" * 27}\n  Solar positions (terrain preloaded)\n{"--" * 27}', flush=True
    )
    pre: SolarPreResult = compute_solar_positions_pre(tmy, latitude, longitude)
    return _run_facet_from_terrain_and_pre(
        terrain=terrain,
        pre=pre,
        profile=profile,
        facets=facets,
        pixel_batch_size=pixel_batch_size,
        visualize=visualize,
        t0=t0,
    )


def _run_facet_from_terrain_and_pre(
    terrain: TerrainResult,
    pre: SolarPreResult,
    profile: Tiff_Meta_Data,
    facets: list,
    pixel_batch_size: int,
    visualize: bool,
    t0: float,
) -> dict:
    """Shared tail: assemble solar -> facet table -> accumulation -> TIFFs."""
    print(
        f'[{time.time() - t0:5.1f}s] Assembling solar result + beam_blk ...', flush=True
    )
    # Combine the raw solar-position precompute with terrain's horizon data
    # to get the far-field beam-blocking flag (solar.beam_blk).
    solar: SolarResult = assemble_solar_result(pre, terrain.horizon)

    print(f'\n{"--" * 27}\n  Facet pixel table\n{"--" * 27}', flush=True)
    # Turn the raw facet polygons into the flat pixel table the kernel needs.
    facet_table = build_facet_pixel_table(terrain.H, terrain.W, facets)

    print(f'\n{"--" * 27}\n  Facet accumulation\n{"--" * 27}', flush=True)
    # Run the physics kernel + per-facet aggregation. hour_chunk is derived
    # inside run_facet_accumulation, per pixel batch, from that batch's own
    # pixel count -- never threaded through here.
    facet_result = run_facet_accumulation(
        terrain,
        solar,
        facet_table,
        pixel_batch_size=pixel_batch_size,
        t0=t0,
    )

    out: dict = {}

    print(f'\n{"--" * 27}\n  Encode sparse TIFFs\n{"--" * 27}', flush=True)
    # Rebuild full-size rasters (facet pixels filled in, everything else
    # NaN/nodata) so the results can be exported as GeoTIFFs for viewing.
    annual_raster = scatter_to_raster(
        terrain.H,
        terrain.W,
        facet_result.union_px,
        facet_result.annual_dc_wh_per_m2_union / np.float32(1000.0),
        fill=NODATA_SENTINEL_FOR_IRRADIANCE,
    )
    monthly_raster = scatter_to_raster(
        terrain.H,
        terrain.W,
        facet_result.union_px,
        facet_result.monthly_dc_wh_per_m2_union / np.float32(1000.0),
        fill=NODATA_SENTINEL_FOR_IRRADIANCE,
    )

    profile['nodata'] = NODATA_SENTINEL_FOR_IRRADIANCE

    out['annual_dc_tiff_b64'] = encode_tif_b64(
        annual_raster, profile, units='kwh_per_kwp_per_year'
    )
    out['monthly_dc_tiff_b64'] = encode_tif_b64(
        monthly_raster, profile, units='kwh_per_kwp_per_year'
    )
    out['annual_dc_tiff_data'] = {'data': annual_raster, 'metadata': profile}
    # out['monthly_dc_tiff_data'] = {'data': monthly_raster, 'metadata': profile}

    if visualize:
        # Everything below is optional debug/QA plotting, only run when
        # explicitly asked for.
        month_labels = [
            'Jan',
            'Feb',
            'Mar',
            'Apr',
            'May',
            'Jun',
            'Jul',
            'Aug',
            'Sep',
            'Oct',
            'Nov',
            'Dec',
        ]

        from .visualize import (
            plot_tiff_b64,
            plot_dsm_overlay,
        )

        out_dir = os.path.join(_SCRIPT_DIR, 'facet_plots')
        os.makedirs(out_dir, exist_ok=True)

        # 3. Visual overlay -- DSM vs. annual shading raster, same grid.
        plot_dsm_overlay(
            dsm=terrain.dsm,
            overlay_array=annual_raster,
            title='Annual DC yield (kWh/kWp)',
            out_path=os.path.join(out_dir, 'dsm_annual_overlay.png'),
        )

        # existing single-raster plots
        annual_path = os.path.join(out_dir, 'annual_dc.png')
        monthly_path = os.path.join(out_dir, 'monthly_dc.png')
        plot_tiff_b64(
            out['annual_dc_tiff_b64'],
            title='Annual DC yield (kWh/kWp)',
            out_path=annual_path,
        )
        plot_tiff_b64(
            out['monthly_dc_tiff_b64'],
            title='Monthly DC yield (kWh/kWp)',
            out_path=monthly_path,
            band_labels=month_labels,
        )
        print(f'[visualize] wrote plots to {out_dir}', flush=True)

    print(
        f'[{time.time() - t0:5.1f}s] Done -- {time.time() - t0:.1f}s total', flush=True
    )
    return out


def resolve_dsm(func):  # type: ignore[no-untyped-def]
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        dsm = bound.arguments.get('dsm')
        dsm_metadata = bound.arguments.get('dsm_metadata')
        if isinstance(dsm, str):
            tiff = read_tiff_file(dsm)
            bound.arguments['dsm'] = np.asarray(tiff['data'], dtype=np.float32)
            bound.arguments['dsm_metadata'] = tiff['metadata']
        elif dsm is None or dsm_metadata is None:
            raise ValueError(
                'dsm must be a URL (str), or an ndarray paired with dsm_metadata'
            )
        return func(*bound.args, **bound.kwargs)

    return wrapper


@resolve_dsm
def run_facet_shading(
    latitude: float,
    longitude: float,
    facets: list,
    weather_data: dict,
    dsm: 'Optional[str | np.ndarray]' = None,
    dsm_metadata: Optional[dict] = None,
    pixel_batch_size: int = _DEFAULT_PIXEL_BATCH_SIZE,
    visualize: bool = False,
) -> dict:
    """
    Single public entry point for facet-area DC accumulation.

    Terrain is computed fresh via `run_terrain_analysis`.

    Solar-position precompute always runs fresh (cheap, per-request,
    depends on this call's weather/lat-lon so it's never reusable across
    calls) concurrently with whichever terrain path was selected, via a
    2-worker thread pool -- identical concurrency to the old run_facet_dsm.

    """
    if dsm is None:
        raise ValueError('run_facet_shading: DSM is required for terrain analysis')

    from energy.pvlib_core import weather_dict_to_dataframe

    t0 = time.time()
    _terrain: dict = {}
    _solar: dict = {}

    def _terrain_worker() -> None:
        # ── ray-march terrain analysis ──
        nominal_cell_size = float(abs(dsm_metadata['transform'].a))
        cell_size = corrected_cell_size(
            nominal_cell_size, dsm_metadata.get('crs'), latitude
        )
        # Only ray-march the output pixels the accumulation consumes: the
        # bounding rectangle of all facet polygons (for house-bbox that's
        # the expanded bbox). Rays still read the full dsm, so computed
        # pixels are bit-identical to the full-grid pass. See
        # run_terrain_analysis `region` and its cache caveat.
        H, W = dsm.shape
        # An explicit `region` (passed by the incremental merge) confines the
        # ray-march to just the pixels that still lack irradiance; otherwise
        # fall back to the bbox of all facet polygons.
        terrain_region = _facet_pixel_region(facets, H, W)
        terrain = run_terrain_analysis(
            dsm,
            cell_size,
            latitude,
            longitude,
            nodata=dsm_metadata.get('nodata'),
            t0=t0,
            region=terrain_region,
        )
        _terrain['result'] = terrain
        _terrain['profile'] = dsm_metadata
        _terrain['freshly_computed'] = True

    def _solar_worker() -> None:
        irradiance = weather_dict_to_dataframe(
            weather_data.get('weather', weather_data)
        )
        # _debug_save_tmy(irradiance, latitude, longitude)  # <-- added

        _solar['pre'] = compute_solar_positions_pre(irradiance, latitude, longitude)

    from . import perf

    print(f'\n{"--" * 27}\n  Terrain || Solar positions\n{"--" * 27}', flush=True)
    # Wall time of the terrain||solar phase as a whole. The ray-march inside it
    # is timed separately by terrain.svf_and_horizon, so this minus that is the
    # backend-independent part (DSM read, slope/aspect, solar precompute).
    _t_ts = perf.tic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_terrain = pool.submit(_terrain_worker)
        fut_solar = pool.submit(_solar_worker)
        futures_wait([fut_terrain, fut_solar])
    perf.toc('terrain || solar phase (wall)', _t_ts)
    fut_terrain.result()  # re-raise any exception from the terrain thread
    fut_solar.result()  # re-raise any exception from the solar thread

    terrain: TerrainResult = _terrain['result']
    profile = _terrain['profile']
    pre: SolarPreResult = _solar['pre']

    out = _run_facet_from_terrain_and_pre(
        terrain=terrain,
        pre=pre,
        profile=profile,
        facets=facets,
        pixel_batch_size=pixel_batch_size,
        visualize=visualize,
        t0=t0,
    )

    # ── Persist freshly-computed terrain only once accumulation has  ──
    # ── succeeded. Preferred: stream the npz straight down the fd-3    ──
    # ── binary pipe Node wired up (no disk touched); fall back to a    ──
    # ── shared-disk temp file when fd 3 isn't available.               ──

    # if _terrain['freshly_computed']:
    # persist_terrain_result(
    #     terrain, profile=profile, terrain_fd=terrain_fd, t0=t0
    # )
    # Only the disk fallback hands a path back for Node to upload; the
    # fd-3 path streams into an S3 upload Node already owns, so there is
    # nothing to surface in the JSON channel.
    # if persisted.get('terrain_local_path'):
    #     out['terrainLocalPath'] = persisted['terrain_local_path']

    return out


# ============================================================
# INCREMENTAL MERGE -- add a new facet to an existing flux raster
# ============================================================
#
# When a facet is added on the add/remove-facet flow, the house-bbox raster was
# already accumulated over the bounding box + margin, so any facet that lands
# fully inside that region already has its per-pixel DC values in the existing
# annual/monthly flux GeoTIFFs -- no recompute needed, just reuse the URL.
#
# A facet (or part of one) that lands OUTSIDE that region has NODATA pixels in
# the existing raster. For those, accumulate DC over just the facet grown by a
# few pixels (FACET_BUFFER_PX) against the SAME cached terrain, splice the result
# into the existing rasters, re-encode and hand the new base64 back to Node to
# upload + persist.


def _is_nodata(arr: np.ndarray, nodata: float) -> np.ndarray:
    """Boolean mask of pixels that carry no real value (NODATA or NaN)."""
    return np.isnan(arr) | (arr == np.float32(nodata))


def _read_raster(url: str) -> tuple[np.ndarray, dict, float]:
    """Read a flux GeoTIFF URL into (array, profile, nodata). Single-band rasters
    come back (H, W); multi-band as (bands, H, W)."""
    tiff = read_tiff_file(url)
    arr = np.asarray(tiff['data'], dtype=np.float32)
    profile = tiff['metadata']
    nodata = profile.get('nodata')
    nodata = (
        float(nodata) if nodata is not None else float(NODATA_SENTINEL_FOR_IRRADIANCE)
    )
    return arr, profile, nodata


def _decode_monthly_b64(b64: str) -> np.ndarray:
    """Decode a base64 multi-band GeoTIFF back into a (bands, H, W) float32 array
    (run_facet_shading only hands back the monthly raster as base64, not ndarray)."""
    with rasterio.open(io.BytesIO(base64.b64decode(b64))) as ds:
        return ds.read().astype(np.float32)


def _facet_pixels(coords: list, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize a facet polygon ([x, y] = [col, row] verts) to (rows, cols)
    pixel indices, clipped to the (H, W) grid. Same convention as
    build_facet_pixel_table."""
    verts = np.asarray(coords, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 2 or verts.shape[0] < 3:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return sk_polygon(verts[:, 1], verts[:, 0], shape=(H, W))


def _buffer_facet(facet: dict, buffer_px: float) -> dict:
    """Return a copy of `facet` whose polygon is grown outward by `buffer_px`
    pixels. Coordinates stay in [x, y] = [col, row] pixel space, so buffering
    in pixel units is exact."""
    coords = facet.get('concave_hull_coords_simplified')
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = poly.buffer(0)
    grown = poly.buffer(buffer_px)
    # A positive buffer of a single polygon is a Polygon; guard the rare
    # MultiPolygon by taking the largest part.
    if grown.geom_type == 'MultiPolygon':
        grown = max(grown.geoms, key=lambda g: g.area)
    return {
        **facet,
        'concave_hull_coords_simplified': [list(xy) for xy in grown.exterior.coords],
    }


def merge_new_facets_accumulation(
    latitude: float,
    longitude: float,
    facets: list,
    annual_flux_url: str,
    monthly_flux_url: str,
    weather_data: dict,
    dsm: 'Optional[str | np.ndarray]' = None,
    dsm_metadata: Optional[dict] = None,
    facet_buffer_px: float = FACET_BUFFER_PX,
    pixel_batch_size: int = _DEFAULT_PIXEL_BATCH_SIZE,
) -> dict:
    """
    Incrementally fold newly added facets into the existing annual + monthly flux
    rasters, only recomputing DC for facets that fall outside the already
    accumulated (house-bbox) region.

    Coverage test: the existing annual raster holds real values exactly where DC
    was already accumulated and NODATA elsewhere, so a facet is "already covered"
    iff every pixel it rasterizes to is non-NODATA. Facets that are fully covered
    are skipped; the rest are grown by `facet_buffer_px` and accumulated against
    the cached terrain (`terrain_url`), then spliced into the existing rasters.

    Returns
    -------
    dict:
      * changed: bool -- False when every facet was already covered (nothing
        recomputed; the caller keeps the existing URLs untouched)
      * recomputed_facet_ids: list[str]
      * annual_dc_tiff_b64 / monthly_dc_tiff_b64: merged rasters (only when
        changed) for the caller to upload + persist
    """
    # Read the annual + monthly rasters concurrently -- each _read_raster call
    # downloads its GeoTIFF over the network, so running them in parallel avoids
    # paying the two downloads back-to-back.
    with ThreadPoolExecutor(max_workers=2) as pool:
        annual_future = pool.submit(_read_raster, annual_flux_url)
        monthly_future = pool.submit(_read_raster, monthly_flux_url)
        annual_arr, annual_profile, annual_nodata = annual_future.result()
        monthly_arr, monthly_profile, _ = monthly_future.result()
    H, W = annual_arr.shape[-2], annual_arr.shape[-1]

    # Snapshot which existing-raster pixels are empty BEFORE splicing. Terrain
    # and the final splice are both confined to these pixels, so previously
    # accumulated (covered) pixels are never re-computed or overwritten.
    existing_nodata = _is_nodata(annual_arr, annual_nodata)

    # Split facets into already-covered vs needs-recompute (buffered), tracking
    # the bbox of the still-empty pixels that actually drive a recompute.
    to_recompute: list = []
    recomputed_ids: list = []

    for facet in facets:
        coords = facet.get('concave_hull_coords_simplified')
        if not coords:
            continue
        rr, cc = _facet_pixels(coords, H, W)
        if rr.size == 0:
            # Polygon falls entirely outside the DSM grid -- nothing to accumulate.
            continue
        facet_nodata = existing_nodata[rr, cc]
        if not facet_nodata.any():
            # Every pixel already has a value -> facet is inside the accumulated
            # region, reuse the existing raster as-is.
            continue
        to_recompute.append(_buffer_facet(facet, facet_buffer_px))
        recomputed_ids.append(str(facet.get('id', len(recomputed_ids))))

    if not to_recompute:
        print(
            'MERGE_FACET_ACCUMULATION: all facets already covered -- no recompute',
            flush=True,
        )
        return {'changed': False, 'recomputed_facet_ids': []}

    print(
        f'MERGE_FACET_ACCUMULATION: recomputing {len(to_recompute)} facet(s) '
        f'outside the accumulated region (+{facet_buffer_px}px buffer), ',
        flush=True,
    )

    facet_result = run_facet_shading(
        latitude=latitude,
        longitude=longitude,
        facets=to_recompute,
        weather_data=weather_data,
        dsm=dsm,
        dsm_metadata=dsm_metadata,
        pixel_batch_size=pixel_batch_size,
        visualize=False,
    )

    new_annual = np.asarray(
        facet_result['annual_dc_tiff_data']['data'], dtype=np.float32
    )
    new_monthly = _decode_monthly_b64(facet_result['monthly_dc_tiff_b64'])

    if new_annual.shape != annual_arr.shape:
        raise ValueError(
            f'merge_new_facets_accumulation: recomputed raster {new_annual.shape} '
            f'does not match existing annual raster {annual_arr.shape} -- the '
            'terrain grid and the flux raster grid must be identical'
        )

    # Splice ONLY the pixels that were empty in the existing raster and now
    # carry a recomputed value. Restricting to `existing_nodata` protects the
    # already accumulated pixels: with terrain confined to the empty region,
    # any covered pixel of a buffered facet falls outside the ray-march and
    # would otherwise accumulate garbage (svf/horizon = 0) that must not be
    # written back over good data.
    fill_mask = existing_nodata & ~_is_nodata(
        new_annual, NODATA_SENTINEL_FOR_IRRADIANCE
    )
    annual_arr[fill_mask] = new_annual[fill_mask]
    if new_monthly.shape[-2:] == monthly_arr.shape[-2:]:
        monthly_arr[:, fill_mask] = new_monthly[:, fill_mask]
    else:
        raise ValueError(
            f'merge_new_facets_accumulation: recomputed monthly raster '
            f'{new_monthly.shape} misaligned with existing {monthly_arr.shape}'
        )

    annual_profile['nodata'] = NODATA_SENTINEL_FOR_IRRADIANCE
    monthly_profile['nodata'] = NODATA_SENTINEL_FOR_IRRADIANCE
    return {
        'changed': True,
        'recomputed_facet_ids': recomputed_ids,
        'annual_dc_tiff_b64': encode_tif_b64(
            annual_arr, annual_profile, units='kwh_per_kwp_per_year'
        ),
        'monthly_dc_tiff_b64': encode_tif_b64(
            monthly_arr, monthly_profile, units='kwh_per_kwp_per_year'
        ),
    }


def run_facet_accumulation_main(data: dict) -> dict:
    """Node-facing action entry point for the incremental facet merge."""
    return merge_new_facets_accumulation(
        latitude=data['latitude'],
        longitude=data['longitude'],
        facets=data['facets'],
        annual_flux_url=data['annualFluxUrl'],
        monthly_flux_url=data['monthlyFluxUrl'],
        weather_data=data['weatherData'],
        dsm=data.get('dsm'),
        dsm_metadata=data.get('dsm_metadata'),
        facet_buffer_px=data.get('facet_buffer_px', FACET_BUFFER_PX),
        pixel_batch_size=data.get('pixel_batch_size', _DEFAULT_PIXEL_BATCH_SIZE),
    )


def _debug_save_tmy(tmy: pd.DataFrame, latitude: float, longitude: float) -> None:
    """
    One-shot debug capture of the real TMY dataframe, gated behind an env
    var so it never fires in normal production traffic. Set
    SAVE_TMY_DEBUG=1 (and optionally SAVE_TMY_PATH) for exactly the
    request(s) you want to capture, then unset it again.

    Defaults to the same folder as this file (facet_accumulation.py),
    not the process's cwd and not /tmp -- so it's always right next to
    the source, wherever the process was actually launched from.
    """
    if os.environ.get('SAVE_TMY_DEBUG', '1') != '1':
        return
    try:
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'tmy_debug.pkl'
        )
        path = os.environ.get('SAVE_TMY_PATH', default_path)
        with open(path, 'wb') as f:
            pickle.dump({'tmy': tmy, 'latitude': latitude, 'longitude': longitude}, f)
        print(
            f'[SAVE_TMY_DEBUG] wrote {len(tmy)} rows to {os.path.abspath(path)}',
            flush=True,
        )
    except Exception as e:  # never break the real request over a debug dump
        print(f'[SAVE_TMY_DEBUG] failed: {e}', flush=True)
