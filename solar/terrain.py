"""
terrain.py — Terrain analysis module.

Responsibilities
----------------
1. fetch_horizon        — far-field PVGIS terrain horizon (360 azimuth buckets)
2. compute_slope_aspect — Horn (1981) 3x3 gradient
3. svf_and_horizon      — combined SVF + near-field horizon map (single ray-march pass)
4. TerrainResult        — typed dataclass returned to all callers

Shadow casting (build_shadow_stack, cast_shadow_batch) has been removed.
The per-pixel near-field horizon map stored in TerrainResult replaces the
shadow stack with an O(1) float16 lookup per pixel per timestep.

Optimisations
-------------
[PERF-1]  Single ray-march loop per direction.  SVF is derived analytically
          from the horizon_map result — no separate SVF ray-march.
          Formula: svf[px] = mean(1 - sin(atan(hor[px, :]))).
          Eliminates the old 32-dir SVF loop (was 32+72=104 dirs total).

[PERF-2]  Kernel allocates float32; Python wrapper casts to float16 after
          return.  Numba cannot allocate float16 arrays natively in @njit.

[PERF-3]  Default n_dirs=72 (5 degree az resolution).
          Memory: 72 x 1M x 2 bytes = 137 MB float16.
          Shadow edge shift at 5 deg az, el=7 deg, 1.5 m obstacle: ~10 px.
          Use n_dirs=180 for finer spatial accuracy.

[PERF-4]  float16 in TerrainResult; accumulation converts to float32 just
          before the Numba kernel call (Numba float16 arg restriction).

[PERF-6]  [LOAD-BALANCE-2026-07] `_horizon_kernel`'s outer `prange` is
          flattened over the FULL (row, col) region instead of splitting
          only by row. A region that is few-rows-tall but many-columns-
          wide (a plausible narrow facet/house-bbox rectangle) used to
          under-utilize available cores under a row-only prange -- e.g. a
          3-row region only ever used 3 threads no matter how many cores
          were free. Flattening `(r1-r0) x (c1-c0)` into one
          `range(n_region_px)` lets Numba divide the work evenly across
          however many pixels exist, for any region shape. Each pixel's
          horizon computation only reads the raw `dsm` (never another
          pixel's OUTPUT), so this reordering is bit-identical to the
          row-major version -- only the iteration order changes, never
          the result. Full-grid calls (thousands of rows) were never
          affected by the original bug; this only matters for the
          bbox/facet `region=` path.
"""

from __future__ import annotations

import json
import math
import time
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

import numba
import numpy as np
from numba import njit, prange

# constants
# If the tallest background horizon angle is less than 0.5°,
# we treat it as flat — not worth using.
# Basically "if the mountains are barely visible, ignore them."
_HORIZON_THRESHOLD_DEG: float = 0.5
# How far out (in meters) each ray will march.
# 35 meters is chosen as a safe distance for a 1000×1000 pixel map at 0.1 meters per pixel.
# Beyond this distance obstacles are too far to matter much.
_RADIUS: float = 35.0  # safe diagonal for 1000x1000 px at 0.1 m/px
_SVF_DIRS: int = 32  # kept for API compat - not used in ray-march [PERF-1]
_HOR_DIRS: int = 120  # 5 deg az resolution; use 180 for finer spatial maps


@dataclass
class TerrainResult:
    """
    All terrain-derived arrays for a single DSM.

    Produced by run_terrain_analysis() and consumed by downstream
    accumulation pipelines.

    Attributes
    ----------
    dsm         : (H, W) float32      -- cleaned DSM elevation in metres
    slope       : (H, W) float32      -- slope in radians from horizontal
    aspect      : (H, W) float32      -- aspect in radians, 0=N clockwise
    svf         : (H, W) float32      -- sky-view factor, 0 (blocked) to 1 (open)
    horizon     : (360,) float32      -- far-field PVGIS horizon elevation, radians
    horizon_map : (H*W, n_dirs) float16
        Per-pixel near-field horizon -- max tan(elevation) seen in each of
        n_dirs equally-spaced azimuth directions within svf_radius_m.
        Stored flattened so the accumulation kernel indexes directly as
        horizon_map[px, az_bucket] without reshaping.
        Shadow test: tan(sun_el) < horizon_map[px, az_bucket].
        SVF is derived from this array [PERF-1] -- no separate SVF ray-march.
    n_dirs      : int   -- number of azimuth directions in horizon_map
    cell_size   : float -- metres per pixel
    H, W        : int   -- DSM height and width in pixels
    """

    dsm: np.ndarray  # (H, W)        float32
    slope: np.ndarray  # (H, W)        float32 radians
    aspect: np.ndarray  # (H, W)        float32 radians, 0=N clockwise
    svf: np.ndarray  # (H, W)        float32 0 to 1
    # The far background horizon angles from PVGIS, one per compass degree.
    # Tells us how high distant mountains block the sky in each direction.
    horizon: np.ndarray  # (360,)        float32 far-field radians
    # The near-field horizon for every pixel in every direction.
    # Stored as tan values in float16 to save memory.
    # Used later for shadow testing — if sun elevation is less than this value, the pixel is in shadow.
    horizon_map: np.ndarray  # (H*W, n_dirs) float16 near-field tan(el)
    n_dirs: int
    cell_size: float
    H: int
    W: int


def fetch_horizon(lat: float, lon: float, timeout: int = 12) -> Tuple[np.ndarray, bool]:
    """
    Fetch the far-field terrain horizon from PVGIS.

    Returns
    -------
    horizon : (360,) float32 -- horizon elevation in radians per degree azimuth
    used    : bool           -- False if flat (PVGIS unavailable or lat > 60)
    """
    FLAT = np.zeros(360, dtype=np.float32)
    if lat > 60.0:
        return FLAT, False
    url = (
        f'https://re.jrc.ec.europa.eu/api/v5_2/printhorizon'
        f'?lat={lat}&lon={lon}&outputformat=json'
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            pts = json.loads(r.read())['outputs']['horizon_profile']

        # Extract azimuth angles and horizon elevations from the downloaded data
        az = np.array([p['A'] for p in pts], dtype=np.float64)
        h = np.array([math.radians(p['H_hor']) for p in pts], dtype=np.float64)

        # PVGIS doesn't give exactly one value per degree
        # — it gives irregularly spaced points.
        # So we interpolate to fill in all 360 integer degrees evenly.
        # period=360 handles the wraparound from 359° back to 0°.
        out = np.zeros(360, dtype=np.float32)
        for d in range(360):
            out[d] = float(np.interp(d, az, h, period=360))

        # If the tallest horizon angle is less than 0.5°,
        # the terrain is essentially flat.
        # Return zeros and False — not worth using this data.
        if float(np.degrees(out.max())) <= _HORIZON_THRESHOLD_DEG:
            return FLAT, False
        return out, True
    except Exception:
        return FLAT, False


def compute_slope_aspect(
    dsm: np.ndarray,
    cell_size: float,
    region: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Horn (1981) 3x3 finite-difference gradient.
    fits a 3×3 neighbourhood of elevation values around each pixel
    using weighted finite differences.
    The gradient in x and y (dz_dx, dz_dy) gives
    the slope magnitude via arctan(hypot(...))
    and the aspect via arctan2.
    Weights are 2× for the cardinal neighbours,
    1× for diagonals — that's the Horn weighting.
    Cost is O(H×W) with region=None, purely vectorised NumPy.

    Parameters
    ----------
    dsm       : (H, W) array
    cell_size : metres per pixel
    region    : optional (r0, r1, c0, c1) output pixel rectangle -- same
                convention as terrain.svf_and_horizon's `region`. None =
                full grid. When given, only the 3x3 neighbourhood needed
                to fill that rectangle is read (dsm sliced to the region
                + 1px halo, clamped to the DSM edge), so a small
                house-bbox/facet rectangle on a huge DSM costs O(region)
                instead of O(H×W). Only immediate neighbours are ever
                needed (unlike the ray-march's `max_r`), so this is
                bit-identical to the full-grid pass for every pixel inside
                the region; pixels outside the region are left 0, matching
                svf_and_horizon's convention.

    Returns
    -------
    slope  : (H, W) float32 -- radians from horizontal
    aspect : (H, W) float32 -- radians, 0=North, clockwise
    """
    H, W = dsm.shape
    if region is None:
        r0, r1, c0, c1 = 0, H, 0, W
    else:
        r0, r1, c0, c1 = region
        r0 = max(0, min(int(r0), H))
        r1 = max(0, min(int(r1), H))
        c0 = max(0, min(int(c0), W))
        c1 = max(0, min(int(c1), W))
        if r1 <= r0 or c1 <= c0:  # empty/degenerate -> fall back to full grid
            r0, r1, c0, c1 = 0, H, 0, W

    full_region = (r0, r1, c0, c1) == (0, H, 0, W)

    # Slice the DSM down to just the region (+1px halo on every side, clamped
    # to the DSM edge) before padding -- the Horn kernel below only ever
    # reads a pixel's immediate 8 neighbours, so this window is all it needs.
    sr0, sr1 = max(0, r0 - 1), min(H, r1 + 1)
    sc0, sc1 = max(0, c0 - 1), min(W, c1 + 1)
    dsm_win = dsm[sr0:sr1, sc0:sc1]

    # Add a 1-pixel border around the window by copying edge values outward.
    # This way every pixel (including edge pixels) has a full 3×3 neighbourhood to work with.
    # Without this, edge pixels would have no neighbours on one side.
    z = np.pad(dsm_win, 1, mode='edge').astype(np.float32)

    a = z[:-2, :-2]  # top-left neighbour
    b = z[:-2, 1:-1]  # top-centre neighbour
    c_ = z[:-2, 2:]  # top-right neighbour
    d = z[1:-1, :-2]  # middle-left neighbour
    f = z[1:-1, 2:]  # middle-right neighbour
    g = z[2:, :-2]  # bottom-left neighbour
    h = z[2:, 1:-1]  # bottom-centre neighbour
    i = z[2:, 2:]  # bottom-right neighbour

    # Slope in the East-West direction - (x)  North-South direction(y).
    # The Horn method weights the cardinal neighbours (f and d) twice as much as diagonal ones (c, i, a, g).
    # Dividing by 8×cell_size converts pixel differences to a proper gradient (rise over run in meters).
    dz_dx = ((c_ + 2 * f + i) - (a + 2 * d + g)) / (8 * cell_size)
    dz_dy = ((g + 2 * h + i) - (a + 2 * b + c_)) / (8 * cell_size)

    # Combine the East-West and North-South gradients into one overall slope angle.
    # hypot computes the magnitude of the 2D gradient vector.
    # arctan converts rise-over-run to an angle in radians.
    slope_win = np.arctan(np.hypot(dz_dx, dz_dy))

    # Calculate which direction the slope faces.
    # arctan2 gives the mathematical angle,
    # but we need compass bearing (0=North, clockwise).
    # The π/2 - part rotates from math convention (0=East, counterclockwise)
    # to compass convention (0=North, clockwise).
    # The % 2π keeps it in the range 0 to 2π.
    aspect_win = (np.pi / 2 - np.arctan2(dz_dy, -dz_dx)) % (2 * np.pi)

    # Completely flat pixels have no meaningful aspect direction.
    # Set them to 0 (North) by convention to avoid garbage values from near-zero gradients.
    aspect_win[slope_win < 1e-6] = 0.0
    slope_win = slope_win.astype(np.float32)
    aspect_win = aspect_win.astype(np.float32)

    if full_region:
        return slope_win, aspect_win

    # Scatter the region's rows/cols back into full-size (H, W) outputs,
    # zero elsewhere -- same "outside the region stays 0" convention as
    # svf_and_horizon, and keeps the flat-index (row*W+col) contract that
    # downstream `terrain.slope.ravel()[union_px]` relies on.
    slope = np.zeros((H, W), dtype=np.float32)
    aspect = np.zeros((H, W), dtype=np.float32)
    slope[r0:r1, c0:c1] = slope_win[r0 - sr0 : r1 - sr0, c0 - sc0 : c1 - sc0]
    aspect[r0:r1, c0:c1] = aspect_win[r0 - sr0 : r1 - sr0, c0 - sc0 : c1 - sc0]
    return slope, aspect


@njit(parallel=True, fastmath=True, cache=True)
def _horizon_kernel(
    dsm: np.ndarray,  # (H, W) float32
    cell_size: float,
    far_horizon: np.ndarray,  # (360,) float32
    n_dirs: int,
    max_r: int,
    r0: int,  # target region row start (inclusive)
    r1: int,  # target region row end   (exclusive)
    c0: int,  # target region col start (inclusive)
    c1: int,  # target region col end   (exclusive)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [PERF-1] Single row-parallel ray-march: SVF and horizon map in one pass.

    Only OUTPUT pixels in the rectangle [r0:r1, c0:c1] are computed -- the
    rest of svf_out/hor_out stay zero. Rays still read the FULL dsm for
    occlusion (obstacles outside the region up to max_r are honoured), so the
    horizon_map / svf for every computed pixel is bit-identical to the
    full-grid pass. The region is the bbox/facet pixel bounds; everything the
    accumulation consumes (union_px) lies inside it. Flat indexing px = i*W+j
    is preserved so downstream `horizon_map[union_px]` is unchanged.

    [PERF-6] [LOAD-BALANCE-2026-07] The outer `prange` is flattened over the
    ENTIRE region (all (r1-r0)*(c1-c0) pixels in one 1-D range), not just
    over rows. A row-only `prange(r0, r1)` means a region only ever gets as
    many active threads as it has rows -- a narrow-but-wide facet/house-bbox
    rectangle (e.g. 3 rows x 2000 cols) would only use 3 threads regardless
    of how many cores are free. Flattening removes that ceiling: Numba
    divides the single flat range evenly across all configured threads no
    matter the region's aspect ratio. Each pixel's horizon computation only
    reads the raw `dsm` input (never another pixel's computed OUTPUT), so
    this reordering changes nothing about the result -- only which thread
    computes which pixel, and in what order. Full-grid calls (thousands of
    rows) were already well-parallelized under the old row-only split; this
    only changes behavior for the bbox/facet `region=` path.

    For each pixel, marches in n_dirs directions tracking max tan(elevation).
    SVF is accumulated in the same loop -- no extra ray-march cost.

    Walk outward step s = 1, 2, 3 ... up to max_r pixels
    At each step compute dz / (s × cell_size) —
        this is tan(elevation angle) to that obstacle
    Track only the best (maximum) value seen
    Early exit:
        if (dsm_max − z0) / (s × cell_size) ≤ best,
        no future obstacle can ever beat the current winner, so break.
        This is the critical speedup — in open terrain it exits very early
    Step coarsening: after s=20, step doubles, then quadruples etc.
    At large distance the angular error introduced by skipping pixels is negligible (1 pixel / large s ≈ same angle as 2 pixels / 2×s),
    so spatial accuracy is preserved while cutting iterations

    Returns
    -------
    svf_out : (H, W)        float32
    hor_out : (H*W, n_dirs) float32 -- cast to float16 by Python wrapper
    """
    H, W = dsm.shape
    n_px = H * W

    # will hold the tallest elevation on the map, start at negative infinity
    dsm_max = np.float32(-1e30)
    for _i in range(H):
        for _j in range(W):
            if dsm[_i, _j] > dsm_max:
                dsm_max = dsm[_i, _j]

    # these three arrays will hold one value per ray direction,
    # filled in the loop below

    # how many rows to move per step in each direction
    dr_arr = np.empty(n_dirs, dtype=numba.float64)
    # how many columns to move per step in each direction
    dc_arr = np.empty(n_dirs, dtype=numba.float64)
    # how high is the sky already blocked in each direction
    ff_arr = np.empty(n_dirs, dtype=numba.float64)

    for k in range(n_dirs):
        az = k * 2.0 * math.pi / n_dirs
        az_deg_f = (
            math.degrees(az) % 360.0
        )  # far_horizon lookup table is indexed by integer degrees (0 to 359).
        az0 = int(az_deg_f) % 360  # integer degree just below the angle
        az1 = (az0 + 1) % 360  # integer degree just above the angle
        frac = az_deg_f - int(az_deg_f)  # how far between az0 and az1 is az_deg_f

        # blend between the two nearest background horizon values to get a smooth angle for this exact direction
        ff_h = float(far_horizon[az0]) * (1.0 - frac) + float(far_horizon[az1]) * frac
        # convert background horizon angle to tan value, store it ready for comparison inside ray loop
        # if no background horizon exists (angle = 0), store 0
        ff_arr[k] = math.tan(ff_h) if ff_h > 0.0 else 0.0

        # Convert the angle az into row and column movement per step.
        # Think of a compass:
        # Pointing North (az=0): move -1 row (upward in the grid),
        # 0 columns → cos(0)=1 so dr=-1, sin(0)=0 so dc=0
        # Pointing East (az=90°): move 0 rows,
        # +1 column → cos(90°)=0 so dr=0, sin(90°)=1 so dc=1
        dr_arr[k] = -math.cos(az)
        dc_arr[k] = math.sin(az)

    # output sky view factor grid, one value per pixel, starts at 0
    svf_out = np.zeros((H, W), dtype=numba.float32)
    # output horizon angle grid, one value per pixel per direction
    hor_out = np.zeros((n_px, n_dirs), dtype=numba.float32)
    # precompute 1/n_dirs so we multiply instead of divide inside the loop
    inv = 1.0 / n_dirs

    # ── flattened region iteration ─────────────────────────────
    # region_w columns per row -> flat index decomposes as (i, j) = (r0 +
    # flat // region_w, c0 + flat % region_w). This is the only change from
    # the old `for i in prange(r0, r1): for j in range(c0, c1):` structure --
    # everything inside the per-pixel body below is untouched.
    region_h = r1 - r0
    region_w = c1 - c0
    n_region_px = region_h * region_w

    for flat in prange(n_region_px):
        i = r0 + flat // region_w
        j = c0 + flat % region_w

        z0 = dsm[i, j]
        px = i * W + j  # flat 1D index of this pixel, used to write into hor_out
        acc = 0.0  # accumulates sky visibility across all directions, divided by n_dirs at the end
        for k in range(n_dirs):
            dr = dr_arr[k]
            dc = dc_arr[k]
            best = ff_arr[
                k
            ]  # steepest horizon angle seen so far, starts at background horizon floor
            s = 1  # current distance along the ray in pixels, starts at 1
            step = 1  # how many pixels to jump each iteration, starts at 1, grows larger further out
            while s <= max_r:
                # early exit: even if the tallest point on the entire map were at distance s
                # its angle still cant beat our current best, so nothing further out can either, stop
                if (dsm_max - z0) / (s * cell_size) <= best:
                    break
                ii = int(round(i + dr * s))
                jj = int(round(j + dc * s))

                # if we have walked off the edge of the map, stop this ray
                if ii < 0 or ii >= H or jj < 0 or jj >= W:
                    break

                # height difference between sampled pixel and our standing height
                dz = dsm[ii, jj] - z0
                if dz > 0.0:
                    # tan of elevation angle to this obstacle,
                    # This is simply rise (dz) over run (s * cell_size = distance in meters).
                    t = dz / (s * cell_size)
                    if (
                        t > best
                    ):  # if this obstacle is steeper than anything seen so far
                        best = t  # update best

                s += step  # move to next sample point along the ray

                # step coarsening: the further out we are the bigger the jumps
                # close up check every pixel, far out skip more and more
                # angular error from skipping is negligible at large distances
                if s > 160:
                    step = 16  # beyond 160 pixels, jump 16 at a time
                elif s > 80:
                    step = 8  # beyond 80 pixels, jump 8 at a time
                elif s > 40:
                    step = 4  # beyond 40 pixels, jump 4 at a time
                elif s > 20:
                    step = 2  # beyond 20 pixels, jump 2 at a time

            # store the final steepest horizon angle for this direction
            hor_out[px, k] = np.float32(best)
            # [PERF] algebraic sin(atan(x)) = x / sqrt(1+x^2) -- avoids
            # a transcendental sin+atan pair per (pixel, direction) in
            # favour of one sqrt; verified bit-exact to double precision.
            acc += 1.0 - (best / math.sqrt(1.0 + best * best))

        # average sky visibility across all directions
        v = acc * inv
        # store sky view factor, cap at 1.0 in case of floating point drift
        svf_out[i, j] = v if v <= 1.0 else 1.0

    # return sky view factor map and horizon angle map
    return svf_out, hor_out


def svf_and_horizon(
    dsm: np.ndarray,
    cell_size: float,
    n_dirs: int = _HOR_DIRS,
    max_radius_m: float = _RADIUS,
    horizon: Optional[np.ndarray] = None,
    region: Optional[Tuple[int, int, int, int]] = None,
    use_gpu: Optional[bool] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-pixel SVF and near-field horizon map in a single pass.

    Parameters
    ----------
    dsm          : (H, W) float32
    cell_size    : metres per pixel
    svf_dirs     : ignored (kept for API compat) [PERF-1]
    n_dirs       : azimuth directions (72=5 deg az, 180=2 deg az)
    max_radius_m : ray-march cutoff in metres
    horizon      : (360,) far-field PVGIS horizon; zeros if None
    region       : (r0, r1, c0, c1) output pixel rectangle to compute; None =
                   full grid. Rays still read the full dsm for occlusion, so
                   computed pixels are bit-identical to the full-grid pass;
                   pixels outside the region stay zero.
    use_gpu      : force the CUDA ray-march (True), force CPU (False), or auto-
                   detect (None -> use GPU when a CUDA device is available).
                   GPU results are numerically equivalent, not bit-identical.

    Returns
    -------
    svf         : (H, W)        float32 -- sky-view factor per pixel
    horizon_map : (H*W, n_dirs) float16 -- max tan(elevation) per direction
    """
    # ── GPU dispatch (CUDA ray-march); CPU @njit path is the fallback ─────
    from . import gpu_kernels, perf

    if use_gpu is None:
        use_gpu = gpu_kernels.GPU_AVAILABLE

    print(f'GPU AVAILABLE {gpu_kernels.GPU_AVAILABLE}', flush=True)

    if use_gpu and gpu_kernels.GPU_AVAILABLE:
        with perf.stage('horizon+svf ray-march'):
            return gpu_kernels.svf_and_horizon_gpu(
                dsm, cell_size, n_dirs, max_radius_m, horizon, region
            )

    dsm_ = np.ascontiguousarray(dsm, dtype=np.float32)
    H, W = dsm_.shape

    # If no horizon was provided, use flat zeros (no background blocking).
    hz = (
        np.ascontiguousarray(horizon, dtype=np.float32)
        if horizon is not None
        else np.zeros(360, dtype=np.float32)
    )

    # Clamp the output region to the grid; None -> whole grid.
    if region is None:
        r0, r1, c0, c1 = 0, H, 0, W
    else:
        r0, r1, c0, c1 = region
        r0 = max(0, min(int(r0), H))
        r1 = max(0, min(int(r1), H))
        c0 = max(0, min(int(c0), W))
        c1 = max(0, min(int(c1), W))
        if r1 <= r0 or c1 <= c0:  # empty/degenerate -> fall back to full grid
            r0, r1, c0, c1 = 0, H, 0, W

    # Convert the max radius from meters to pixels
    max_r = max(1, int(round(max_radius_m / cell_size)))
    with perf.stage('horizon+svf ray-march'):
        svf, hor = _horizon_kernel(dsm_, cell_size, hz, n_dirs, max_r, r0, r1, c0, c1)
    return svf, hor.astype(np.float16)


def _warmup_svf_jit(cell_size: float, n_dirs: int = _HOR_DIRS) -> None:
    """Force Numba to compile _horizon_kernel before the timed terrain run."""
    tiny = np.ones((16, 16), dtype=np.float32) + np.random.rand(16, 16).astype(
        np.float32
    )
    hz = np.zeros(360, dtype=np.float32)
    _horizon_kernel(tiny, cell_size, hz, min(n_dirs, 8), 24, 0, 16, 0, 16)


def horizon_beam_blocking(
    el_rad: np.ndarray,
    az_rad: np.ndarray,
    horizon: np.ndarray,
) -> np.ndarray:
    """
    is the sun higher or lower than the terrain at that compass direction?
    -----------------------------------------------------------------------

    Sun at 10° elevation, terrain horizon at 5°
        → sun clears the terrain → no blocking → full beam hits the panel
    Sun at 3° elevation, terrain horizon at 5°
        → sun is behind the mountain → blocked → no direct beam that hour
    ------------------------------------------------------------------------

    Per-hour far-field beam-blocking flag with bilinear azimuth interpolation.

    Parameters
    ----------
    el_rad  : (n_hours,) float64 -- apparent solar elevation in radians
    az_rad  : (n_hours,) float64 -- solar azimuth in radians
    horizon : (360,)    float32  -- far-field horizon elevation in radians

    Returns
    -------
    (n_hours,) float32 -- 1.0 where sun elevation < horizon angle, else 0.0
    """
    # Convert solar azimuth from radians to degrees for all hours at once.
    az_deg_f = np.degrees(az_rad) % 360.0
    az0 = np.floor(az_deg_f).astype(np.intp) % 360
    az1 = (az0 + 1) % 360
    frac = (az_deg_f - np.floor(az_deg_f)).astype(np.float32)

    # Interpolate the background horizon angle at the exact solar azimuth
    # for each hour.
    h_interp = horizon[az0] * (1.0 - frac) + horizon[az1] * frac

    # Compare sun elevation to terrain horizon.
    # If sun is lower than the horizon angle, it is blocked (1.0).
    # If sun is higher, it clears the terrain (0.0).
    # Returns an array of 1s and 0s, one per hour.
    return (el_rad < h_interp).astype(np.float32)


def run_terrain_analysis(
    dsm_raw: np.ndarray,
    cell_size: float,
    latitude: float,
    longitude: float,
    nodata: Optional[float] = None,
    svf_radius_m: float = _RADIUS,
    hor_dirs: int = _HOR_DIRS,
    t0: Optional[float] = None,
    region: Optional[Tuple[int, int, int, int]] = None,
) -> TerrainResult:
    """
    Full terrain analysis pipeline for one DSM.

    Stages
    ------
    1. Clean nodata pixels (nanmedian fill).
    2. JIT warm-up before network I/O.
    3. Fetch far-field horizon from PVGIS.
    4. Horn (1981) slope + aspect.
    5. Combined SVF + horizon map (single row-parallel Numba pass).

    Parameters
    ----------
    dsm_raw      : (H, W) array -- raw DSM values (cast to float32 internally)
    cell_size    : metres per pixel
    latitude     : decimal degrees
    longitude    : decimal degrees
    nodata       : sentinel value replaced by nanmedian if present
    svf_dirs     : ignored; kept for API compat [PERF-1]
    svf_radius_m : ray-march radius in metres (35 m safe for 100 m tile)
    hor_dirs     : horizon map directions (72=5 deg az, 180=2 deg az)
    t0           : pipeline start time for log timestamps
    region       : optional (r0, r1, c0, c1) output pixel rectangle for BOTH
                   the SVF + horizon-map ray-march AND the Horn slope/aspect
                   gradient. None = full grid. When given, only those output
                   pixels are computed (the ray-march still reads the full
                   dsm out to max_r for occlusion, and slope/aspect still
                   read a 1px halo around the region, so computed pixels are
                   bit-identical to the full-grid pass either way); pixels
                   outside the region are left zero in slope, aspect, AND
                   svf/horizon_map. Use the bbox/facet pixel bounds so every
                   consumed union pixel lies inside it. As of [PERF-6], the
                   ray-march parallelizes over the FULL region (not just its
                   rows), so narrow-but-wide regions use all available cores
                   same as any other shape. Slope/aspect are cheap vectorised
                   NumPy either way, but skipping the full H×W grid still
                   avoids allocating/computing arrays far larger than the
                   region actually needed (relevant on a large DSM with a
                   small house-bbox region).

                   CAVEAT: a region-limited TerrainResult is only valid for
                   that region -- do NOT cache/reuse it for a different facet
                   set. (The terrain_url reuse path is currently disabled; if
                   re-enabled, compute full-grid or key the cache by region.)

    Returns
    -------
    TerrainResult
        All terrain arrays ready for downstream accumulation.
        horizon_map is (H*W, hor_dirs) float16.
    """
    if t0 is None:
        t0 = time.time()

    dsm = np.asarray(dsm_raw, dtype=np.float32)
    if nodata is not None:
        bad = dsm == nodata
        if bad.any():
            dsm[bad] = float(np.nanmedian(dsm[~bad]))

    H, W = dsm.shape
    print(
        f'[{time.time() - t0:5.1f}s] DSM {H}x{W} px  cell={cell_size:.4f} m  '
        f'elev {dsm.min():.1f}-{dsm.max():.1f} m',
        flush=True,
    )

    from . import gpu_kernels, perf

    # Warm-up is timed separately from the ray-march: on GPU it's the per-process
    # PTX compile, on CPU the (disk-cached) Numba JIT -- comparing the two
    # backends' real work means keeping this out of the kernel numbers.
    if gpu_kernels.GPU_AVAILABLE:
        print(f'[{time.time() - t0:5.1f}s] Horizon CUDA warm-up ...', flush=True)
        with perf.stage('kernel warm-up (compile)'):
            gpu_kernels.warmup_gpu(cell_size, hor_dirs)
    else:
        print(f'[{time.time() - t0:5.1f}s] Horizon JIT warm-up ...', flush=True)
        with perf.stage('kernel warm-up (compile)'):
            _warmup_svf_jit(cell_size, hor_dirs)

    print(f'[{time.time() - t0:5.1f}s] Far-field horizon (PVGIS) ...', flush=True)
    horizon, used_pvgis = fetch_horizon(latitude, longitude)
    if used_pvgis:
        print(
            f'[{time.time() - t0:5.1f}s] Horizon: max={np.degrees(horizon.max()):.2f} deg',
            flush=True,
        )
    else:
        print(
            f'[{time.time() - t0:5.1f}s] Horizon: flat (PVGIS unavailable)', flush=True
        )

    print(f'[{time.time() - t0:5.1f}s] Slope + aspect ...', flush=True)
    slope, aspect = compute_slope_aspect(dsm, cell_size, region=region)

    hor_mb = H * W * hor_dirs * 2 // (1024 * 1024)
    print(
        f'[{time.time() - t0:5.1f}s] Horizon map + SVF '
        f'({hor_dirs} dirs, r={svf_radius_m} m, ~{hor_mb} MB) ...',
        flush=True,
    )

    svf, horizon_map = svf_and_horizon(
        dsm,
        cell_size,
        n_dirs=hor_dirs,
        max_radius_m=svf_radius_m,
        horizon=horizon,
        region=region,
    )
    print(
        f'[{time.time() - t0:5.1f}s] Done  svf mean={svf.mean():.3f}  min={svf.min():.3f}',
        flush=True,
    )

    return TerrainResult(
        dsm=dsm,
        slope=slope,
        aspect=aspect,
        svf=svf,
        horizon=horizon,
        horizon_map=horizon_map,
        n_dirs=hor_dirs,
        cell_size=cell_size,
        H=H,
        W=W,
    )