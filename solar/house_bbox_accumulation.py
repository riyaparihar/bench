"""
house_bbox_accumulation.py -- DC accumulation over the house bounding box + margin.

Motivation
----------
`facet_accumulation.py` accumulates DC yield per roof *facet*, so it can only
start once every roof-segment polygon has been extracted. But for a whole-house
shading picture we only need one rectangular region: the house bounding box
(the bounds of the roof outer-boundary polygon) grown by a small margin so the
surroundings that cast shadows onto the roof are included.

That region is known the instant the outer boundary is computed -- well before
any per-facet work -- so this pipeline can be kicked off much earlier and overlap
the roof-segment / edge / setback stages.

Reuse
-----
A rectangle is just a 4-corner polygon, and the facet machinery already knows how
to rasterize an arbitrary polygon into DSM pixels, run the shared Perez + Faiman
`_poa_accum_kernel` over the union of those pixels, and encode the sparse result
raster to base64 GeoTIFFs. So instead of duplicating the kernel loop and the
terrain-reuse/persist orchestration, this module builds the expanded bbox as a
single-facet list and delegates straight to `run_facet_shading`.

`pixel_batch_size` below is passed straight through to
facet_accumulation.run_facet_accumulation -- see accumulation_kernels.py's
module docstring for the pixel_batch_size vs hour_chunk distinction. This
module never touches hour_chunk itself; that's derived downstream, per
pixel batch, inside run_facet_accumulation.

Output (identical shape to the facet / full-DSM pipelines, so nothing downstream
changes):
  * annual_dc_tiff_b64   -- annual DC GeoTIFF, base64 (kWh/kWp/year)
  * monthly_dc_tiff_b64  -- 12-band monthly DC GeoTIFF, base64
  * annual_dc_tiff_data  -- {'data': ndarray, 'metadata': profile} (Tiff_Data_Result)
  * terrainLocalPath     -- fresh-terrain npz path for Node to upload (fresh path only)
"""

from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
from shapely import Polygon, box

from .constants import FACET_BUFFER_PX, HOUSE_BBOX_EXPAND_PCT
from .facet_accumulation import (
    _DEFAULT_PIXEL_BATCH_SIZE,
    _buffer_facet,
    run_facet_shading,
)

# Bounds ordering used throughout: (min_x, min_y, max_x, max_y) in DSM pixel
# space, where x = column and y = row. This matches
# shapely Polygon.bounds and the [x, y] = [col, row] vertex convention that
# build_facet_pixel_table expects for `concave_hull_coords_simplified`.


def expand_bbox(
    bounds: 'tuple[float, float, float, float] | list[float]',
    expand_pct: float = HOUSE_BBOX_EXPAND_PCT,
) -> tuple[float, float, float, float]:
    """
    Grow a (min_x, min_y, max_x, max_y) bounding box by `expand_pct` of its
    width/height on every side.

    e.g. expand_pct=0.10 adds a 10% margin on each edge, so a 100x60 box becomes
    120x72 centered on the same point. Not clamped here -- `skimage.draw.polygon`
    clips to the raster shape downstream, so out-of-range corners are harmless.
    """
    min_x, min_y, max_x, max_y = bounds
    pad_x = (max_x - min_x) * expand_pct
    pad_y = (max_y - min_y) * expand_pct
    return (min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y)


def bbox_to_rect_facet(
    bounds: 'tuple[float, float, float, float] | list[float]',
) -> dict:
    """
    Build a single "facet" dict whose polygon is the rectangle described by
    `bounds`, ready to hand to build_facet_pixel_table via run_facet_shading.

    Vertices are [x, y] = [col, row], clockwise from the top-left corner.
    """
    min_x, min_y, max_x, max_y = bounds
    return {
        'id': 'house_bbox',
        'concave_hull_coords_simplified': [
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y],
        ],
    }


def _capture_bench_input(**kwargs) -> None:
    """
    One-shot capture of a REAL run_house_bbox_shading call, gated behind an
    env var so it never fires in normal traffic. Set CAPTURE_BENCH_INPUT=1
    (and optionally CAPTURE_BENCH_PATH) for exactly the request you want to
    snapshot, then unset it again -- mirrors the SAVE_TMY_DEBUG pattern in
    facet_accumulation.py.

    Pickles the *exact* kwargs this function received -- same lat/lng, same
    house_bounding_box, same weather_data object, same already-resolved dsm
    ndarray + dsm_metadata -- so a later benchmark run replays production
    input byte-for-byte instead of a synthetic approximation. The real
    request still proceeds normally after the dump; this never blocks or
    alters behavior.
    """
    if os.environ.get('CAPTURE_BENCH_INPUT') != '1':
        return
    try:
        default_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'bench_input.pkl'
        )
        path = os.environ.get('CAPTURE_BENCH_PATH', default_path)
        with open(path, 'wb') as f:
            pickle.dump(kwargs, f)
        print(
            f'[CAPTURE_BENCH_INPUT] wrote run_house_bbox_shading input to '
            f'{os.path.abspath(path)}',
            flush=True,
        )
    except Exception as e:  # never break the real request over a debug dump
        print(f'[CAPTURE_BENCH_INPUT] failed: {e}', flush=True)


def run_house_bbox_shading(
    latitude: float,
    longitude: float,
    house_bounding_box: tuple[float, float, float, float] | list[float],
    weather_data: dict,
    dsm: Optional[str | np.ndarray] = None,
    dsm_metadata: Optional[dict] = None,
    expand_pct: float = HOUSE_BBOX_EXPAND_PCT,
    pixel_batch_size: int = _DEFAULT_PIXEL_BATCH_SIZE,
    visualize: bool = False,
) -> dict:
    """
    Run shadow-aware DC accumulation over the house bounding box grown by
    `expand_pct`, returning the same dict shape as run_facet_shading.

    Terrain handling (reuse vs fresh-compute + persist), solar precompute, the
    physics kernel and TIFF encoding are all delegated to run_facet_shading with
    a single rectangle "facet" -- see facet_accumulation.run_facet_shading for
    the terrain_url / dsm contract and the returned keys.

    hour_chunk (hours per kernel call) is never a parameter here: it's derived
    inside run_facet_accumulation, per pixel batch, from that batch's own
    pixel count (see resolve_hour_chunk in accumulation_kernels.py) -- a house
    bbox is small, so it typically collapses to the whole year in one Numba
    launch, with no manual tuning needed. `pixel_batch_size` (RAM-bounding,
    independent axis) is the only chunking knob exposed here.
    """
    _capture_bench_input(
        latitude=latitude,
        longitude=longitude,
        house_bounding_box=house_bounding_box,
        weather_data=weather_data,
        dsm=dsm,
        dsm_metadata=dsm_metadata,
        expand_pct=expand_pct,
        pixel_batch_size=pixel_batch_size,
    )

    if house_bounding_box is None:
        raise ValueError('run_house_bbox_shading: house_bounding_box is required')

    expanded = expand_bbox(house_bounding_box, expand_pct)
    print(
        f'HOUSE_BBOX_SHADING: bounds={tuple(house_bounding_box)} '
        f'expand_pct={expand_pct} -> {expanded}',
        flush=True,
    )
    rect_facet = bbox_to_rect_facet(expanded)

    return run_facet_shading(
        latitude=latitude,
        longitude=longitude,
        facets=[rect_facet],
        weather_data=weather_data,
        dsm=dsm,
        dsm_metadata=dsm_metadata,
        pixel_batch_size=pixel_batch_size,
        visualize=visualize,
    )


# ============================================================
# HOUSE BBOX + OUTLIER FACETS -- single-pass initial accumulation
# ============================================================
#
# The house-bbox pipeline (house_bbox_accumulation.run_house_bbox_shading)
# accumulates DC over just the expanded house bounding box. Any roof facet that
# sticks out past that rectangle (e.g. a detached garage, a wing the bbox margin
# didn't reach) has NODATA pixels there and would later have to be folded in by
# merge_new_facets_accumulation.
#
# This method does both up front in ONE accumulation pass: it treats the expanded
# bbox as one big region AND adds every facet that is NOT fully inside that bbox
# as its own (buffered) region, so the initial rasters already cover the bbox plus
# every out-of-box facet -- no follow-up merge needed for those.


def build_bbox_with_outlier_facets(
    house_bounding_box: 'tuple[float, float, float, float] | list[float]',
    facets: list,
    expand_pct: float = HOUSE_BBOX_EXPAND_PCT,
    facet_buffer_px: float = FACET_BUFFER_PX,
) -> tuple[list, tuple, list]:
    """
    Build the region list for a single-pass initial accumulation: the expanded
    house bounding box as one region, plus every facet NOT fully contained in
    that bbox, each grown outward by `facet_buffer_px` pixels.

    A facet already inside the expanded bbox is skipped -- the bbox region
    already covers its pixels, so accumulating it again would be redundant.

    Returns
    -------
    (regions, expanded_bounds, outlier_facet_ids)
      * regions            : [rect_facet, *buffered_outlier_facets] ready for
                             run_facet_shading
      * expanded_bounds    : the expanded (min_x, min_y, max_x, max_y) box
      * outlier_facet_ids  : ids of the facets that landed outside the bbox
    """

    bbox_poly = None
    regions: list = []
    outlier_ids: list = []
    expanded = None

    if house_bounding_box is not None:
        expanded = expand_bbox(house_bounding_box, expand_pct)
        bbox_poly = box(*expanded)  # (min_x, min_y, max_x, max_y) in [col, row] space
        regions.append(bbox_to_rect_facet(expanded))

    for idx, facet in enumerate(facets):
        coords = facet.get('concave_hull_coords_simplified')
        if not coords or len(coords) < 3:
            continue  # no usable polygon -- nothing to accumulate

        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)

        # Fully inside the expanded bbox -> already covered by the rect region.
        if bbox_poly is not None and bbox_poly.contains(poly):
            continue

        regions.append(_buffer_facet(facet, facet_buffer_px))
        outlier_ids.append(str(facet.get('id', idx)))

    return regions, expanded, outlier_ids


def run_bbox_with_outlier_facets_shading(
    latitude: float,
    longitude: float,
    house_bounding_box: 'tuple[float, float, float, float] | list[float]',
    facets: list,
    weather_data: dict,
    dsm: np.ndarray,
    dsm_metadata: Optional[dict] = None,
    expand_pct: float = HOUSE_BBOX_EXPAND_PCT,
    facet_buffer_px: float = FACET_BUFFER_PX,
    pixel_batch_size: int = _DEFAULT_PIXEL_BATCH_SIZE,
    visualize: bool = False,
) -> dict:
    """
    Single-pass initial accumulation over the expanded house bounding box PLUS
    every facet that falls outside it (each buffered by `facet_buffer_px`).

    Delegates to run_facet_shading with the combined region list, so terrain
    reuse/fresh-compute, solar precompute, the physics kernel and TIFF encoding
    are all shared -- see run_facet_shading for the terrain_url / dsm contract
    and returned keys. Adds `outlierFacetIds` to the returned dict.

    hour_chunk is derived inside run_facet_accumulation from each pixel
    batch's own pixel count (see resolve_hour_chunk in
    accumulation_kernels.py) -- if outlier facets make this combined region
    large enough to span multiple `pixel_batch_size` batches, hour_chunk is
    computed correctly for each batch independently, shrinking back toward
    full_dsm_accumulation's own default automatically as needed.
    """
    regions, expanded, outlier_ids = build_bbox_with_outlier_facets(
        house_bounding_box,
        facets,
        expand_pct=expand_pct,
        facet_buffer_px=facet_buffer_px,
    )

    print(
        f'BBOX_WITH_OUTLIER_FACETS: bounds={tuple(house_bounding_box)} '
        f'expand_pct={expand_pct} -> {expanded}  '
        f'outlier facets (+{facet_buffer_px}px buffer): {len(outlier_ids)} '
        f'of {len(facets)}',
        flush=True,
    )

    out = run_facet_shading(
        latitude=latitude,
        longitude=longitude,
        facets=regions,
        weather_data=weather_data,
        dsm=dsm,
        dsm_metadata=dsm_metadata,
        pixel_batch_size=pixel_batch_size,
        visualize=visualize,
    )
    out['outlierFacetIds'] = outlier_ids
    return out
