"""
accumulation_kernels.py -- Shared Numba POA + DC accumulation kernels.

Single home for the pixel-parallel physics kernels used by every
accumulation pipeline in this package, so the Perez + Faiman algebra lives
in exactly one place and all callers stay numerically consistent:

  * full_dsm_accumulation.py -- runs `_poa_accum_kernel` over the whole grid.
  * facet_accumulation.py    -- runs `_poa_accum_kernel` over the union of
                                facet pixels only.
  * panel_accumulation.py    -- runs `_panel_accum_kernel` (shaded/unshaded
                                POA + hourly DC + per-panel reduction) for
                                per-panel yield and solar access.

Shared physics core
-------------------
Both kernels compute the SAME per-(pixel, timestep) physics: cos(AOI) ->
near-field shadow test -> Perez beam/diffuse/ground POA -> Faiman DC. That
per-pixel-per-hour computation lives once in `_poa_dc_core`, an
`inline='always'` Numba device function returning (poa_shd, poa_unshd, dc).
The two kernels differ ONLY in loop order, reduction, and outputs:

  _poa_accum_kernel   -> pixel-outer, time-inner; accumulates annual + monthly
                         DC per pixel. Ignores poa_unshd.
  _panel_accum_kernel -> time-outer, pixel-inner; reduces each panel's pixels
                         in-kernel to per-panel hourly + monthly DC and
                         monthly shaded/unshaded POA (the last two only exist
                         to feed the panel solar-access ratio -- that extra
                         output, plus the per-panel reduction needed for
                         hourly yield, is the whole reason the panel kernel is
                         a separate loop shape rather than a flag on the other)

-------------------------------------------------------------------
There are two independent things every accumulation pipeline chops into
pieces before calling a kernel:

  * `hour_chunk`        -- HOW MANY HOURS one kernel call processes.
                           Produced by `resolve_hour_chunk()` below.
  * `pixel_batch_size`  -- HOW MANY PIXELS one RAM-bounded batch covers,
                           before that batch is itself further split into
                           hour_chunk-sized kernel calls. Only
                           facet_accumulation.py uses this axis today --
                           full_dsm_accumulation chops pixels via its own
                           `strip_rows`, and panel_accumulation never
                           chops pixels at all (its union is always small)..
"""

from __future__ import annotations

import numba
import numpy as np
from numba import njit, prange

from .solar_positions import (
    _FAIMAN_U0,  # noqa: F401  -- re-exported for callers building Phi2
    _FAIMAN_U1,  # noqa: F401
    _ETA_REF,  # noqa: F401
    _GAMMA_PDC_DEFAULT,  # noqa: F401
    _DEFAULT_ALBEDO,  # noqa: F401
)
import math

# ============================================================
# NAMED CONSTANTS  (shared by every accumulation pipeline)
# ============================================================

# Bytes per float32 element -- used only for the human-readable MB
# estimates printed to logs, not for any array allocation itself.
_BYTES_PER_FLOAT32 = 4

# 5 deg solar elevation is the conventional cutoff below which
# near-horizon irradiance geometry is treated as numerically unreliable.
# Used to floor cos(zenith) before it's used as a Perez circumsolar
# normalisation divisor, which would blow up toward infinity as
# cos_z -> 0 right at the true horizon.
_COS_Z_FLOOR_85DEG = np.float32(math.cos(math.radians(85.0)))
_SIN_Z_EPSILON = np.float32(1e-6)

# Sentinel value substituted for tan(sun elevation) whenever sin_z is
# below _SIN_Z_EPSILON. Deliberately huge so the shadow test
# (tan_sun_el < horizon_map[...]) always evaluates to "not shadowed" --
# physically correct, since a sun directly overhead has no meaningful
# shadow-casting angle for any realistic near-field obstruction.
_TAN_SUN_EL_OVERHEAD_SENTINEL = np.float32(1e6)

# STC (Standard Test Conditions) reference cell temperature in the
# Faiman thermal model -- efficiency is defined relative to this
# baseline and drops as cell temperature rises above it.
_STC_TEMP_DEGC = np.float32(25.0)


# ============================================================
# DYNAMIC HOUR-CHUNK SIZING  (facet / house-bbox / panel pipelines)
# ============================================================
#
# full_dsm_accumulation keeps a static hour_chunk=120: its strip_rows
# loop already bounds per-launch pixel count to a known, fixed tile shape
# (~1000x1000 px), so 120 was hand-tuned for that one shape and never
# needs to move.
#
# The facet / house-bbox / panel pipelines don't have that luxury -- their
# per-call pixel count (n_px, e.g. n_union or a single pixel batch) can be
# anywhere from a few hundred (a single small facet, a house bbox) to a
# large fraction of the full DSM (many outlier facets folded into one
# merge). A single static hour_chunk is either too small (wasted Numba
# launch overhead on tiny regions) or unnecessarily conservative (a huge
# region still launching many times when it could do a handful).
#
# resolve_hour_chunk() picks hour_chunk so the TOTAL per-launch work
# (pixels x hours) 
# CALLER CONTRACT -- read this before calling: `n_px` must be the pixel
# count actually handed to the kernel IN A SINGLE CALL. If a pipeline also
# batches pixels (see pixel_batch_size below), call resolve_hour_chunk()
# with that batch's own pixel count, INSIDE the pixel-batching loop -- not
# once, up front, with the region's total pixel count. Sizing hour_chunk
# off a bigger number than the kernel will actually see under-shoots the
# per-launch target and causes more kernel calls than necessary.
#
# Safe to do because hour_chunk scales no allocation in either kernel: the
# per-pixel accumulator is a scalar (_poa_accum_kernel) and the sliced
# hour-arrays are zero-copy views -- see each kernel's own docstring.
# The only real trade being tuned is launch overhead (favors bigger
# hour_chunk) vs. how long one worker thread is pinned inside a single
# kernel call (favors smaller hour_chunk, for fairness under concurrent
# requests) -- this keeps both bounded by anchoring to the proven point.

_TARGET_PX_HOURS_PER_LAUNCH = 30_000_000  # pixels x hours per kernel call
_MIN_HOUR_CHUNK = 120  # floor -- matches full_dsm_accumulation's own default


def resolve_hour_chunk(n_px: int, n_daylight: int) -> int:
    """
    Pick hour_chunk (daylight-hours-per-kernel-call) so that
    n_px * hour_chunk stays near _TARGET_PX_HOURS_PER_LAUNCH, clamped to
    [_MIN_HOUR_CHUNK, n_daylight].

    `n_px` MUST be the pixel count of the actual kernel call this
    hour_chunk will be used for -- e.g. a single pixel_batch_size-sized
    batch, not a larger enclosing region that gets further subdivided.
    Called once per pixel batch, right before that batch's accumulation
    loop, so every call site gets a value sized to the pixels it is
    actually about to process.
    """
    n_daylight = max(1, n_daylight)
    if n_px <= 0:
        return min(_MIN_HOUR_CHUNK, n_daylight)
    raw = _TARGET_PX_HOURS_PER_LAUNCH // n_px
    return max(_MIN_HOUR_CHUNK, min(raw, n_daylight))


# ============================================================
# SHARED PHYSICS CORE  (one pixel, one timestep)
# ============================================================


@njit(inline='always', fastmath=True, cache=True)
def _poa_dc_core(
    # ── temporal scalars for this timestep ────────────────────────────
    dni: float,  # float32  W/m²  direct normal irradiance
    cos_z: float,  # float32        cos(solar zenith)
    beam_blk: float,  # float32        far-field beam blocking flag (0/1)
    sun_ew: float,  # float32  sin(zenith)*cos(azimuth)
    sun_ns: float,  # float32  sin(zenith)*sin(azimuth)
    tan_sun_el: float,  # float32  tan(sun elevation)
    hor_at_bucket: float,  # float32  horizon_map[px, az_bucket] tan(elevation)
    C1: float,  # float32  DHI_t * (1 - F1_t)
    C2: float,  # float32  DHI_t * F1_t / cos_z_clamped_t
    C3: float,  # float32  DHI_t * F2_t
    Phi1: float,  # float32  1 + gamma_pdc*(temp_air_t - 25)
    Phi2: float,  # float32  gamma_pdc*(1-eta_ref) / (U0 + U1*wind_t)
    ghi_albedo: float,  # float32  GHI_t * albedo
    # ── spatial scalars for this pixel ────────────────────────────────
    cos_slope: float,  # float32
    sin_slope: float,  # float32
    proj_ew: float,  # float32  sin(slope)*cos(aspect), PURELY SPATIAL
    proj_ns: float,  # float32  sin(slope)*sin(aspect), PURELY SPATIAL
    svf: float,  # float32  sky-view factor 0-1
):
    """
    Perez POA + Faiman DC for a single (pixel, timestep).

    Division-free, redundant-multiply-free: every quantity that depends on
    time ONLY (C1/C2/C3, Phi1/Phi2, ghi_albedo, sun_ew/ns, tan_sun_el) or
    pixel ONLY (proj_ew/ns, cos/sin_slope, svf) is precomputed by the caller.

    Returns
    -------
    (poa_shd, poa_unshd, dc) : float32
        poa_shd   -- POA irradiance WITH the near-field shadow applied
                     (the physically received irradiance -> DC yield).
        poa_unshd -- POA irradiance WITHOUT the near-field shadow (far-field
                     beam blocking still applied). Denominator for the panel
                     solar-access ratio; the full/facet kernel ignores it.
        dc        -- temperature-corrected DC (poa_shd * Faiman factor).

    Shadow test:
        in_shadow = tan(sun_elevation) < horizon_map[px, az_bucket]

    cos(angle of incidence):
        cos_aoi = cos_z*cos_slope + proj_ew*sun_ew + proj_ns*sun_ns

    Perez diffuse (decoupled):
        diffuse = svf*C1 + cos_aoi*C2 + sin_slope*C3

    Faiman correction (decoupled, no intermediate T_cell, no division):
        corr = Phi1 + poa*Phi2
    """
    # ── cos(angle of incidence) -- precomputed spatial projections ──────
    # dot product of sun unit vector and panel normal unit vector:
    #   sun    = (sin_z*cos_az, sin_z*sin_az, cos_z)
    #   normal = (sin_slope*cos_aspect, sin_slope*sin_aspect, cos_slope)
    cos_aoi = cos_z * cos_slope + proj_ew * sun_ew + proj_ns * sun_ns
    cos_aoi = min(np.float32(1.0), max(np.float32(0.0), cos_aoi))

    # ── near-field shadow test (O(1) horizon-map lookup) ────────────────
    # tan(sun_elevation) = cos_z / sin_z — monotone, avoids arctan.
    # If sun elevation < horizon elevation → pixel is in shadow.
    in_shadow = tan_sun_el < hor_at_bucket
    shd_mult = np.float32(1.0) - np.float32(in_shadow)

    # ── beam component ──────────────────────────────────────────────────
    # DNI [W/m²] projected onto the tilted surface with cos_aoi.
    # (1 - beam_blk): far-field site horizon (distant mountains/terrain).
    # shd_mult:       near-field obstruction (chimney/dormer/neighbour px).
    # beam_shd keeps the exact expression order of the original full/facet
    # kernel so that path stays numerically unchanged; beam_unshd drops only
    # the near-field shadow, feeding poa_unshd for solar access.
    beam_shd = dni * cos_aoi * shd_mult * (np.float32(1.0) - beam_blk)
    beam_unshd = dni * cos_aoi * (np.float32(1.0) - beam_blk)

    # ── diffuse component -- decoupled Perez, division-free ─────────────
    #   svf*C1        isotropic sky (scaled by sky-view factor)
    #   cos_aoi*C2    circumsolar (C2 folds in DHI*F1 / cos_z_clamped)
    #   sin_slope*C3  horizon brightening (C3 folds in DHI*F2)
    diffuse = svf * C1 + cos_aoi * C2 + sin_slope * C3
    if diffuse < 0.0:
        diffuse = np.float32(0.0)

    # ── ground-reflected component ──────────────────────────────────────
    ground = ghi_albedo * (np.float32(1.0) - svf)

    poa_shd = beam_shd + diffuse + ground
    poa_unshd = beam_unshd + diffuse + ground

    # ── Faiman thermal correction -- decoupled, division-free ───────────
    thermal_efficiency_factor = Phi1 + poa_shd * Phi2
    if thermal_efficiency_factor < 0.0:
        thermal_efficiency_factor = np.float32(0.0)

    dc = poa_shd * thermal_efficiency_factor
    return poa_shd, poa_unshd, dc


# ============================================================
# FULL/FACET KERNEL  (pixel-outer; annual + monthly DC per pixel)
# ============================================================


@njit(parallel=True, fastmath=True, cache=True)
def _poa_accum_kernel(
    n_chunk: int,
    # ── per-chunk time-series (n_chunk,) ──────────────────────────────
    dni_c: np.ndarray,  # float32  W/m²  direct normal irradiance
    cos_z_c: np.ndarray,  # float32        cos(solar zenith)
    beam_blk_c: np.ndarray,  # float32        far-field beam blocking flag (0/1)
    month_c: np.ndarray,  # int32          0-based month index
    sun_ew_c: np.ndarray,  # float32  sin(zenith)*cos(azimuth)
    sun_ns_c: np.ndarray,  # float32  sin(zenith)*sin(azimuth)
    tan_sun_el_c: np.ndarray,  # float32  tan(sun elevation)
    az_bucket_c: np.ndarray,  # int32    azimuth bucket index
    # ── pre-factored Perez diffuse coefficients, PURELY TEMPORAL ──────
    C1_c: np.ndarray,  # float32  DHI_t * (1 - F1_t)
    C2_c: np.ndarray,  # float32  DHI_t * F1_t / cos_z_clamped_t
    C3_c: np.ndarray,  # float32  DHI_t * F2_t
    # ── pre-factored Faiman thermal coefficients, PURELY TEMPORAL ─────
    Phi1_c: np.ndarray,  # float32  1 + gamma_pdc*(temp_air_t - 25)
    Phi2_c: np.ndarray,  # float32  gamma_pdc*(1-eta_ref) / (U0 + U1*wind_t)
    ghi_albedo_c: np.ndarray,  # float32  GHI_t * albedo
    # ── near-field horizon map for this pixel strip ───────────────────
    horizon_map_flat: np.ndarray,  # float32  (n_px_strip, n_dirs)  tan(elevation)
    n_dirs: int,
    # ── static per-pixel arrays for this strip (n_px_strip,) ──────────
    cos_slope_flat: np.ndarray,  # float32
    sin_slope_flat: np.ndarray,  # float32
    proj_ew_flat: np.ndarray,  # float32  sin(slope)*cos(aspect), PURELY SPATIAL
    proj_ns_flat: np.ndarray,  # float32  sin(slope)*sin(aspect), PURELY SPATIAL
    svf_flat: np.ndarray,  # float32  sky-view factor 0-1
    # ── accumulators written in-place (n_px_strip,) ───────────────────
    annual_dc_wh_per_m2_flat: np.ndarray,  # float32
    monthly_dc_wh_per_m2_flat: np.ndarray,  # float32  (12, n_px_strip)
) -> None:
    """
    Pixel-parallel Perez POA + Faiman DC accumulation over one hour_chunk.

    Thin loop around `_poa_dc_core`: pixel-outer / time-inner, accumulating
    annual + monthly DC per pixel. The shaded-POA-only DC is all this path
    needs, so the core's poa_shd/poa_unshd are discarded.

    `n_chunk` here is this call's hour_chunk (see resolve_hour_chunk) --
    named n_chunk in the signature since it's a generic "how many time
    steps this call covers", but every caller derives it from hour_chunk.
    """
    n_px = cos_slope_flat.shape[0]

    for px in prange(n_px):
        cos_slope = cos_slope_flat[px]
        sin_slope = sin_slope_flat[px]
        proj_ew = proj_ew_flat[px]
        proj_ns = proj_ns_flat[px]
        svf = svf_flat[px]

        dc_wh_per_m2_acc = np.float32(0.0)

        for t in range(n_chunk):
            _, _, dc_wh_per_m2 = _poa_dc_core(
                dni_c[t],
                cos_z_c[t],
                beam_blk_c[t],
                sun_ew_c[t],
                sun_ns_c[t],
                tan_sun_el_c[t],
                horizon_map_flat[px, az_bucket_c[t]],
                C1_c[t],
                C2_c[t],
                C3_c[t],
                Phi1_c[t],
                Phi2_c[t],
                ghi_albedo_c[t],
                cos_slope,
                sin_slope,
                proj_ew,
                proj_ns,
                svf,
            )

            dc_wh_per_m2_acc += dc_wh_per_m2
            monthly_dc_wh_per_m2_flat[month_c[t], px] += dc_wh_per_m2

        annual_dc_wh_per_m2_flat[px] += dc_wh_per_m2_acc


def _warmup_accumulation_jit(n_dirs: int = 8) -> None:
    """
    Compile _poa_accum_kernel with a tiny dummy call before the timed run.

    n_dirs=8 and the array sizes below are arbitrary -- only the shapes
    matter for triggering Numba's JIT compilation, not the values, since
    this call's output is discarded.
    """
    _n = 16  # arbitrary small size, only needed to give Numba real arrays to compile against
    _hmap = np.zeros((_n, n_dirs), dtype=np.float32)
    _fl = np.ones(_n, dtype=np.float32)
    _zf = np.zeros(_n, dtype=np.float32)
    _ann = np.zeros(_n, dtype=np.float32)
    _mon = np.zeros((12, _n), dtype=np.float32)
    _poa_accum_kernel(
        2,
        np.float32([800.0, 700.0]),  # dni
        np.float32([0.6, 0.5]),  # cos_z
        np.float32([0.0, 0.0]),  # beam_blk
        np.int32([0, 1]),  # month
        np.float32([0.5, 0.4]),  # sun_ew
        np.float32([0.3, 0.2]),  # sun_ns
        np.float32([2.0, 1.5]),  # tan_sun_el
        np.int32([0, 1]),  # az_bucket
        np.float32([70.0, 60.0]),  # C1 (DHI*(1-F1))
        np.float32([30.0, 25.0]),  # C2 (DHI*F1/cos_z_clamped)
        np.float32([2.0, 1.0]),  # C3 (DHI*F2)
        np.float32([0.9, 0.95]),  # Phi1
        np.float32([-0.0001, -0.00012]),  # Phi2
        np.float32([180.0, 160.0]),  # ghi_albedo (GHI*albedo)
        _hmap,
        n_dirs,
        _fl,  # cos_slope
        _zf,  # sin_slope
        _fl,  # proj_ew
        _zf,  # proj_ns
        _fl,  # svf
        _ann,  # annual_dc_wh_per_m2_flat
        _mon,  # monthly_dc_wh_per_m2_flat
    )


# Warm up the full/facet kernel at import time so the first timed run of any
# pipeline that imports this module does not pay the JIT compile cost inline.
# (The panel kernel is warmed by panel_accumulation.run_panel_accumulation,
# which owns the only pipeline that uses it.)
_warmup_accumulation_jit()


def poa_accum(*args) -> None:
    """
    Dispatcher for the full/facet POA+DC accumulation.

    Routes to the CUDA kernel (gpu_kernels.poa_accum_gpu) when a CUDA device is
    available, otherwise the CPU `_poa_accum_kernel` Numba path. Both accumulate
    in-place into the annual/monthly arrays with identical semantics; GPU output
    is numerically equivalent, not bit-identical. Arg order matches
    `_poa_accum_kernel` exactly so callers can swap this in directly.
    """
    from . import gpu_kernels

    if gpu_kernels.GPU_AVAILABLE:
        gpu_kernels.poa_accum_gpu(*args)
    else:
        _poa_accum_kernel(*args)


def panel_accum(*args) -> None:
    """
    Dispatcher for the panel POA+DC accumulation.

    Routes to gpu_kernels.panel_accum_gpu when a CUDA device is available,
    otherwise the CPU `_panel_accum_kernel`. Same in-place semantics; arg order
    matches `_panel_accum_kernel` exactly so callers can swap this in directly.
    """
    from . import gpu_kernels

    if gpu_kernels.GPU_AVAILABLE:
        gpu_kernels.panel_accum_gpu(*args)
    else:
        _panel_accum_kernel(*args)


# ============================================================
# PANEL KERNEL  (time-outer; per-panel hourly/monthly DC + solar access)
# ============================================================


@njit(parallel=True, fastmath=True, cache=True)
def _panel_accum_kernel(
    n_chunk: int,
    # ── per-chunk time-series (n_chunk,) ──────────────────────────────
    dni_c: np.ndarray,  # float32
    cos_z_c: np.ndarray,  # float32
    beam_blk_c: np.ndarray,  # float32  far-field blocking
    month_c: np.ndarray,  # int32    0-based month
    hour_idx_c: np.ndarray,  # int32    absolute hour index
    sun_ew_c: np.ndarray,  # float32  sin(zenith)*cos(azimuth)
    sun_ns_c: np.ndarray,  # float32  sin(zenith)*sin(azimuth)
    tan_sun_el_c: np.ndarray,  # float32  tan(sun elevation)
    az_bucket_c: np.ndarray,  # int32    azimuth bucket index
    # ── pre-factored Perez + Faiman coefficients, PURELY TEMPORAL ─────
    C1_c: np.ndarray,  # float32  DHI_t * (1 - F1_t)
    C2_c: np.ndarray,  # float32  DHI_t * F1_t / cos_z_clamped_t
    C3_c: np.ndarray,  # float32  DHI_t * F2_t
    Phi1_c: np.ndarray,  # float32  1 + gamma_pdc*(temp_air_t - 25)
    Phi2_c: np.ndarray,  # float32  gamma_pdc*(1-eta_ref) / (U0 + U1*wind_t)
    ghi_albedo_c: np.ndarray,  # float32  GHI_t * albedo
    # ── compact panel-pixel horizon slice (n_union, n_dirs) ───────────
    # [KEY] Only panel pixels — index 0..n_union-1, sequential, cache-hot.
    hor_union: np.ndarray,  # float32  (n_union, n_dirs)
    n_dirs: int,
    # ── panel pixel table — union of all panel pixels (n_union,) ──────
    cos_slope: np.ndarray,  # float32
    sin_slope: np.ndarray,  # float32
    proj_ew: np.ndarray,  # float32  sin(slope)*cos(aspect), PURELY SPATIAL
    proj_ns: np.ndarray,  # float32  sin(slope)*sin(aspect), PURELY SPATIAL
    svf: np.ndarray,  # float32
    # ── panel slice boundaries ────────────────────────────────────────
    panel_starts: np.ndarray,  # int32  (n_panels,)
    panel_ends: np.ndarray,  # int32  (n_panels,)
    n_panels: int,
    # ── accumulators written in-place ─────────────────────────────────
    hourly_dc: np.ndarray,  # (n_panels, n_hours)  float32
    monthly_dc: np.ndarray,  # (n_panels, 12)       float32
    monthly_poa_shaded: np.ndarray,  # (n_panels, 12)       float32
    monthly_poa_unshaded: np.ndarray,  # (n_panels, 12)       float32
) -> None:
    """
    Pixel-parallel Perez POA + Faiman DC over n_chunk timesteps, reduced to
    per-panel totals in-kernel.

    Same `_poa_dc_core` physics as the full/facet kernel, but the loop is
    time-outer / pixel-inner so that each timestep's per-pixel results can be
    averaged over each panel's pixels immediately -- giving per-panel HOURLY
    DC (hourly_dc[pi, h]) without ever materialising an (n_px, n_hours) array.

    [SOLAR-ACCESS]  poa_shd and poa_unshd are accumulated per panel per month;
    their ratio is the panel's solar access (computed by the caller).

    Outer prange: pixels (n_union, parallel).
    Per-panel reduction: sequential over n_panels (tiny) after the pixel prange.

    `n_chunk` here is this call's hour_chunk (see resolve_hour_chunk).
    """
    n_union = cos_slope.shape[0]

    # ── Per-pixel temp buffers ─────────────────────────────────────────
    poa_shd_px = np.empty(n_union, dtype=numba.float32)
    poa_unshd_px = np.empty(n_union, dtype=numba.float32)
    dc_shd_px = np.empty(n_union, dtype=numba.float32)

    for t in range(n_chunk):
        m = month_c[t]
        h = hour_idx_c[t]

        # ── Pixel-parallel POA + DC (prange over n_union) ─────────────
        for px in prange(n_union):
            p_shd, p_unshd, dc = _poa_dc_core(
                dni_c[t],
                cos_z_c[t],
                beam_blk_c[t],
                sun_ew_c[t],
                sun_ns_c[t],
                tan_sun_el_c[t],
                hor_union[px, az_bucket_c[t]],
                C1_c[t],
                C2_c[t],
                C3_c[t],
                Phi1_c[t],
                Phi2_c[t],
                ghi_albedo_c[t],
                cos_slope[px],
                sin_slope[px],
                proj_ew[px],
                proj_ns[px],
                svf[px],
            )
            poa_shd_px[px] = p_shd
            poa_unshd_px[px] = p_unshd
            dc_shd_px[px] = dc

        # ── Per-panel mean reduction (sequential, n_panels << n_union) ─
        for pi in range(n_panels):
            s = panel_starts[pi]
            e = panel_ends[pi]
            cnt = numba.float32(e - s)
            s_sum = np.float32(0.0)
            u_sum = np.float32(0.0)
            dc_sum = np.float32(0.0)
            for px in range(s, e):
                s_sum += poa_shd_px[px]
                u_sum += poa_unshd_px[px]
                dc_sum += dc_shd_px[px]
            inv = np.float32(1.0) / cnt
            hourly_dc[pi, h] += dc_sum * inv
            monthly_dc[pi, m] += dc_sum * inv
            monthly_poa_shaded[pi, m] += s_sum * inv
            monthly_poa_unshaded[pi, m] += u_sum * inv
