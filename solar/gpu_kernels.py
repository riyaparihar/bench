"""
gpu_kernels.py — CUDA (numba.cuda) ports of the three hot shading kernels.

The CPU versions
    (referenced as terrain._horizon_kernel, accumulation_kernels._poa_accum_kernel, accumulation_kernels._panel_accum_kernel)
    are @njit functions parallelized with Numba's prange —
    each CPU core picks up one loop iteration (e.g., one pixel) at a time,
    and there might be a handful of cores (say 8–32) working in parallel.

The GPU versions
    replace that with thousands of simultaneous threads,
    one per unit of work, launched all at once.
    The pattern throughout the file is:

1. Host (Python/CPU) code prepares NumPy arrays and copies them to device memory (cuda.to_device(...)).
2. A kernel (@cuda.jit-decorated function) runs once per thread on the GPU.
    cuda.grid(1) gives each thread a unique 1D index (gid),
    which is decoded into
    "which pixel / direction / panel / timestep am I responsible for."
3. Results are copied back to the host (.copy_to_host()).

This "upload → compute → download" round trip over PCIe
is the main extra cost that doesn't exist on CPU
(where everything already lives in RAM) —
which is exactly why PoaGpuSession exists
(explained below): to avoid paying that cost every chunk.

Design & rationale: see GPU_MIGRATION_PLAN.md in this directory.

Fallback contract
-----------------
`GPU_AVAILABLE` is False whenever CUDA is unusable (no toolkit, no device,
import error). Every public entry point here is only meant to be called when
`GPU_AVAILABLE` is True; the CPU `@njit` kernels in terrain.py /
accumulation_kernels.py remain the reference implementation and the fallback.
Callers dispatch with:

    from . import gpu_kernels
    if gpu_kernels.GPU_AVAILABLE:
        svf, hor = gpu_kernels.svf_and_horizon_gpu(...)
    else:
        svf, hor = <existing CPU path>

Newer-GPU compute-capability note
----------------------------------
On a GPU newer than the installed CUDA toolchain recognizes (e.g. this
package's verified RTX 5060 / Blackwell / sm_120, against `numba-cuda`
0.30.4 whose NVRTC only knows targets up to compute_87), compiling
architecture-specific PTX for the real device fails outright with
`Can't use arch-specific compute_120a with closest found compute capability
compute_87`. The fix lives in `solar/__init__.py`, not here:
`os.environ.setdefault('NUMBA_FORCE_CUDA_CC', '8.7')`, set before this
module's `from numba import cuda` runs. This forces PTX generation at a
supported older target; the CUDA driver JIT-recompiles that PTX for the
actual device at kernel-launch time (PTX forward compatibility), so kernels
run correctly on newer hardware at the cost of missing architecture-specific
instruction scheduling for that generation. Revisit/remove once a
`numba-cuda` release adds native support for the deployed GPU's compute
capability.

Numerical note
--------------
GPU FMA + fastmath reorder float ops, so results are numerically equivalent to
the CPU kernels but NOT bit-identical. Validate with rtol~1e-4, not array_equal
-- confirmed in production: max relative diff ~3e-7 between CPU and GPU
annual_dc on real data, well within tolerance.

Operational note -- per-request compile cost
---------------------------------------------
If the process invoking this module is short-lived (spawned fresh per
request, e.g. via Node's PythonShell rather than a long-running worker),
`warmup_gpu()`'s first-launch PTX compile (~0.9s observed) is paid on EVERY
request, unlike the CPU kernels' `cache=True`, which persists across
processes via Numba's on-disk cache. No `cache=True` equivalent exists here
for CUDA kernels in this numba-cuda version. Negligible for large jobs,
proportionally larger for small ones. See GPU_MIGRATION_PLAN.md for the
long-lived-worker follow-up this implies.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# ── CUDA availability gate ─────────────────────────────────────────────────
# `SHADING_FORCE_CPU=1` (read in perf.py) pins GPU_AVAILABLE to False so the
# whole pipeline takes its CPU @njit paths -- that's how the CPU half of a
# CPU-vs-GPU timing comparison is produced, without touching any call site.
from .perf import FORCE_CPU as _FORCE_CPU, _out as _perf_out

try:
    from numba import cuda

    GPU_AVAILABLE: bool = (not _FORCE_CPU) and bool(cuda.is_available())
    if _FORCE_CPU:
        # via perf._out so this can be kept off stdout for entry points whose
        # stdout is a machine-read result payload (SHADING_PERF_STDERR=1).
        print(
            '[gpu_kernels] SHADING_FORCE_CPU set -- CUDA disabled, using CPU kernels',
            file=_perf_out(),
            flush=True,
        )
except Exception as e:  # pragma: no cover - numba.cuda import/driver failure
    print(f'[gpu_kernels] CUDA unavailable: {e}', flush=True)
    cuda = None  # type: ignore[assignment]
    GPU_AVAILABLE = False


# Default CUDA block size (threads per block).
# 256 is a safe general default.
# Every kernel launch below computes
# blocks = (total_work + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK (ceiling division)
# so there are enough blocks × threads
# to cover all the work, then launches kernel[blocks, _THREADS_PER_BLOCK](...).
_THREADS_PER_BLOCK: int = 256


# ============================================================
# KERNEL 1 — HORIZON RAY-MARCH  (one thread per (pixel, direction))
# ============================================================

if GPU_AVAILABLE:

    @cuda.jit(fastmath=True)
    def _horizon_kernel_cuda(
        dsm,  # (H, W)   float32
        cell_size,  # float32
        dsm_max,  # float32  precomputed on host
        dr_arr,  # (n_dirs,) float64  precomputed on host
        dc_arr,  # (n_dirs,) float64  precomputed on host
        ff_arr,  # (n_dirs,) float64  precomputed on host (tan floor)
        n_dirs,
        max_r,
        r0,
        r1,
        c0,
        c1,
        H,
        W,
        hor_out,  # (n_px, n_dirs) float32  written per (px, k)
        svf_acc,  # (n_px,)        float32  atomic-accumulated ((sky view factor accumulator)
    ):
        """
        One thread computes one (pixel, direction) ray-march.
        CPU version (implied):
            a prange over pixels, and inside that,
            a Python for loop over n_dirs directions per pixel
            — so one CPU thread does all directions for a pixel sequentially.

        GPU version:
            one thread per (pixel, direction) pair
            — much finer-grained parallelism.
            Instead of a thread doing ~120 directions in sequence,
            120 different threads each do 1 direction, simultaneously.
        """
        # this thread's global flat index.
        gid = cuda.grid(1)
        region_w = c1 - c0
        n_region_px = (r1 - r0) * region_w
        # total threads needed = pixels × directions.
        total = n_region_px * n_dirs

        if gid >= total:
            # If a thread's gid is past total
            # (happens because block counts are rounded up),
            # it does nothing and returns
            # — this is boilerplate "bounds guard" in every CUDA kernel
            return

        # ---- decode `gid` back into "which pixel, which direction" ----.
        # flat: the pixel's index within the region;
        flat = gid // n_dirs
        # k : which of the n_dirs directions (e.g. direction 37 of 120) this thread handles.
        k = gid % n_dirs
        # i,j are the row/col of that pixel in the full DSM,
        # offset by the region's start (r0, c0).
        i = r0 + flat // region_w
        j = c0 + flat % region_w

        # elevation of the source pixel.
        z0 = dsm[i, j]
        # flattened index into the pixel-major
        # output arrays (hor_out, svf_acc),
        # since GPU kernels prefer 1D flat arrays over 2D indexing for output.
        px = i * W + j

        # precomputed once on the host
        # the row/col step direction
        # (basically -cos(azimuth), sin(azimuth)) for direction k
        dr = dr_arr[k]
        dc = dc_arr[k]

        # running maximum horizon slope found so far along this ray.
        best = ff_arr[k]  # background horizon floor for direction k

        s = 1  # distance in cells along the ray, from 1 up to max_r (max search radius in cells)
        step = 1
        while s <= max_r:
            # early exit: tallest point on the map can't beat current best at distance s
            if (dsm_max - z0) / (s * cell_size) <= best:
                break

            # the terrain cell at distance s along direction (dr, dc),
            # rounded to nearest integer cell.
            ii = int(round(i + dr * s))
            jj = int(round(j + dc * s))

            # If that cell is off the map, stop.
            if ii < 0 or ii >= H or jj < 0 or jj >= W:
                break

            # how much higher that cell is than the source pixel.
            # If it creates a bigger slope (t) than the current best,
            # update best.
            dz = dsm[ii, jj] - z0
            if dz > 0.0:
                t = dz / (s * cell_size)
                if t > best:
                    best = t

            # s grows, take bigger jumps (1 → 2 → 4 → 8 → 16 cells per step)
            # since far-away terrain needs less angular precision.
            # This is a performance trick to avoid checking every single cell out to the max radius.
            s += step
            if s > 160:
                step = 16
            elif s > 80:
                step = 8
            elif s > 40:
                step = 4
            elif s > 20:
                step = 2

        hor_out[px, k] = np.float32(best)
        # sin(atan(best)) = best / sqrt(1 + best^2); sky visibility term
        cuda.atomic.add(svf_acc, px, 1.0 - (best / math.sqrt(1.0 + best * best)))

    @cuda.jit(fastmath=True)
    def _svf_finalize_cuda(svf_acc, svf_out, inv, r0, r1, c0, c1, W):
        """
        svf_out[i,j] = min(svf_acc[px] / n_dirs, 1.0) for region pixels.
        runs after kernel 1 finishes
            necessary because all n_dirs atomic adds for a pixel must be done
            before we safely divide/finalize it; can't guarantee ordering within one kernel launch.
            One thread per pixel (not per pixel×direction). inv = 1/n_dirs,
            so this converts the summed contribution into an average,
            then clamps to a max of 1.0.
            This mirrors the CPU code's final svf = min(sum/n_dirs, 1.0) step,
            just as its own GPU pass.
        """

        gid = cuda.grid(1)
        region_w = c1 - c0
        n_region_px = (r1 - r0) * region_w
        if gid >= n_region_px:
            return
        i = r0 + gid // region_w
        j = c0 + gid % region_w
        v = svf_acc[i * W + j] * inv
        svf_out[i, j] = v if v <= 1.0 else np.float32(1.0)


def svf_and_horizon_gpu(
    dsm: np.ndarray,
    cell_size: float,
    n_dirs: int,
    max_radius_m: float,
    horizon: Optional[np.ndarray] = None,
    region: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU equivalent of terrain.svf_and_horizon().

    Same inputs/outputs and same region semantics: rays read the FULL dsm for
    occlusion; only output pixels inside `region` are computed, the rest stay
    zero. Returns (svf float32 (H,W), horizon_map float16 (H*W, n_dirs)).

    Only call when GPU_AVAILABLE is True.
    """
    if not GPU_AVAILABLE:  # defensive; callers should gate on GPU_AVAILABLE
        raise RuntimeError('svf_and_horizon_gpu called without a CUDA device')

    dsm_ = np.ascontiguousarray(dsm, dtype=np.float32)
    H, W = dsm_.shape
    n_px = H * W

    hz = (
        np.ascontiguousarray(horizon, dtype=np.float32)
        if horizon is not None
        else np.zeros(360, dtype=np.float32)
    )

    if region is None:
        r0, r1, c0, c1 = 0, H, 0, W
    else:
        r0, r1, c0, c1 = region
        r0 = max(0, min(int(r0), H))
        r1 = max(0, min(int(r1), H))
        c0 = max(0, min(int(c0), W))
        c1 = max(0, min(int(c1), W))
        if r1 <= r0 or c1 <= c0:
            r0, r1, c0, c1 = 0, H, 0, W

    max_r = max(1, int(round(max_radius_m / cell_size)))

    # ── host-side precompute (was the head of the CPU kernel) ─────────────
    dsm_max = float(dsm_.max())
    dr_arr = np.empty(n_dirs, dtype=np.float64)
    dc_arr = np.empty(n_dirs, dtype=np.float64)
    ff_arr = np.empty(n_dirs, dtype=np.float64)
    for k in range(n_dirs):
        az = k * 2.0 * math.pi / n_dirs
        az_deg_f = math.degrees(az) % 360.0
        az0 = int(az_deg_f) % 360
        az1 = (az0 + 1) % 360
        frac = az_deg_f - int(az_deg_f)
        ff_h = float(hz[az0]) * (1.0 - frac) + float(hz[az1]) * frac
        ff_arr[k] = math.tan(ff_h) if ff_h > 0.0 else 0.0
        dr_arr[k] = -math.cos(az)
        dc_arr[k] = math.sin(az)

    # ── device arrays ─────────────────────────────────────────────────────
    d_dsm = cuda.to_device(dsm_)
    d_dr = cuda.to_device(dr_arr)
    d_dc = cuda.to_device(dc_arr)
    d_ff = cuda.to_device(ff_arr)
    d_hor = cuda.device_array((n_px, n_dirs), dtype=np.float32)
    d_svf_acc = cuda.to_device(np.zeros(n_px, dtype=np.float32))
    d_svf = cuda.to_device(np.zeros((H, W), dtype=np.float32))

    # ── launch ray-march: one thread per (pixel, direction) ───────────────
    n_region_px = (r1 - r0) * (c1 - c0)
    total = n_region_px * n_dirs
    blocks = (total + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _horizon_kernel_cuda[blocks, _THREADS_PER_BLOCK](
        d_dsm,
        np.float32(cell_size),
        np.float32(dsm_max),
        d_dr,
        d_dc,
        d_ff,
        n_dirs,
        max_r,
        r0,
        r1,
        c0,
        c1,
        H,
        W,
        d_hor,
        d_svf_acc,
    )

    # ── finalize SVF ──────────────────────────────────────────────────────
    fin_blocks = (n_region_px + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _svf_finalize_cuda[fin_blocks, _THREADS_PER_BLOCK](
        d_svf_acc, d_svf, np.float32(1.0 / n_dirs), r0, r1, c0, c1, W
    )

    svf = d_svf.copy_to_host()
    hor = d_hor.copy_to_host().astype(np.float16)
    return svf, hor


# ============================================================
# KERNEL 2 — POA / DC ACCUMULATION  (one thread per pixel)
# ============================================================

if GPU_AVAILABLE:
    # device=True
    #   means this isn't launched from the host
    #   — it's a helper function called from inside another kernel,
    #   like a regular function but compiled to run on-device

    # inline=True
    #   asks the compiler to inline it (avoid function-call overhead)
    #   since it's called once per timestep per pixel
    @cuda.jit(device=True, fastmath=True, inline=True)
    def _poa_dc_core_cuda(
        dni,
        cos_z,
        beam_blk,
        sun_ew,
        sun_ns,
        tan_sun_el,
        hor_at_bucket,
        C1,
        C2,
        C3,
        Phi1,
        Phi2,
        ghi_albedo,
        cos_slope,
        sin_slope,
        proj_ew,
        proj_ns,
        svf,
    ):
        """Device port of accumulation_kernels._poa_dc_core (body unchanged)."""
        cos_aoi = cos_z * cos_slope + proj_ew * sun_ew + proj_ns * sun_ns
        cos_aoi = min(np.float32(1.0), max(np.float32(0.0), cos_aoi))

        in_shadow = tan_sun_el < hor_at_bucket
        shd_mult = np.float32(1.0) - np.float32(in_shadow)

        beam_shd = dni * cos_aoi * shd_mult * (np.float32(1.0) - beam_blk)
        beam_unshd = dni * cos_aoi * (np.float32(1.0) - beam_blk)

        diffuse = svf * C1 + cos_aoi * C2 + sin_slope * C3
        if diffuse < 0.0:
            diffuse = np.float32(0.0)

        ground = ghi_albedo * (np.float32(1.0) - svf)

        poa_shd = beam_shd + diffuse + ground
        poa_unshd = beam_unshd + diffuse + ground

        tef = Phi1 + poa_shd * Phi2
        if tef < 0.0:
            tef = np.float32(0.0)

        dc = poa_shd * tef
        return poa_shd, poa_unshd, dc

    @cuda.jit(fastmath=True)
    def _poa_accum_kernel_cuda(
        n_chunk,
        dni_c,
        cos_z_c,
        beam_blk_c,
        month_c,
        sun_ew_c,
        sun_ns_c,
        tan_sun_el_c,
        az_bucket_c,
        C1_c,
        C2_c,
        C3_c,
        Phi1_c,
        Phi2_c,
        ghi_albedo_c,
        horizon_map_flat,
        n_dirs,
        cos_slope_flat,
        sin_slope_flat,
        proj_ew_flat,
        proj_ns_flat,
        svf_flat,
        annual_dc_wh_per_m2_flat,  # (n_px,)     accumulated in-place
        monthly_dc_wh_per_m2_flat,  # (12, n_px)  accumulated in-place
    ):
        """
        One thread per pixel; sequential over n_chunk timesteps.

        CPU version (implied):
            prange over pixels; each thread/core owns one pixel and loops sequentially over all n_chunk timesteps for that pixel,
            accumulating annual/monthly totals.

        GPU version: identical structure — one thread per pixel, looping sequentially over timesteps inside the thread
            (this part is not further parallelized, unlike Kernel 1's per-direction split):
        """
        px = cuda.grid(1)
        n_px = cos_slope_flat.shape[0]
        if px >= n_px:
            return

        cos_slope = cos_slope_flat[px]
        sin_slope = sin_slope_flat[px]
        proj_ew = proj_ew_flat[px]
        proj_ns = proj_ns_flat[px]
        svf = svf_flat[px]

        dc_acc = np.float32(0.0)
        for t in range(n_chunk):
            _, _, dc = _poa_dc_core_cuda(
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
            dc_acc += dc
            monthly_dc_wh_per_m2_flat[month_c[t], px] += dc

        annual_dc_wh_per_m2_flat[px] += dc_acc


# Uploads everything
# — temporal series and per-pixel statics
# — to the device on every call,
# launches the kernel, downloads results, writes them back into the caller's NumPy arrays in place.
# This mirrors the CPU function's signature and in-place-accumulation contract exactly (a drop-in replacement),
# but it's flagged in the docstring as inefficient for the real calling pattern:
# the real pipeline calls this function repeatedly (once per hour-chunk, inside a loop, inside another loop over pixel batches),
# and each call re-uploads the full temporal series and per-pixel statics from scratch and downloads the accumulators
# — even though the temporal series doesn't change between calls.
#
# That's pure PCIe overhead that doesn't exist on CPU (RAM access is just fast, always).
# PoaGpuSession: This solves that inefficiency:
#
# stateless host wrapper
def poa_accum_gpu(
    n_chunk,
    dni_c,
    cos_z_c,
    beam_blk_c,
    month_c,
    sun_ew_c,
    sun_ns_c,
    tan_sun_el_c,
    az_bucket_c,
    C1_c,
    C2_c,
    C3_c,
    Phi1_c,
    Phi2_c,
    ghi_albedo_c,
    horizon_map_flat,
    n_dirs,
    cos_slope_flat,
    sin_slope_flat,
    proj_ew_flat,
    proj_ns_flat,
    svf_flat,
    annual_dc_wh_per_m2_flat,
    monthly_dc_wh_per_m2_flat,
) -> None:
    """
    Drop-in GPU replacement for accumulation_kernels._poa_accum_kernel.

    Identical signature and in-place-accumulation semantics: annual/monthly
    accumulators are read back, added to, and written into the caller's arrays.
    Only call when GPU_AVAILABLE is True.
    """
    if not GPU_AVAILABLE:  # defensive
        raise RuntimeError('poa_accum_gpu called without a CUDA device')

    n_px = cos_slope_flat.shape[0]

    def _c32(a):
        return np.ascontiguousarray(a, dtype=np.float32)

    # temporal series (small) + per-pixel static arrays -> device
    d_dni = cuda.to_device(_c32(dni_c))
    d_cos_z = cuda.to_device(_c32(cos_z_c))
    d_beam = cuda.to_device(_c32(beam_blk_c))
    d_month = cuda.to_device(np.ascontiguousarray(month_c, dtype=np.int32))
    d_sun_ew = cuda.to_device(_c32(sun_ew_c))
    d_sun_ns = cuda.to_device(_c32(sun_ns_c))
    d_tan = cuda.to_device(_c32(tan_sun_el_c))
    d_azb = cuda.to_device(np.ascontiguousarray(az_bucket_c, dtype=np.int32))
    d_C1 = cuda.to_device(_c32(C1_c))
    d_C2 = cuda.to_device(_c32(C2_c))
    d_C3 = cuda.to_device(_c32(C3_c))
    d_Phi1 = cuda.to_device(_c32(Phi1_c))
    d_Phi2 = cuda.to_device(_c32(Phi2_c))
    d_gha = cuda.to_device(_c32(ghi_albedo_c))

    d_hor = cuda.to_device(_c32(horizon_map_flat))
    d_cos_s = cuda.to_device(_c32(cos_slope_flat))
    d_sin_s = cuda.to_device(_c32(sin_slope_flat))
    d_pew = cuda.to_device(_c32(proj_ew_flat))
    d_pns = cuda.to_device(_c32(proj_ns_flat))
    d_svf = cuda.to_device(_c32(svf_flat))

    # accumulators: seed the device copy with existing host values so the
    # kernel's in-place += extends any prior chunk's totals.
    d_annual = cuda.to_device(_c32(annual_dc_wh_per_m2_flat))
    d_monthly = cuda.to_device(_c32(monthly_dc_wh_per_m2_flat))

    blocks = (n_px + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _poa_accum_kernel_cuda[blocks, _THREADS_PER_BLOCK](
        int(n_chunk),
        d_dni,
        d_cos_z,
        d_beam,
        d_month,
        d_sun_ew,
        d_sun_ns,
        d_tan,
        d_azb,
        d_C1,
        d_C2,
        d_C3,
        d_Phi1,
        d_Phi2,
        d_gha,
        d_hor,
        int(n_dirs),
        d_cos_s,
        d_sin_s,
        d_pew,
        d_pns,
        d_svf,
        d_annual,
        d_monthly,
    )

    # write results back into the caller's arrays in place
    annual_dc_wh_per_m2_flat[:] = d_annual.copy_to_host()
    monthly_dc_wh_per_m2_flat[:] = d_monthly.copy_to_host()


# ============================================================
# RESIDENT SESSION — full/facet POA over a strip/batch loop
# ============================================================


class PoaGpuSession:
    """
    Data-residency wrapper for the full/facet POA+DC accumulation loop.

    The CPU pipelines iterate an OUTER pixel strip/batch loop and, inside it, an
    INNER hour-chunk loop that re-passes the same static arrays every chunk.
    `poa_accum_gpu` (the stateless drop-in) would re-upload the full temporal
    series AND the per-pixel statics on every chunk, and copy accumulators back
    every chunk -- most of the wall time becomes PCIe traffic on small batches.

    This session eliminates that:
      * __init__      uploads the full temporal series ONCE (shared by all
                      strips/batches).
      * run_batch()   uploads one batch's per-pixel statics ONCE, loops the
                      hour-chunks launching the kernel on DEVICE slices of the
                      resident temporal arrays (no host<->device traffic in the
                      inner loop), and copies the batch accumulators back ONCE.

    Numerically identical to calling poa_accum_gpu per chunk; the accumulation
    order is the same (chunks summed in order into the batch accumulators).

    Only construct when GPU_AVAILABLE is True.
    """

    def __init__(
        self,
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
    ):
        if not GPU_AVAILABLE:  # defensive
            raise RuntimeError('PoaGpuSession created without a CUDA device')

        def _f(a):
            return cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))

        def _i(a):
            return cuda.to_device(np.ascontiguousarray(a, dtype=np.int32))

        # full temporal series, resident for the whole accumulation
        self._dni = _f(dni_all)
        self._cos_z = _f(cos_z_all)
        self._blk = _f(blk_all)
        self._month = _i(month_all)
        self._sun_ew = _f(sun_ew_all)
        self._sun_ns = _f(sun_ns_all)
        self._tan = _f(tan_sun_el_all)
        self._azb = _i(az_bucket_all)
        self._C1 = _f(C1_all)
        self._C2 = _f(C2_all)
        self._C3 = _f(C3_all)
        self._Phi1 = _f(Phi1_all)
        self._Phi2 = _f(Phi2_all)
        self._gha = _f(ghi_albedo_all)
        self._n_daylight = int(len(dni_all))

    def run_batch(
        self,
        hor_batch,  # (n_px, n_dirs) float
        n_dirs,
        cos_slope,
        sin_slope,
        proj_ew,
        proj_ns,
        svf,  # (n_px,) each
        chunk_size,
    ):
        """
        Accumulate one pixel batch over all daylight hours on-device.

        Returns (annual_dc (n_px,), monthly_dc (12, n_px)) as host float32
        arrays -- the batch's fresh totals (caller writes them into its
        full-size union/grid accumulators).
        """

        def _f(a):
            return cuda.to_device(np.ascontiguousarray(a, dtype=np.float32))

        n_px = int(cos_slope.shape[0])

        # batch statics -> device once
        d_hor = _f(hor_batch)
        d_cs = _f(cos_slope)
        d_ss = _f(sin_slope)
        d_pew = _f(proj_ew)
        d_pns = _f(proj_ns)
        d_sv = _f(svf)

        # batch accumulators start at zero, live on device across all chunks
        d_ann = cuda.to_device(np.zeros(n_px, dtype=np.float32))
        d_mon = cuda.to_device(np.zeros((12, n_px), dtype=np.float32))

        blocks = (n_px + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
        for t0 in range(0, self._n_daylight, chunk_size):
            t1 = min(t0 + chunk_size, self._n_daylight)
            nc = t1 - t0
            sl = slice(t0, t1)
            # slice the RESIDENT device arrays -- no host<->device copy here
            _poa_accum_kernel_cuda[blocks, _THREADS_PER_BLOCK](
                nc,
                self._dni[sl],
                self._cos_z[sl],
                self._blk[sl],
                self._month[sl],
                self._sun_ew[sl],
                self._sun_ns[sl],
                self._tan[sl],
                self._azb[sl],
                self._C1[sl],
                self._C2[sl],
                self._C3[sl],
                self._Phi1[sl],
                self._Phi2[sl],
                self._gha[sl],
                d_hor,
                int(n_dirs),
                d_cs,
                d_ss,
                d_pew,
                d_pns,
                d_sv,
                d_ann,
                d_mon,
            )

        return d_ann.copy_to_host(), d_mon.copy_to_host()


# ============================================================
# KERNEL 3 — PANEL ACCUMULATION  (one thread per (panel, timestep))
# ============================================================

if GPU_AVAILABLE:

    @cuda.jit(fastmath=True)
    def _panel_accum_kernel_cuda(
        n_chunk,
        dni_c,
        cos_z_c,
        beam_blk_c,
        month_c,
        hour_idx_c,
        sun_ew_c,
        sun_ns_c,
        tan_sun_el_c,
        az_bucket_c,
        C1_c,
        C2_c,
        C3_c,
        Phi1_c,
        Phi2_c,
        ghi_albedo_c,
        hor_union,
        n_dirs,
        cos_slope,
        sin_slope,
        proj_ew,
        proj_ns,
        svf,
        panel_starts,
        panel_ends,
        n_panels,
        hourly_dc,  # (n_panels, n_hours)
        monthly_dc,  # (n_panels, 12)
        monthly_poa_shaded,  # (n_panels, 12)
        monthly_poa_unshaded,  # (n_panels, 12)
    ):
        """
        One thread per (panel, timestep). Each thread reduces its panel's
        pixels for one timestep -- no shared temp buffers (unlike the CPU
        kernel, which materialises poa_*_px across pixels then reduces).

        hourly_dc[pi, h] is written by exactly one thread ((pi, t) -> unique
        (pi, h)), so no atomic. monthly_* accumulate over all timesteps sharing
        a month for the same panel, so they use cuda.atomic.add.
        """
        gid = cuda.grid(1)
        total = n_panels * n_chunk
        if gid >= total:
            return
        pi = gid // n_chunk
        t = gid % n_chunk

        m = month_c[t]
        h = hour_idx_c[t]
        s = panel_starts[pi]
        e = panel_ends[pi]
        cnt = np.float32(e - s)
        if cnt <= 0.0:
            return

        s_sum = np.float32(0.0)
        u_sum = np.float32(0.0)
        dc_sum = np.float32(0.0)
        for px in range(s, e):
            p_shd, p_unshd, dc = _poa_dc_core_cuda(
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
            s_sum += p_shd
            u_sum += p_unshd
            dc_sum += dc

        inv = np.float32(1.0) / cnt
        hourly_dc[pi, h] += dc_sum * inv  # unique (pi, h) -> no atomic
        cuda.atomic.add(monthly_dc, (pi, m), dc_sum * inv)
        cuda.atomic.add(monthly_poa_shaded, (pi, m), s_sum * inv)
        cuda.atomic.add(monthly_poa_unshaded, (pi, m), u_sum * inv)


def panel_accum_gpu(
    n_chunk,
    dni_c,
    cos_z_c,
    beam_blk_c,
    month_c,
    hour_idx_c,
    sun_ew_c,
    sun_ns_c,
    tan_sun_el_c,
    az_bucket_c,
    C1_c,
    C2_c,
    C3_c,
    Phi1_c,
    Phi2_c,
    ghi_albedo_c,
    hor_union,
    n_dirs,
    cos_slope,
    sin_slope,
    proj_ew,
    proj_ns,
    svf,
    panel_starts,
    panel_ends,
    n_panels,
    hourly_dc,
    monthly_dc,
    monthly_poa_shaded,
    monthly_poa_unshaded,
) -> None:
    """
    Drop-in GPU replacement for accumulation_kernels._panel_accum_kernel.

    Identical signature and in-place-accumulation semantics. Only call when
    GPU_AVAILABLE is True.
    """
    if not GPU_AVAILABLE:  # defensive
        raise RuntimeError('panel_accum_gpu called without a CUDA device')

    def _c32(a):
        return np.ascontiguousarray(a, dtype=np.float32)

    def _c32i(a):
        return np.ascontiguousarray(a, dtype=np.int32)

    d = dict(
        dni=cuda.to_device(_c32(dni_c)),
        cos_z=cuda.to_device(_c32(cos_z_c)),
        beam=cuda.to_device(_c32(beam_blk_c)),
        month=cuda.to_device(_c32i(month_c)),
        hour=cuda.to_device(_c32i(hour_idx_c)),
        sew=cuda.to_device(_c32(sun_ew_c)),
        sns=cuda.to_device(_c32(sun_ns_c)),
        tan=cuda.to_device(_c32(tan_sun_el_c)),
        azb=cuda.to_device(_c32i(az_bucket_c)),
        C1=cuda.to_device(_c32(C1_c)),
        C2=cuda.to_device(_c32(C2_c)),
        C3=cuda.to_device(_c32(C3_c)),
        Phi1=cuda.to_device(_c32(Phi1_c)),
        Phi2=cuda.to_device(_c32(Phi2_c)),
        gha=cuda.to_device(_c32(ghi_albedo_c)),
        hor=cuda.to_device(_c32(hor_union)),
        cs=cuda.to_device(_c32(cos_slope)),
        ss=cuda.to_device(_c32(sin_slope)),
        pew=cuda.to_device(_c32(proj_ew)),
        pns=cuda.to_device(_c32(proj_ns)),
        sv=cuda.to_device(_c32(svf)),
        ps=cuda.to_device(_c32i(panel_starts)),
        pe=cuda.to_device(_c32i(panel_ends)),
    )
    d_hourly = cuda.to_device(_c32(hourly_dc))
    d_mon = cuda.to_device(_c32(monthly_dc))
    d_mps = cuda.to_device(_c32(monthly_poa_shaded))
    d_mpu = cuda.to_device(_c32(monthly_poa_unshaded))

    total = n_panels * int(n_chunk)
    blocks = (total + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK
    _panel_accum_kernel_cuda[blocks, _THREADS_PER_BLOCK](
        int(n_chunk),
        d['dni'],
        d['cos_z'],
        d['beam'],
        d['month'],
        d['hour'],
        d['sew'],
        d['sns'],
        d['tan'],
        d['azb'],
        d['C1'],
        d['C2'],
        d['C3'],
        d['Phi1'],
        d['Phi2'],
        d['gha'],
        d['hor'],
        int(n_dirs),
        d['cs'],
        d['ss'],
        d['pew'],
        d['pns'],
        d['sv'],
        d['ps'],
        d['pe'],
        int(n_panels),
        d_hourly,
        d_mon,
        d_mps,
        d_mpu,
    )

    hourly_dc[:] = d_hourly.copy_to_host()
    monthly_dc[:] = d_mon.copy_to_host()
    monthly_poa_shaded[:] = d_mps.copy_to_host()
    monthly_poa_unshaded[:] = d_mpu.copy_to_host()


# ============================================================
# WARM-UP  (compile PTX before the timed run, mirror CPU warmups)
# ============================================================


def warmup_gpu(cell_size: float = 0.1, n_dirs: int = 8) -> None:
    """Trigger CUDA JIT of both kernels with tiny dummy launches."""
    if not GPU_AVAILABLE:
        return
    tiny = np.ones((16, 16), dtype=np.float32) + np.random.rand(16, 16).astype(
        np.float32
    )
    svf_and_horizon_gpu(tiny, cell_size, min(n_dirs, 8), 2.4, None, None)

    _n = 16
    poa_accum_gpu(
        4,
        *([np.zeros(4, np.float32)] * 3),  # dni, cos_z, beam
        np.zeros(4, np.int32),  # month
        *([np.zeros(4, np.float32)] * 3),  # sun_ew, sun_ns, tan
        np.zeros(4, np.int32),  # az_bucket
        *([np.zeros(4, np.float32)] * 6),  # C1..C3, Phi1, Phi2, ghi_albedo
        np.zeros((_n, n_dirs), np.float32),
        n_dirs,
        *([np.ones(_n, np.float32)] * 2),  # cos_slope, sin_slope
        *([np.zeros(_n, np.float32)] * 3),  # proj_ew, proj_ns, svf
        np.zeros(_n, np.float32),
        np.zeros((12, _n), np.float32),
    )

    # panel kernel
    panel_accum_gpu(
        4,
        *([np.zeros(4, np.float32)] * 3),  # dni, cos_z, beam
        np.zeros(4, np.int32),  # month
        np.arange(4, dtype=np.int32),  # hour_idx
        *([np.zeros(4, np.float32)] * 3),  # sun_ew, sun_ns, tan
        np.zeros(4, np.int32),  # az_bucket
        *([np.zeros(4, np.float32)] * 6),  # C1..C3, Phi1, Phi2, ghi_albedo
        np.zeros((_n, n_dirs), np.float32),
        n_dirs,
        *([np.ones(_n, np.float32)] * 2),  # cos_slope, sin_slope
        *([np.zeros(_n, np.float32)] * 3),  # proj_ew, proj_ns, svf
        np.array([0, 8], np.int32),
        np.array([8, _n], np.int32),
        2,  # panel bounds
        np.zeros((2, 8), np.float32),  # hourly_dc (n_hours=8)
        *([np.zeros((2, 12), np.float32)] * 3),  # monthly_dc, poa_shaded, poa_unshaded
    )
