"""
perf.py -- CPU-vs-GPU timing instrumentation for the shading pipeline.

Purpose
-------
Run the same job twice -- once forced onto the CPU @njit kernels, once on the
CUDA kernels -- and get a comparable, per-stage breakdown printed to stdout.

Forcing the backend
-------------------
Set `SHADING_FORCE_CPU=1` in the environment to make
`gpu_kernels.GPU_AVAILABLE` False, which sends every dispatch site
(terrain.svf_and_horizon, accumulation_kernels.poa_accum / panel_accum,
full_dsm/facet PoaGpuSession) down its CPU path. Unset (or 0) = normal
auto-detect.

    SHADING_FORCE_CPU=1 python shading_main.py ...   # CPU run
    python shading_main.py ...                       # GPU run (if a device exists)

Reading the output
------------------
Each instrumented stage prints one line when it finishes:

    [perf][CPU] horizon+svf ray-march          12.480s
    [perf][GPU] horizon+svf ray-march           0.910s

and `report()` prints a summary table of every stage plus a total at the end
of a run. Stage timings are cumulative: a stage entered N times (e.g. one
strip/batch per call) reports the SUM of all N and the call count, so the
CPU and GPU numbers are directly comparable even when the two paths chunk
the work differently.

GPU-timing correctness
----------------------
CUDA kernel launches are asynchronous, so naive wall-clock around a launch
measures only the enqueue. `stage()` calls `cuda.synchronize()` before
stopping the clock whenever the GPU backend is active, so the reported time
includes actual kernel execution (and any pending PCIe copies).
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager

# ── backend selection ──────────────────────────────────────────────────────
# Read once at import; gpu_kernels consults this to decide GPU_AVAILABLE.
FORCE_CPU: bool = os.environ.get('SHADING_FORCE_CPU', '').strip().lower() in (
    '1',
    'true',
    'yes',
    'on',
)


def _out():
    """Where timing lines go.

    Default stdout, alongside the pipeline's other progress logs. Set
    SHADING_PERF_STDERR=1 for entry points whose stdout IS a machine-read
    payload -- calculateLayoutFromRoofSegmentMask writes its result JSON to
    fd 1, and a table appended there would corrupt what Node parses.
    """
    if os.environ.get('SHADING_PERF_STDERR', '').strip().lower() in (
        '1',
        'true',
        'yes',
        'on',
    ):
        return sys.stderr
    return sys.stdout


def backend() -> str:
    """'GPU' or 'CPU' -- what the dispatch sites will actually use."""
    from . import gpu_kernels

    return 'GPU' if gpu_kernels.GPU_AVAILABLE else 'CPU'


def _sync_if_gpu() -> None:
    """Block until queued CUDA work completes, so timings are real."""
    from . import gpu_kernels

    if gpu_kernels.GPU_AVAILABLE and gpu_kernels.cuda is not None:
        try:
            gpu_kernels.cuda.synchronize()
        except Exception:  # pragma: no cover - never let timing break a run
            pass


# ── cumulative stage registry: label -> [total_seconds, n_calls] ───────────
_STAGES: 'dict[str, list]' = {}


@contextmanager
def stage(label: str, quiet: bool = False):
    """
    Time a block and record it under `label`.

    quiet=True suppresses the per-entry line (use inside hot loops entered
    many times); the stage still accumulates into the final report().
    """
    _sync_if_gpu()  # don't charge this stage for the previous one's tail
    t = time.perf_counter()
    try:
        yield
    finally:
        _sync_if_gpu()
        dt = time.perf_counter() - t
        rec = _STAGES.setdefault(label, [0.0, 0])
        rec[0] += dt
        rec[1] += 1
        if not quiet:
            print(f'[perf][{backend()}] {label:<38} {dt:8.3f}s', file=_out(), flush=True)


def tic() -> float:
    """
    Start a manual timer (paired with `toc`).

    Use instead of `stage()` where the timed region has two indented
    alternative branches (GPU vs CPU) that would need re-indenting to fit in
    a `with` block. Synchronizes CUDA first, same as `stage()`.
    """
    _sync_if_gpu()
    return time.perf_counter()


def toc(label: str, t_start: float, quiet: bool = True) -> float:
    """Stop a `tic()` timer, record it under `label`, return the elapsed s."""
    _sync_if_gpu()
    dt = time.perf_counter() - t_start
    mark(label, dt)
    if not quiet:
        print(f'[perf][{backend()}] {label:<38} {dt:8.3f}s', file=_out(), flush=True)
    return dt


def mark(label: str, seconds: float) -> None:
    """Record a stage timed by the caller (no context manager available)."""
    rec = _STAGES.setdefault(label, [0.0, 0])
    rec[0] += seconds
    rec[1] += 1


def reset() -> None:
    """Clear all recorded stages (for back-to-back runs in one process)."""
    _STAGES.clear()


def report(title: str = 'shading timing') -> None:
    """Print the cumulative per-stage table and the summed total."""
    b = backend()
    print(f'\n{"=" * 64}', file=_out(), flush=True)
    print(f'  {title} -- backend: {b}'
          f'{"  (forced via SHADING_FORCE_CPU)" if FORCE_CPU else ""}', file=_out(), flush=True)
    print(f'{"=" * 64}', file=_out(), flush=True)
    if not _STAGES:
        print('  (no stages recorded)', file=_out(), flush=True)
        return
    total = 0.0
    for label, (secs, n) in _STAGES.items():
        total += secs
        calls = f'x{n}' if n > 1 else ''
        print(f'  {label:<38} {secs:9.3f}s {calls:>6}', file=_out(), flush=True)
    print(f'  {"-" * 62}', file=_out(), flush=True)
    print(f'  {"TOTAL (instrumented stages)":<38} {total:9.3f}s', file=_out(), flush=True)
    print(f'{"=" * 64}\n', file=_out(), flush=True)


def _quiet() -> bool:
    return os.environ.get('SHADING_PERF_QUIET', '').strip().lower() in (
        '1',
        'true',
        'yes',
        'on',
    )


def _atexit_report() -> None:  # pragma: no cover - process-teardown hook
    if not _STAGES:
        return
    if not _quiet():
        report('shading timing (at exit)')
    # SHADING_PERF_JSON=<path> also dumps the machine-readable snapshot, so a
    # benchmark driver can collect timings from an entry point whose stdout is
    # a result payload (e.g. calculateLayoutFromRoofSegmentMask writes JSON to
    # fd 1) instead of having to parse the printed table.
    path = os.environ.get('SHADING_PERF_JSON', '').strip()
    if path:
        try:
            with open(path, 'w') as fh:
                json.dump(snapshot(), fh)
        except Exception as e:
            print(f'[perf] could not write {path}: {e}', file=_out(), flush=True)


# Print the table automatically when the process ends, so every entry point
# (calculateLayoutFromRoofSegmentMask, dsm_pipeline, ground_mount, a Node
# PythonShell invocation, ...) gets the breakdown without being modified.
# SHADING_PERF_QUIET=1 suppresses the table but still writes SHADING_PERF_JSON;
# SHADING_PERF_STDERR=1 keeps the table but moves it off stdout.
import atexit  # noqa: E402

atexit.register(_atexit_report)


def snapshot() -> 'dict[str, dict]':
    """Machine-readable copy of the registry, e.g. to dump as JSON."""
    return {
        'backend': backend(),
        'forced_cpu': FORCE_CPU,
        'stages': {k: {'seconds': v[0], 'calls': v[1]} for k, v in _STAGES.items()},
    }
