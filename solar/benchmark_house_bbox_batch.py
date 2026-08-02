"""
benchmark_house_bbox_batch.py -- replay ALL captured designs through
run_house_bbox_shading in one process, so JIT/PTX warm-up is paid once and
amortized the same way it would be on a real long-lived server (a per-design
process restart would unfairly re-pay compile cost on every single design).

Usage
-----
    python -m yourpackage.benchmark_house_bbox_batch ./bench_inputs/

    # CPU path, same machine:
    SHADING_FORCE_CPU=1 python -m yourpackage.benchmark_house_bbox_batch ./bench_inputs/

Expects ./bench_inputs/ to contain bench_input_<id>.pkl files (as produced
by prepare_bench_inputs.py). Prints one BENCH_RESULT line per design plus a
final BENCH_SUMMARY line.
"""

from __future__ import annotations

import glob
import json
import os
import pickle
import sys
import time


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python -m yourpackage.benchmark_house_bbox_batch <bench_inputs_dir>', file=sys.stderr)
        sys.exit(1)

    bench_dir = sys.argv[1]
    pkl_paths = sorted(glob.glob(os.path.join(bench_dir, 'bench_input_*.pkl')))
    if not pkl_paths:
        print(f'no bench_input_*.pkl files found in {bench_dir}', file=sys.stderr)
        sys.exit(1)

    from . import gpu_kernels
    from . import perf
    from .house_bbox_accumulation import run_house_bbox_shading

    force_cpu = os.environ.get('SHADING_FORCE_CPU') == '1'
    if not force_cpu and not gpu_kernels.GPU_AVAILABLE:
        print(
            '[bench] WARNING: SHADING_FORCE_CPU is not set but GPU_AVAILABLE '
            'is False -- every design below will silently run on CPU. Check '
            'nvidia-smi / NUMBA_FORCE_CUDA_CC first.',
            file=sys.stderr,
            flush=True,
        )
    mode = 'cpu' if (force_cpu or not gpu_kernels.GPU_AVAILABLE) else 'gpu'
    print(f'[bench] mode={mode}  designs={len(pkl_paths)}', flush=True)

    results = []
    for pkl_path in pkl_paths:
        design_id = os.path.basename(pkl_path).replace('bench_input_', '').replace('.pkl', '')
        with open(pkl_path, 'rb') as f:
            kwargs = pickle.load(f)

        perf.reset()  # isolate this design's stage timings from the previous one
        t0 = time.perf_counter()
        result = run_house_bbox_shading(**kwargs)
        elapsed = time.perf_counter() - t0

        stages = perf.snapshot()['stages']  # {'label': {'seconds':..,'calls':..}, ...}
        row = {
            'BENCH_RESULT': True,
            'design_id': design_id,
            'mode': mode,
            'gpu_available': gpu_kernels.GPU_AVAILABLE,
            'elapsed_seconds': round(elapsed, 4),
            'terrain_solar_seconds': round(stages.get('terrain || solar phase (wall)', {}).get('seconds', 0.0), 4),
            'ray_march_seconds': round(stages.get('horizon+svf ray-march', {}).get('seconds', 0.0), 4),
            'compile_warmup_seconds': round(stages.get('kernel warm-up (compile)', {}).get('seconds', 0.0), 4),
            'accumulation_seconds': round(stages.get('POA+DC accumulation (facets)', {}).get('seconds', 0.0), 4),
            'annual_dc_tiff_present': bool(result.get('annual_dc_tiff_b64')),
        }
        results.append(row)
        print('BENCH_RESULT ' + json.dumps(row), flush=True)

    total = sum(r['elapsed_seconds'] for r in results)
    summary = {
        'BENCH_SUMMARY': True,
        'mode': mode,
        'n_designs': len(results),
        'total_seconds': round(total, 4),
        'mean_seconds': round(total / len(results), 4),
        'total_terrain_solar_seconds': round(sum(r['terrain_solar_seconds'] for r in results), 4),
        'total_ray_march_seconds': round(sum(r['ray_march_seconds'] for r in results), 4),
        'total_compile_warmup_seconds': round(sum(r['compile_warmup_seconds'] for r in results), 4),
        'total_accumulation_seconds': round(sum(r['accumulation_seconds'] for r in results), 4),
        'per_design': {r['design_id']: r['elapsed_seconds'] for r in results},
    }
    print('BENCH_SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
