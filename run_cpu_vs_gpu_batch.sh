#!/usr/bin/env bash
# run_cpu_vs_gpu_batch.sh -- runs benchmark_house_bbox_batch.py twice (GPU,
# then CPU) against the SAME bench_inputs/ directory on THIS machine, and
# prints a per-design + aggregate comparison.
#
# Usage:
#   ./run_cpu_vs_gpu_batch.sh ./bench_inputs yourpackage.benchmark_house_bbox_batch

set -euo pipefail

BENCH_DIR="${1:?usage: run_cpu_vs_gpu_batch.sh <bench_inputs_dir> <module.path>}"
MODULE="${2:?usage: run_cpu_vs_gpu_batch.sh <bench_inputs_dir> <module.path>}"

echo "=== GPU pass ==="
GPU_OUT=$(python -m "$MODULE" "$BENCH_DIR" | tee /dev/stderr)
GPU_SUMMARY=$(echo "$GPU_OUT" | grep '^BENCH_SUMMARY ')

echo
echo "=== CPU pass (SHADING_FORCE_CPU=1) ==="
CPU_OUT=$(SHADING_FORCE_CPU=1 python -m "$MODULE" "$BENCH_DIR" | tee /dev/stderr)
CPU_SUMMARY=$(echo "$CPU_OUT" | grep '^BENCH_SUMMARY ')

echo
echo "=== Per-design + aggregate summary ==="
python3 - "$GPU_SUMMARY" "$CPU_SUMMARY" <<'EOF'
import json, sys

def parse(line):
    return json.loads(line.split("BENCH_SUMMARY ", 1)[1])

gpu = parse(sys.argv[1])
cpu = parse(sys.argv[2])

if gpu['mode'] != 'gpu':
    print("/!\\ GPU pass did not actually use the GPU path -- check nvidia-smi / NUMBA_FORCE_CUDA_CC. This comparison is not valid.")

print(f"{'design':<10}{'gpu (s)':<12}{'cpu (s)':<12}{'speedup':<10}")
for did in gpu['per_design']:
    g = gpu['per_design'][did]
    c = cpu['per_design'].get(did)
    speedup = f"{c/g:.2f}x" if c and g else "n/a"
    print(f"{did:<10}{g:<12}{c if c is not None else 'n/a':<12}{speedup:<10}")

print()
print(f"Total   -- gpu: {gpu['total_seconds']}s   cpu: {cpu['total_seconds']}s   speedup: {cpu['total_seconds']/gpu['total_seconds']:.2f}x")
print(f"Mean/design -- gpu: {gpu['mean_seconds']}s   cpu: {cpu['mean_seconds']}s")

print()
print("=== Stage breakdown (summed across all designs) ===")
print("NOTE: terrain and solar-position run CONCURRENTLY on two threads,")
print("so they're measured as one combined wall-clock phase, not two")
print("separate numbers that would sum correctly.")
print()
stage_rows = [
    ("terrain || solar phase (wall)", "total_terrain_solar_seconds"),
    ("  of which: horizon+svf ray-march", "total_ray_march_seconds"),
    ("kernel warm-up (compile)", "total_compile_warmup_seconds"),
    ("POA+DC accumulation", "total_accumulation_seconds"),
]
print(f"{'stage':<34}{'gpu (s)':<12}{'cpu (s)':<12}{'speedup':<10}")
for label, key in stage_rows:
    g_s = gpu.get(key, 0.0)
    c_s = cpu.get(key, 0.0)
    speedup = f"{c_s/g_s:.2f}x" if g_s else "n/a"
    print(f"{label:<34}{g_s:<12}{c_s:<12}{speedup:<10}")
EOF
