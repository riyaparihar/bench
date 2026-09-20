# CPU vs GPU shading benchmark — package layout

## Before you run anything

Two files in `solar/` are **placeholders**, not your real code (they were
never uploaded to me, so I reconstructed only what I could infer from the
other files' docstrings/imports):

- `solar/__init__.py` — real one sets `NUMBA_FORCE_CUDA_CC` before the CUDA
  import; stub only has that one line.
- `solar/constants.py` — real one defines `HOUSE_BBOX_EXPAND_PCT` and
  `FACET_BUFFER_PX`; stub has placeholder values. **Also check whether your
  real `constants.py` actually lives top-level (sibling to `solar/`) rather
  than inside `solar/`** — see the comment inside the stub file for why.

**Replace both with your real files before running.**

## What goes where

```
bench/
├── README.md                          (this file)
├── run_cpu_vs_gpu_batch.sh            → run on the REMOTE box
├── pyproject.toml                     → run on the REMOTE box (deps)
├── bench_inputs/                      → EMPTY. Put your bench_input_*.pkl here.
├── energy/
│   ├── __init__.py
│   └── pvlib_core.py                  → REMOTE box
├── solar/
│   ├── __init__.py                    
│   ├── constants.py                   
│   ├── perf.py                        → REMOTE box
│   ├── gpu_kernels.py                 → REMOTE box
│   ├── terrain.py                     → REMOTE box
│   ├── solar_positions.py             → REMOTE box
│   ├── accumulation_kernels.py        → REMOTE box
│   ├── facet_accumulation.py          → REMOTE box
│   ├── house_bbox_accumulation.py     → REMOTE box (patched w/ capture hook)
│   └── benchmark_house_bbox_batch.py  → REMOTE box
└── local_only/
    ├── prepare_bench_inputs.py        → LOCAL machine only, do NOT upload
    └── designs.json                   → LOCAL machine only, do NOT upload
```

`local_only/` needs network access to your DSM URLs and NSRDB files, which
the remote test box doesn't need at all — it only needs the resulting
`.pkl` files. Keep it off the remote box.

## Step 0 — locally, build the pickles

```bash
cd local_only/
pip install rasterio requests pandas --break-system-packages
python prepare_bench_inputs.py designs.json ../bench_inputs/
```

Fill in real values in `designs.json` first (id, lat, lng, nsrdb_file_pth,
dsm_url, hbbox) — see the comments inside `prepare_bench_inputs.py` for the
exact schema.

## Step 1 — upload to the vast.ai instance

From one level above this `bench/` folder:

```bash
scp -P <port> -r bench/{run_cpu_vs_gpu_batch.sh,pyproject.toml,bench_inputs,energy,solar} \
    root@<host>:/root/bench/
```

(intentionally excludes `local_only/` and `README.md`)

## Step 2 — on the instance

```bash
cd /root/bench
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
export NUMBA_FORCE_CUDA_CC=<whatever nvidia-smi just showed you>   # e.g. 6.1 for GTX 1070

source /venv/main/bin/activate      # this template's preinstalled venv
uv pip install -r pyproject.toml    # or pip install, matching your real setup

chmod +x run_cpu_vs_gpu_batch.sh
./run_cpu_vs_gpu_batch.sh ./bench_inputs solar.benchmark_house_bbox_batch
```

This runs the GPU pass, then the CPU pass (`SHADING_FORCE_CPU=1`), prints a
per-design + aggregate comparison table, and — if the `vastai` CLI and
`$CONTAINER_ID` are available (they are, in the CUDA Dev Environment
template) — **stops the instance automatically** at the end so you don't
have to remember to.

## What's inside a `bench_input_<id>.pkl`

A pickled Python `dict` with exactly the kwargs `run_house_bbox_shading`
takes:

| key | type | content |
|---|---|---|
| `latitude` | `float` | design's latitude |
| `longitude` | `float` | design's longitude |
| `house_bounding_box` | `tuple[float, float, float, float]` | `(min_x, min_y, max_x, max_y)` in DSM pixel space (col, row) |
| `weather_data` | `pd.DataFrame` or `dict` | your NSRDB weather, as loaded from `nsrdb_file_pth` — whatever object your pipeline normally passes through |
| `dsm` | `np.ndarray` | the DSM raster's first band, read from `dsm_url` |
| `dsm_metadata` | `dict` | `{'transform': affine.Affine, 'crs': rasterio.crs.CRS, 'nodata': float or None}` — read off the same raster |

Nothing else is in there — no facet polygons, no roof-segment data, since
`run_house_bbox_shading` only needs the rectangle it builds internally from
`house_bounding_box` + `expand_pct`. If you later want to benchmark
`run_bbox_with_outlier_facets_shading` instead (bbox + outlier roof facets),
the pickle would additionally need a `facets` key — not included here since
you asked specifically about the house-bbox path.
