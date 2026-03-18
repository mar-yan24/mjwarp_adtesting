# mjwarp_adtesting

Autodifferentiation testing suite for [mujoco_warp](https://github.com/google-deepmind/mujoco_warp). Validates AD correctness, performance, and visual fidelity across three dimensions:

- **Mathematical correctness**: AD vs finite-difference gradient comparison, Taylor convergence, chain rule verification
- **Performance overhead**: Wall-clock timing, GPU memory, scaling with DOFs and batch size
- **Visual verification**: Trajectory parity (AD on/off), gradient sensitivity heatmaps

## Setup

```bash
# Install in editable mode with all extras
pip install -e ".[visual,analysis]"
```

Requires `mujoco_warp` installed or available at `C:\Projects\mujoco_warp` (configurable via `--mujoco-warp-root` or `MUJOCO_WARP_ROOT`).

## Usage

```bash
# Run all non-slow tests
pytest mjwarp_adtest/ -v -m "not slow"

# Run by category
pytest mjwarp_adtest/math_tests/ -v -m math
pytest mjwarp_adtest/perf_tests/ -v -m perf --save-results
pytest mjwarp_adtest/visual_tests/ -v -m visual --save-results

# Skip tests requiring SM 70+ GPU
pytest mjwarp_adtest/ -v -m "not gpu_sm70"

# Run on CPU
pytest mjwarp_adtest/ -v --cpu

# Generate report from saved results
python scripts/generate_report.py --results-dir results
```

## Test Categories

| Marker | Description |
|--------|------------|
| `math` | Mathematical correctness (AD vs FD, Taylor, chain rule) |
| `perf` | Performance benchmarking (overhead, memory, scaling) |
| `visual` | Visual verification (trajectory parity, gradient viz) |
| `slow` | Long-running tests (humanoid, batch scaling) |
| `gpu_sm70` | Requires CUDA compute capability 7.0+ |

## Configuration

Command-line options:

| Option | Default | Description |
|--------|---------|------------|
| `--fd-eps` | 1e-3 | Finite-difference perturbation |
| `--fd-tol` | 1e-3 | AD vs FD tolerance |
| `--contact-fd-tol` | 1e-2 | Tolerance for contact tests |
| `--results-dir` | results | Output directory |
| `--save-results` | false | Save results to files |
| `--mujoco-warp-root` | C:/Projects/mujoco_warp | mujoco_warp source root |

Environment variable overrides: `ADTEST_FD_EPS`, `ADTEST_FD_TOL`, `ADTEST_CONTACT_FD_TOL`, `MUJOCO_WARP_ROOT`, `ADTEST_RESULTS_DIR`.
