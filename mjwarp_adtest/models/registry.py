"""Model registry: resolves model names to XML strings or file paths."""

import os

from mjwarp_adtest.config import ADTestConfig
from mjwarp_adtest.models.xml_defs import ALL_MODELS

# Benchmark models that live as files in the mujoco_warp repo
_BENCHMARK_PATHS = {
  "humanoid": "mujoco_warp/test_data/humanoid/humanoid.xml",
  "collision": "mujoco_warp/test_data/collision.xml",
  "primitives": "mujoco_warp/test_data/primitives.xml",
  "constraints": "mujoco_warp/test_data/constraints.xml",
  "convex_collision": "mujoco_warp/test_data/convex_collision/box100.xml",
  "hfield": "mujoco_warp/test_data/hfield/hfield.xml",
  "flex_rope": "mujoco_warp/test_data/flex/rope.xml",
}


def get_model(name: str, config: ADTestConfig | None = None) -> str:
  """Returns XML string for inline models, or file path for benchmark models.

  Args:
    name: Model name (e.g., "simple_hinge", "humanoid").
    config: ADTestConfig for resolving benchmark paths.

  Returns:
    XML string or absolute file path.
  """
  if name in ALL_MODELS:
    return ALL_MODELS[name]

  if name in _BENCHMARK_PATHS:
    if config is None:
      config = ADTestConfig()
    path = os.path.join(config.mujoco_warp_root, _BENCHMARK_PATHS[name])
    if not os.path.exists(path):
      raise FileNotFoundError(
        f"Benchmark model '{name}' not found at {path}. "
        f"Set --mujoco-warp-root or MUJOCO_WARP_ROOT."
      )
    return path

  raise KeyError(
    f"Unknown model '{name}'. Available: "
    f"{sorted(list(ALL_MODELS.keys()) + list(_BENCHMARK_PATHS.keys()))}"
  )
