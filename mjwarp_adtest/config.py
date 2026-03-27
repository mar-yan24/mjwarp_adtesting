"""Configuration dataclasses for the AD and determinism testing suites."""

import dataclasses
import os


@dataclasses.dataclass
class DeterminismTestConfig:
  nruns: int = 3
  short_horizon: int = 10
  medium_horizon: int = 100
  long_horizon: int = 500
  perf_nstep: int = 200
  perf_warmup: int = 10
  overhead_threshold: float = 2.0
  mujoco_warp_root: str = "C:/Projects/mujoco_warp"
  results_dir: str = "results"

  def __post_init__(self):
    self.nruns = int(os.environ.get("DETTEST_NRUNS", self.nruns))
    self.short_horizon = int(
      os.environ.get("DETTEST_SHORT_HORIZON", self.short_horizon)
    )
    self.medium_horizon = int(
      os.environ.get("DETTEST_MEDIUM_HORIZON", self.medium_horizon)
    )
    self.long_horizon = int(
      os.environ.get("DETTEST_LONG_HORIZON", self.long_horizon)
    )
    self.mujoco_warp_root = os.environ.get(
      "MUJOCO_WARP_ROOT", self.mujoco_warp_root
    )
    self.results_dir = os.environ.get("DETTEST_RESULTS_DIR", self.results_dir)


@dataclasses.dataclass
class ADTestConfig:
  fd_eps: float = 1e-3
  fd_tol: float = 1e-3
  contact_fd_tol: float = 1e-2
  mujoco_warp_root: str = "C:/Projects/mujoco_warp"
  results_dir: str = "results"
  default_nworld: int = 1
  perf_nworld: int = 4096
  perf_nstep: int = 1000
  warmup_steps: int = 5
  render_width: int = 640
  render_height: int = 480
  taylor_h_values: tuple = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)

  def __post_init__(self):
    # Allow env overrides
    self.fd_eps = float(os.environ.get("ADTEST_FD_EPS", self.fd_eps))
    self.fd_tol = float(os.environ.get("ADTEST_FD_TOL", self.fd_tol))
    self.contact_fd_tol = float(
      os.environ.get("ADTEST_CONTACT_FD_TOL", self.contact_fd_tol)
    )
    self.mujoco_warp_root = os.environ.get(
      "MUJOCO_WARP_ROOT", self.mujoco_warp_root
    )
    self.results_dir = os.environ.get("ADTEST_RESULTS_DIR", self.results_dir)
