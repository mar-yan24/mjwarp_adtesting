"""Configuration dataclass for the AD testing suite."""

import dataclasses
import os


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
