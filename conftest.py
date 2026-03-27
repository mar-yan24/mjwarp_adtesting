"""Root pytest conftest for the AD testing suite.

Mirrors mujoco_warp's conftest options and adds AD-specific configuration.
"""

import os

import pytest
import warp as wp

from mjwarp_adtest.config import ADTestConfig, DeterminismTestConfig


def pytest_addoption(parser):
  # --- mujoco_warp-compatible options ---
  parser.addoption(
    "--cpu", action="store_true", default=False, help="run tests on CPU"
  )
  parser.addoption(
    "--verify_cuda",
    action="store_true",
    default=False,
    help="enable CUDA error checking",
  )
  parser.addoption(
    "--kernel_cache_dir",
    action="store",
    default=None,
    help="directory for compiled warp kernel cache",
  )
  parser.addoption(
    "--lineinfo",
    action="store_true",
    default=False,
    help="add line info to warp kernels",
  )
  parser.addoption(
    "--optimization_level",
    action="store",
    default=None,
    type=int,
    help="set wp.config.optimization_level",
  )
  parser.addoption(
    "--debug_mode",
    action="store_true",
    default=False,
    help="debug mode compilation",
  )

  # --- AD test-specific options ---
  parser.addoption(
    "--fd-eps",
    action="store",
    default=1e-3,
    type=float,
    help="finite-difference epsilon",
  )
  parser.addoption(
    "--fd-tol",
    action="store",
    default=1e-3,
    type=float,
    help="AD vs FD tolerance",
  )
  parser.addoption(
    "--contact-fd-tol",
    action="store",
    default=1e-2,
    type=float,
    help="AD vs FD tolerance for contact tests",
  )
  parser.addoption(
    "--results-dir",
    action="store",
    default="results",
    help="directory for test output",
  )
  parser.addoption(
    "--save-results",
    action="store_true",
    default=False,
    help="save test results to files",
  )
  parser.addoption(
    "--mujoco-warp-root",
    action="store",
    default="C:/Projects/mujoco_warp",
    help="path to mujoco_warp source root",
  )

  # --- Determinism test-specific options ---
  parser.addoption(
    "--det-nruns",
    action="store",
    default=3,
    type=int,
    help="number of repeated runs for determinism checks",
  )
  parser.addoption(
    "--det-nsteps",
    action="store",
    default=10,
    type=int,
    help="default step count for determinism tests",
  )


def pytest_configure(config):
  kernel_cache_dir = config.getoption("--kernel_cache_dir")
  if kernel_cache_dir:
    wp.config.kernel_cache_dir = os.path.expanduser(kernel_cache_dir)
  if config.getoption("--cpu"):
    wp.set_device("cpu")
  if config.getoption("--verify_cuda"):
    wp.config.verify_cuda = True
  if config.getoption("--lineinfo"):
    wp.config.lineinfo = True
  opt_level = config.getoption("--optimization_level")
  if opt_level is not None:
    wp.config.optimization_level = opt_level
  if config.getoption("--debug_mode"):
    wp.config.mode = "debug"


@pytest.fixture(scope="session")
def ad_config(request):
  """Session-scoped ADTestConfig built from command-line options."""
  return ADTestConfig(
    fd_eps=request.config.getoption("--fd-eps"),
    fd_tol=request.config.getoption("--fd-tol"),
    contact_fd_tol=request.config.getoption("--contact-fd-tol"),
    mujoco_warp_root=request.config.getoption("--mujoco-warp-root"),
    results_dir=request.config.getoption("--results-dir"),
  )


@pytest.fixture(scope="session")
def det_config(request):
  """Session-scoped DeterminismTestConfig built from command-line options."""
  return DeterminismTestConfig(
    nruns=request.config.getoption("--det-nruns"),
    short_horizon=request.config.getoption("--det-nsteps"),
    mujoco_warp_root=request.config.getoption("--mujoco-warp-root"),
    results_dir=request.config.getoption("--results-dir"),
  )


@pytest.fixture(scope="session")
def results_dir(ad_config, request):
  """Creates timestamped results directory when --save-results is set."""
  if not request.config.getoption("--save-results"):
    return None

  import datetime

  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  base = os.path.join(ad_config.results_dir, timestamp)

  for subdir in ("math", "perf", "visual", "reports"):
    os.makedirs(os.path.join(base, subdir), exist_ok=True)

  return base
