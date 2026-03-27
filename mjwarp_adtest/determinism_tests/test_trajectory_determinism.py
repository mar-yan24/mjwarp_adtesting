"""Multi-step trajectory determinism tests.

Tests that trajectories remain bitwise identical over medium (100-step) and
long (500-step) horizons, where chaotic dynamics would amplify any
non-determinism from unsorted contacts.
"""

import numpy as np
import pytest

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  requires_gpu_sm70,
  run_simulation_trajectory,
)

pytestmark = [pytest.mark.determinism, requires_gpu_sm70]


class TestMediumHorizon:
  """100-step trajectory determinism."""

  @pytest.mark.parametrize(
    "model_id",
    [
      pytest.param("collision.xml", id="collision"),
      pytest.param("humanoid/humanoid.xml", id="humanoid"),
    ],
  )
  def test_medium_horizon_trajectory(self, model_id, nruns, medium_horizon):
    """Full trajectory (qpos, qvel at each step) identical across runs."""
    trajectories = []
    for _ in range(nruns):
      qpos_traj, qvel_traj = run_simulation_trajectory(
        model_id,
        nworld=1,
        nsteps=medium_horizon,
        deterministic=True,
      )
      trajectories.append((qpos_traj, qvel_traj))

    for run in range(1, nruns):
      np.testing.assert_array_equal(
        trajectories[0][0],
        trajectories[run][0],
        err_msg=f"qpos trajectory differs: run 0 vs run {run}",
      )
      np.testing.assert_array_equal(
        trajectories[0][1],
        trajectories[run][1],
        err_msg=f"qvel trajectory differs: run 0 vs run {run}",
      )


class TestLongHorizonChaotic:
  """Long-horizon tests where chaos amplifies non-determinism."""

  @pytest.mark.slow
  @pytest.mark.parametrize(
    "model_id",
    [
      pytest.param("humanoid/humanoid.xml", id="humanoid"),
      pytest.param("collision.xml", id="collision"),
    ],
  )
  def test_long_horizon_chaotic_trajectory(
    self, model_id, nruns, long_horizon
  ):
    """500-step trajectory remains bitwise identical under chaotic dynamics."""
    trajectories = []
    for _ in range(nruns):
      qpos_traj, qvel_traj = run_simulation_trajectory(
        model_id,
        nworld=1,
        nsteps=long_horizon,
        deterministic=True,
      )
      trajectories.append((qpos_traj, qvel_traj))

    for run in range(1, nruns):
      np.testing.assert_array_equal(
        trajectories[0][0],
        trajectories[run][0],
        err_msg=f"qpos diverged at long horizon: run 0 vs run {run}",
      )
      np.testing.assert_array_equal(
        trajectories[0][1],
        trajectories[run][1],
        err_msg=f"qvel diverged at long horizon: run 0 vs run {run}",
      )


class TestDivergenceWithoutDeterminism:
  """Sanity check that the flag actually matters on non-deterministic hardware."""

  @pytest.mark.xfail(
    strict=False,
    reason=(
      "On some hardware/drivers, GPU results are already deterministic "
      "without explicit sorting. This test validates that the flag has "
      "an observable effect when hardware non-determinism is present."
    ),
  )
  @pytest.mark.parametrize(
    "model_id",
    [
      pytest.param("collision.xml", id="collision"),
    ],
  )
  def test_trajectory_divergence_without_determinism(self, model_id):
    """Without deterministic flag, trajectories may differ across runs."""
    nruns = 5
    nsteps = 50

    trajectories = []
    for _ in range(nruns):
      qpos_traj, _ = run_simulation_trajectory(
        model_id,
        nworld=1,
        nsteps=nsteps,
        deterministic=False,
      )
      trajectories.append(qpos_traj)

    # Check if ANY pair of runs diverge.
    any_different = False
    for i in range(1, nruns):
      if not np.array_equal(trajectories[0], trajectories[i]):
        any_different = True
        break

    assert any_different, (
      "All runs were identical even without deterministic flag. "
      "This is expected on some hardware but means we can't verify "
      "the flag's effect here."
    )
