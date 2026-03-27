"""Tests that each integrator type is deterministic with contact sorting.

Validates Euler, RK4, and ImplicitFast integrators all produce bitwise
identical trajectories when deterministic mode is enabled.
"""

import numpy as np
import pytest

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  collect_full_state,
  requires_gpu_sm70,
  run_simulation,
  assert_states_bitwise_equal,
)

pytestmark = [pytest.mark.determinism, requires_gpu_sm70]

_NSTEPS = 50

_MODELS = [
  pytest.param("collision.xml", id="collision"),
  pytest.param("humanoid/humanoid.xml", id="humanoid"),
]

# IntegratorType enum values from mujoco_warp types.
_INTEGRATORS = [
  pytest.param(0, id="euler"),        # EULER = 0
  pytest.param(1, id="rk4"),          # RK4 = 1
  pytest.param(3, id="implicitfast"), # IMPLICITFAST = 3 (2 is IMPLICIT, unsupported)
]


class TestIntegratorDeterminism:
  """Each integrator produces deterministic results."""

  @pytest.mark.parametrize("integrator", _INTEGRATORS)
  @pytest.mark.parametrize("model_id", [p.values[0] for p in _MODELS])
  def test_integrator_deterministic(self, model_id, integrator, nruns):
    """Trajectory is bitwise identical across runs for given integrator."""
    overrides = {"opt.integrator": integrator}

    results = []
    for _ in range(nruns):
      m, d = run_simulation(
        model_id, 1, _NSTEPS, deterministic=True, overrides=overrides
      )
      results.append(collect_full_state(m, d))

    for run in range(1, nruns):
      assert_states_bitwise_equal(
        results[0],
        results[run],
        fields=["qpos", "qvel", "qacc", "qfrc_constraint"],
      )
