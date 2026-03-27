"""Tests that solver and constraint outputs are deterministic.

Validates that Newton and CG solvers produce identical constraint forces,
and that solver iteration count is consistent across deterministic runs.
"""

import numpy as np
import pytest

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  make_det_fixture,
  requires_gpu_sm70,
  run_simulation,
)

pytestmark = [pytest.mark.determinism, requires_gpu_sm70]

_COLLISION_MODELS = [
  pytest.param("collision.xml", id="collision"),
  pytest.param("humanoid/humanoid.xml", id="humanoid"),
]


def _collect_solver_state(m, d):
  """Collect solver-relevant state from Data."""
  nefc = int(d.nefc.numpy()[0])
  return {
    "nefc": nefc,
    "qfrc_constraint": d.qfrc_constraint.numpy().copy(),
    "efc_force": d.efc.force.numpy().copy(),
    "efc_type": d.efc.type.numpy().copy(),
    "efc_pos": d.efc.pos.numpy().copy(),
    "efc_D": d.efc.D.numpy().copy(),
    "efc_aref": d.efc.aref.numpy().copy(),
    "solver_niter": d.solver_niter.numpy().copy()
    if hasattr(d, "solver_niter")
    else None,
  }


class TestNewtonSolverDeterminism:
  """Newton solver produces deterministic results."""

  @pytest.mark.parametrize("model_id", [p.values[0] for p in _COLLISION_MODELS])
  def test_newton_solver_deterministic(self, model_id, nruns):
    """Constraint forces are bitwise identical with Newton solver."""
    overrides = {"opt.solver": 2}  # NEWTON = 2
    results = []
    for _ in range(nruns):
      m, d = run_simulation(
        model_id, 1, 10, deterministic=True, overrides=overrides
      )
      results.append(_collect_solver_state(m, d))

    for run in range(1, nruns):
      assert results[0]["nefc"] == results[run]["nefc"]
      np.testing.assert_array_equal(
        results[0]["qfrc_constraint"],
        results[run]["qfrc_constraint"],
        err_msg=f"qfrc_constraint differs: run 0 vs {run}",
      )
      np.testing.assert_array_equal(
        results[0]["efc_force"],
        results[run]["efc_force"],
        err_msg=f"efc_force differs: run 0 vs {run}",
      )


class TestCGSolverDeterminism:
  """CG solver produces deterministic results."""

  @pytest.mark.parametrize("model_id", [p.values[0] for p in _COLLISION_MODELS])
  def test_cg_solver_deterministic(self, model_id, nruns):
    """Constraint forces are bitwise identical with CG solver."""
    overrides = {"opt.solver": 1}  # CG = 1
    results = []
    for _ in range(nruns):
      m, d = run_simulation(
        model_id, 1, 10, deterministic=True, overrides=overrides
      )
      results.append(_collect_solver_state(m, d))

    for run in range(1, nruns):
      assert results[0]["nefc"] == results[run]["nefc"]
      np.testing.assert_array_equal(
        results[0]["qfrc_constraint"],
        results[run]["qfrc_constraint"],
        err_msg=f"qfrc_constraint differs: run 0 vs {run}",
      )
      np.testing.assert_array_equal(
        results[0]["efc_force"],
        results[run]["efc_force"],
        err_msg=f"efc_force differs: run 0 vs {run}",
      )


class TestSolverIterationDeterminism:
  """Solver iteration count is deterministic."""

  @pytest.mark.parametrize("model_id", [p.values[0] for p in _COLLISION_MODELS])
  def test_solver_niter_deterministic(self, model_id, nruns):
    """Number of solver iterations is identical across deterministic runs."""
    results = []
    for _ in range(nruns):
      m, d = run_simulation(model_id, 1, 10, deterministic=True)
      if hasattr(d, "solver_niter"):
        results.append(d.solver_niter.numpy().copy())
      else:
        pytest.skip("solver_niter not available on Data")

    for run in range(1, nruns):
      np.testing.assert_array_equal(
        results[0],
        results[run],
        err_msg=f"solver_niter differs: run 0 vs {run}",
      )
