"""Tests that the full forward/step pipeline is deterministic.

Goes beyond contact-level verification to check that qpos, qvel, qacc,
forces, body positions, and constraint data are all bitwise identical
across repeated deterministic runs.
"""

import numpy as np
import pytest

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  collect_full_state,
  make_det_fixture,
  requires_gpu_sm70,
  run_n_times_and_compare,
  run_simulation,
  assert_states_bitwise_equal,
  STATE_FIELDS,
)

pytestmark = [pytest.mark.determinism, requires_gpu_sm70]

_COLLISION_MODELS = [
  pytest.param("collision.xml", id="collision"),
  pytest.param("humanoid/humanoid.xml", id="humanoid"),
]

_NWORLDS = [1, 4]


class TestFullStateDeterminism:
  """Full simulation state is deterministic after step()."""

  @pytest.mark.parametrize("nworld", _NWORLDS)
  @pytest.mark.parametrize("model_id", [p.values[0] for p in _COLLISION_MODELS])
  def test_full_state_after_step(self, model_id, nworld, nruns):
    """All state fields are bitwise identical across runs."""
    results = run_n_times_and_compare(
      model_id,
      nworld=nworld,
      nsteps=10,
      nruns=nruns,
      deterministic=True,
    )
    assert results[0]["nacon"] > 0, f"No contacts for {model_id}"

  @pytest.mark.parametrize("model_id", [p.values[0] for p in _COLLISION_MODELS])
  def test_constraint_data_deterministic(self, model_id, nruns):
    """Constraint fields are bitwise identical after deterministic step."""

    def collect_constraints(m, d):
      nefc = int(d.nefc.numpy()[0])
      state = {}
      state["nefc"] = nefc
      state["efc_type"] = d.efc.type.numpy().copy()
      state["efc_pos"] = d.efc.pos.numpy().copy()
      state["efc_aref"] = d.efc.aref.numpy().copy()
      state["efc_D"] = d.efc.D.numpy().copy()
      state["efc_force"] = d.efc.force.numpy().copy()
      return state

    results = []
    for _ in range(nruns):
      m, d = run_simulation(model_id, 1, 10, deterministic=True)
      results.append(collect_constraints(m, d))

    for run in range(1, nruns):
      assert results[0]["nefc"] == results[run]["nefc"], (
        f"nefc mismatch: {results[0]['nefc']} vs {results[run]['nefc']}"
      )
      for field in (
        "efc_type",
        "efc_pos",
        "efc_aref",
        "efc_D",
        "efc_force",
      ):
        np.testing.assert_array_equal(
          results[0][field],
          results[run][field],
          err_msg=f"{field} differs: run 0 vs run {run}",
        )


class TestForwardOnlyDeterminism:
  """forward() without integration is deterministic."""

  @pytest.mark.parametrize("model_id", [p.values[0] for p in _COLLISION_MODELS])
  def test_forward_only_deterministic(self, model_id, nruns):
    """Calling forward() without step produces identical state."""
    results = []
    for _ in range(nruns):
      _, _, m, d = make_det_fixture(model_id, nworld=1, deterministic=True)
      mjw.forward(m, d)
      results.append(collect_full_state(m, d))

    for run in range(1, nruns):
      assert_states_bitwise_equal(results[0], results[run])

  @pytest.mark.parametrize("model_id", [p.values[0] for p in _COLLISION_MODELS])
  def test_fwd_position_only(self, model_id, nruns):
    """fwd_position() alone (narrowest collision slice) is deterministic."""
    results = []
    for _ in range(nruns):
      _, _, m, d = make_det_fixture(model_id, nworld=1, deterministic=True)
      mjw.fwd_position(m, d)
      state = {}
      state["nacon"] = int(d.nacon.numpy()[0])
      state["xpos"] = d.xpos.numpy().copy()
      state["xquat"] = d.xquat.numpy().copy()
      from mjwarp_adtest.determinism_tests.conftest import (
        CONTACT_FIELDS,
        collect_contacts_complete,
      )

      contacts = collect_contacts_complete(d, nacon=state["nacon"])
      for k, v in contacts.items():
        state[f"contact_{k}"] = v
      results.append(state)

    for run in range(1, nruns):
      assert_states_bitwise_equal(results[0], results[run])
