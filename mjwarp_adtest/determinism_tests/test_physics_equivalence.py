"""Tests that deterministic ON vs OFF produce equivalent physics.

Sorting contacts should not change the numerical results - only the ordering
in contact arrays. The integrated state, energy, and contact counts should
be identical (or very close) between the two modes.
"""

import numpy as np
import pytest

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  collect_full_state,
  requires_gpu_sm70,
  run_simulation,
  assert_states_close,
)

pytestmark = [pytest.mark.determinism, requires_gpu_sm70]

_MODELS = [
  pytest.param("collision.xml", id="collision"),
  pytest.param("humanoid/humanoid.xml", id="humanoid"),
]

_NWORLDS = [1, 4]


class TestPhysicsEquivalence:
  """Deterministic ON vs OFF should produce identical physics."""

  @pytest.mark.parametrize("model_id", [p.values[0] for p in _MODELS])
  def test_on_off_same_physics(self, model_id):
    """Integrated state is numerically close between ON and OFF."""
    nsteps = 10

    m_on, d_on = run_simulation(model_id, 1, nsteps, deterministic=True)
    state_on = collect_full_state(m_on, d_on)

    m_off, d_off = run_simulation(model_id, 1, nsteps, deterministic=False)
    state_off = collect_full_state(m_off, d_off)

    # State fields should be very close - sorting reorders contacts but
    # the solver may accumulate in a different order, producing small
    # floating-point differences (observed ~1e-5 for qacc on humanoid).
    assert_states_close(
      state_on,
      state_off,
      atol=1e-3,
      rtol=1e-3,
      fields=["qpos", "qvel", "qacc", "qfrc_bias", "qfrc_smooth"],
    )

  @pytest.mark.parametrize("nworld", _NWORLDS)
  @pytest.mark.parametrize("model_id", [p.values[0] for p in _MODELS])
  def test_contact_count_equivalence(self, model_id, nworld):
    """nacon is identical between ON and OFF from the same initial state."""
    nsteps = 10

    _, d_on = run_simulation(model_id, nworld, nsteps, deterministic=True)
    nacon_on = int(d_on.nacon.numpy()[0])

    _, d_off = run_simulation(model_id, nworld, nsteps, deterministic=False)
    nacon_off = int(d_off.nacon.numpy()[0])

    assert nacon_on == nacon_off, (
      f"nacon differs: ON={nacon_on}, OFF={nacon_off}"
    )

  @pytest.mark.parametrize("model_id", [p.values[0] for p in _MODELS])
  def test_energy_conservation_equivalence(self, model_id):
    """Energy (potential + kinetic) is identical between ON and OFF."""
    nsteps = 10

    _, d_on = run_simulation(model_id, 1, nsteps, deterministic=True)
    energy_on = d_on.energy.numpy()[0].copy()

    _, d_off = run_simulation(model_id, 1, nsteps, deterministic=False)
    energy_off = d_off.energy.numpy()[0].copy()

    np.testing.assert_allclose(
      energy_on,
      energy_off,
      atol=1e-6,
      rtol=1e-6,
      err_msg="Energy differs between deterministic ON and OFF",
    )
