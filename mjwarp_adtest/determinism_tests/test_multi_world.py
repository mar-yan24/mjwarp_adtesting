"""Tests for multi-world determinism and isolation.

Validates that contacts are sorted per-world, that worlds don't interfere
with each other, and that world 0 state is consistent regardless of batch size.
"""

import numpy as np
import pytest

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  collect_contacts_complete,
  collect_full_state,
  make_det_fixture,
  requires_gpu_sm70,
  run_n_times_and_compare,
  run_simulation,
  assert_states_bitwise_equal,
)

pytestmark = [pytest.mark.determinism, requires_gpu_sm70]


class TestPerWorldContactOrder:
  """Each world's contacts are in sorted order."""

  def test_per_world_contacts_sorted(self, nruns):
    """Contacts within each world are sorted by (geom0, geom1)."""
    nworld = 8
    m, d = run_simulation(
      "collision.xml", nworld=nworld, nsteps=10, deterministic=True
    )
    nacon = int(d.nacon.numpy()[0])
    assert nacon > 0

    geom = d.contact.geom.numpy()[:nacon]
    worldid = d.contact.worldid.numpy()[:nacon]

    # Group contacts by world and verify each world is sorted.
    for w in range(nworld):
      mask = worldid == w
      world_geom = geom[mask]
      if len(world_geom) < 2:
        continue
      for i in range(1, len(world_geom)):
        key_prev = (int(world_geom[i - 1, 0]), int(world_geom[i - 1, 1]))
        key_curr = (int(world_geom[i, 0]), int(world_geom[i, 1]))
        assert key_prev <= key_curr, (
          f"World {w}: contacts not sorted at index {i}: "
          f"{key_prev} > {key_curr}"
        )

  def test_per_world_contacts_deterministic(self, nruns):
    """Per-world contact data is bitwise identical across runs."""
    nworld = 4
    results = []
    for _ in range(nruns):
      m, d = run_simulation(
        "collision.xml", nworld=nworld, nsteps=10, deterministic=True
      )
      nacon = int(d.nacon.numpy()[0])
      contacts = collect_contacts_complete(d, nacon)
      results.append(contacts)

    for run in range(1, nruns):
      assert results[0]["nacon"] == results[run]["nacon"]
      np.testing.assert_array_equal(
        results[0]["geom"], results[run]["geom"]
      )
      np.testing.assert_array_equal(
        results[0]["worldid"], results[run]["worldid"]
      )


class TestMultiWorldIsolation:
  """Sorting in one world does not corrupt another."""

  def test_multi_world_isolation(self, nruns):
    """Perturbing world 0 does not affect worlds 1-3 contacts."""
    nworld = 4
    nsteps = 10

    # Reference run: no perturbation.
    m_ref, d_ref = run_simulation(
      "collision.xml", nworld, nsteps, deterministic=True
    )
    nacon_ref = int(d_ref.nacon.numpy()[0])
    geom_ref = d_ref.contact.geom.numpy()[:nacon_ref].copy()
    worldid_ref = d_ref.contact.worldid.numpy()[:nacon_ref].copy()

    # Perturbed run: modify world 0's qpos before simulation.
    _, _, m_pert, d_pert = make_det_fixture(
      "collision.xml", nworld=nworld, deterministic=True
    )
    qpos = d_pert.qpos.numpy()
    qpos[0] += 0.1  # Perturb world 0 only.
    d_pert.qpos.assign(qpos)
    for _ in range(nsteps):
      mjw.step(m_pert, d_pert)

    nacon_pert = int(d_pert.nacon.numpy()[0])
    geom_pert = d_pert.contact.geom.numpy()[:nacon_pert].copy()
    worldid_pert = d_pert.contact.worldid.numpy()[:nacon_pert].copy()

    # Worlds 1-3 contacts should be identical between ref and perturbed.
    for w in range(1, nworld):
      mask_ref = worldid_ref == w
      mask_pert = worldid_pert == w
      np.testing.assert_array_equal(
        geom_ref[mask_ref],
        geom_pert[mask_pert],
        err_msg=f"World {w} contacts differ after perturbing world 0",
      )


class TestNWorldScaling:
  """World 0 state is consistent regardless of batch size."""

  @pytest.mark.parametrize("nworld", [1, 2, 4, 8, 16])
  def test_nworld_scaling_determinism(self, nworld):
    """World 0's qpos/qvel is identical at different nworld values."""
    nsteps = 10
    m, d = run_simulation(
      "collision.xml", nworld=nworld, nsteps=nsteps, deterministic=True
    )
    qpos_w0 = d.qpos.numpy()[0].copy()
    qvel_w0 = d.qvel.numpy()[0].copy()

    # Compare against nworld=1 baseline.
    m_base, d_base = run_simulation(
      "collision.xml", nworld=1, nsteps=nsteps, deterministic=True
    )
    qpos_base = d_base.qpos.numpy()[0].copy()
    qvel_base = d_base.qvel.numpy()[0].copy()

    np.testing.assert_array_equal(
      qpos_w0,
      qpos_base,
      err_msg=f"World 0 qpos differs at nworld={nworld} vs nworld=1",
    )
    np.testing.assert_array_equal(
      qvel_w0,
      qvel_base,
      err_msg=f"World 0 qvel differs at nworld={nworld} vs nworld=1",
    )
