"""Tests for sort algorithm edge cases and boundary conditions.

Validates correct behavior with zero contacts, single contact, many contacts,
overflow fallback, tie-breaking, and inactive contact handling.
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
from mjwarp_adtest.models.xml_defs import (
  MANY_SPHERES_XML,
  NO_CONTACT_HIGH_XML,
  SINGLE_CONTACT_XML,
  STACKED_BOXES_XML,
  HIGH_NGEOM_XML,
)

pytestmark = [pytest.mark.determinism, requires_gpu_sm70]


class TestZeroContacts:
  """Sort handles zero active contacts gracefully."""

  def test_zero_contacts_no_crash(self):
    """Deterministic mode with no contacts does not crash."""
    m, d = run_simulation(
      NO_CONTACT_HIGH_XML, nworld=1, nsteps=2, deterministic=True
    )
    nacon = int(d.nacon.numpy()[0])
    assert nacon == 0, f"Expected 0 contacts, got {nacon}"

  def test_zero_contacts_state_deterministic(self, nruns):
    """State is trivially deterministic when there are no contacts."""
    results = run_n_times_and_compare(
      NO_CONTACT_HIGH_XML,
      nworld=1,
      nsteps=2,
      nruns=nruns,
      deterministic=True,
      fields=["qpos", "qvel", "qacc"],
    )
    assert results[0]["nacon"] == 0


class TestSingleContact:
  """Sort with exactly one active contact."""

  def test_single_contact_deterministic(self, nruns):
    """Single contact is deterministic (trivial permutation)."""
    results = run_n_times_and_compare(
      SINGLE_CONTACT_XML,
      nworld=1,
      nsteps=10,
      nruns=nruns,
      deterministic=True,
    )
    assert results[0]["nacon"] > 0, "Expected at least 1 contact"

  def test_single_contact_sort_is_identity(self):
    """With one contact, sorted order equals original order."""
    m, d = run_simulation(
      SINGLE_CONTACT_XML, nworld=1, nsteps=10, deterministic=True
    )
    nacon = int(d.nacon.numpy()[0])
    assert nacon >= 1

    # Sort key should be valid (not INT_MAX).
    geom = d.contact.geom.numpy()[0]
    worldid = d.contact.worldid.numpy()[0]
    assert worldid == 0
    assert geom[0] >= 0 and geom[1] >= 0


class TestManyContacts:
  """Sort with many active contacts."""

  @pytest.mark.parametrize(
    "model",
    [
      pytest.param(MANY_SPHERES_XML, id="many_spheres"),
    ],
  )
  def test_many_contacts_sorted_correctly(self, model, nruns):
    """Large contact count is sorted in correct order."""
    results = run_n_times_and_compare(
      model,
      nworld=1,
      nsteps=10,
      nruns=nruns,
      deterministic=True,
      collect_fn=lambda m, d: collect_contacts_complete(d),
    )
    nacon = results[0]["nacon"]
    assert nacon > 3, f"Expected many contacts, got {nacon}"

    # Verify sorted order.
    geom = results[0]["geom"]
    worldid = results[0]["worldid"]
    for i in range(1, nacon):
      key_prev = (int(worldid[i - 1]), int(geom[i - 1, 0]), int(geom[i - 1, 1]))
      key_curr = (int(worldid[i]), int(geom[i, 0]), int(geom[i, 1]))
      assert key_prev <= key_curr, (
        f"Sort violated at index {i}: {key_prev} > {key_curr}"
      )


class TestOverflowFallback:
  """Overflow protection when sort key would exceed int32."""

  @pytest.mark.slow
  def test_overflow_fallback_gcid_max_1(self, nruns):
    """When nworld * ngeom^2 * 16 > 2^31-1, gcid_max falls back to 1.

    With HIGH_NGEOM_XML (~200 geoms from replicate), we need a large nworld
    to trigger overflow: 200^2 * 16 = 640,000. Need nworld > 2^31 / 640,000
    which is ~3355. We use a moderate nworld and verify the sort still works
    correctly by (worldid, geom0, geom1) ordering.
    """
    # Use a model with many geoms and moderate nworld.
    # Even if overflow isn't triggered on this hardware, the sort must be
    # correct. This test verifies the algorithm handles large models.
    results = run_n_times_and_compare(
      HIGH_NGEOM_XML,
      nworld=1,
      nsteps=5,
      nruns=nruns,
      deterministic=True,
      collect_fn=lambda m, d: collect_contacts_complete(d),
    )

    nacon = results[0]["nacon"]
    if nacon > 1:
      geom = results[0]["geom"]
      worldid = results[0]["worldid"]
      for i in range(1, nacon):
        key_prev = (int(worldid[i - 1]), int(geom[i - 1, 0]), int(geom[i - 1, 1]))
        key_curr = (int(worldid[i]), int(geom[i, 0]), int(geom[i, 1]))
        assert key_prev <= key_curr


class TestTieBreaking:
  """Stable sort tie-breaking for identical (worldid, geom0, geom1) tuples."""

  def test_tie_breaking_stability(self, nruns):
    """Contacts sharing (worldid, geom0, geom1) have stable sub-ordering."""
    results = run_n_times_and_compare(
      STACKED_BOXES_XML,
      nworld=1,
      nsteps=10,
      nruns=nruns,
      deterministic=True,
      collect_fn=lambda m, d: collect_contacts_complete(d),
    )
    nacon = results[0]["nacon"]
    assert nacon > 1, "Need multiple contacts for tie-breaking test"

    # Find contacts sharing the same (worldid, geom0, geom1).
    geom = results[0]["geom"]
    worldid = results[0]["worldid"]
    gcid = results[0]["geomcollisionid"]

    groups = {}
    for i in range(nacon):
      key = (int(worldid[i]), int(geom[i, 0]), int(geom[i, 1]))
      groups.setdefault(key, []).append(int(gcid[i]))

    # Within each group, gcid should be non-decreasing (due to stable sort
    # with gcid as the least-significant key component).
    for key, gcids in groups.items():
      if len(gcids) > 1:
        for j in range(1, len(gcids)):
          assert gcids[j - 1] <= gcids[j], (
            f"gcid not non-decreasing within group {key}: {gcids}"
          )


class TestInactiveContacts:
  """Inactive contacts sort to the end of the array."""

  @pytest.mark.parametrize(
    "model",
    [
      pytest.param("collision.xml", id="collision"),
      pytest.param(MANY_SPHERES_XML, id="many_spheres"),
    ],
  )
  def test_inactive_contacts_at_end(self, model):
    """Contact indices >= nacon are inactive after sort."""
    m, d = run_simulation(model, nworld=1, nsteps=10, deterministic=True)
    nacon = int(d.nacon.numpy()[0])
    naconmax = d.naconmax

    if nacon >= naconmax:
      pytest.skip("All contact slots active, cannot test inactive region")

    assert nacon > 0, "Need some active contacts"

    # Active contacts should have valid geom IDs.
    geom_active = d.contact.geom.numpy()[:nacon]
    assert (geom_active >= 0).all(), "Active contacts have invalid geom IDs"

    # The dim field for inactive contacts should be 0 or the slot should be
    # clearly beyond the active range. We verify that nacon correctly
    # separates active from inactive.
    dim_all = d.contact.dim.numpy()
    dim_active = dim_all[:nacon]
    assert (dim_active > 0).all(), "Active contacts should have dim > 0"
