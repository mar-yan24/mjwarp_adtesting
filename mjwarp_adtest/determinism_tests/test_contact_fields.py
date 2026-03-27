"""Tests that all permuted contact fields are deterministic after sorting.

Closes the coverage gap where the existing determinism_test.py only checks
7 of the 17 permuted fields. Also validates sort key monotonicity, efc_address
validity, and contact count consistency.
"""

import numpy as np
import pytest

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  CONTACT_FIELDS,
  collect_contacts_complete,
  make_det_fixture,
  requires_gpu_sm70,
  run_n_times_and_compare,
  run_simulation,
)

pytestmark = [pytest.mark.determinism, requires_gpu_sm70]


# Models that generate contacts.
_CONTACT_MODELS = [
  pytest.param("collision.xml", id="collision"),
  pytest.param("humanoid/humanoid.xml", id="humanoid"),
]

_INLINE_CONTACT_MODELS = [
  pytest.param("many_spheres", id="many_spheres"),
  pytest.param("mixed_geom_types", id="mixed_geom_types"),
]

_NWORLDS = [1, 4]


def _get_model_source(model_id):
  """Resolve model_id to path or XML string."""
  from mjwarp_adtest.models.xml_defs import DETERMINISM_MODELS

  if model_id in DETERMINISM_MODELS:
    return DETERMINISM_MODELS[model_id]
  return model_id


class TestAllPermutedFields:
  """Verify all 17 contact fields that _sort_contacts permutes."""

  @pytest.mark.parametrize("nworld", _NWORLDS)
  @pytest.mark.parametrize(
    "model_id",
    ["collision.xml", "humanoid/humanoid.xml", "many_spheres", "mixed_geom_types"],
  )
  def test_all_permuted_fields_deterministic(self, model_id, nworld, nruns):
    """All 17 contact fields are bitwise identical across repeated runs."""
    source = _get_model_source(model_id)

    def collect_contacts(m, d):
      return collect_contacts_complete(d)

    results = run_n_times_and_compare(
      source,
      nworld=nworld,
      nsteps=10,
      nruns=nruns,
      deterministic=True,
      collect_fn=collect_contacts,
    )

    # Verify contacts were generated.
    assert results[0]["nacon"] > 0, f"No contacts for {model_id} nworld={nworld}"

  @pytest.mark.parametrize(
    "model_id", ["collision.xml", "humanoid/humanoid.xml"]
  )
  def test_efc_address_valid_after_sort(self, model_id):
    """efc_address values are valid constraint indices or -1 after sort."""
    m, d = run_simulation(model_id, nworld=1, nsteps=10, deterministic=True)
    nacon = int(d.nacon.numpy()[0])
    if nacon == 0:
      pytest.skip("No contacts generated")

    efc_address = d.contact.efc_address.numpy()[:nacon]
    nefc = int(d.nefc.numpy()[0])

    # Each efc_address entry should be -1 (no constraint) or in [0, nefc).
    valid = (efc_address == -1) | ((efc_address >= 0) & (efc_address < nefc))
    assert valid.all(), (
      f"Invalid efc_address values found. Range: [{efc_address.min()}, "
      f"{efc_address.max()}], nefc={nefc}"
    )

  @pytest.mark.parametrize(
    "model_id", ["collision.xml", "stacked_boxes"]
  )
  def test_sort_key_full_monotonicity(self, model_id):
    """Full composite key (worldid, geom0, geom1, gcid) is non-decreasing."""
    source = _get_model_source(model_id)
    m, d = run_simulation(source, nworld=1, nsteps=10, deterministic=True)
    nacon = int(d.nacon.numpy()[0])
    assert nacon > 1, f"Need >1 contacts for ordering test, got {nacon}"

    geom = d.contact.geom.numpy()[:nacon]
    worldid = d.contact.worldid.numpy()[:nacon]
    gcid = d.contact.geomcollisionid.numpy()[:nacon]

    for i in range(1, nacon):
      key_prev = (int(worldid[i - 1]), int(geom[i - 1, 0]), int(geom[i - 1, 1]), int(gcid[i - 1]))
      key_curr = (int(worldid[i]), int(geom[i, 0]), int(geom[i, 1]), int(gcid[i]))
      assert key_prev <= key_curr, (
        f"Sort order violated at index {i}: {key_prev} > {key_curr}"
      )

  def test_deterministic_flag_default_false(self):
    """The deterministic flag defaults to False."""
    _, _, m, _ = make_det_fixture("collision.xml", deterministic=False)
    # Undo our explicit set - check the model default.
    from mujoco_warp import test_data

    _, _, m_default, _ = test_data.fixture(path="collision.xml")
    assert not m_default.opt.deterministic

  @pytest.mark.parametrize("nworld", _NWORLDS)
  @pytest.mark.parametrize("model_id", ["collision.xml", "humanoid/humanoid.xml"])
  def test_nacon_identical_across_runs(self, model_id, nworld, nruns):
    """Contact count is identical across deterministic runs."""
    results = []
    for _ in range(nruns):
      m, d = run_simulation(model_id, nworld, nsteps=10, deterministic=True)
      results.append(int(d.nacon.numpy()[0]))

    assert results[0] > 0, f"No contacts for {model_id}"
    assert all(r == results[0] for r in results), (
      f"nacon varies across runs: {results}"
    )
