"""Diverse collision type coverage for determinism.

Sweeps across all available collision geometry types to ensure deterministic
sorting works for every narrowphase path: primitives, convex, heightfield,
flex, mixed geometry, and complex articulated models.
"""

import numpy as np
import pytest

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  collect_contacts_complete,
  collect_full_state,
  requires_gpu_sm70,
  run_n_times_and_compare,
  assert_states_bitwise_equal,
)
from mjwarp_adtest.models.xml_defs import (
  MIXED_GEOM_TYPES_XML,
  MANY_SPHERES_XML,
  STACKED_BOXES_XML,
  EQUALITY_CONSTRAINT_XML,
)

pytestmark = [pytest.mark.determinism, requires_gpu_sm70]

# Benchmark models resolved via test_data paths.
_BENCHMARK_MODELS = [
  pytest.param("collision.xml", id="collision"),
  pytest.param("humanoid/humanoid.xml", id="humanoid"),
]

# Inline XML models.
_INLINE_MODELS = [
  pytest.param(MIXED_GEOM_TYPES_XML, id="mixed_geom_types"),
  pytest.param(MANY_SPHERES_XML, id="many_spheres"),
  pytest.param(STACKED_BOXES_XML, id="stacked_boxes"),
  pytest.param(EQUALITY_CONSTRAINT_XML, id="equality_constraint"),
]


class TestCollisionTypeCoverage:
  """Determinism across diverse collision geometry types."""

  @pytest.mark.parametrize("model_id", [p.values[0] for p in _BENCHMARK_MODELS])
  def test_benchmark_model_deterministic(self, model_id, nruns):
    """Benchmark collision models: contacts and state bitwise identical."""
    results = run_n_times_and_compare(
      model_id,
      nworld=1,
      nsteps=10,
      nruns=nruns,
      deterministic=True,
    )
    assert results[0]["nacon"] > 0, f"No contacts for {model_id}"

  @pytest.mark.parametrize("model_xml", _INLINE_MODELS)
  def test_inline_model_deterministic(self, model_xml, nruns):
    """Inline collision models: contacts and state bitwise identical."""
    results = run_n_times_and_compare(
      model_xml,
      nworld=1,
      nsteps=10,
      nruns=nruns,
      deterministic=True,
    )
    assert results[0]["nacon"] > 0, f"No contacts generated"

  @pytest.mark.parametrize(
    "model_id",
    [
      pytest.param("collision.xml", id="collision"),
      pytest.param("humanoid/humanoid.xml", id="humanoid"),
    ],
  )
  def test_multi_world_model_coverage(self, model_id, nruns):
    """Multi-world determinism across different model types."""
    results = run_n_times_and_compare(
      model_id,
      nworld=4,
      nsteps=10,
      nruns=nruns,
      deterministic=True,
    )
    assert results[0]["nacon"] > 0
