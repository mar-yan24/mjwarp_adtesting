"""Robustness and stress tests for deterministic contact sorting.

Tests flag toggling mid-simulation, repeated forward calls, and other
edge-case usage patterns.
"""

import numpy as np
import pytest

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  collect_full_state,
  make_det_fixture,
  requires_gpu_sm70,
  assert_states_bitwise_equal,
)

pytestmark = [pytest.mark.determinism, requires_gpu_sm70]


class TestFlagToggling:
  """Deterministic flag can be toggled mid-simulation."""

  def test_flag_toggle_mid_simulation(self, nruns):
    """After toggling OFF->ON mid-simulation, the ON segment is deterministic."""
    nsteps_off = 10
    nsteps_on = 10

    results = []
    for _ in range(nruns):
      _, _, m, d = make_det_fixture(
        "collision.xml", nworld=1, deterministic=False
      )
      # Run without determinism first.
      for _ in range(nsteps_off):
        mjw.step(m, d)

      # Enable determinism and run more steps.
      m.opt.deterministic = True
      for _ in range(nsteps_on):
        mjw.step(m, d)

      results.append(collect_full_state(m, d))

    # The post-toggle segment should be deterministic.
    for run in range(1, nruns):
      assert_states_bitwise_equal(
        results[0],
        results[run],
        fields=["qpos", "qvel", "qacc"],
      )

  def test_flag_toggle_on_off_on(self, nruns):
    """Toggle True->False->True: final segment is deterministic."""
    results = []
    for _ in range(nruns):
      _, _, m, d = make_det_fixture(
        "collision.xml", nworld=1, deterministic=True
      )
      # Phase 1: deterministic ON.
      for _ in range(10):
        mjw.step(m, d)

      # Phase 2: deterministic OFF.
      m.opt.deterministic = False
      for _ in range(10):
        mjw.step(m, d)

      # Phase 3: deterministic ON again.
      m.opt.deterministic = True
      for _ in range(10):
        mjw.step(m, d)

      results.append(collect_full_state(m, d))

    # The final state should be deterministic.
    for run in range(1, nruns):
      assert_states_bitwise_equal(
        results[0],
        results[run],
        fields=["qpos", "qvel", "qacc"],
      )


class TestRepeatedForward:
  """Multiple forward() calls without step() are deterministic."""

  def test_repeated_forward_calls(self, nruns):
    """Calling forward() 5 times without step() produces identical state."""
    results = []
    for _ in range(nruns):
      _, _, m, d = make_det_fixture(
        "collision.xml", nworld=1, deterministic=True
      )
      for _ in range(5):
        mjw.forward(m, d)
      results.append(collect_full_state(m, d))

    for run in range(1, nruns):
      assert_states_bitwise_equal(results[0], results[run])
