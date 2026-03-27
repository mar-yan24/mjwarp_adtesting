"""Performance overhead measurement for deterministic contact sorting.

Measures wall-clock overhead of deterministic=True vs False for both the
full step() pipeline and the isolated collision() call.
"""

import time

import pytest
import warp as wp

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  make_det_fixture,
  requires_gpu_sm70,
)

pytestmark = [pytest.mark.determinism, pytest.mark.perf, requires_gpu_sm70]

_WARMUP = 10
_NSTEPS = 200


class _PerfTimer:
  """GPU-aware timing context manager."""

  def __init__(self):
    self.elapsed = 0.0

  def __enter__(self):
    wp.synchronize()
    self._start = time.perf_counter()
    return self

  def __exit__(self, *args):
    wp.synchronize()
    self._end = time.perf_counter()
    self.elapsed = self._end - self._start


class TestStepOverhead:
  """Wall-clock overhead of deterministic step()."""

  @pytest.mark.parametrize(
    "model_id",
    [
      pytest.param("collision.xml", id="collision"),
      pytest.param("humanoid/humanoid.xml", id="humanoid"),
    ],
  )
  def test_step_overhead(self, model_id, det_config):
    """Deterministic step overhead is within acceptable bounds."""
    # Baseline: deterministic OFF.
    _, _, m_off, d_off = make_det_fixture(
      model_id, nworld=1, deterministic=False
    )
    for _ in range(_WARMUP):
      mjw.step(m_off, d_off)
    with _PerfTimer() as t_off:
      for _ in range(_NSTEPS):
        mjw.step(m_off, d_off)

    # Deterministic ON.
    _, _, m_on, d_on = make_det_fixture(
      model_id, nworld=1, deterministic=True
    )
    for _ in range(_WARMUP):
      mjw.step(m_on, d_on)
    with _PerfTimer() as t_on:
      for _ in range(_NSTEPS):
        mjw.step(m_on, d_on)

    overhead = t_on.elapsed / max(t_off.elapsed, 1e-9)
    print(
      f"\n{model_id}: OFF={t_off.elapsed:.4f}s, "
      f"ON={t_on.elapsed:.4f}s, overhead={overhead:.2f}x"
    )

    assert overhead < det_config.overhead_threshold, (
      f"Deterministic overhead {overhead:.2f}x exceeds threshold "
      f"{det_config.overhead_threshold}x for {model_id}"
    )

  @pytest.mark.parametrize(
    "model_id",
    [
      pytest.param("collision.xml", id="collision"),
    ],
  )
  def test_collision_only_overhead(self, model_id, det_config):
    """Isolated collision() overhead from sorting."""
    _, _, m_off, d_off = make_det_fixture(
      model_id, nworld=1, deterministic=False
    )
    # Initialize positions so collision has something to process.
    mjw.fwd_position(m_off, d_off)
    with _PerfTimer() as t_off:
      for _ in range(_NSTEPS):
        mjw.fwd_position(m_off, d_off)

    _, _, m_on, d_on = make_det_fixture(
      model_id, nworld=1, deterministic=True
    )
    mjw.fwd_position(m_on, d_on)
    with _PerfTimer() as t_on:
      for _ in range(_NSTEPS):
        mjw.fwd_position(m_on, d_on)

    overhead = t_on.elapsed / max(t_off.elapsed, 1e-9)
    print(
      f"\n{model_id} fwd_position: OFF={t_off.elapsed:.4f}s, "
      f"ON={t_on.elapsed:.4f}s, overhead={overhead:.2f}x"
    )

    assert overhead < det_config.overhead_threshold
