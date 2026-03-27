"""Scaling characteristics of deterministic contact sorting.

Measures how sort overhead grows with contact count and nworld batch size.
"""

import time

import pytest
import warp as wp

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  make_det_fixture,
  requires_gpu_sm70,
)
from mjwarp_adtest.models.xml_defs import MANY_SPHERES_XML

pytestmark = [
  pytest.mark.determinism,
  pytest.mark.perf,
  pytest.mark.slow,
  requires_gpu_sm70,
]

_WARMUP = 5
_NSTEPS = 100


class _PerfTimer:
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


class TestContactCountScaling:
  """Sort overhead as a function of active contact count."""

  @pytest.mark.parametrize(
    "model,expected_contacts",
    [
      pytest.param("collision.xml", "few", id="few_contacts"),
      pytest.param(MANY_SPHERES_XML, "many", id="many_contacts"),
    ],
  )
  def test_contact_count_scaling(self, model, expected_contacts):
    """Measure step time with varying contact counts."""
    _, _, m_on, d_on = make_det_fixture(model, nworld=1, deterministic=True)
    for _ in range(_WARMUP):
      mjw.step(m_on, d_on)
    nacon = int(d_on.nacon.numpy()[0])

    with _PerfTimer() as t:
      for _ in range(_NSTEPS):
        mjw.step(m_on, d_on)

    time_per_step = t.elapsed / _NSTEPS
    print(
      f"\n{expected_contacts} contacts (nacon={nacon}): "
      f"{time_per_step*1000:.3f} ms/step"
    )


class TestNWorldScaling:
  """Sort overhead as nworld increases."""

  @pytest.mark.parametrize(
    "nworld",
    [
      pytest.param(1, id="nworld_1"),
      pytest.param(16, id="nworld_16"),
      pytest.param(256, id="nworld_256"),
      pytest.param(1024, id="nworld_1024"),
      pytest.param(4096, id="nworld_4096"),
    ],
  )
  def test_nworld_scaling(self, nworld):
    """Measure step time as batch size grows."""
    _, _, m, d = make_det_fixture(
      "collision.xml", nworld=nworld, deterministic=True
    )
    for _ in range(_WARMUP):
      mjw.step(m, d)

    with _PerfTimer() as t:
      for _ in range(_NSTEPS):
        mjw.step(m, d)

    time_per_step = t.elapsed / _NSTEPS
    nacon = int(d.nacon.numpy()[0])
    print(
      f"\nnworld={nworld}, nacon={nacon}: "
      f"{time_per_step*1000:.3f} ms/step"
    )
