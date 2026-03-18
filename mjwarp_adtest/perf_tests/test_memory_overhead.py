"""GPU memory overhead comparison: AD on vs AD off.

Measures memory footprint of Data objects with and without gradient tracking.
"""

import numpy as np
import pytest
import warp as wp

import mujoco
import mujoco_warp as mjw

from mjwarp_adtest.fixtures.loss_functions import sum_xpos_kernel
from mjwarp_adtest.models.xml_defs import (
  SIMPLE_HINGE_XML,
  THREE_LINK_HINGE_XML,
  FIVE_LINK_CHAIN_XML,
)
from mjwarp_adtest.perf_tests.conftest import PerfTimer

_MEMORY_MODELS = [
  pytest.param(SIMPLE_HINGE_XML, "simple_hinge", id="simple_hinge"),
  pytest.param(THREE_LINK_HINGE_XML, "three_link_hinge", id="three_link_hinge"),
  pytest.param(FIVE_LINK_CHAIN_XML, "five_link_chain", id="five_link_chain"),
]

_CSV_HEADER = ["model", "mem_no_ad_bytes", "mem_ad_bytes", "overhead_bytes", "overhead_ratio"]


def _estimate_data_memory(d):
  """Estimate memory used by a Data object's warp arrays."""
  total = 0
  for name in dir(d):
    if name.startswith("_"):
      continue
    attr = getattr(d, name, None)
    if isinstance(attr, wp.array):
      total += attr.capacity
  return total


@pytest.mark.perf
class TestMemoryOverhead:
  @pytest.mark.parametrize("xml,model_name", _MEMORY_MODELS)
  def test_data_memory_overhead(self, xml, model_name, csv_writer):
    """Compare memory footprint of make_data vs make_diff_data."""
    mjm = mujoco.MjModel.from_xml_string(xml)

    d_base = mjw.make_data(mjm)
    mem_base = _estimate_data_memory(d_base)

    d_diff = mjw.make_diff_data(mjm)
    mem_diff = _estimate_data_memory(d_diff)

    overhead = mem_diff - mem_base
    ratio = mem_diff / max(mem_base, 1)

    csv_writer(
      "memory_overhead.csv",
      [model_name, mem_base, mem_diff, overhead, f"{ratio:.2f}"],
      header=_CSV_HEADER,
    )

  @pytest.mark.parametrize("xml,model_name", _MEMORY_MODELS)
  def test_tape_memory(self, xml, model_name, csv_writer):
    """Measure additional memory from tape recording + backward pass."""
    if wp.get_device().is_cuda and wp.get_device().arch < 70:
      pytest.skip("tile kernels require sm_70+")

    from mjwarp_adtest.fixtures.data_factory import make_ad_fixture

    mjm, mjd, m, d = make_ad_fixture(xml=xml, keyframe=0)

    mem_before = _estimate_data_memory(d)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      wp.launch(
        sum_xpos_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[d.xpos, loss],
      )
    tape.backward(loss=loss)

    mem_after = _estimate_data_memory(d)
    tape_overhead = mem_after - mem_before

    csv_writer(
      "memory_overhead.csv",
      [f"{model_name}_tape", mem_before, mem_after, tape_overhead, "N/A"],
      header=_CSV_HEADER,
    )

    tape.zero()
