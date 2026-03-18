"""Wall-clock overhead comparison: AD on vs AD off.

For each model, times N steps with gradients disabled vs enabled,
reports the overhead ratio, and separates JIT compilation time.
"""

import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src.grad import enable_grad, disable_grad

from mjwarp_adtest.fixtures.data_factory import make_ad_fixture, make_baseline_fixture
from mjwarp_adtest.fixtures.loss_functions import sum_xpos_kernel
from mjwarp_adtest.models.xml_defs import (
  SIMPLE_HINGE_XML,
  SIMPLE_SLIDE_XML,
  THREE_LINK_HINGE_XML,
  FIVE_LINK_CHAIN_XML,
)
from mjwarp_adtest.perf_tests.conftest import PerfTimer

_OVERHEAD_MODELS = [
  pytest.param(SIMPLE_HINGE_XML, "simple_hinge", id="simple_hinge"),
  pytest.param(SIMPLE_SLIDE_XML, "simple_slide", id="simple_slide"),
  pytest.param(THREE_LINK_HINGE_XML, "three_link_hinge", id="three_link_hinge"),
  pytest.param(FIVE_LINK_CHAIN_XML, "five_link_chain", id="five_link_chain"),
]

_CSV_HEADER = [
  "model", "nworld", "nstep", "time_no_ad", "time_ad", "overhead_ratio", "jit_time",
]


@pytest.mark.perf
@pytest.mark.gpu_sm70
class TestADOverhead:
  @pytest.mark.parametrize("xml,model_name", _OVERHEAD_MODELS)
  def test_step_overhead(self, xml, model_name, ad_config, csv_writer):
    """Measure wall-clock overhead of AD-enabled step() vs baseline."""
    if wp.get_device().is_cuda and wp.get_device().arch < 70:
      pytest.skip("step() tile kernels require sm_70+")
    nstep = 100  # Fewer steps for overhead test (not full perf_nstep)
    nworld = 1

    # --- Baseline (no AD) ---
    mjm, mjd, m, d_base = make_baseline_fixture(xml=xml, keyframe=0, nworld=nworld)

    # Warmup (includes JIT)
    with PerfTimer() as jit_timer:
      for _ in range(ad_config.warmup_steps):
        mjw.step(m, d_base)

    # Timed run
    # Re-create to get clean state
    _, _, _, d_base = make_baseline_fixture(xml=xml, keyframe=0, nworld=nworld)
    with PerfTimer() as base_timer:
      for _ in range(nstep):
        mjw.step(m, d_base)

    # --- AD enabled ---
    _, _, _, d_ad = make_ad_fixture(xml=xml, keyframe=0, nworld=nworld)

    # Warmup with tape
    for _ in range(ad_config.warmup_steps):
      loss = wp.zeros(1, dtype=float, requires_grad=True)
      tape = wp.Tape()
      with tape:
        mjw.step(m, d_ad)
        wp.launch(
          sum_xpos_kernel,
          dim=(d_ad.nworld, m.nbody),
          inputs=[d_ad.xpos, loss],
        )
      tape.backward(loss=loss)
      tape.zero()

    # Timed run
    _, _, _, d_ad = make_ad_fixture(xml=xml, keyframe=0, nworld=nworld)
    with PerfTimer() as ad_timer:
      for _ in range(nstep):
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        tape = wp.Tape()
        with tape:
          mjw.step(m, d_ad)
          wp.launch(
            sum_xpos_kernel,
            dim=(d_ad.nworld, m.nbody),
            inputs=[d_ad.xpos, loss],
          )
        tape.backward(loss=loss)
        tape.zero()

    overhead = ad_timer.elapsed / max(base_timer.elapsed, 1e-9)

    csv_writer(
      "ad_overhead.csv",
      [model_name, nworld, nstep, f"{base_timer.elapsed:.4f}",
       f"{ad_timer.elapsed:.4f}", f"{overhead:.2f}", f"{jit_timer.elapsed:.4f}"],
      header=_CSV_HEADER,
    )

    # Sanity: AD should not be more than 100x slower
    assert overhead < 100, (
      f"AD overhead {overhead:.1f}x exceeds 100x threshold for {model_name}"
    )
