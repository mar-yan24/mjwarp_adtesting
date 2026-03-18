"""Phase 1.5 conditional wp.clone() cost measurement.

Measures the overhead of euler() with requires_grad=True vs False,
isolating the cost of the conditional clone guards added in Phase 1.5.
"""

import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw

from mjwarp_adtest.fixtures.data_factory import make_ad_fixture, make_baseline_fixture
from mjwarp_adtest.perf_tests.conftest import PerfTimer

_CLONE_MODELS = [
  pytest.param(
    """
    <mujoco>
      <option gravity="0 0 -9.81" jacobian="sparse">
        <flag contact="disable" constraint="disable"/>
      </option>
      <worldbody>
        <body>
          <joint name="j0" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.1" mass="1"/>
          <body pos="0 0 -0.5">
            <joint name="j1" type="hinge" axis="0 1 0"/>
            <geom type="sphere" size="0.1" mass="1"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <motor joint="j0" gear="1"/>
        <motor joint="j1" gear="1"/>
      </actuator>
      <keyframe>
        <key qpos="0.5 -0.3" qvel="0.1 -0.2" ctrl="0.5 -0.5"/>
      </keyframe>
    </mujoco>
    """,
    "simple_hinge",
    id="simple_hinge",
  ),
]

_CSV_HEADER = [
  "model", "nstep", "time_no_grad", "time_grad", "clone_overhead_ratio",
]


@pytest.mark.perf
@pytest.mark.gpu_sm70
class TestCloneOverhead:
  @pytest.mark.parametrize("xml,model_name", _CLONE_MODELS)
  def test_euler_clone_overhead(self, xml, model_name, ad_config, csv_writer):
    """Measure euler() cost with requires_grad True vs False."""
    if wp.get_device().is_cuda and wp.get_device().arch < 70:
      pytest.skip("forward() tile kernels require sm_70+")
    nstep = 200

    # Without grad (no clone guards)
    mjm, _, m, d_base = make_baseline_fixture(xml=xml, keyframe=0)
    # Warmup: run forward first to set up state
    mjw.forward(m, d_base)
    for _ in range(ad_config.warmup_steps):
      mjw.euler(m, d_base)

    _, _, _, d_base = make_baseline_fixture(xml=xml, keyframe=0)
    mjw.forward(m, d_base)
    with PerfTimer() as base_timer:
      for _ in range(nstep):
        mjw.euler(m, d_base)

    # With grad (clone guards active)
    _, _, _, d_ad = make_ad_fixture(xml=xml, keyframe=0)
    mjw.forward(m, d_ad)
    for _ in range(ad_config.warmup_steps):
      mjw.euler(m, d_ad)

    _, _, _, d_ad = make_ad_fixture(xml=xml, keyframe=0)
    mjw.forward(m, d_ad)
    with PerfTimer() as ad_timer:
      for _ in range(nstep):
        mjw.euler(m, d_ad)

    overhead = ad_timer.elapsed / max(base_timer.elapsed, 1e-9)

    csv_writer(
      "clone_overhead.csv",
      [model_name, nstep, f"{base_timer.elapsed:.4f}",
       f"{ad_timer.elapsed:.4f}", f"{overhead:.2f}"],
      header=_CSV_HEADER,
    )
