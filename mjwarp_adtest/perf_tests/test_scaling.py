"""Scaling tests: how AD overhead changes with DOFs and batch size.

Tests DOF scaling (1-DOF slide through 5-DOF chain) and
batch scaling (nworld from 1 to 8192) on a fixed model.
"""

import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw

from mjwarp_adtest.fixtures.data_factory import make_ad_fixture, make_baseline_fixture
from mjwarp_adtest.fixtures.loss_functions import sum_xpos_kernel
from mjwarp_adtest.models.xml_defs import (
  SIMPLE_SLIDE_XML,
  SIMPLE_HINGE_XML,
  THREE_LINK_HINGE_XML,
  FIVE_LINK_CHAIN_XML,
)
from mjwarp_adtest.perf_tests.conftest import PerfTimer

_DOF_MODELS = [
  pytest.param(SIMPLE_SLIDE_XML, "simple_slide", 1, id="1dof"),
  pytest.param(SIMPLE_HINGE_XML, "simple_hinge", 2, id="2dof"),
  pytest.param(THREE_LINK_HINGE_XML, "three_link_hinge", 3, id="3dof"),
  pytest.param(FIVE_LINK_CHAIN_XML, "five_link_chain", 5, id="5dof"),
]

_SCALING_CSV_HEADER = [
  "model", "ndof", "nworld", "nstep", "time_no_ad", "time_ad", "overhead_ratio",
]


@pytest.mark.perf
@pytest.mark.slow
@pytest.mark.gpu_sm70
class TestDOFScaling:
  @pytest.mark.parametrize("xml,model_name,ndof", _DOF_MODELS)
  def test_dof_scaling(self, xml, model_name, ndof, ad_config, csv_writer):
    """Measure AD overhead as DOFs increase."""
    if wp.get_device().is_cuda and wp.get_device().arch < 70:
      pytest.skip("step() tile kernels require sm_70+")
    nstep = 50
    nworld = 1

    # Baseline
    mjm, _, m, d_base = make_baseline_fixture(xml=xml, keyframe=0, nworld=nworld)
    for _ in range(ad_config.warmup_steps):
      mjw.step(m, d_base)

    _, _, _, d_base = make_baseline_fixture(xml=xml, keyframe=0, nworld=nworld)
    with PerfTimer() as base_timer:
      for _ in range(nstep):
        mjw.step(m, d_base)

    # AD
    _, _, _, d_ad = make_ad_fixture(xml=xml, keyframe=0, nworld=nworld)
    for _ in range(ad_config.warmup_steps):
      loss = wp.zeros(1, dtype=float, requires_grad=True)
      tape = wp.Tape()
      with tape:
        mjw.step(m, d_ad)
        wp.launch(sum_xpos_kernel, dim=(d_ad.nworld, m.nbody), inputs=[d_ad.xpos, loss])
      tape.backward(loss=loss)
      tape.zero()

    _, _, _, d_ad = make_ad_fixture(xml=xml, keyframe=0, nworld=nworld)
    with PerfTimer() as ad_timer:
      for _ in range(nstep):
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        tape = wp.Tape()
        with tape:
          mjw.step(m, d_ad)
          wp.launch(sum_xpos_kernel, dim=(d_ad.nworld, m.nbody), inputs=[d_ad.xpos, loss])
        tape.backward(loss=loss)
        tape.zero()

    overhead = ad_timer.elapsed / max(base_timer.elapsed, 1e-9)

    csv_writer(
      "scaling.csv",
      [model_name, ndof, nworld, nstep,
       f"{base_timer.elapsed:.4f}", f"{ad_timer.elapsed:.4f}", f"{overhead:.2f}"],
      header=_SCALING_CSV_HEADER,
    )


_BATCH_SIZES = [1, 16, 256, 1024, 4096]


@pytest.mark.perf
@pytest.mark.slow
@pytest.mark.gpu_sm70
class TestBatchScaling:
  @pytest.mark.parametrize("nworld", _BATCH_SIZES, ids=[f"nw{n}" for n in _BATCH_SIZES])
  def test_batch_scaling(self, nworld, ad_config, csv_writer):
    """Measure AD overhead as nworld increases (fixed model)."""
    if wp.get_device().is_cuda and wp.get_device().arch < 70:
      pytest.skip("step() tile kernels require sm_70+")
    xml = SIMPLE_HINGE_XML
    model_name = "simple_hinge"
    nstep = 20

    # Baseline
    mjm, _, m, d_base = make_baseline_fixture(xml=xml, keyframe=0, nworld=nworld)
    for _ in range(ad_config.warmup_steps):
      mjw.step(m, d_base)

    _, _, _, d_base = make_baseline_fixture(xml=xml, keyframe=0, nworld=nworld)
    with PerfTimer() as base_timer:
      for _ in range(nstep):
        mjw.step(m, d_base)

    # AD
    _, _, _, d_ad = make_ad_fixture(xml=xml, keyframe=0, nworld=nworld)
    for _ in range(ad_config.warmup_steps):
      loss = wp.zeros(1, dtype=float, requires_grad=True)
      tape = wp.Tape()
      with tape:
        mjw.step(m, d_ad)
        wp.launch(sum_xpos_kernel, dim=(d_ad.nworld, m.nbody), inputs=[d_ad.xpos, loss])
      tape.backward(loss=loss)
      tape.zero()

    _, _, _, d_ad = make_ad_fixture(xml=xml, keyframe=0, nworld=nworld)
    with PerfTimer() as ad_timer:
      for _ in range(nstep):
        loss = wp.zeros(1, dtype=float, requires_grad=True)
        tape = wp.Tape()
        with tape:
          mjw.step(m, d_ad)
          wp.launch(sum_xpos_kernel, dim=(d_ad.nworld, m.nbody), inputs=[d_ad.xpos, loss])
        tape.backward(loss=loss)
        tape.zero()

    overhead = ad_timer.elapsed / max(base_timer.elapsed, 1e-9)
    ndof = mjm.nv

    csv_writer(
      "scaling.csv",
      [model_name, ndof, nworld, nstep,
       f"{base_timer.elapsed:.4f}", f"{ad_timer.elapsed:.4f}", f"{overhead:.2f}"],
      header=_SCALING_CSV_HEADER,
    )
