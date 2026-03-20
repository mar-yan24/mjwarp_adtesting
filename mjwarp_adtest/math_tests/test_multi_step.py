"""Multi-step gradient propagation tests.

Verifies that gradients remain stable and correct when propagated
through multiple consecutive step() calls via a single tape.
"""

import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw

from mjwarp_adtest.fixtures.data_factory import make_ad_fixture
from mjwarp_adtest.fixtures.finite_difference import fd_gradient
from mjwarp_adtest.fixtures.loss_functions import sum_xpos_kernel
from mjwarp_adtest.models.xml_defs import SIMPLE_HINGE_XML, SIMPLE_SLIDE_XML


def _skip_if_no_sm70():
  if wp.get_device().is_cuda and wp.get_device().arch < 70:
    pytest.skip("tile kernels (cuSolverDx) require sm_70+")


_MULTI_STEP_MODELS = [
  pytest.param(SIMPLE_HINGE_XML, id="simple_hinge"),
  pytest.param(SIMPLE_SLIDE_XML, id="simple_slide"),
]


@pytest.mark.math
@pytest.mark.gpu_sm70
class TestMultiStepGrad:
  @pytest.mark.xfail(
    reason="Multi-step tape AD: intermediate state arrays are overwritten "
    "across step() calls, so backward produces zero gradients. "
    "Single-step AD works correctly (see TestFullStepGrad).",
    strict=True,
  )
  @pytest.mark.parametrize("xml", _MULTI_STEP_MODELS)
  @pytest.mark.parametrize("nsteps", [2, 5], ids=["2step", "5step"])
  def test_multi_step_gradient(self, xml, nsteps, ad_config):
    """Gradient through multiple consecutive step() calls."""
    _skip_if_no_sm70()
    mjm, mjd, m, d = make_ad_fixture(xml=xml, keyframe=0)

    # AD gradient through nsteps
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      for _ in range(nsteps):
        mjw.step(m, d)
      wp.launch(
        sum_xpos_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[d.xpos, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    # Verify gradient is finite (no explosion/vanishing)
    assert np.all(np.isfinite(ad_grad)), (
      f"Multi-step ({nsteps}) gradient has NaN/Inf: {ad_grad}"
    )
    assert np.any(ad_grad != 0), (
      f"Multi-step ({nsteps}) gradient is all zeros"
    )

    # FD comparison
    def eval_loss(ctrl_np):
      _, _, _, d_fd = make_ad_fixture(xml=xml, keyframe=0)
      d_fd.ctrl = wp.array(ctrl_np.reshape(1, -1), dtype=float)
      for _ in range(nsteps):
        mjw.step(m, d_fd)
      l = wp.zeros(1, dtype=float)
      wp.launch(
        sum_xpos_kernel,
        dim=(d_fd.nworld, m.nbody),
        inputs=[d_fd.xpos, l],
      )
      return l.numpy()[0]

    ctrl_np = mjd.ctrl.copy()
    fd_grad = fd_gradient(eval_loss, ctrl_np, eps=ad_config.fd_eps)

    # Multi-step accumulates error, so use relaxed tolerance
    tol = ad_config.fd_tol * nsteps
    np.testing.assert_allclose(
      ad_grad, fd_grad, atol=tol, rtol=tol,
      err_msg=f"Multi-step ({nsteps}) AD vs FD mismatch",
    )

  @pytest.mark.parametrize("xml", _MULTI_STEP_MODELS)
  def test_gradient_does_not_explode(self, xml, ad_config):
    """Verify gradients don't grow exponentially over 10 steps."""
    _skip_if_no_sm70()
    mjm, mjd, m, d = make_ad_fixture(xml=xml, keyframe=0)

    nsteps = 10
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      for _ in range(nsteps):
        mjw.step(m, d)
      wp.launch(
        sum_xpos_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[d.xpos, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    assert np.all(np.isfinite(ad_grad)), (
      f"10-step gradient has NaN/Inf: {ad_grad}"
    )
    # Gradient magnitude should be reasonable (not exploded)
    assert np.max(np.abs(ad_grad)) < 1e6, (
      f"10-step gradient magnitude too large: {np.max(np.abs(ad_grad)):.2e}"
    )
