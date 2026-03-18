"""Phase 2 constraint solver implicit differentiation tests.

Tests AD through step() with active contacts using Newton solver,
including sparse vs dense Jacobian mode and multiple simultaneous contacts.
"""

import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src.grad import enable_grad

from mjwarp_adtest.fixtures.data_factory import make_ad_fixture
from mjwarp_adtest.fixtures.finite_difference import fd_gradient
from mjwarp_adtest.fixtures.loss_functions import sum_qpos_kernel, sum_xpos_kernel
from mjwarp_adtest.models.xml_defs import (
  CONTACT_SLIDE_DENSE_XML,
  CONTACT_SLIDE_XML,
  MULTI_CONTACT_XML,
  NO_CONTACT_HIGH_XML,
  SIMPLE_HINGE_XML,
)


def _skip_if_no_sm70():
  if wp.get_device().is_cuda and wp.get_device().arch < 70:
    pytest.skip("tile kernels (cuSolverDx) require sm_70+")


@pytest.mark.math
@pytest.mark.gpu_sm70
class TestSolverAdjointContact:
  def test_contact_step_sparse(self, ad_config):
    """dL/dctrl through step() with active contacts (sparse Jacobian)."""
    _skip_if_no_sm70()
    xml = CONTACT_SLIDE_XML
    mjm, mjd, m, d = make_ad_fixture(xml=xml)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      wp.launch(
        sum_qpos_kernel,
        dim=(d.nworld, mjm.nq),
        inputs=[d.qpos, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    def eval_loss(ctrl_np):
      _, _, _, d_fd = make_ad_fixture(xml=xml)
      d_fd.ctrl = wp.array(ctrl_np.reshape(1, -1), dtype=float)
      mjw.step(m, d_fd)
      l = wp.zeros(1, dtype=float)
      wp.launch(
        sum_qpos_kernel, dim=(d_fd.nworld, mjm.nq), inputs=[d_fd.qpos, l],
      )
      return l.numpy()[0]

    ctrl_np = mjd.ctrl.copy()
    fd_grad = fd_gradient(eval_loss, ctrl_np, eps=ad_config.fd_eps)

    np.testing.assert_allclose(
      ad_grad, fd_grad,
      atol=ad_config.contact_fd_tol, rtol=ad_config.contact_fd_tol,
    )

  def test_contact_step_dense(self, ad_config):
    """dL/dctrl through step() with active contacts (dense Jacobian)."""
    _skip_if_no_sm70()
    xml = CONTACT_SLIDE_DENSE_XML
    mjm, mjd, m, d = make_ad_fixture(xml=xml)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      wp.launch(
        sum_qpos_kernel,
        dim=(d.nworld, mjm.nq),
        inputs=[d.qpos, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    def eval_loss(ctrl_np):
      _, _, _, d_fd = make_ad_fixture(xml=xml)
      d_fd.ctrl = wp.array(ctrl_np.reshape(1, -1), dtype=float)
      mjw.step(m, d_fd)
      l = wp.zeros(1, dtype=float)
      wp.launch(
        sum_qpos_kernel, dim=(d_fd.nworld, mjm.nq), inputs=[d_fd.qpos, l],
      )
      return l.numpy()[0]

    ctrl_np = mjd.ctrl.copy()
    fd_grad = fd_gradient(eval_loss, ctrl_np, eps=ad_config.fd_eps)

    np.testing.assert_allclose(
      ad_grad, fd_grad,
      atol=ad_config.contact_fd_tol, rtol=ad_config.contact_fd_tol,
    )


@pytest.mark.math
@pytest.mark.gpu_sm70
class TestSolverAdjointNoContact:
  def test_no_active_constraints(self, ad_config):
    """No active contacts: solver adjoint matches unconstrained result."""
    _skip_if_no_sm70()
    xml = NO_CONTACT_HIGH_XML
    mjm, mjd, m, d = make_ad_fixture(xml=xml)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      wp.launch(
        sum_qpos_kernel,
        dim=(d.nworld, mjm.nq),
        inputs=[d.qpos, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    def eval_loss(ctrl_np):
      _, _, _, d_fd = make_ad_fixture(xml=xml)
      d_fd.ctrl = wp.array(ctrl_np.reshape(1, -1), dtype=float)
      mjw.step(m, d_fd)
      l = wp.zeros(1, dtype=float)
      wp.launch(
        sum_qpos_kernel, dim=(d_fd.nworld, mjm.nq), inputs=[d_fd.qpos, l],
      )
      return l.numpy()[0]

    ctrl_np = mjd.ctrl.copy()
    fd_grad = fd_gradient(eval_loss, ctrl_np, eps=ad_config.fd_eps)

    np.testing.assert_allclose(
      ad_grad, fd_grad,
      atol=ad_config.fd_tol, rtol=ad_config.fd_tol,
    )

  def test_identity_unconstrained(self, ad_config):
    """njmax==0 (constraints disabled): identity pass-through."""
    _skip_if_no_sm70()
    xml = SIMPLE_HINGE_XML
    mjm, mjd, m, d = make_ad_fixture(xml=xml, keyframe=0)

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
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    def eval_loss(ctrl_np):
      _, _, _, d_fd = make_ad_fixture(xml=xml, keyframe=0)
      d_fd.ctrl = wp.array(ctrl_np.reshape(1, -1), dtype=float)
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

    np.testing.assert_allclose(
      ad_grad, fd_grad, atol=ad_config.fd_tol, rtol=ad_config.fd_tol,
    )


@pytest.mark.math
@pytest.mark.gpu_sm70
class TestMultiContact:
  def test_multi_sphere_contact(self, ad_config):
    """Multiple simultaneous contacts: 3 spheres on a plane."""
    _skip_if_no_sm70()
    xml = MULTI_CONTACT_XML
    mjm, mjd, m, d = make_ad_fixture(xml=xml)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.step(m, d)
      wp.launch(
        sum_qpos_kernel,
        dim=(d.nworld, mjm.nq),
        inputs=[d.qpos, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    def eval_loss(ctrl_np):
      _, _, _, d_fd = make_ad_fixture(xml=xml)
      d_fd.ctrl = wp.array(ctrl_np.reshape(1, -1), dtype=float)
      mjw.step(m, d_fd)
      l = wp.zeros(1, dtype=float)
      wp.launch(
        sum_qpos_kernel, dim=(d_fd.nworld, mjm.nq), inputs=[d_fd.qpos, l],
      )
      return l.numpy()[0]

    ctrl_np = mjd.ctrl.copy()
    fd_grad = fd_gradient(eval_loss, ctrl_np, eps=ad_config.fd_eps)

    np.testing.assert_allclose(
      ad_grad, fd_grad,
      atol=ad_config.contact_fd_tol, rtol=ad_config.contact_fd_tol,
    )
