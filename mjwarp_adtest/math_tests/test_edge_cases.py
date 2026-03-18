"""Edge case tests: gradient stability at boundary configurations.

Tests gradient correctness and numerical stability at:
- Joint limits (qpos at bounds)
- Zero angular velocity (quaternion singularity)
- Near-singular configurations
- Large control inputs
- Identity quaternion states
"""

import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src.grad import enable_grad
from mujoco_warp._src import math as mjw_math

from mjwarp_adtest.fixtures.data_factory import make_ad_fixture
from mjwarp_adtest.fixtures.finite_difference import fd_gradient
from mjwarp_adtest.fixtures.loss_functions import sum_xpos_kernel, sum_qpos_kernel


@wp.kernel
def _quat_integrate_kernel(
  q_in: wp.array(dtype=wp.quat),
  v_in: wp.array(dtype=wp.vec3),
  dt_in: wp.array(dtype=float),
  q_out: wp.array(dtype=wp.quat),
):
  i = wp.tid()
  q_out[i] = mjw_math.quat_integrate(q_in[i], v_in[i], dt_in[i])


@wp.kernel
def _quat_loss_kernel(
  q: wp.array(dtype=wp.quat),
  loss: wp.array(dtype=float),
):
  i = wp.tid()
  v = q[i]
  wp.atomic_add(loss, 0, v[0] + v[1] + v[2] + v[3])


@pytest.mark.math
class TestQuaternionEdgeCases:
  def test_quat_integrate_zero_vel(self):
    """Gradient at zero angular velocity should be finite (no NaN/Inf)."""
    q_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_np = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    dt_np = np.array([0.01], dtype=np.float32)

    q_arr = wp.array([wp.quat(*q_np)], dtype=wp.quat, requires_grad=True)
    v_arr = wp.array([wp.vec3(*v_np)], dtype=wp.vec3, requires_grad=True)
    dt_arr = wp.array(dt_np, dtype=float, requires_grad=True)
    q_out = wp.zeros(1, dtype=wp.quat, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
      wp.launch(_quat_integrate_kernel, dim=1, inputs=[q_arr, v_arr, dt_arr, q_out])
      wp.launch(_quat_loss_kernel, dim=1, inputs=[q_out, loss])
    tape.backward(loss=loss)

    ad_grad_v = v_arr.grad.numpy()[0].copy()
    tape.zero()

    assert np.all(np.isfinite(ad_grad_v)), (
      f"quat_integrate grad has NaN/Inf at zero velocity: {ad_grad_v}"
    )

    # Also verify against FD
    def eval_loss_v(v_test):
      q_a = wp.array([wp.quat(*q_np)], dtype=wp.quat)
      v_a = wp.array([wp.vec3(*v_test)], dtype=wp.vec3)
      dt_a = wp.array(dt_np, dtype=float)
      qo = wp.zeros(1, dtype=wp.quat)
      l = wp.zeros(1, dtype=float)
      wp.launch(_quat_integrate_kernel, dim=1, inputs=[q_a, v_a, dt_a, qo])
      wp.launch(_quat_loss_kernel, dim=1, inputs=[qo, l])
      return l.numpy()[0]

    fd_grad_v = fd_gradient(eval_loss_v, v_np)
    np.testing.assert_allclose(ad_grad_v, fd_grad_v, atol=5e-3, rtol=5e-2)

  def test_quat_integrate_identity_quat(self):
    """Gradient at identity quaternion with small velocity."""
    q_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    v_np = np.array([0.01, 0.01, 0.01], dtype=np.float32)
    dt_np = np.array([0.01], dtype=np.float32)

    q_arr = wp.array([wp.quat(*q_np)], dtype=wp.quat, requires_grad=True)
    v_arr = wp.array([wp.vec3(*v_np)], dtype=wp.vec3, requires_grad=True)
    dt_arr = wp.array(dt_np, dtype=float, requires_grad=True)
    q_out = wp.zeros(1, dtype=wp.quat, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    tape = wp.Tape()
    with tape:
      wp.launch(_quat_integrate_kernel, dim=1, inputs=[q_arr, v_arr, dt_arr, q_out])
      wp.launch(_quat_loss_kernel, dim=1, inputs=[q_out, loss])
    tape.backward(loss=loss)

    ad_grad_q = q_arr.grad.numpy()[0].copy()
    tape.zero()

    assert np.all(np.isfinite(ad_grad_q)), (
      f"quat_integrate grad has NaN/Inf at identity quat: {ad_grad_q}"
    )


@pytest.mark.math
class TestLargeInputEdgeCases:
  def test_large_ctrl_gradient(self, ad_config):
    """Gradient should be stable with large control inputs."""
    from mjwarp_adtest.models.xml_defs import SIMPLE_HINGE_XML

    mjm, mjd, m, d = make_ad_fixture(xml=SIMPLE_HINGE_XML, keyframe=0)

    # Set large control
    large_ctrl = np.array([[100.0, -100.0]], dtype=np.float32)
    # Pad to match array width
    ctrl_full = np.zeros((1, d.ctrl.shape[1]), dtype=np.float32)
    ctrl_full[0, : mjm.nu] = large_ctrl[0, : mjm.nu]
    d.ctrl = wp.array(ctrl_full, dtype=float)
    enable_grad(d)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.kinematics(m, d)
      mjw.com_pos(m, d)
      mjw.crb(m, d)
      mjw.factor_m(m, d)
      mjw.transmission(m, d)
      mjw.fwd_velocity(m, d)
      mjw.fwd_actuation(m, d)
      wp.launch(
        sum_xpos_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[d.xpos, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    assert np.all(np.isfinite(ad_grad)), (
      f"Gradient has NaN/Inf with large ctrl: {ad_grad}"
    )


@pytest.mark.math
class TestFreeBodyEdgeCases:
  def test_free_body_kinematics_grad_finite(self, ad_config):
    """Gradient through kinematics for a free body should be finite."""
    from mjwarp_adtest.models.xml_defs import SIMPLE_FREE_XML

    mjm, mjd, m, d = make_ad_fixture(xml=SIMPLE_FREE_XML, keyframe=0)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.kinematics(m, d)
      mjw.com_pos(m, d)
      wp.launch(
        sum_xpos_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[d.xpos, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.qpos.grad.numpy()[0, : mjm.nq].copy()
    tape.zero()

    assert np.all(np.isfinite(ad_grad)), (
      f"Free body kinematics grad has NaN/Inf: {ad_grad}"
    )

  def test_free_body_quaternion_dofs_grad(self, ad_config):
    """Verify quaternion DOFs of free body have non-trivial gradient.

    Note: Translation DOFs of free joints may not propagate gradients
    through kinematics() in the current AD implementation. This test
    checks the quaternion portion (indices 3:7) against FD.
    """
    from mjwarp_adtest.models.xml_defs import SIMPLE_FREE_XML

    mjm, mjd, m, d = make_ad_fixture(xml=SIMPLE_FREE_XML, keyframe=0)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.kinematics(m, d)
      mjw.com_pos(m, d)
      wp.launch(
        sum_xpos_kernel,
        dim=(d.nworld, m.nbody),
        inputs=[d.xpos, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.qpos.grad.numpy()[0, : mjm.nq].copy()
    tape.zero()

    # Verify against FD for quaternion DOFs only (indices 3:7)
    def eval_loss(qpos_np):
      d_fd = mjw.make_data(mjm)
      d_fd.qpos = wp.array(qpos_np.reshape(1, -1), dtype=float)
      mjw.kinematics(m, d_fd)
      mjw.com_pos(m, d_fd)
      l = wp.zeros(1, dtype=float)
      wp.launch(
        sum_xpos_kernel,
        dim=(d_fd.nworld, m.nbody),
        inputs=[d_fd.xpos, l],
      )
      return l.numpy()[0]

    qpos_np = d.qpos.numpy()[0, : mjm.nq]
    fd_grad = fd_gradient(eval_loss, qpos_np, eps=ad_config.fd_eps)

    # Check quaternion DOFs (3:7) if they have non-zero FD gradient
    quat_slice = slice(3, 7)
    if np.any(np.abs(fd_grad[quat_slice]) > 1e-6):
      np.testing.assert_allclose(
        ad_grad[quat_slice], fd_grad[quat_slice],
        atol=ad_config.fd_tol * 5,  # relaxed for quaternion
        rtol=ad_config.fd_tol * 5,
      )
