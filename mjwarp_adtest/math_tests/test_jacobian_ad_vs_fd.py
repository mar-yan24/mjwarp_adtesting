"""AD vs finite-difference gradient comparison across models and pipeline stages.

This is the core correctness test. For each combination of model, input variable,
and pipeline stage, we compare the AD gradient against central-difference FD.
"""

import json
import os

import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src.grad import enable_grad

from mjwarp_adtest.fixtures.data_factory import make_ad_fixture
from mjwarp_adtest.fixtures.finite_difference import fd_gradient
from mjwarp_adtest.fixtures.loss_functions import (
  sum_qacc_kernel,
  sum_qfrc_actuator_kernel,
  sum_qfrc_bias_kernel,
  sum_qpos_kernel,
  sum_xpos_kernel,
)
from mjwarp_adtest.models.xml_defs import (
  FIVE_LINK_CHAIN_XML,
  SIMPLE_HINGE_XML,
  SIMPLE_SLIDE_XML,
  SPRING_DAMPER_XML,
  THREE_LINK_HINGE_XML,
)


# ---- Kinematics: dL/dqpos through kinematics() ----

_KINEMATICS_MODELS = [
  pytest.param(SIMPLE_HINGE_XML, id="simple_hinge"),
  pytest.param(SIMPLE_SLIDE_XML, id="simple_slide"),
  pytest.param(THREE_LINK_HINGE_XML, id="three_link_hinge"),
  pytest.param(FIVE_LINK_CHAIN_XML, id="five_link_chain"),
  pytest.param(SPRING_DAMPER_XML, id="spring_damper"),
]


@pytest.mark.math
class TestKinematicsGrad:
  @pytest.mark.parametrize("xml", _KINEMATICS_MODELS)
  def test_dL_dqpos_kinematics(self, xml, ad_config):
    """dL/dqpos through kinematics(): loss = sum(xpos)."""
    mjm, mjd, m, d = make_ad_fixture(xml=xml, keyframe=0)

    # AD gradient
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

    # FD gradient
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

    np.testing.assert_allclose(
      ad_grad, fd_grad, atol=ad_config.fd_tol, rtol=ad_config.fd_tol,
    )


# ---- Velocity: dL/dqvel through fwd_velocity() ----

_VELOCITY_MODELS = [
  pytest.param(THREE_LINK_HINGE_XML, id="three_link_hinge"),
  pytest.param(SIMPLE_SLIDE_XML, id="simple_slide"),
  pytest.param(FIVE_LINK_CHAIN_XML, id="five_link_chain"),
]


@pytest.mark.math
class TestVelocityGrad:
  @pytest.mark.parametrize("xml", _VELOCITY_MODELS)
  def test_dL_dqvel_fwd_velocity(self, xml, ad_config):
    """dL/dqvel through fwd_velocity(): loss = sum(qfrc_bias)."""
    mjm, mjd, m, d = make_ad_fixture(xml=xml, keyframe=0)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
      mjw.kinematics(m, d)
      mjw.com_pos(m, d)
      mjw.crb(m, d)
      mjw.factor_m(m, d)
      mjw.transmission(m, d)
      mjw.fwd_velocity(m, d)
      wp.launch(
        sum_qfrc_bias_kernel,
        dim=(d.nworld, m.nv),
        inputs=[d.qfrc_bias, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.qvel.grad.numpy()[0, : mjm.nv].copy()
    tape.zero()

    def eval_loss(qvel_np):
      d_fd = mjw.make_data(mjm)
      wp.copy(d_fd.qpos, d.qpos)
      d_fd.qvel = wp.array(qvel_np.reshape(1, -1), dtype=float)
      mjw.kinematics(m, d_fd)
      mjw.com_pos(m, d_fd)
      mjw.crb(m, d_fd)
      mjw.factor_m(m, d_fd)
      mjw.transmission(m, d_fd)
      mjw.fwd_velocity(m, d_fd)
      l = wp.zeros(1, dtype=float)
      wp.launch(
        sum_qfrc_bias_kernel,
        dim=(d_fd.nworld, m.nv),
        inputs=[d_fd.qfrc_bias, l],
      )
      return l.numpy()[0]

    qvel_np = d.qvel.numpy()[0, : mjm.nv]
    fd_grad = fd_gradient(eval_loss, qvel_np, eps=ad_config.fd_eps)

    np.testing.assert_allclose(
      ad_grad, fd_grad, atol=ad_config.fd_tol, rtol=ad_config.fd_tol,
    )


# ---- Actuation: dL/dctrl through fwd_actuation() ----

_ACTUATION_MODELS = [
  pytest.param(SIMPLE_HINGE_XML, id="simple_hinge"),
  pytest.param(FIVE_LINK_CHAIN_XML, id="five_link_chain"),
  pytest.param(SPRING_DAMPER_XML, id="spring_damper"),
]


@pytest.mark.math
class TestActuationGrad:
  @pytest.mark.parametrize("xml", _ACTUATION_MODELS)
  def test_dL_dctrl_fwd_actuation(self, xml, ad_config):
    """dL/dctrl through fwd_actuation(): loss = sum(qfrc_actuator)."""
    mjm, mjd, m, d = make_ad_fixture(xml=xml, keyframe=0)

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
        sum_qfrc_actuator_kernel,
        dim=(d.nworld, m.nv),
        inputs=[d.qfrc_actuator, loss],
      )
    tape.backward(loss=loss)
    ad_grad = d.ctrl.grad.numpy()[0, : mjm.nu].copy()
    tape.zero()

    def eval_loss(ctrl_np):
      d_fd = mjw.make_data(mjm)
      wp.copy(d_fd.qpos, d.qpos)
      wp.copy(d_fd.qvel, d.qvel)
      d_fd.ctrl = wp.array(ctrl_np.reshape(1, -1), dtype=float)
      mjw.kinematics(m, d_fd)
      mjw.com_pos(m, d_fd)
      mjw.crb(m, d_fd)
      mjw.factor_m(m, d_fd)
      mjw.transmission(m, d_fd)
      mjw.fwd_velocity(m, d_fd)
      mjw.fwd_actuation(m, d_fd)
      l = wp.zeros(1, dtype=float)
      wp.launch(
        sum_qfrc_actuator_kernel,
        dim=(d_fd.nworld, m.nv),
        inputs=[d_fd.qfrc_actuator, l],
      )
      return l.numpy()[0]

    ctrl_np = d.ctrl.numpy()[0, : mjm.nu]
    fd_grad = fd_gradient(eval_loss, ctrl_np, eps=ad_config.fd_eps)

    np.testing.assert_allclose(
      ad_grad, fd_grad, atol=ad_config.fd_tol, rtol=ad_config.fd_tol,
    )


# ---- Full step: dL/dctrl through step() ----

_STEP_MODELS = [
  pytest.param(SIMPLE_HINGE_XML, id="simple_hinge"),
  pytest.param(SIMPLE_SLIDE_XML, id="simple_slide"),
  pytest.param(THREE_LINK_HINGE_XML, id="three_link_hinge"),
  pytest.param(FIVE_LINK_CHAIN_XML, id="five_link_chain"),
]


@pytest.mark.math
@pytest.mark.gpu_sm70
class TestFullStepGrad:
  @pytest.mark.parametrize("xml", _STEP_MODELS)
  def test_dL_dctrl_step(self, xml, ad_config):
    """dL/dctrl through full step(): loss = sum(xpos)."""
    if wp.get_device().is_cuda and wp.get_device().arch < 70:
      pytest.skip("tile kernels require sm_70+")

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


# ---- Humanoid (benchmark model) ----


@pytest.mark.math
@pytest.mark.slow
@pytest.mark.gpu_sm70
class TestHumanoidGrad:
  def test_humanoid_kinematics_grad(self, ad_config):
    """dL/dqpos through kinematics() on the 27-DOF humanoid."""
    if wp.get_device().is_cuda and wp.get_device().arch < 70:
      pytest.skip("tile kernels require sm_70+")

    from mjwarp_adtest.models.registry import get_model

    model_path = get_model("humanoid", config=ad_config)
    mjm, mjd, m, d = make_ad_fixture(path=model_path, keyframe=0)

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

    np.testing.assert_allclose(
      ad_grad, fd_grad, atol=ad_config.fd_tol, rtol=ad_config.fd_tol,
    )


# ---- Results saving ----


@pytest.fixture(autouse=True)
def _save_results(request, results_dir):
  """Save test outcomes to JSON when --save-results is set."""
  yield
  if results_dir is None:
    return
  # Only save on test completion
  report = getattr(request.node, "_report", None)
  if report is None:
    return
  result_file = os.path.join(results_dir, "math", "jacobian_comparison.json")
  entry = {
    "test": request.node.nodeid,
    "passed": not request.node.rep_call.failed if hasattr(request.node, "rep_call") else True,
  }
  existing = []
  if os.path.exists(result_file):
    with open(result_file) as f:
      existing = json.load(f).get("tests", [])
  existing.append(entry)
  with open(result_file, "w") as f:
    json.dump({"tests": existing}, f, indent=2)
