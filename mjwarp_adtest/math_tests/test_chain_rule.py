"""Chain rule tests: verify gradient composition across module boundaries.

Computes Jacobians of individual pipeline stages and multiplies them,
then compares against end-to-end AD gradient through the full pipeline.
"""

import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw

from mjwarp_adtest.fixtures.data_factory import make_ad_fixture
from mjwarp_adtest.fixtures.finite_difference import fd_jacobian
from mjwarp_adtest.fixtures.loss_functions import sum_xpos_kernel
from mjwarp_adtest.models.xml_defs import SIMPLE_HINGE_XML, THREE_LINK_HINGE_XML

_CHAIN_MODELS = [
  pytest.param(SIMPLE_HINGE_XML, id="simple_hinge"),
  pytest.param(THREE_LINK_HINGE_XML, id="three_link_hinge"),
]


@pytest.mark.math
class TestChainRule:
  @pytest.mark.parametrize("xml", _CHAIN_MODELS)
  def test_kinematics_chain_rule(self, xml, ad_config):
    """Verify J_kinematics composed matches end-to-end AD.

    Computes dxpos/dqpos via FD Jacobian on kinematics alone,
    then compares the resulting dL/dqpos = J^T * dL/dxpos against AD.
    """
    mjm, mjd, m, d = make_ad_fixture(xml=xml, keyframe=0)
    nq = mjm.nq
    nbody = mjm.nbody

    # End-to-end AD gradient: dL/dqpos
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
    ad_grad = d.qpos.grad.numpy()[0, :nq].copy()
    tape.zero()

    # FD Jacobian of kinematics: dxpos/dqpos
    # xpos is (nbody, 3) flattened to nbody*3
    def kinematics_fn(qpos_np):
      d_fd = mjw.make_data(mjm)
      d_fd.qpos = wp.array(qpos_np.reshape(1, -1), dtype=float)
      mjw.kinematics(m, d_fd)
      mjw.com_pos(m, d_fd)
      xpos = d_fd.xpos.numpy()[0]  # (nbody, 3)
      return xpos.ravel()

    qpos_np = d.qpos.numpy()[0, :nq]
    jac = fd_jacobian(kinematics_fn, qpos_np, n_outputs=nbody * 3, eps=ad_config.fd_eps)

    # dL/dxpos = [1, 1, 1, ...] since loss = sum of all xpos components
    dL_dxpos = np.ones(nbody * 3)

    # Chain rule: dL/dqpos = J^T @ dL/dxpos
    chain_grad = jac.T @ dL_dxpos

    np.testing.assert_allclose(
      ad_grad, chain_grad,
      atol=ad_config.fd_tol * 2,  # slightly relaxed for composed errors
      rtol=ad_config.fd_tol * 2,
    )
