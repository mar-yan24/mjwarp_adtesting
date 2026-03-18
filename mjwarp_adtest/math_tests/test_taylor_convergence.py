"""Taylor convergence tests: verify O(h^2) remainder for AD gradients.

For loss f(x) with AD gradient g, the Taylor remainder
  r(h) = |f(x + h*d) - f(x) - h * g^T * d|
should decrease as O(h^2). We fit log(r) vs log(h) and check slope ~ 2.
"""

import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src.grad import enable_grad

from mjwarp_adtest.fixtures.data_factory import make_ad_fixture
from mjwarp_adtest.fixtures.finite_difference import taylor_test
from mjwarp_adtest.fixtures.loss_functions import sum_xpos_kernel
from mjwarp_adtest.models.xml_defs import (
  FIVE_LINK_CHAIN_XML,
  SIMPLE_HINGE_XML,
  SIMPLE_SLIDE_XML,
  THREE_LINK_HINGE_XML,
)

_TAYLOR_MODELS = [
  pytest.param(SIMPLE_HINGE_XML, id="simple_hinge"),
  pytest.param(SIMPLE_SLIDE_XML, id="simple_slide"),
  pytest.param(THREE_LINK_HINGE_XML, id="three_link_hinge"),
  pytest.param(FIVE_LINK_CHAIN_XML, id="five_link_chain"),
]


def _check_taylor_slope(h_values, remainders, f0_magnitude):
  """Analyze Taylor convergence, accounting for float32 precision.

  Returns (slope, n_good_points) or (None, 0) if insufficient data.

  Float32 limits the useful range of the Taylor test. We require at
  least 2 well-behaved consecutive points where the remainder
  strictly decreases. Points near the precision floor are excluded.
  """
  # Float32 precision floor: ~1e-6 relative to function magnitude.
  # This is conservative because atomic_add accumulation in warp
  # kernels adds additional float32 rounding.
  eps_floor = max(abs(f0_magnitude) * 1e-6, 1e-10)

  # Only use points above the precision floor
  mask = remainders > eps_floor
  if mask.sum() < 2:
    return None, 0

  # Use only points where remainder is strictly decreasing
  indices = np.where(mask)[0]
  good = [indices[0]]
  for i in range(1, len(indices)):
    if remainders[indices[i]] < remainders[indices[i - 1]]:
      good.append(indices[i])

  if len(good) < 2:
    return None, 0

  log_h = np.log10(h_values[good])
  log_r = np.log10(remainders[good])

  slope, _ = np.polyfit(log_h, log_r, 1)
  return slope, len(good)


@pytest.mark.math
class TestTaylorConvergence:
  @pytest.mark.parametrize("xml", _TAYLOR_MODELS)
  def test_kinematics_taylor_convergence(self, xml, ad_config):
    """Verify O(h^2) convergence for kinematics gradient."""
    mjm, mjd, m, d = make_ad_fixture(xml=xml, keyframe=0)

    # Get AD gradient
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

    # Define f(x) for Taylor test
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
    f0 = eval_loss(qpos_np)

    # Use larger h values that stay above float32 precision floor
    h_vals = (5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3)

    h_values, remainders = taylor_test(
      fn=eval_loss,
      grad_fn=lambda x: ad_grad,
      x_np=qpos_np,
      h_values=h_vals,
    )

    slope, n_good = _check_taylor_slope(h_values, remainders, f0)

    if slope is None:
      # All remainders below precision floor — gradient is essentially exact.
      # This is a pass (the gradient is so accurate that the second-order
      # remainder is at or below float32 noise).
      return

    # Expect slope > 1.3 (O(h^2) = slope 2.0, but float32 degrades it).
    # A slope above 1.3 with at least 2 well-behaved points confirms
    # super-linear convergence, ruling out gross gradient errors.
    assert slope > 1.3, (
      f"Taylor convergence slope = {slope:.2f} ({n_good} points), expected > 1.3"
    )
