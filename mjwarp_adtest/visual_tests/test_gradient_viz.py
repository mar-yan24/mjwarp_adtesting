"""Gradient visualization: sensitivity maps and heatmaps.

Generates matplotlib visualizations of gradient magnitude across DOFs
and timesteps. These are documentation artifacts, not assertion-based tests.
"""

import os

import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw

from mjwarp_adtest.fixtures.data_factory import make_ad_fixture
from mjwarp_adtest.fixtures.loss_functions import sum_xpos_kernel
from mjwarp_adtest.models.xml_defs import SIMPLE_HINGE_XML, THREE_LINK_HINGE_XML

_VIZ_MODELS = [
  pytest.param(SIMPLE_HINGE_XML, "simple_hinge", id="simple_hinge"),
  pytest.param(THREE_LINK_HINGE_XML, "three_link_hinge", id="three_link_hinge"),
]

_VIZ_NSTEPS = 20


def _has_matplotlib():
  try:
    import matplotlib
    return True
  except ImportError:
    return False


@pytest.mark.visual
class TestGradientVisualization:
  @pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
  @pytest.mark.parametrize("xml,model_name", _VIZ_MODELS)
  def test_gradient_heatmap(self, xml, model_name, ad_config, gradient_maps_dir):
    """Generate gradient sensitivity heatmap: DOF x timestep."""
    if gradient_maps_dir is None:
      pytest.skip("--save-results not set")

    mjm, _, m, d = make_ad_fixture(xml=xml, keyframe=0)
    nq = mjm.nq

    # Collect gradients over timesteps
    grad_matrix = []
    for step_i in range(_VIZ_NSTEPS):
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
      grad = d.qpos.grad.numpy()[0, :nq].copy()
      grad_matrix.append(np.abs(grad))
      tape.zero()

      # Advance state (without tape for clean forward step)
      mjw.step(m, d)

    grad_matrix = np.array(grad_matrix)  # (nsteps, nq)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
      grad_matrix.T,
      aspect="auto",
      cmap="viridis",
      interpolation="nearest",
    )
    ax.set_xlabel("Timestep")
    ax.set_ylabel("DOF index")
    ax.set_title(f"Gradient magnitude |dL/dqpos| — {model_name}")
    fig.colorbar(im, ax=ax, label="|gradient|")
    plt.tight_layout()

    filepath = os.path.join(gradient_maps_dir, f"heatmap_{model_name}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)

  @pytest.mark.skipif(not _has_matplotlib(), reason="matplotlib not installed")
  @pytest.mark.parametrize("xml,model_name", _VIZ_MODELS)
  def test_gradient_bar_chart(self, xml, model_name, ad_config, gradient_maps_dir):
    """Generate bar chart of relative gradient sensitivity per DOF."""
    if gradient_maps_dir is None:
      pytest.skip("--save-results not set")

    mjm, _, m, d = make_ad_fixture(xml=xml, keyframe=0)
    nq = mjm.nq

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
    grad = d.qpos.grad.numpy()[0, :nq].copy()
    tape.zero()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    dof_labels = [f"q{i}" for i in range(nq)]
    ax.bar(dof_labels, np.abs(grad))
    ax.set_xlabel("DOF")
    ax.set_ylabel("|dL/dqpos|")
    ax.set_title(f"Gradient sensitivity — {model_name}")
    plt.tight_layout()

    filepath = os.path.join(gradient_maps_dir, f"sensitivity_{model_name}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
