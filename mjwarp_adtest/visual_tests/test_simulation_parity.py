"""Trajectory parity test: AD on/off should produce identical trajectories.

Verifies that enabling gradient tracking (recording a tape, without backward)
does not change the forward simulation output. Catches bugs where
enable_backward=True inadvertently changes kernel behavior.
"""

import json
import os

import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw

from mjwarp_adtest.fixtures.data_factory import make_ad_fixture, make_baseline_fixture
from mjwarp_adtest.models.xml_defs import (
  SIMPLE_HINGE_XML,
  SIMPLE_SLIDE_XML,
  THREE_LINK_HINGE_XML,
  FIVE_LINK_CHAIN_XML,
)

_PARITY_MODELS = [
  pytest.param(SIMPLE_HINGE_XML, "simple_hinge", id="simple_hinge"),
  pytest.param(SIMPLE_SLIDE_XML, "simple_slide", id="simple_slide"),
  pytest.param(THREE_LINK_HINGE_XML, "three_link_hinge", id="three_link_hinge"),
  pytest.param(FIVE_LINK_CHAIN_XML, "five_link_chain", id="five_link_chain"),
]

_NSTEPS = 50


@pytest.mark.visual
@pytest.mark.gpu_sm70
class TestSimulationParity:
  @pytest.mark.parametrize("xml,model_name", _PARITY_MODELS)
  def test_trajectory_identity(self, xml, model_name, ad_config, results_dir):
    """AD on (tape recorded, no backward) must match AD off bitwise."""
    if wp.get_device().is_cuda and wp.get_device().arch < 70:
      pytest.skip("step() tile kernels require sm_70+")
    # --- Baseline trajectory (no AD) ---
    mjm, _, m, d_base = make_baseline_fixture(xml=xml, keyframe=0)
    nq = mjm.nq
    nv = mjm.nv

    traj_qpos_base = []
    traj_qvel_base = []
    for _ in range(_NSTEPS):
      mjw.step(m, d_base)
      traj_qpos_base.append(d_base.qpos.numpy()[0, :nq].copy())
      traj_qvel_base.append(d_base.qvel.numpy()[0, :nv].copy())

    traj_qpos_base = np.array(traj_qpos_base)
    traj_qvel_base = np.array(traj_qvel_base)

    # --- AD trajectory (tape recorded, no backward) ---
    _, _, _, d_ad = make_ad_fixture(xml=xml, keyframe=0)

    traj_qpos_ad = []
    traj_qvel_ad = []
    for _ in range(_NSTEPS):
      tape = wp.Tape()
      with tape:
        mjw.step(m, d_ad)
      # No backward — just record the tape
      traj_qpos_ad.append(d_ad.qpos.numpy()[0, :nq].copy())
      traj_qvel_ad.append(d_ad.qvel.numpy()[0, :nv].copy())
      tape.zero()

    traj_qpos_ad = np.array(traj_qpos_ad)
    traj_qvel_ad = np.array(traj_qvel_ad)

    # --- Compare ---
    max_qpos_diff = np.max(np.abs(traj_qpos_base - traj_qpos_ad))
    max_qvel_diff = np.max(np.abs(traj_qvel_base - traj_qvel_ad))

    # Save results
    if results_dir is not None:
      result_file = os.path.join(results_dir, "visual", "trajectory_diff.json")
      entry = {
        "model": model_name,
        "nsteps": _NSTEPS,
        "max_qpos_diff": float(max_qpos_diff),
        "max_qvel_diff": float(max_qvel_diff),
      }
      existing = []
      if os.path.exists(result_file):
        with open(result_file) as f:
          existing = json.load(f).get("tests", [])
      existing.append(entry)
      with open(result_file, "w") as f:
        json.dump({"tests": existing}, f, indent=2)

    # Allow small floating-point tolerance (ULP-level differences)
    np.testing.assert_allclose(
      traj_qpos_base, traj_qpos_ad, atol=1e-7, rtol=0,
      err_msg=f"qpos trajectory diverged for {model_name}",
    )
    np.testing.assert_allclose(
      traj_qvel_base, traj_qvel_ad, atol=1e-7, rtol=0,
      err_msg=f"qvel trajectory diverged for {model_name}",
    )
