"""Wrappers around test_data.fixture with gradient tracking."""

import os

import mujoco_warp as mjw
from mujoco_warp import test_data
from mujoco_warp._src.grad import enable_grad


def make_ad_fixture(xml=None, path=None, keyframe=0, nworld=1):
  """Returns (mjm, mjd, m, d) with gradient tracking enabled.

  Args:
    xml: Inline XML string.
    path: Path to MJCF file (relative to test_data root, or absolute).
    keyframe: Keyframe index or name for initialization.
    nworld: Number of parallel worlds.

  Returns:
    Tuple of (MjModel, MjData, Model, Data) with gradients enabled on Data.
  """
  kwargs = dict(keyframe=keyframe, nworld=nworld)
  if xml is not None:
    kwargs["xml"] = xml
  elif path is not None:
    # If absolute path, load via xml string from file
    if os.path.isabs(path):
      with open(path) as f:
        kwargs["xml"] = f.read()
    else:
      kwargs["path"] = path
  else:
    raise ValueError("Either xml or path must be provided.")

  mjm, mjd, m, d = test_data.fixture(**kwargs)
  enable_grad(d)
  return mjm, mjd, m, d


def make_baseline_fixture(xml=None, path=None, keyframe=0, nworld=1):
  """Returns (mjm, mjd, m, d) without gradient tracking.

  Same interface as make_ad_fixture but does not enable gradients.
  """
  kwargs = dict(keyframe=keyframe, nworld=nworld)
  if xml is not None:
    kwargs["xml"] = xml
  elif path is not None:
    if os.path.isabs(path):
      with open(path) as f:
        kwargs["xml"] = f.read()
    else:
      kwargs["path"] = path
  else:
    raise ValueError("Either xml or path must be provided.")

  return test_data.fixture(**kwargs)
