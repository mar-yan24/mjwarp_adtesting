"""Determinism test fixtures and helpers.

Provides factory functions for creating deterministic fixtures, collecting
full simulation state, and asserting bitwise or approximate equality across
repeated simulation runs.
"""

import os

import numpy as np
import pytest
import warp as wp

import mujoco_warp as mjw
from mujoco_warp import test_data

from mjwarp_adtest.config import DeterminismTestConfig


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nruns(det_config):
  return det_config.nruns


@pytest.fixture
def short_horizon(det_config):
  return det_config.short_horizon


@pytest.fixture
def medium_horizon(det_config):
  return det_config.medium_horizon


@pytest.fixture
def long_horizon(det_config):
  return det_config.long_horizon


# ---------------------------------------------------------------------------
# GPU capability helpers
# ---------------------------------------------------------------------------

requires_gpu_sm70 = pytest.mark.skipif(
  not wp.is_cuda_available() or wp.get_device("cuda:0").arch < 70,
  reason="Requires CUDA SM 70+ (tile kernels)",
)


# ---------------------------------------------------------------------------
# Fixture factory
# ---------------------------------------------------------------------------


def make_det_fixture(path_or_xml, nworld=1, deterministic=True, overrides=None):
  """Create (mjm, mjd, m, d) with deterministic flag set.

  Args:
    path_or_xml: Either a path relative to mujoco_warp test_data, an absolute
      file path, or an inline XML string.
    nworld: Number of parallel simulation worlds.
    deterministic: Value for m.opt.deterministic.
    overrides: Optional dict of model field overrides.

  Returns:
    Tuple of (MjModel, MjData, Model, Data).
  """
  kwargs = dict(nworld=nworld)
  if overrides:
    kwargs["overrides"] = overrides

  if path_or_xml.lstrip().startswith("<"):
    kwargs["xml"] = path_or_xml
  elif os.path.isabs(path_or_xml):
    with open(path_or_xml) as f:
      kwargs["xml"] = f.read()
  else:
    kwargs["path"] = path_or_xml

  mjm, mjd, m, d = test_data.fixture(**kwargs)
  m.opt.deterministic = deterministic
  return mjm, mjd, m, d


# ---------------------------------------------------------------------------
# Simulation runners
# ---------------------------------------------------------------------------


def run_simulation(path_or_xml, nworld, nsteps, deterministic, overrides=None):
  """Run a simulation and return (m, d) after nsteps."""
  _, _, m, d = make_det_fixture(
    path_or_xml, nworld, deterministic, overrides=overrides
  )
  for _ in range(nsteps):
    mjw.step(m, d)
  return m, d


def run_simulation_trajectory(
  path_or_xml, nworld, nsteps, deterministic, overrides=None
):
  """Run simulation and return trajectory of (qpos, qvel) at each step."""
  _, _, m, d = make_det_fixture(
    path_or_xml, nworld, deterministic, overrides=overrides
  )
  qpos_traj = []
  qvel_traj = []
  for _ in range(nsteps):
    mjw.step(m, d)
    qpos_traj.append(d.qpos.numpy().copy())
    qvel_traj.append(d.qvel.numpy().copy())
  return np.array(qpos_traj), np.array(qvel_traj)


# ---------------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------------

# All 17 contact fields that are permuted by _sort_contacts.
CONTACT_FIELDS = (
  "dist",
  "pos",
  "frame",
  "includemargin",
  "friction",
  "solref",
  "solreffriction",
  "solimp",
  "dim",
  "geom",
  "flex",
  "vert",
  "efc_address",
  "worldid",
  "type",
  "geomcollisionid",
)

# Core simulation state fields on Data.
STATE_FIELDS = (
  "qpos",
  "qvel",
  "qacc",
  "qfrc_bias",
  "qfrc_smooth",
  "qacc_smooth",
  "qfrc_constraint",
  "xpos",
  "xquat",
  "sensordata",
)


def collect_contacts_complete(d, nacon=None):
  """Extract all 17 contact fields as numpy arrays.

  Args:
    d: MJWarp Data object.
    nacon: Number of active contacts. If None, reads from d.nacon.

  Returns:
    Dict mapping field name to numpy array (sliced to nacon).
  """
  if nacon is None:
    nacon = int(d.nacon.numpy()[0])
  result = {"nacon": nacon}
  for field in CONTACT_FIELDS:
    arr = getattr(d.contact, field).numpy()
    result[field] = arr[:nacon].copy()
  return result


def collect_full_state(m, d):
  """Extract full simulation state as numpy arrays.

  Returns:
    Dict with state fields, contact fields, and nacon.
  """
  state = {}
  for field in STATE_FIELDS:
    arr = getattr(d, field)
    state[field] = arr.numpy().copy()

  nacon = int(d.nacon.numpy()[0])
  state["nacon"] = nacon
  for field in CONTACT_FIELDS:
    arr = getattr(d.contact, field).numpy()
    state[f"contact_{field}"] = arr[:nacon].copy()

  return state


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_states_bitwise_equal(state_a, state_b, fields=None):
  """Assert all (or selected) fields are bitwise identical.

  Args:
    state_a: Dict of field -> numpy array.
    state_b: Dict of field -> numpy array.
    fields: Optional iterable of field names to check. If None, checks all
      keys present in both dicts.
  """
  if fields is None:
    fields = sorted(set(state_a.keys()) & set(state_b.keys()))
  for field in fields:
    if field == "nacon":
      assert state_a["nacon"] == state_b["nacon"], (
        f"nacon mismatch: {state_a['nacon']} vs {state_b['nacon']}"
      )
      continue
    np.testing.assert_array_equal(
      state_a[field],
      state_b[field],
      err_msg=f"Bitwise mismatch in field '{field}'",
    )


def assert_states_close(state_a, state_b, atol=1e-10, rtol=1e-10, fields=None):
  """Assert all (or selected) fields are numerically close.

  Args:
    state_a: Dict of field -> numpy array.
    state_b: Dict of field -> numpy array.
    atol: Absolute tolerance.
    rtol: Relative tolerance.
    fields: Optional iterable of field names to check.
  """
  if fields is None:
    fields = sorted(set(state_a.keys()) & set(state_b.keys()))
  for field in fields:
    if field == "nacon":
      assert state_a["nacon"] == state_b["nacon"], (
        f"nacon mismatch: {state_a['nacon']} vs {state_b['nacon']}"
      )
      continue
    a = state_a[field]
    b = state_b[field]
    if np.issubdtype(a.dtype, np.integer):
      np.testing.assert_array_equal(
        a, b, err_msg=f"Integer field '{field}' differs"
      )
    else:
      np.testing.assert_allclose(
        a, b, atol=atol, rtol=rtol, err_msg=f"Numerical mismatch in '{field}'"
      )


# ---------------------------------------------------------------------------
# Convenience: run N times and compare
# ---------------------------------------------------------------------------


def run_n_times_and_compare(
  path_or_xml,
  nworld,
  nsteps,
  nruns,
  deterministic=True,
  fields=None,
  collect_fn=None,
  overrides=None,
):
  """Run simulation nruns times and assert bitwise equality.

  Args:
    path_or_xml: Model path or XML string.
    nworld: Number of parallel worlds.
    nsteps: Simulation steps.
    nruns: Number of repeated runs.
    deterministic: Deterministic flag value.
    fields: Fields to compare (None = all).
    collect_fn: Custom collection function(m, d) -> dict. Defaults to
      collect_full_state.
    overrides: Optional model overrides dict.

  Returns:
    List of collected state dicts (for further inspection by caller).
  """
  if collect_fn is None:
    collect_fn = collect_full_state

  results = []
  for _ in range(nruns):
    m, d = run_simulation(
      path_or_xml, nworld, nsteps, deterministic, overrides=overrides
    )
    results.append(collect_fn(m, d))

  for run in range(1, nruns):
    assert_states_bitwise_equal(results[0], results[run], fields=fields)

  return results
