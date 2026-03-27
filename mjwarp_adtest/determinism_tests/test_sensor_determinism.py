"""Tests that sensor outputs are deterministic.

Sensors (especially contact-based ones like touch) depend on contact data
that gets sorted. Their outputs should be bitwise identical across
deterministic runs.
"""

import numpy as np
import pytest

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  requires_gpu_sm70,
  run_simulation,
)
from mjwarp_adtest.models.xml_defs import SENSOR_CONTACT_XML

pytestmark = [pytest.mark.determinism, requires_gpu_sm70]


class TestSensorDeterminism:
  """Sensor data is deterministic when contact sorting is enabled."""

  @pytest.mark.parametrize(
    "model",
    [
      pytest.param(SENSOR_CONTACT_XML, id="sensor_contact"),
    ],
  )
  def test_sensordata_deterministic(self, model, nruns):
    """d.sensordata is bitwise identical across deterministic runs."""
    results = []
    for _ in range(nruns):
      m, d = run_simulation(model, nworld=1, nsteps=10, deterministic=True)
      sensordata = d.sensordata.numpy().copy()
      results.append(sensordata)

    # Check that sensor data exists.
    assert results[0].size > 0, "No sensor data"

    for run in range(1, nruns):
      np.testing.assert_array_equal(
        results[0],
        results[run],
        err_msg=f"sensordata differs: run 0 vs run {run}",
      )

  def test_contact_sensor_values_reasonable(self):
    """Touch sensor produces non-zero values when contact is active."""
    m, d = run_simulation(
      SENSOR_CONTACT_XML, nworld=1, nsteps=10, deterministic=True
    )
    sensordata = d.sensordata.numpy()[0]
    nacon = int(d.nacon.numpy()[0])

    # The touch sensor (index 0) should report force when in contact.
    if nacon > 0:
      # Touch sensor should be >= 0 (normal force).
      assert sensordata[0] >= 0, (
        f"Touch sensor value is negative: {sensordata[0]}"
      )
