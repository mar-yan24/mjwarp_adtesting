"""Reusable warp kernel loss functions for AD testing.

All kernels use wp.atomic_add to accumulate into a scalar loss array,
matching the pattern established in grad_test.py.
"""

import warp as wp


@wp.kernel
def sum_xpos_kernel(
  xpos_in: wp.array2d(dtype=wp.vec3),
  loss: wp.array(dtype=float),
):
  """loss += sum of all body positions (x + y + z)."""
  worldid, bodyid = wp.tid()
  v = xpos_in[worldid, bodyid]
  wp.atomic_add(loss, 0, v[0] + v[1] + v[2])


@wp.kernel
def sum_qacc_kernel(
  qacc_in: wp.array2d(dtype=float),
  loss: wp.array(dtype=float),
):
  """loss += sum of all generalized accelerations."""
  worldid, dofid = wp.tid()
  wp.atomic_add(loss, 0, qacc_in[worldid, dofid])


@wp.kernel
def sum_qpos_kernel(
  qpos_in: wp.array2d(dtype=float),
  loss: wp.array(dtype=float),
):
  """loss += sum of all generalized positions."""
  worldid, qid = wp.tid()
  wp.atomic_add(loss, 0, qpos_in[worldid, qid])


@wp.kernel
def sum_qvel_kernel(
  qvel_in: wp.array2d(dtype=float),
  loss: wp.array(dtype=float),
):
  """loss += sum of all generalized velocities."""
  worldid, dofid = wp.tid()
  wp.atomic_add(loss, 0, qvel_in[worldid, dofid])


@wp.kernel
def sum_qfrc_bias_kernel(
  qfrc_bias_in: wp.array2d(dtype=float),
  loss: wp.array(dtype=float),
):
  """loss += sum of bias forces (Coriolis + gravity)."""
  worldid, dofid = wp.tid()
  wp.atomic_add(loss, 0, qfrc_bias_in[worldid, dofid])


@wp.kernel
def sum_qfrc_actuator_kernel(
  qfrc_actuator_in: wp.array2d(dtype=float),
  loss: wp.array(dtype=float),
):
  """loss += sum of actuator forces."""
  worldid, dofid = wp.tid()
  wp.atomic_add(loss, 0, qfrc_actuator_in[worldid, dofid])
