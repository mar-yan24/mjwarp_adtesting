"""Memory overhead estimation for deterministic contact sorting.

Estimates the temporary buffer memory allocated by _sort_contacts:
sort key/index arrays and 16 contact field copies.
"""

import pytest
import warp as wp

import mujoco_warp as mjw

from mjwarp_adtest.determinism_tests.conftest import (
  make_det_fixture,
  requires_gpu_sm70,
)

pytestmark = [pytest.mark.determinism, pytest.mark.perf, requires_gpu_sm70]


class TestSortBufferMemory:
  """Estimate memory allocated by _sort_contacts temporaries."""

  @pytest.mark.parametrize(
    "model_id",
    [
      pytest.param("collision.xml", id="collision"),
      pytest.param("humanoid/humanoid.xml", id="humanoid"),
    ],
  )
  def test_sort_buffer_memory(self, model_id):
    """Estimate sort buffer memory relative to total contact memory."""
    _, _, m, d = make_det_fixture(model_id, nworld=1, deterministic=True)
    mjw.step(m, d)

    naconmax = d.naconmax

    # Sort buffers: 2 * naconmax ints for sort_keys and sort_indices
    # (doubled for radix sort internal working space).
    sort_buffer_bytes = 2 * 2 * naconmax * 4  # 2 arrays * 2x capacity * int32

    # Temporary contact field copies (16 fields).
    # Estimate sizes based on field dtypes and naconmax.
    field_sizes = {
      "dist": naconmax * 4,           # float
      "pos": naconmax * 3 * 4,        # vec3
      "frame": naconmax * 9 * 4,      # mat33
      "includemargin": naconmax * 4,   # float
      "friction": naconmax * 5 * 4,    # vec5
      "solref": naconmax * 2 * 4,      # vec2
      "solreffriction": naconmax * 2 * 4,  # vec2
      "solimp": naconmax * 5 * 4,      # vec5
      "dim": naconmax * 4,            # int
      "geom": naconmax * 2 * 4,       # vec2i
      "flex": naconmax * 2 * 4,       # vec2i
      "vert": naconmax * 2 * 4,       # vec2i
      "worldid": naconmax * 4,        # int
      "type": naconmax * 4,           # int
      "geomcollisionid": naconmax * 4,  # int
      "efc_address": naconmax * d.contact.efc_address.numpy().shape[1] * 4,
    }
    tmp_buffer_bytes = sum(field_sizes.values())

    total_sort_bytes = sort_buffer_bytes + tmp_buffer_bytes
    total_sort_mb = total_sort_bytes / (1024 * 1024)

    print(
      f"\n{model_id} (naconmax={naconmax}):"
      f"\n  Sort key buffers: {sort_buffer_bytes / 1024:.1f} KB"
      f"\n  Tmp field copies: {tmp_buffer_bytes / 1024:.1f} KB"
      f"\n  Total sort memory: {total_sort_mb:.2f} MB"
    )

    # Sanity check: sort memory should be reasonable.
    assert total_sort_mb < 512, (
      f"Sort buffer memory {total_sort_mb:.1f} MB seems excessive"
    )
