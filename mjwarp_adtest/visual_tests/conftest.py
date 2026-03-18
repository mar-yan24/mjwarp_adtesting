"""Visual test fixtures: render setup, output directories."""

import os

import pytest


@pytest.fixture
def visual_output_dir(results_dir):
  """Returns path for visual test outputs, or None if not saving."""
  if results_dir is None:
    return None
  path = os.path.join(results_dir, "visual")
  os.makedirs(path, exist_ok=True)
  return path


@pytest.fixture
def gradient_maps_dir(visual_output_dir):
  """Returns path for gradient heatmap outputs."""
  if visual_output_dir is None:
    return None
  path = os.path.join(visual_output_dir, "gradient_maps")
  os.makedirs(path, exist_ok=True)
  return path
