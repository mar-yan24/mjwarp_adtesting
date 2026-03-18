"""Performance test fixtures: warmup, timing utilities."""

import csv
import os
import time

import pytest
import warp as wp


@pytest.fixture
def warmup_steps(ad_config):
  return ad_config.warmup_steps


@pytest.fixture
def perf_nstep(ad_config):
  return ad_config.perf_nstep


@pytest.fixture
def perf_nworld(ad_config):
  return ad_config.perf_nworld


class PerfTimer:
  """Context manager for GPU-aware timing."""

  def __init__(self):
    self.elapsed = 0.0

  def __enter__(self):
    wp.synchronize()
    self._start = time.perf_counter()
    return self

  def __exit__(self, *args):
    wp.synchronize()
    self._end = time.perf_counter()
    self.elapsed = self._end - self._start


@pytest.fixture
def perf_timer():
  return PerfTimer


def append_csv(filepath, row, header=None):
  """Append a row to a CSV file, writing header if file is new."""
  write_header = not os.path.exists(filepath)
  os.makedirs(os.path.dirname(filepath), exist_ok=True)
  with open(filepath, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header and header:
      writer.writerow(header)
    writer.writerow(row)


@pytest.fixture
def csv_writer(results_dir):
  """Returns a helper to append rows to result CSVs."""
  def _write(filename, row, header=None):
    if results_dir is None:
      return
    filepath = os.path.join(results_dir, "perf", filename)
    append_csv(filepath, row, header=header)
  return _write
