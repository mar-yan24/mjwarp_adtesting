#!/usr/bin/env python
"""Convenience script to run the full AD test suite."""

import subprocess
import sys


def main():
  args = sys.argv[1:]

  # Default: run all non-slow tests with verbose output
  cmd = [
    sys.executable, "-m", "pytest",
    "mjwarp_adtest/",
    "-v",
    "-m", "not slow",
  ] + args

  print(f"Running: {' '.join(cmd)}")
  result = subprocess.run(cmd)
  sys.exit(result.returncode)


if __name__ == "__main__":
  main()
