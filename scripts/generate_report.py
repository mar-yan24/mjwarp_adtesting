#!/usr/bin/env python
"""Generate summary report from saved test results."""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mjwarp_adtest.analysis.report_generator import generate_report


def main():
  parser = argparse.ArgumentParser(description="Generate AD test report")
  parser.add_argument(
    "--results-dir",
    default="results",
    help="Path to results directory (or timestamped subdirectory)",
  )
  parser.add_argument(
    "--output",
    default=None,
    help="Output file path (default: results_dir/reports/summary.md)",
  )
  args = parser.parse_args()

  results_dir = args.results_dir

  # If results_dir points to the top-level "results/" dir, find the latest timestamp
  if os.path.basename(results_dir) == "results" or not any(
    os.path.isdir(os.path.join(results_dir, sub))
    for sub in ("math", "perf", "visual")
  ):
    subdirs = sorted(
      [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
        and d[0].isdigit()
      ]
    )
    if subdirs:
      results_dir = os.path.join(results_dir, subdirs[-1])
      print(f"Using latest results: {results_dir}")
    else:
      print(f"No timestamped result directories found in {results_dir}")
      sys.exit(1)

  report = generate_report(results_dir, output_path=args.output)
  output_path = args.output or os.path.join(results_dir, "reports", "summary.md")
  print(f"Report generated: {output_path}")
  print()
  print(report)


if __name__ == "__main__":
  main()
