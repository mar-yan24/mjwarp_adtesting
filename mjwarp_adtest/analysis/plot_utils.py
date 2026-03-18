"""Matplotlib helpers for performance and math result plots."""

import os

import numpy as np


def _ensure_matplotlib():
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  return plt


def plot_overhead_bar(csv_path, output_path):
  """Generate bar chart of AD overhead ratios from ad_overhead.csv."""
  plt = _ensure_matplotlib()
  import csv

  models = []
  ratios = []
  with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
      models.append(row["model"])
      ratios.append(float(row["overhead_ratio"]))

  fig, ax = plt.subplots(figsize=(10, 5))
  ax.bar(models, ratios, color="steelblue")
  ax.set_ylabel("AD Overhead Ratio (AD time / baseline time)")
  ax.set_title("AD Wall-Clock Overhead by Model")
  ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
  plt.xticks(rotation=45, ha="right")
  plt.tight_layout()
  fig.savefig(output_path, dpi=150)
  plt.close(fig)


def plot_scaling(csv_path, output_path, x_col="ndof", group_col=None):
  """Generate scaling plot from scaling.csv."""
  plt = _ensure_matplotlib()
  import csv

  data = []
  with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
      data.append(row)

  if not data:
    return

  x_vals = [float(row[x_col]) for row in data]
  ratios = [float(row["overhead_ratio"]) for row in data]

  fig, ax = plt.subplots(figsize=(8, 5))
  ax.plot(x_vals, ratios, "o-", color="steelblue")
  ax.set_xlabel(x_col.replace("_", " ").title())
  ax.set_ylabel("AD Overhead Ratio")
  ax.set_title(f"AD Overhead Scaling vs {x_col.replace('_', ' ').title()}")
  ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
  plt.tight_layout()
  fig.savefig(output_path, dpi=150)
  plt.close(fig)


def plot_taylor_convergence(h_values, remainders, output_path, title="Taylor Convergence"):
  """Plot Taylor convergence: log(remainder) vs log(h)."""
  plt = _ensure_matplotlib()

  mask = remainders > 1e-15
  if mask.sum() < 2:
    return

  fig, ax = plt.subplots(figsize=(6, 5))
  ax.loglog(h_values[mask], remainders[mask], "o-", color="steelblue", label="Remainder")

  # Reference O(h^2) line
  h_ref = h_values[mask]
  r_ref = remainders[mask][0] * (h_ref / h_ref[0]) ** 2
  ax.loglog(h_ref, r_ref, "--", color="gray", alpha=0.7, label="O(h²) reference")

  ax.set_xlabel("h")
  ax.set_ylabel("|f(x+hd) - f(x) - h·g^T·d|")
  ax.set_title(title)
  ax.legend()
  plt.tight_layout()
  fig.savefig(output_path, dpi=150)
  plt.close(fig)
