#!/usr/bin/env python3
"""
metric_grid.py
==============
Stitch the consolidated single-image-per-condition plots into one grid per
denoising arm: rows = metric (lr_curve, per_symptom_auroc, per_symptom_f1),
columns = thresholds (27 / 37 / 50). One final PNG per arm.

INPUT (consolidated reporting folder)
-------------------------------------
Reads from final_plots/ (produced by consolidate_results.py):
    final_plots/lr_curve_{original|denoised}_{thr}.png
    final_plots/per_symptom_auroc_{original|denoised}_{thr}.png
    final_plots/per_symptom_f1_{original|denoised}_{thr}.png

Rendered images are ARRANGED, not redrawn. Missing tiles render as a blank
cell labelled "missing"; row/column headers are added on the margins.

USAGE
-----
    python metric_grid.py --plots_dir reporting/final_plots
    # writes metric_grid_original.png and metric_grid_denoised.png there
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

THRESHOLDS = ["27", "37", "50"]
ARMS = ["original", "denoised"]

# (filename stem, row label shown on the grid) — one row each
METRICS = [
    ("lr_curve", "learning rate"),
    ("per_symptom_auroc", "AUROC"),
    ("per_symptom_f1", "F1"),
]


def build_grid(plots_dir: Path, arm: str, out_path: Path):
    nrow, ncol = len(METRICS), len(THRESHOLDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 3.0),
                             squeeze=False)

    n_found = 0
    for i, (stem, label) in enumerate(METRICS):
        for j, thr in enumerate(THRESHOLDS):
            ax = axes[i][j]
            ax.axis("off")
            img_path = plots_dir / f"{stem}_{arm}_{thr}.png"
            if img_path.exists():
                ax.imshow(mpimg.imread(img_path))
                n_found += 1
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        color="#999", fontsize=11, transform=ax.transAxes)

            if i == 0:
                ax.set_title(f"threshold > {thr}", fontsize=14, pad=8)
            if j == 0:
                ax.text(-0.04, 0.5, label, ha="right", va="center",
                        rotation=90, fontsize=13, transform=ax.transAxes)

    fig.suptitle(f"Metrics — {arm}  (rows: metric · cols: threshold)",
                 fontsize=16, y=0.997)
    fig.tight_layout(rect=[0.02, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}  ({n_found}/{nrow*ncol} tiles present)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plots_dir", default="reporting/final_plots")
    ap.add_argument("--out_dir", default=None,
                    help="where to write grids (default: same as --plots_dir)")
    args = ap.parse_args()

    plots_dir = Path(args.plots_dir)
    out_dir = Path(args.out_dir) if args.out_dir else plots_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in ARMS:
        print(f"\n=== {arm} ===")
        build_grid(plots_dir, arm, out_dir / f"metric_grid_{arm}.png")


if __name__ == "__main__":
    main()
