#!/usr/bin/env python3
"""
loss_curve_grid.py
==================
Stitch the consolidated loss-curve PNGs into one grid per denoising arm:
rows = outer folds, columns = thresholds (27 / 37 / 50). One final PNG per arm.

INPUT (consolidated reporting folder)
-------------------------------------
Reads the loss curves produced by consolidate_results.py in final_plots/:
    final_plots/loss_curve_outer{N}_{original|denoised}_{thr}.png

These are rendered images, so this script ARRANGES them (it does not redraw).
Each tile keeps its own axes/title; row and column headers are added on the
margins. Missing tiles render as a blank cell labelled "missing".

USAGE
-----
    python loss_curve_grid.py --plots_dir reporting/final_plots
    # writes loss_curve_grid_original.png and loss_curve_grid_denoised.png there
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

THRESHOLDS = ["27", "37", "50"]
ARMS = ["original", "denoised"]


def discover_folds(plots_dir: Path, arm: str) -> list[int]:
    """Find which outer-fold numbers exist for this arm (across any threshold)."""
    folds = set()
    for p in plots_dir.glob(f"loss_curve_outer*_{arm}_*.png"):
        m = re.search(r"loss_curve_outer(\d+)_", p.name)
        if m:
            folds.add(int(m.group(1)))
    return sorted(folds)


def build_grid(plots_dir: Path, arm: str, out_path: Path):
    folds = discover_folds(plots_dir, arm)
    if not folds:
        print(f"  [skip] {arm}: no loss_curve_outer*_{arm}_*.png found")
        return

    nrow, ncol = len(folds), len(THRESHOLDS)
    # each loss png is ~7x4.5 in; scale the grid to keep tiles legible
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.2, nrow * 2.8),
                             squeeze=False)

    n_found = 0
    for i, fold in enumerate(folds):
        for j, thr in enumerate(THRESHOLDS):
            ax = axes[i][j]
            ax.axis("off")
            img_path = plots_dir / f"loss_curve_outer{fold}_{arm}_{thr}.png"
            if img_path.exists():
                ax.imshow(mpimg.imread(img_path))
                n_found += 1
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        color="#999", fontsize=11, transform=ax.transAxes)

            if i == 0:
                ax.set_title(f"threshold > {thr}", fontsize=14, pad=8)
            if j == 0:
                ax.text(-0.04, 0.5, f"fold {fold}", ha="right", va="center",
                        rotation=90, fontsize=13, transform=ax.transAxes)

    fig.suptitle(f"Loss curves — {arm}  (rows: outer fold · cols: threshold)",
                 fontsize=16, y=0.997)
    fig.tight_layout(rect=[0.02, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}  ({n_found}/{nrow*ncol} tiles present)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plots_dir", default="reporting/final_plots",
                    help="folder with consolidated loss_curve_outer*_{arm}_{thr}.png")
    ap.add_argument("--out_dir", default=None,
                    help="where to write grids (default: same as --plots_dir)")
    args = ap.parse_args()

    plots_dir = Path(args.plots_dir)
    out_dir = Path(args.out_dir) if args.out_dir else plots_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in ARMS:
        print(f"\n=== {arm} ===")
        build_grid(plots_dir, arm, out_dir / f"loss_curve_grid_{arm}.png")


if __name__ == "__main__":
    main()
