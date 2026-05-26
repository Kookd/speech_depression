#!/usr/bin/env python3
"""
confusion_grid.py
=================
Regenerate clean confusion-matrix grids from the CONSOLIDATED out-of-fold
predictions: one grid per denoising arm (original, denoised), symptoms as ROWS,
thresholds as COLUMNS (27 / 37 / 50). Shared color scale, edge-only labels,
publication quality for embedding in a Quarto (.qmd) HTML report.

DATA SOURCE (consolidated reporting folder)
-------------------------------------------
Reads from the final_tables/ folder produced by consolidate_results.py:
    final_tables/oof_predictions_{original|denoised}_{thr}.csv   (symptom, y_true, y_prob)
    final_tables/optimal_thresholds_{original|denoised}_{thr}.csv (symptom, f1_optimal_threshold)

The binary call uses each model's F1-optimal threshold (fallback 0.5).
Grids are written to the final_plots/ folder by default.

USAGE
-----
    python confusion_grid.py --tables_dir reporting/final_tables \
        --out_dir reporting/final_plots
    python confusion_grid.py --normalize      # row-normalized rates vs raw counts
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

THRESHOLDS = ["27", "37", "50"]
ARMS = ["original", "denoised"]

# canonical symptom display order (rows). Must match the `symptom` strings in
# the oof CSVs after threshold-suffix stripping; edit if your names differ.
SYMPTOM_ORDER = [
    "Anhedonia", "Depressed mood", "Sleep disturbance", "Fatigue",
    "Appetite changes", "Worthlessness", "Concentration", "Psychomotor",
    "Suicidality",
]


def strip_threshold(name: str) -> str:
    """'Sleep disturbance (>37.5)' -> 'Sleep disturbance'."""
    return re.sub(r"\s*\(>[\d.]+\)\s*$", "", str(name)).strip()


def load_condition(tables_dir: Path, arm: str, thr: str):
    """Return {symptom: (y_true, y_prob, cut)} for one condition, or None."""
    oof_path = tables_dir / f"oof_predictions_{arm}_{thr}.csv"
    if not oof_path.exists():
        return None
    oof = pd.read_csv(oof_path)
    oof["symptom_clean"] = oof["symptom"].map(strip_threshold)

    cuts = {}
    thr_path = tables_dir / f"optimal_thresholds_{arm}_{thr}.csv"
    if thr_path.exists():
        td = pd.read_csv(thr_path)
        for _, r in td.iterrows():
            cuts[strip_threshold(r["symptom"])] = float(r["f1_optimal_threshold"])

    out = {}
    for sym, g in oof.groupby("symptom_clean"):
        out[sym] = (g["y_true"].to_numpy(),
                    g["y_prob"].to_numpy(),
                    cuts.get(sym, 0.5))
    return out


def build_grid(arm: str, data_by_thr: dict, out_path: Path, normalize: bool):
    nrow, ncol = len(SYMPTOM_ORDER), len(THRESHOLDS)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 2.4),
                             squeeze=False)

    for i, sym in enumerate(SYMPTOM_ORDER):
        for j, thr in enumerate(THRESHOLDS):
            ax = axes[i][j]
            cond = data_by_thr.get(thr)
            cell = cond.get(sym) if cond else None

            if cell is None or len(np.unique(cell[0])) < 2:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        color="#999", fontsize=11, transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
            else:
                yt, yp, cut = cell
                cm = confusion_matrix(yt, (yp >= cut).astype(int), labels=[0, 1])
                disp = cm.astype(float)
                if normalize:
                    rows = disp.sum(axis=1, keepdims=True)
                    disp = np.divide(disp, rows, out=np.zeros_like(disp),
                                     where=rows > 0)
                ax.imshow(disp, cmap="Blues", vmin=0,
                          vmax=1 if normalize else (cm.max() if cm.max() else 1))
                fmt = "{:.2f}" if normalize else "{:d}"
                for (r, c), v in np.ndenumerate(cm):
                    shown = (disp[r, c] if normalize else int(v))
                    ax.text(c, r, fmt.format(shown), ha="center", va="center",
                            fontsize=9,
                            color="white" if disp[r, c] > (disp.max() * 0.6) else "black")
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(["0", "1"], fontsize=7)
                ax.set_yticklabels(["0", "1"], fontsize=7)

            if i == 0:
                ax.set_title(f"> {thr}", fontsize=12, pad=10)
            if j == 0:
                ax.set_ylabel(sym, fontsize=10, rotation=0, ha="right",
                              va="center", labelpad=10)

    metric = "row-normalized" if normalize else "counts"
    fig.suptitle(f"Confusion matrices — {arm} ({metric})\n"
                 f"rows: true / cols: predicted, at F1-optimal threshold",
                 fontsize=13, y=0.995)
    fig.supxlabel("predicted", fontsize=11)
    fig.tight_layout(rect=[0.02, 0.02, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables_dir", default="reporting/final_tables",
                    help="folder with consolidated oof_predictions_*.csv files")
    ap.add_argument("--out_dir", default="reporting/final_plots")
    ap.add_argument("--normalize", action="store_true",
                    help="show row-normalized rates instead of raw counts")
    args = ap.parse_args()

    tables_dir = Path(args.tables_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for arm in ARMS:
        print(f"\n=== {arm} ===")
        data_by_thr = {}
        for thr in THRESHOLDS:
            cond = load_condition(tables_dir, arm, thr)
            if cond is None:
                print(f"  [missing] oof_predictions_{arm}_{thr}.csv not in {tables_dir}")
                continue
            data_by_thr[thr] = cond
            print(f"  loaded {arm} > {thr}")
        if not data_by_thr:
            print(f"  [skip] {arm}: nothing to plot")
            continue
        build_grid(arm, data_by_thr,
                   out_dir / f"confusion_grid_{arm}.png", args.normalize)


if __name__ == "__main__":
    main()
