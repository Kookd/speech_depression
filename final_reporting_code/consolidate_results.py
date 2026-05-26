#!/usr/bin/env python3
"""
consolidate_results.py
======================
Collect the final models, tables, and plots from the 6 training conditions
(3 thresholds x {original, denoised}) into three clean reporting folders:

    models/         <- the 6 .pt files, renamed {original|denoised}_{thr}.pt
    final_tables/   <- key CSVs, renamed *_{original|denoised}_{thr}.csv
    final_plots/    <- selected plots, renamed *_{original|denoised}_{thr}.png

SOURCE LAYOUT EXPECTED
----------------------
    results/v{thr}/{original|denoised}/job_{id}/  ... CSVs + final_inference_model.pt
    results/v{thr}/{original|denoised}/job_{id}/plots/  ... plots

By default the NEWEST job_ directory per condition is used (by mtime). The
script prints its choices and asks for confirmation before copying. Override
any condition with --job v27:denoised=8684588 (repeatable).

USAGE
-----
    python consolidate_results.py --results_dir results --out_dir reporting
    python consolidate_results.py --results_dir results --out_dir reporting \
        --job v27:denoised=8684588 --job v50:original=8690011
    # add --force to overwrite existing files, --yes to skip confirmation
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

THRESHOLDS = ["27", "37", "50"]
ARMS = ["original", "denoised"]

# CSVs to copy (real pipeline filenames). value = output basename stem.
TABLE_FILES = {
    "optimal_thresholds.csv": "optimal_thresholds",
    "per_fold_metrics.csv": "per_fold_metrics",
    "selected_hyperparams.csv": "selected_hyperparams",
    "summary_metrics.csv": "summary_metrics",
    "oof_predictions.csv": "oof_predictions",
}

# Fixed (non per-symptom) plots: copied as-is with a tag appended.
FIXED_PLOTS = ["per_symptom_auroc.png", "per_symptom_f1.png", "lr_curve.png"]
# loss_curve_outer{N}.png handled by glob. confusion_*.png handled by glob.


def parse_job_overrides(items: list[str]) -> dict:
    """['v27:denoised=8684588'] -> {('27','denoised'): '8684588'}"""
    out = {}
    for it in items or []:
        m = re.match(r"v(\d+):(original|denoised)=(.+)", it.strip())
        if not m:
            sys.exit(f"Bad --job format: {it!r}. Expected v27:denoised=JOBID")
        out[(m.group(1), m.group(2))] = m.group(3)
    return out


def pick_job_dir(cond_dir: Path, override: str | None) -> Path | None:
    """Choose the job_ directory for a condition: explicit override, else newest."""
    if not cond_dir.is_dir():
        return None
    job_dirs = sorted([d for d in cond_dir.iterdir()
                       if d.is_dir() and d.name.startswith("job_")])
    if not job_dirs:
        return None
    if override is not None:
        match = [d for d in job_dirs if override in d.name]
        if not match:
            sys.exit(f"Override job '{override}' not found in {cond_dir}")
        return match[0]
    # newest by modification time
    return max(job_dirs, key=lambda d: d.stat().st_mtime)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="reporting",
                    help="parent dir; models/, final_tables/, final_plots/ created inside")
    ap.add_argument("--job", action="append", default=[],
                    help="override a condition's job, e.g. v27:denoised=8684588 (repeatable)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    overrides = parse_job_overrides(args.job)

    # --- resolve the job dir for each of the 6 conditions ---
    chosen: dict[tuple[str, str], Path] = {}
    print("Resolving job directories per condition:\n")
    missing = []
    for thr in THRESHOLDS:
        for arm in ARMS:
            cond_dir = results_dir / f"v{thr}" / arm
            jobd = pick_job_dir(cond_dir, overrides.get((thr, arm)))
            tag = f"{arm}_{thr}"
            if jobd is None:
                print(f"  [MISSING] {tag:16} <- {cond_dir} (no job_ dir found)")
                missing.append(tag)
            else:
                src = "override" if (thr, arm) in overrides else "newest"
                print(f"  {tag:16} <- {jobd}   ({src})")
                chosen[(thr, arm)] = jobd

    if missing:
        print(f"\n[warn] {len(missing)} condition(s) missing: {', '.join(missing)}")

    if not args.yes:
        resp = input("\nProceed with these selections? [y/N] ").strip().lower()
        if resp != "y":
            sys.exit("Aborted.")

    # --- prepare output dirs ---
    out = Path(args.out_dir)
    models_dir = out / "models"
    tables_dir = out / "final_tables"
    plots_dir = out / "final_plots"
    for d in (models_dir, tables_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    def copy(src: Path, dst: Path):
        if dst.exists() and not args.force:
            print(f"    [skip exists] {dst.name} (use --force to overwrite)")
            return False
        if not src.exists():
            print(f"    [MISSING SRC] {src}")
            return False
        shutil.copy2(src, dst)
        print(f"    {src.name}  ->  {dst.name}")
        return True

    n_models = 0
    for (thr, arm), jobd in chosen.items():
        tag = f"{arm}_{thr}"
        print(f"\n=== {tag} ({jobd}) ===")

        # model
        if copy(jobd / "final_inference_model.pt", models_dir / f"{tag}.pt"):
            n_models += 1

        # tables
        for fname, stem in TABLE_FILES.items():
            copy(jobd / fname, tables_dir / f"{stem}_{tag}.csv")

        # fixed plots
        pdir = jobd / "plots"
        for p in FIXED_PLOTS:
            base = p[:-4]  # strip .png
            copy(pdir / p, plots_dir / f"{base}_{tag}.png")

        # loss curves (variable count)
        for lc in sorted(pdir.glob("loss_curve_outer*.png")):
            copy(lc, plots_dir / f"{lc.stem}_{tag}.png")

        # confusion matrices (per symptom): confusion_<symptom>.png
        for cm in sorted(pdir.glob("confusion_*.png")):
            copy(cm, plots_dir / f"{cm.stem}_{tag}.png")

    # --- invariant check ---
    print("\n" + "=" * 50)
    pt_count = len(list(models_dir.glob("*.pt")))
    print(f"models/ now contains {pt_count} .pt file(s) "
          f"({'OK' if pt_count == 6 else 'EXPECTED 6 — check missing conditions'})")
    print(f"final_tables/: {len(list(tables_dir.glob('*.csv')))} csv files")
    print(f"final_plots/:  {len(list(plots_dir.glob('*.png')))} png files")
    print(f"\nOutput in: {out.resolve()}")


if __name__ == "__main__":
    main()
