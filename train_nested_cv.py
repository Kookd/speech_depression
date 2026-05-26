"""
train_nested_cv.py
==================
Nested cross-validation, split at the PARTICIPANT level, to train and honestly
evaluate a ReLU MLP that predicts 9 binary depression symptoms from pooled
Whisper embeddings.

Design decisions (see README for rationale):
  * Outer loop  -> unbiased performance estimate
  * Inner loop  -> hyperparameter selection (no peeking at outer test)
  * Splitting   -> StratifiedGroupKFold on participant_id, so NO participant
                   appears in both train and test of any split. This is what
                   prevents leakage; folds will be uneven in size and that's OK.
  * Loss        -> BCEWithLogitsLoss with per-symptom pos_weight (imbalance)
  * Schedule    -> linear warmup then cosine annealing
  * Scaling     -> StandardScaler fit on TRAIN ONLY inside each split
  * Final model -> retrained on ALL data with the most-frequently-selected
                   hyperparameters, then bundled into an InferencePipeline.

Outputs (written to --outdir):
  plots/loss_curve_outer{k}.png         train/val BCE per epoch (each outer fold)
  plots/lr_curve.png                    learning rate per epoch
  plots/per_symptom_auroc.png           AUROC per symptom (out-of-fold)
  plots/per_symptom_f1.png              F1 per symptom (out-of-fold)
  plots/calibration_<symptom>.png       reliability curve per symptom
  plots/confusion_<symptom>.png         confusion matrix per symptom
  plots/pr_curve_<symptom>.png          precision-recall curve per symptom
  oof_predictions.csv                   held-out prob+pred per recording/symptom
  per_fold_metrics.csv                  metrics per outer fold per symptom
  selected_hyperparams.csv              chosen HPs per outer fold
  optimal_thresholds.csv                F1-optimal threshold per symptom
  summary_metrics.csv                   mean +/- std across folds per symptom
  final_inference_model.pt              the deployable pipeline (see predict.py)
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter
from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / SLURM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from model import InferencePipeline, SymptomMLP

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
#  Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
#  Display names + filename slugs
#
#  The SLURM script passes the REAL csv column names (e.g. PHQ.1_f27). We parse
#  the PHQ item number + dichotomization version out of each column and build a
#  clean human-readable label for plot titles and CSVs, and a filesystem-safe
#  slug for plot filenames.
# --------------------------------------------------------------------------- #
SYMPTOM_NAMES = {
    1: "Anhedonia",
    2: "Depressed mood",
    3: "Sleep disturbance",
    4: "Fatigue",
    5: "Appetite changes",
    6: "Worthlessness",
    7: "Concentration",
    8: "Psychomotor",
    9: "Suicidality",
}

# label appended per dichotomization version. "" = show nothing (original cut).
# Adjust the _f37 cut to whatever threshold that version actually uses.
VERSION_LABELS = {
    "f50": "",
    "f27": " (>27)",
    "f37": " (>37)",
}


def pretty_name(col: str) -> str:
    """PHQ.1_f27 -> 'Anhedonia (>27)'. Unrecognized -> returned unchanged."""
    m = re.match(r"PHQ\.(\d+)_(\w+)", col)
    if not m:
        return col
    num, ver = int(m.group(1)), m.group(2)
    base = SYMPTOM_NAMES.get(num, col)
    return base + VERSION_LABELS.get(ver, f" ({ver})")


def slugify(label: str) -> str:
    """'Anhedonia (>27)' -> 'anhedonia_27'. Lowercase, spaces->_, safe chars."""
    s = label.lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]", "", s)   # drop parens, >, ., etc.
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #
def load_data(
    features_csv: str,
    labels_csv: str,
    filename_col: str,
    participant_col: str,
    symptom_cols: list[str],
    embedding_start_col: int,
):
    """Join features + labels on filename, return X, Y, groups, names.

    features_csv : cols [filename, path, emb_0 ... emb_511]
    labels_csv   : must contain filename_col, participant_col, and symptom_cols
    """
    feats = pd.read_csv(features_csv)
    labels = pd.read_csv(labels_csv)

    fname_feat = feats.columns[0]  # first col = filename per your layout
    emb_cols = list(feats.columns[embedding_start_col:])
    if len(emb_cols) != 512:
        warnings.warn(
            f"Expected 512 embedding columns, found {len(emb_cols)}. "
            "Proceeding with what was found — check embedding_start_col."
        )

    merged = feats.merge(
        labels, left_on=fname_feat, right_on=filename_col, how="inner"
    )
    if len(merged) == 0:
        raise ValueError(
            "Join produced 0 rows. Check that filenames match between the "
            f"features file column '{fname_feat}' and labels column "
            f"'{filename_col}'."
        )
    n_dropped = len(feats) - len(merged)
    if n_dropped > 0:
        warnings.warn(f"{n_dropped} feature rows had no matching label and were dropped.")

    X = merged[emb_cols].to_numpy(dtype=np.float32)
    Y = merged[symptom_cols].to_numpy(dtype=np.float32)
    groups = merged[participant_col].to_numpy()

    # --- GUARD: labels must be binary {0,1} ---
    uniq = np.unique(Y[~np.isnan(Y)])
    if not np.all(np.isin(uniq, [0.0, 1.0])):
        raise ValueError(
            f"Label columns must be binary 0/1 but contain values {uniq}. "
            "Did you pass the continuous (0-100) score instead of the "
            "dichotomized column? Dichotomize in R (>27 -> 1) first."
        )
    if np.isnan(Y).any():
        raise ValueError("Found NaN in label columns. Clean these in R first.")

    print(f"[data] {X.shape[0]} recordings, {X.shape[1]} embedding dims, "
          f"{len(np.unique(groups))} participants, {Y.shape[1]} symptoms")
    for j, name in enumerate(symptom_cols):
        pos = int(Y[:, j].sum())
        print(f"        {name:>12}: {pos:4d} positives "
              f"({100*pos/len(Y):5.1f}%)")
    return X, Y, groups, symptom_cols


# --------------------------------------------------------------------------- #
#  LR schedule: linear warmup -> cosine annealing
# --------------------------------------------------------------------------- #
def make_lr_lambda(warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return lr_lambda


# --------------------------------------------------------------------------- #
#  pos_weight for imbalance (computed on the training split only)
# --------------------------------------------------------------------------- #
def compute_pos_weight(Y_train: np.ndarray) -> torch.Tensor:
    pos = Y_train.sum(axis=0)
    neg = Y_train.shape[0] - pos
    # avoid div-by-zero for symptoms with no positives in this split
    pos_weight = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  Train one model (returns trained model + per-epoch train/val loss + lr trace)
# --------------------------------------------------------------------------- #
def train_one(
    X_tr, Y_tr, X_val, Y_val,
    hidden_dims, dropout, lr, weight_decay,
    epochs, warmup_epochs, batch_size, device,
    record_lr=False,
    patience=10, restore_best=True,
):
    """Train an MLP. Early stopping: track the epoch with lowest validation
    loss, keep those weights, and (if restore_best) load them back at the end
    so the returned model is the best-generalizing checkpoint, not the last
    (most-overfit) epoch. `patience` stops training if val loss hasn't improved
    for that many epochs. Returns the actual stopping epoch as `best_epoch`.
    """
    import copy

    in_dim = X_tr.shape[1]
    n_out = Y_tr.shape[1]
    model = SymptomMLP(in_dim, hidden_dims, n_out, dropout).to(device)

    pos_weight = compute_pos_weight(Y_tr).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_epochs, epochs)
    )

    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    Ytr = torch.tensor(Y_tr, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    Yva = torch.tensor(Y_val, dtype=torch.float32, device=device)

    n = Xtr.shape[0]
    train_losses, val_losses, lr_trace = [], [], []

    best_val = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_since_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            logits = model(Xtr[idx])
            loss = criterion(logits, Ytr[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        train_losses.append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xva), Yva).item()
        val_losses.append(val_loss)
        if record_lr:
            lr_trace.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

        # --- early stopping bookkeeping ---
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if patience is not None and epochs_since_improve >= patience:
                break

    if restore_best:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses, lr_trace, best_epoch



# --------------------------------------------------------------------------- #
#  Metrics helpers that degrade gracefully when a fold has no positives
# --------------------------------------------------------------------------- #
def safe_auroc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def safe_ap(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return average_precision_score(y_true, y_prob)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features_csv", required=True)
    p.add_argument("--labels_csv", required=True,
                   help="CSV with filename, participant id, and 9 binary symptom cols.")
    p.add_argument("--outdir", default="results")
    p.add_argument("--filename_col", default="filename")
    p.add_argument("--participant_col", default="participant_id")
    p.add_argument("--symptom_cols", nargs=9, required=True,
                   help="The 9 binary symptom column names, in order.")
    p.add_argument("--embedding_start_col", type=int, default=2,
                   help="0-indexed column where the 512 embeddings begin (default 2).")
    p.add_argument("--outer_folds", type=int, default=5)
    p.add_argument("--inner_folds", type=int, default=4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping: stop if val loss hasn't improved for "
                        "this many epochs. The best-val-loss checkpoint is "
                        "always restored regardless of where it stopped.")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--whisper_model_name", default="base")
    p.add_argument("--denoised", action="store_true",
                   help="Set if these embeddings were extracted from DENOISED "
                        "audio. Stamped into the saved model so the app applies "
                        "the matching preprocessing at inference.")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    outdir = Path(args.outdir)
    (outdir / "plots").mkdir(parents=True, exist_ok=True)

    X, Y, groups, names = load_data(
        args.features_csv, args.labels_csv, args.filename_col,
        args.participant_col, args.symptom_cols, args.embedding_start_col,
    )
    n_out = Y.shape[1]

    # Map raw csv column names -> clean display labels (used in titles + CSVs),
    # and a parallel slug map for filesystem-safe plot filenames.
    raw_names = names
    names = [pretty_name(c) for c in raw_names]
    slug_of = {disp: slugify(disp) for disp in names}
    print("[names] column -> display:")
    for raw, disp in zip(raw_names, names):
        print(f"        {raw:>12}  ->  {disp}  (file: {slug_of[disp]})")

    # Hyperparameter grid searched in the INNER loop. Kept small on purpose:
    # ~50 participants cannot support a big search without overfitting the CV.
    hp_grid = list(product(
        [(128, 64), (256, 64)],   # hidden_dims
        [0.3, 0.5],               # dropout
        [1e-3, 3e-4],             # lr
        [1e-4],                   # weight_decay
    ))

    # Stratify on a single representative label so folds keep some balance.
    # We use the most prevalent symptom as the stratification target; group
    # constraint (participant) always takes precedence over stratification.
    strat_target = Y[:, int(np.argmax(Y.sum(axis=0)))].astype(int)

    outer = StratifiedGroupKFold(n_splits=args.outer_folds, shuffle=True,
                                 random_state=args.seed)

    oof_rows = []            # out-of-fold predictions
    per_fold_rows = []       # metrics per fold per symptom
    selected_hps = []        # chosen HP per outer fold
    lr_trace_saved = None    # captured once for the LR plot

    for k, (tr_idx, te_idx) in enumerate(
        outer.split(X, strat_target, groups)
    ):
        print(f"\n===== OUTER FOLD {k+1}/{args.outer_folds} =====")
        X_tr_o, X_te = X[tr_idx], X[te_idx]
        Y_tr_o, Y_te = Y[tr_idx], Y[te_idx]
        g_tr_o = groups[tr_idx]
        print(f"  train participants={len(np.unique(g_tr_o))}, "
              f"test participants={len(np.unique(groups[te_idx]))}, "
              f"train n={len(tr_idx)}, test n={len(te_idx)}")

        # ---------------- inner loop: pick hyperparameters ----------------
        strat_inner = Y_tr_o[:, int(np.argmax(Y_tr_o.sum(axis=0)))].astype(int)
        inner = StratifiedGroupKFold(n_splits=args.inner_folds, shuffle=True,
                                     random_state=args.seed)
        hp_scores = {i: [] for i in range(len(hp_grid))}

        for itr_idx, iva_idx in inner.split(X_tr_o, strat_inner, g_tr_o):
            Xi_tr, Xi_va = X_tr_o[itr_idx], X_tr_o[iva_idx]
            Yi_tr, Yi_va = Y_tr_o[itr_idx], Y_tr_o[iva_idx]

            scaler = StandardScaler().fit(Xi_tr)
            Xi_tr_s = scaler.transform(Xi_tr)
            Xi_va_s = scaler.transform(Xi_va)

            for hp_i, (hd, dr, lr, wd) in enumerate(hp_grid):
                model, _, _, _, _ = train_one(
                    Xi_tr_s, Yi_tr, Xi_va_s, Yi_va,
                    hd, dr, lr, wd,
                    args.epochs, args.warmup_epochs, args.batch_size, device,
                    patience=args.patience,
                )
                model.eval()
                with torch.no_grad():
                    probs = torch.sigmoid(
                        model(torch.tensor(Xi_va_s, dtype=torch.float32, device=device))
                    ).cpu().numpy()
                # mean AUROC across symptoms that are defined in this val split
                aurocs = [safe_auroc(Yi_va[:, j], probs[:, j]) for j in range(n_out)]
                hp_scores[hp_i].append(np.nanmean(aurocs))

        mean_hp = {i: np.nanmean(s) for i, s in hp_scores.items()}
        best_hp_i = max(mean_hp, key=mean_hp.get)
        best_hd, best_dr, best_lr, best_wd = hp_grid[best_hp_i]
        print(f"  selected HP: hidden={best_hd} dropout={best_dr} "
              f"lr={best_lr} wd={best_wd} (inner mean AUROC={mean_hp[best_hp_i]:.3f})")
        selected_hps.append({
            "outer_fold": k + 1, "hidden_dims": str(best_hd),
            "dropout": best_dr, "lr": best_lr, "weight_decay": best_wd,
            "inner_mean_auroc": mean_hp[best_hp_i],
        })

        # ---------------- refit on full outer-train, eval on outer-test ----
        scaler = StandardScaler().fit(X_tr_o)
        X_tr_s = scaler.transform(X_tr_o)
        X_te_s = scaler.transform(X_te)

        # Use the last inner val fold's split as the in-training val curve.
        # Simpler & leakage-free: carve a small grouped val out of outer-train.
        inner_for_curve = StratifiedGroupKFold(n_splits=args.inner_folds,
                                               shuffle=True, random_state=args.seed)
        c_tr, c_va = next(inner_for_curve.split(X_tr_s, strat_inner, g_tr_o))
        model, tr_losses, va_losses, lr_trace, best_epoch = train_one(
            X_tr_s[c_tr], Y_tr_o[c_tr], X_tr_s[c_va], Y_tr_o[c_va],
            best_hd, best_dr, best_lr, best_wd,
            args.epochs, args.warmup_epochs, args.batch_size, device,
            record_lr=(lr_trace_saved is None),
            patience=args.patience,
        )
        if lr_trace_saved is None:
            lr_trace_saved = lr_trace

        # loss curve for this fold, with the early-stopping epoch marked
        plt.figure(figsize=(7, 4.5))
        plt.plot(tr_losses, label="train BCE")
        plt.plot(va_losses, label="val BCE")
        plt.axvline(best_epoch, color="grey", ls="--", lw=1,
                    label=f"best epoch ({best_epoch})")
        plt.xlabel("epoch"); plt.ylabel("BCE loss")
        plt.title(f"Loss curve — outer fold {k+1}")
        plt.legend(); plt.tight_layout()
        plt.savefig(outdir / "plots" / f"loss_curve_outer{k+1}.png", dpi=150)
        plt.close()

        # predict on outer test
        model.eval()
        with torch.no_grad():
            te_probs = torch.sigmoid(
                model(torch.tensor(X_te_s, dtype=torch.float32, device=device))
            ).cpu().numpy()

        for row_i, rec_idx in enumerate(te_idx):
            for j, sname in enumerate(names):
                oof_rows.append({
                    "outer_fold": k + 1,
                    "row_index": int(rec_idx),
                    "participant_id": groups[rec_idx],
                    "symptom": sname,
                    "y_true": int(Y_te[row_i, j]),
                    "y_prob": float(te_probs[row_i, j]),
                })

        for j, sname in enumerate(names):
            yt, yp = Y_te[:, j], te_probs[:, j]
            auroc = safe_auroc(yt, yp)
            ap = safe_ap(yt, yp)
            f1 = (f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)
                  if len(np.unique(yt)) > 1 else np.nan)
            per_fold_rows.append({
                "outer_fold": k + 1, "symptom": sname,
                "n_test": len(yt), "n_pos": int(yt.sum()),
                "auroc": auroc, "average_precision": ap, "f1@0.5": f1,
            })

    # ----------------------- aggregate + write CSVs ------------------------
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(outdir / "oof_predictions.csv", index=False)
    per_fold_df = pd.DataFrame(per_fold_rows)
    per_fold_df.to_csv(outdir / "per_fold_metrics.csv", index=False)
    pd.DataFrame(selected_hps).to_csv(outdir / "selected_hyperparams.csv", index=False)

    summary = (per_fold_df.groupby("symptom")
               .agg(auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"),
                    ap_mean=("average_precision", "mean"),
                    f1_mean=("f1@0.5", "mean"), f1_std=("f1@0.5", "std"))
               .reset_index())
    summary.to_csv(outdir / "summary_metrics.csv", index=False)
    print("\n[summary]\n", summary.to_string(index=False))

    # ------------------- F1-optimal thresholds from OOF --------------------
    thr_rows, opt_thresholds = [], []
    for sname in names:
        d = oof_df[oof_df.symptom == sname]
        yt, yp = d.y_true.to_numpy(), d.y_prob.to_numpy()
        if len(np.unique(yt)) > 1:
            prec, rec, thr = precision_recall_curve(yt, yp)
            f1s = np.divide(2 * prec * rec, prec + rec,
                            out=np.zeros_like(prec), where=(prec + rec) > 0)
            best = thr[max(0, np.argmax(f1s) - 1)] if len(thr) else 0.5
        else:
            best = 0.5
        opt_thresholds.append(best)
        thr_rows.append({"symptom": sname, "f1_optimal_threshold": float(best)})
    pd.DataFrame(thr_rows).to_csv(outdir / "optimal_thresholds.csv", index=False)

    # ----------------------------- plots -----------------------------------
    if lr_trace_saved:
        plt.figure(figsize=(7, 4.5))
        plt.plot(lr_trace_saved)
        plt.xlabel("epoch"); plt.ylabel("learning rate")
        plt.title("Learning rate schedule (warmup + cosine)")
        plt.tight_layout()
        plt.savefig(outdir / "plots" / "lr_curve.png", dpi=150)
        plt.close()

    # per-symptom AUROC / F1 bar charts (mean +/- std across folds)
    for metric, mean_c, std_c, fname in [
        ("AUROC", "auroc_mean", "auroc_std", "per_symptom_auroc.png"),
        ("F1@0.5", "f1_mean", "f1_std", "per_symptom_f1.png"),
    ]:
        plt.figure(figsize=(9, 4.5))
        order = summary.sort_values(mean_c, ascending=False)
        plt.bar(order.symptom, order[mean_c],
                yerr=order[std_c].fillna(0), capsize=3)
        plt.axhline(0.5, color="grey", ls="--", lw=1)
        plt.ylabel(metric); plt.xticks(rotation=45, ha="right")
        plt.title(f"{metric} per symptom (out-of-fold, mean +/- std)")
        plt.tight_layout()
        plt.savefig(outdir / "plots" / fname, dpi=150)
        plt.close()

    # calibration, confusion, PR curve per symptom
    for sname in names:
        slug = slug_of[sname]
        d = oof_df[oof_df.symptom == sname]
        yt, yp = d.y_true.to_numpy(), d.y_prob.to_numpy()
        if len(np.unique(yt)) < 2:
            continue

        # calibration
        bins = np.linspace(0, 1, 11)
        binid = np.digitize(yp, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = binid == b
            if m.sum() > 0:
                xs.append(yp[m].mean()); ys.append(yt[m].mean())
        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], "--", color="grey")
        plt.plot(xs, ys, "o-")
        plt.xlabel("mean predicted prob"); plt.ylabel("observed frequency")
        plt.title(f"Calibration — {sname}")
        plt.tight_layout()
        plt.savefig(outdir / "plots" / f"calibration_{slug}.png", dpi=150)
        plt.close()

        # confusion @ 0.5
        cm = confusion_matrix(yt, (yp >= 0.5).astype(int), labels=[0, 1])
        plt.figure(figsize=(4.2, 4))
        plt.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            plt.text(c, r, str(v), ha="center", va="center")
        plt.xticks([0, 1], ["pred 0", "pred 1"])
        plt.yticks([0, 1], ["true 0", "true 1"])
        plt.title(f"Confusion @0.5 — {sname}")
        plt.colorbar(); plt.tight_layout()
        plt.savefig(outdir / "plots" / f"confusion_{slug}.png", dpi=150)
        plt.close()

        # PR curve
        prec, rec, _ = precision_recall_curve(yt, yp)
        plt.figure(figsize=(5, 4.5))
        plt.plot(rec, prec)
        plt.xlabel("recall"); plt.ylabel("precision")
        plt.title(f"PR curve — {sname} (AP={safe_ap(yt, yp):.3f})")
        plt.tight_layout()
        plt.savefig(outdir / "plots" / f"pr_curve_{slug}.png", dpi=150)
        plt.close()

    # --------------- final deployable model on ALL data --------------------
    print("\n[final] retraining on all data with majority-selected HPs...")
    hp_counter = Counter(
        (h["hidden_dims"], h["dropout"], h["lr"], h["weight_decay"])
        for h in selected_hps
    )
    (hd_str, dr, lr, wd), _ = hp_counter.most_common(1)[0]
    best_hd = eval(hd_str)  # str like "(128, 64)" -> tuple; our own data, safe

    final_scaler = StandardScaler().fit(X)
    X_s = final_scaler.transform(X)
    # small grouped holdout just to produce a final loss curve (not for selection)
    fk = StratifiedGroupKFold(n_splits=max(2, args.inner_folds),
                              shuffle=True, random_state=args.seed)
    f_tr, f_va = next(fk.split(X_s, strat_target, groups))
    final_model, ftr, fva, _, final_best_epoch = train_one(
        X_s[f_tr], Y[f_tr], X_s[f_va], Y[f_va],
        best_hd, dr, lr, wd,
        args.epochs, args.warmup_epochs, args.batch_size, device,
        patience=args.patience,
    )
    print(f"[final] early-stopped at epoch {final_best_epoch}; using that "
          "checkpoint (trained on the grouped holdout split) as the deployed "
          "model. Training on a holdout — rather than a val==train pass over "
          "all data — avoids deploying an overfit last-epoch model.")

    pipeline = InferencePipeline(
        model=final_model.cpu(),
        scaler_mean=final_scaler.mean_,
        scaler_scale=final_scaler.scale_,
        symptom_names=names,
        thresholds=np.array(opt_thresholds),
        whisper_model_name=args.whisper_model_name,
        denoised=args.denoised,
    )
    pipeline.save(outdir / "final_inference_model.pt")
    print(f"[final] saved -> {outdir/'final_inference_model.pt'}")
    print("[done] all outputs in", outdir)


if __name__ == "__main__":
    main()