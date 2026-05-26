# Whisper → 9 PHQ symptom predictor (nested CV, participant-level)

Trains a ReLU MLP that predicts **9 binary depression symptoms** from pooled
512-d Whisper (`base`) embeddings, evaluated with **nested cross-validation
split at the participant level** so there is no leakage. Produces a deployable
inference model (audio file → 9 predictions) plus evaluation plots and CSVs.

---

## 0. Files

| File | What it is |
|---|---|
| `train_nested_cv.py` | Main script: nested CV, training, all outputs |
| `model.py` | Network + bundled inference pipeline (edit pooling here) |
| `predict.py` | CLI to run the saved model on audio or an embedding |
| `environment.yml` | Conda env (recommended on the cluster) |
| `requirements.txt` | pip fallback |
| `submit.slurm` | SLURM batch script for the GPU partition |

---

## 1. Prepare your data

You need **two CSVs** in a `data/` folder:

**`data/embeddings.csv`** — your existing file: column 1 = filename,
column 2 = path, columns 3–514 = the 512 embeddings.

**`data/labels.csv`** — produced in R: must contain a **filename** column (to
join on), a **participant id** column, and the **9 binary symptom columns**.
Dichotomize in R first (`> 27.5 → 1`, else `0`). The script **refuses to run**
if any label column contains non-binary values, so you can't accidentally feed
it the continuous 0–100 score.

> The join is on filename. Make sure the filename strings match exactly between
> the two files (same extension, no path prefix differences).

---

## 2. Run on the cluster (SLURM)

```bash
# from the project directory on Discovery
mkdir -p data logs
# put embeddings.csv and labels.csv in data/

# edit submit.slurm: set the partition name and the --symptom_cols to your
# actual 9 column names, then:
sbatch submit.slurm

squeue -u $USER                       # watch the queue
tail -f logs/whisper_phq_<jobid>.out  # watch progress
```

The first run creates the conda env (a few minutes); later runs reuse it.

### Running without SLURM (e.g. a quick local test)
```bash
conda env create -f environment.yml
conda activate whisper_phq
python train_nested_cv.py \
  --features_csv data/embeddings.csv \
  --labels_csv data/labels.csv \
  --symptom_cols PHQ1 PHQ2 PHQ3 PHQ4 PHQ5 PHQ6 PHQ7 PHQ8 PHQ9 \
  --outdir results
```

Key flags: `--outer_folds` (default 5), `--inner_folds` (4), `--epochs` (100),
`--warmup_epochs` (5), `--batch_size` (64), `--seed` (42),
`--embedding_start_col` (2), `--participant_col`, `--filename_col`.

---

## 3. Outputs (in `results/`)

**The deployable model**
- `final_inference_model.pt` — trained on all data. Bundles the network, the
  feature scaler, the symptom names, and F1-optimal per-symptom thresholds.

**Plots** (`results/plots/`)
- `loss_curve_outer{1..K}.png` — train/val BCE per epoch, one per outer fold
- `lr_curve.png` — learning rate per epoch (warmup + cosine)
- `per_symptom_auroc.png`, `per_symptom_f1.png` — mean ± std across folds
- `calibration_<symptom>.png` — are predicted probabilities trustworthy
- `confusion_<symptom>.png` — confusion matrix at threshold 0.5
- `pr_curve_<symptom>.png` — precision-recall (more honest than ROC under imbalance)

**CSVs**
- `oof_predictions.csv` — **the most useful file**: every recording's
  held-out probability + true label per symptom. Use this for any further
  error analysis without retraining.
- `per_fold_metrics.csv` — AUROC / average-precision / F1 per fold per symptom,
  with `n_test` and `n_pos` so you can see which folds had no positives.
- `summary_metrics.csv` — mean ± std per symptom.
- `selected_hyperparams.csv` — which hyperparameters the inner loop chose per fold.
- `optimal_thresholds.csv` — F1-optimal decision threshold per symptom.

---

## 4. Inference on a new audio file

The bundled model re-creates embeddings from raw audio using Whisper, so the
audio path needs `openai-whisper` installed (CPU is fine — you said inference
runs elsewhere).

```bash
python predict.py --model results/final_inference_model.pt --audio diary.wav
```

Or, if you already have a pooled 512-d embedding (skips Whisper):
```bash
python predict.py --model results/final_inference_model.pt --embedding_csv vec.csv
```

Output is JSON: a probability and a binary call for each of the 9 symptoms.

---

## 5. ⚠️ One thing you must verify: pooling consistency

Your training embeddings were pooled **upstream** (in your feature extraction).
The inference path re-pools from raw audio, and the two **must match** or
inference will quietly disagree with training.

The default is **mean-pooling over Whisper encoder frames**. If your upstream
used something else (max-pool, last hidden state, attention), edit the single
function `pool_encoder_states` at the top of `model.py` — nothing else needs to
change.

---

## 6. Honest notes on interpretation

- With ~50 participants, **per-fold metrics are noisy**. Read `summary_metrics.csv`
  as "mean ± std" and expect wide spread, especially for rare symptoms. This is
  the real cost of doing participant-level splitting correctly; diary-level
  splitting would look better and be wrong.
- Some symptoms (e.g. the rarest one) may have folds with **zero positives** in
  the test set; AUROC is undefined there and is reported as `NaN` rather than a
  misleading number. `n_pos` in `per_fold_metrics.csv` shows you where.
- The decision threshold for the binary call is separate from your 27.5 *label*
  cut. The model learns probabilities; `optimal_thresholds.csv` gives you better
  per-symptom cutoffs than a flat 0.5 if you want them.
```
