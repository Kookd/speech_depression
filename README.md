# Whisper embeddings → 9 PHQ symptom predictor (nested CV, participant-level split)

Trains a ReLU multi-layer perceptron (MLP) that predicts
**9 binary depression symptoms** from pooled 512-d Whisper (`base`) embeddings, 
evaluated with **nested cross-validation split at the participant level** so 
there is no leakage. Produces a deployable inference model 
(audio file → 9 predictions) plus evaluation plots and CSVs.

---

## 0. Files

| File | What it is |
|---|---|
| `train_nested_cv.py` | Main script: nested CV, training, all outputs |
| `model.py` | Network + bundled inference pipeline (edit pooling here) |
| `predict.py` | CLI to run the saved model on audio or an embedding |
| `environment.yml` | Conda env (recommended on the cluster) |
| `requirements.txt` | pip fallback |
| `submit.sh` | SLURM batch script for the GPU partition |

---

## 1. Prepare your data

You need **three CSVs** in a `data/` folder:

**`data/embeddings.csv`** — your existing file: column 1 = filename,
columns 2–514 = the 512 embeddings. There are two versions of the embeddings:
original and denoised.

**`data/labels.csv`** — produced in R: must contain a **filename** column (to
join on), a **participant id** column, and the **9 binary symptom columns**.
Dichotomize in R first (`> 27 → 1`, else `0`). The script **refuses to run**
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

# edit submit.sh: set the partition name and the --symptom_cols to your
# actual 9 column names. Add the --denoised flag if you're working with 
# the denoised embeddings, then:

sbatch submit.sh

squeue -u $USER                       # watch the queue
tail -f logs/<jobname>_<jobid>.out  # watch progress
```

The first run creates the conda env (a few minutes); later runs reuse it.

### Running without SLURM (e.g. a quick local test)
```bash
conda env create -f environment.yml
conda activate speech_depression
## set the <threshold> to either 27, 37, or 50 at which symptoms are 
## grouped into positive or negative classes
## set either original or denoised in `outdir` path

python train_nested_cv.py \
    --features_csv   data/embeddings_original.csv \
    --labels_csv     data/labels.csv \
    --filename_col   file \
    --participant_col subject \
    --symptom_cols   PHQ.1_f<threshold> PHQ.2_f<threshold> PHQ.3_f<threshold> PHQ.4_f<threshold> PHQ.5_f<threshold> PHQ.6_f<threshold> PHQ.7_f<threshold> PHQ.8_f<threshold> PHQ.9_f<threshold> \
    --embedding_start_col 1 \
    --outer_folds 5 \
    --inner_folds 4 \
    --epochs 100 \
    --warmup_epochs 5 \
    --batch_size 64 \
    --whisper_model_name base \
    --seed 42 \
    --patience 10 \
    --outdir results/v<threshold>/<original|denoised>/job_${SLURM_JOB_ID}
```

Key flags: `--outer_folds` (default 5), `--inner_folds` (4), `--epochs` (100),
`--warmup_epochs` (5), `--batch_size` (64), `--seed` (42),
`--embedding_start_col` (1), `--participant_col`, `--filename_col`, 
`--denoised`, `--patience`

---

## 3. Outputs (in `results/v<threshold>/<original|denoised>/job_${SLURM_JOB_ID}`)

**The deployable model**
- `final_inference_model.pt` — trained on all data. Bundles the network, the
  feature scaler, the symptom names, and F1-optimal per-symptom thresholds.

**Plots** (`results/v<threshold>/<original|denoised>/job_${SLURM_JOB_ID}/plots`)
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
audio path needs `openai-whisper` installed (CPU is fine for both processing
and inference). 
```bash
python predict.py --model results/final_inference_model.pt --audio diary.wav
```

Or, if you already have a pooled 512-d embedding (skips Whisper):
```bash
python predict.py --model results/final_inference_model.pt --embedding_csv vec.csv
```

Output is JSON: a probability and a binary call for each of the 9 symptoms.

---

## 5. Notes on interpretation

- With 57 participants, **per-fold metrics are noisy**. Read `summary_metrics.csv`
  as "mean ± std" and expect wide spread, especially for rare symptoms. 
- Some symptoms (e.g. the rarest one like suicidality) may have folds with 
  **zero positives** in the test set; AUROC is undefined there and is reported 
  as `NaN` rather than a misleading number. `n_pos` in `per_fold_metrics.csv` 
  shows you where.
- The decision threshold for the binary call is separate from the threshold
  *label* cut. The model learns probabilities; `optimal_thresholds.csv` gives 
  better per-symptom cutoffs than a flat 0.5 if you want them.
```
