#!/bin/bash
#SBATCH --job-name=speech_depression_27
#SBATCH --output=logs/original/v27/%x_%j.out
#SBATCH --error=logs/original/v27/%x_%j.err
#SBATCH --partition=gpuq            # <-- CHANGE to cluster name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00             # generous; ~850 rows trains fast
#SBATCH --mail-type=BEGIN,END,FAIL  # email notifications
#SBATCH --mail-user=david.kook.gr@dartmouth.edu 


# ---------------------------------------------------------------------------
# Dartmouth Discovery / SLURM submission script
#   Submit with:  sbatch submit.slurm
#   Check on it:  squeue -u $USER       /   tail -f logs/whisper_phq_<jobid>.out
# Adjust the partition name, time, and module lines to match your allocation.
# Change lines 11, 29, 30, 32, and 33 to match your Dartmouth email,
# remote project directory, an dconda environment path.
# Edit lines 2 - 4, 71, and 73 
# if you're using denoised or a different threshold (27, 37, or 50).
# ---------------------------------------------------------------------------

set -eo pipefail
export LOGLEVEL="DEBUG"

node=$(hostname -s)
user=$(whoami)
cluster="discovery"
remote_project_dir="/dartfs-hpc/rc/home/n/f0069vn/speech_depression"

source /optnfs/common/miniconda3/etc/profile.d/conda.sh

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/dartfs-hpc/rc/home/n/f0069vn/.conda/envs/speech_depression/lib
conda activate /dartfs-hpc/rc/home/n/f0069vn/.conda/envs/speech_depression

cd $remote_project_dir

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# ---------------------------------------------------------------------------
# RUN. Edit the column names / paths below to match your files.
#   --symptom_cols takes the NINE real CSV column names in order. Pass ONE
#     dichotomization version per run. The script internally maps these to
#     clean display names (e.g. PHQ.1_f27 -> "Anhedonia (>27.5)") for plot
#     titles + CSVs, and clean lowercase filenames (anhedonia_275.png).
#   To compare versions, run three times with different --symptom_cols and
#     different --outdir, e.g.:
#       _f   columns -> --outdir results/v_orig
#       _f27 columns -> --outdir results/v_27
#       _f37 columns -> --outdir results/v_37
#   --embedding_start_col is the 0-indexed column where the 512 embeddings
#     begin (your layout = filename, path, emb... -> start col 2).
# ---------------------------------------------------------------------------

python train_nested_cv.py \
    --features_csv   data/embeddings_original.csv \
    --labels_csv     data/labels.csv \
    --filename_col   file \
    --participant_col src_subject_id \
    --symptom_cols   PHQ.1_f27 PHQ.2_f27 PHQ.3_f27 PHQ.4_f27 PHQ.5_f27 PHQ.6_f27 PHQ.7_f27 PHQ.8_f27 PHQ.9_f27 \
    --embedding_start_col 2 \
    --outer_folds 5 \
    --inner_folds 4 \
    --epochs 100 \
    --warmup_epochs 5 \
    --batch_size 64 \
    --whisper_model_name base \
    --seed 42 \
    --patience 10 \
    --outdir results/v27/original/job_${SLURM_JOB_ID}

echo "Job finished. Outputs in results/v27/original/job_${SLURM_JOB_ID}"
