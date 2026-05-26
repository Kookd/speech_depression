"""
app.py
======
Local Streamlit app for the trained Whisper -> 9-symptom models.

Supports SIX models = {raw, denoised} x {threshold 27, 37, 50}. The user picks
a threshold; the app shows that threshold's RAW and DENOISED models side by
side so the denoising effect is readable down each symptom row.

MODEL DISCOVERY
---------------
Point --models_dir at a folder containing the six .pt files. Each file's
denoised/raw nature is read from metadata stored INSIDE the .pt (the `denoised`
flag), so preprocessing can never mismatch. Filenames are used only to group by
threshold; the convention expected is any name containing the threshold number
and the word 'denoised' or 'raw', e.g.:
    model_raw_27.pt   model_denoised_27.pt
    model_raw_37.pt   model_denoised_37.pt
    model_raw_50.pt   model_denoised_50.pt

RUN
---
    conda activate speech_depression
    streamlit run app.py -- --models_dir models/

  (do the denoiser weight pre-download on a LOGIN node first:
     python -c "from denoiser import pretrained; pretrained.dns64()")

RESEARCH USE ONLY — cross-validated on ~50 participants. NOT a diagnostic tool.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
import re

from model import InferencePipeline


# --------------------------------------------------------------------------- #
#  Args + model discovery
# --------------------------------------------------------------------------- #
def clean_symptom(name: str) -> str:
    """Drop any trailing threshold annotation like ' (>37.5)' from a symptom name."""
    return re.sub(r"\s*\(>[\d.]+\)\s*$", "", name).strip()

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models_dir", default="models")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


def discover_models(models_dir: str) -> dict:
    """Return {threshold: {'raw': path, 'denoised': path}} from the folder."""
    found: dict[str, dict[str, str]] = {}
    for pt in sorted(Path(models_dir).glob("*.pt")):
        name = pt.name.lower()
        m = re.search(r"(\d{2})", name)        # threshold number (27/37/50)
        thr = m.group(1) if m else "?"
        kind = "denoised" if "denoised" in name else "raw"
        found.setdefault(thr, {})[kind] = str(pt)
    return found




@st.cache_resource(show_spinner="Loading model...")
def load_pipeline(model_path: str) -> InferencePipeline:
    return InferencePipeline.load(model_path)


def to_wav_bytes(data: bytes, suffix: str):
    with tempfile.NamedTemporaryFile(suffix=suffix) as src, \
         tempfile.NamedTemporaryFile(suffix=".wav") as dst:
        src.write(data); src.flush()
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", src.name, "-ar", "16000", "-ac", "1", dst.name],
                capture_output=True, check=True,
            )
            return Path(dst.name).read_bytes()
        except Exception:
            return None


# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Audio -> Symptom Predictions", layout="wide")
st.title("Audio Diary -> Depression Symptom Predictions")

st.warning(
    "**Research use only.** These models were cross-validated on ~50 "
    "participants and are **not** diagnostic or screening tools. Outputs are "
    "probabilities, not clinical assessments.",
    icon=":material/priority_high:",
)

args = get_args()
models = discover_models(args.models_dir)
if not models:
    st.error(
        f"No .pt models found in `{args.models_dir}`. "
        "Pass the folder with:  `streamlit run app.py -- --models_dir path/to/models`"
    )
    st.stop()

# --- threshold buttons ---
thresholds = sorted(models.keys())
st.subheader("1. Choose dichotomization threshold")
choice = st.radio(
    "Threshold", thresholds,
    format_func=lambda t: f"> {t}", horizontal=True, label_visibility="collapsed",
)

pair = models[choice]
have_raw = "raw" in pair
have_den = "denoised" in pair
st.caption(
    f"Threshold > {choice} - "
    f"{'raw ✓' if have_raw else 'raw ✗'} · "
    f"{'denoised ✓' if have_den else 'denoised ✗'}"
)

# --- upload ---
st.subheader("2. Upload audio")
uploaded = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "m4a", "flac", "ogg", "webm"],
)

if uploaded is not None:
    data = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix or ".wav"
    wav = to_wav_bytes(data, suffix)
    if wav:
        st.audio(wav, format="audio/wav")
    else:
        st.audio(data)
        st.caption("Couldn't transcode for playback; predictions are unaffected.")

    if st.button("Run prediction", type="primary"):
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            tmp.write(data); tmp.flush()
            audio_path = tmp.name

            results = {}
            with st.spinner("Embedding and predicting..."):
                # RAW model: embed once (no denoise), predict
                if have_raw:
                    raw_pipe = load_pipeline(pair["raw"])
                    try:
                        emb_raw = raw_pipe.embed_audio(audio_path, denoised=False)
                    except FileNotFoundError as e:
                        if "ffmpeg" in str(e):
                            st.error("ffmpeg not found: `conda install -c conda-forge ffmpeg`")
                            st.stop()
                        raise
                    results["raw"] = raw_pipe.predict_from_embedding(emb_raw)

                # DENOISED model: embed once (with denoise), predict
                if have_den:
                    den_pipe = load_pipeline(pair["denoised"])
                    try:
                        emb_den = den_pipe.embed_audio(audio_path, denoised=True)
                    except ModuleNotFoundError:
                        st.error(
                            "denoiser package not installed. On a login node: "
                            "`pip install denoiser` then pre-cache weights with "
                            "`python -c \"from denoiser import pretrained; pretrained.dns64()\"`"
                        )
                        st.stop()
                    results["denoised"] = den_pipe.predict_from_embedding(emb_den)

        # --- side-by-side display ---
        st.subheader(f"Results — threshold > {choice}")
        symptoms = results[next(iter(results))]["symptoms"]
        symptoms_display = [clean_symptom(s) for s in symptoms]
        table = {"symptom": symptoms_display}
        if "raw" in results:
            table["raw — prob"] = [results["raw"]["probabilities"][s] for s in symptoms]
            table["raw — call"] = [
                "present" if results["raw"]["predictions"][s] else "absent" for s in symptoms
            ]
        if "denoised" in results:
            table["denoised — prob"] = [results["denoised"]["probabilities"][s] for s in symptoms]
            table["denoised — call"] = [
                "present" if results["denoised"]["predictions"][s] else "absent" for s in symptoms
            ]
        df = pd.DataFrame(table)

        prob_cols = [c for c in df.columns if c.endswith("prob")]
        st.dataframe(
            df.style.format({c: "{:.3f}" for c in prob_cols}),
            use_container_width=True, hide_index=True,
        )

        # side-by-side probability bar charts
        cols = st.columns(len(results))
        for col, (kind, res) in zip(cols, results.items()):
            with col:
                st.caption(f"{kind} — predicted probabilities")
                chart_df = pd.DataFrame(
                    {"probability": [res["probabilities"][s] for s in symptoms]},
                    index=symptoms_display,
                )
                st.bar_chart(chart_df)

        st.download_button(
            "Download results as CSV",
            df.to_csv(index=False).encode(),
            file_name=f"prediction_{Path(uploaded.name).stem}_thr{choice}.csv",
            mime="text/csv",
        )

        st.caption(
            "Binary calls use each model's F1-optimal per-symptom thresholds "
            "(from out-of-fold predictions), not a flat 0.5. Raw and denoised "
            "models use their respective matching audio preprocessing."
        )
else:
    st.info("Choose a threshold, then upload an audio file.")
