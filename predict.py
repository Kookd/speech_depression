"""
predict.py
==========
Run the saved inference pipeline on an audio file (or a pre-pooled embedding)
and print the 9 symptom probabilities + binary calls.

Examples
--------
# from a raw audio file (requires openai-whisper installed)
python predict.py --model results/final_inference_model.pt --audio diary.wav

# from a pre-pooled 512-d embedding stored as a single-row CSV (no header)
python predict.py --model results/final_inference_model.pt --embedding_csv vec.csv
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from model import InferencePipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path to final_inference_model.pt")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--audio", help="path to an audio file (.wav/.mp3/...)")
    g.add_argument("--embedding_csv", help="single-row CSV of 512 values, no header")
    args = p.parse_args()

    pipe = InferencePipeline.load(args.model)

    if args.audio:
        result = pipe.predict_from_audio(args.audio)
    else:
        vec = np.loadtxt(args.embedding_csv, delimiter=",")
        result = pipe.predict_from_embedding(vec)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
