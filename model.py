"""
model.py
========
Network definition and the self-contained inference pipeline.

The inference pipeline takes a raw audio file -> Whisper `base` encoder ->
pooling -> (saved) scaler -> MLP -> 9 symptom probabilities + binary calls.

IMPORTANT — POOLING MUST MATCH TRAINING
---------------------------------------
The training CSV embeddings were already mean-pooled upstream (in the feature-
extraction step). The inference path below re-creates that pooling from raw
audio. If the two disagree, inference will silently drift from training.

The default here is MEAN-POOLING over the Whisper encoder's time frames. 
If the upstream pipeline used something else (max-pool, last hidden state, attention pooling), 
change ONLY the function `pool_encoder_states` below and everything downstream stays consistent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
#  POOLING — single source of truth. Edit here if upstream pooling differs.
# --------------------------------------------------------------------------- #
def pool_encoder_states(encoder_states: torch.Tensor) -> torch.Tensor:
    """Pool (n_frames, n_dim) Whisper encoder output to a single (n_dim,) vector.

    Default: mean over the time/frame axis.

    encoder_states: tensor of shape (n_frames, n_dim) for a single recording.
    returns:        tensor of shape (n_dim,)
    """
    return encoder_states.mean(dim=0)


# --------------------------------------------------------------------------- #
#  Denoiser (dns64) — matches the training-time denoising for denoised models.
#
#  TRAINING CHAIN for denoised models (confirmed):
#      load_audio (16k mono) -> pad/trim to 30s @ 16k -> denoise(dns64)
#      -> log_mel -> encoder -> mean-pool
#  The denoise step happens AFTER pad/trim, so that order is replicated in
#  embed_audio below. The model is a module-level singleton so it loads once.
#
#  NOTE: dns64 weights download from the internet on first use. On an offline
#  compute node, pre-cache them on a login node first:
#      python -c "from denoiser import pretrained; pretrained.dns64()"
# --------------------------------------------------------------------------- #
SR = 16000  # sample rate used at training when denoising

_denoiser_model = None  # singleton


def get_denoiser():
    global _denoiser_model
    if _denoiser_model is None:
        from denoiser import pretrained

        _denoiser_model = pretrained.dns64()
        _denoiser_model.eval()
    return _denoiser_model


def denoise_waveform(waveform: np.ndarray, sr: int = SR) -> np.ndarray:
    """Apply dns64 denoising to a 1-D waveform, returning a 1-D float32 array."""
    from denoiser.dsp import convert_audio

    model = get_denoiser()
    wav = torch.from_numpy(waveform).unsqueeze(0).unsqueeze(0)  # (1, 1, T)
    wav = convert_audio(wav, sr, model.sample_rate, model.chin)
    with torch.no_grad():
        out = model(wav)[0].squeeze().numpy()
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Network
# --------------------------------------------------------------------------- #
class SymptomMLP(nn.Module):
    """Feed-forward ReLU net for multi-label (9 symptom) prediction.

    Outputs raw logits (no sigmoid) so I can use BCEWithLogitsLoss during
    training for numerical stability. Apply sigmoid at inference for probabilities.
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dims: tuple[int, ...] = (128, 64),
        n_outputs: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dims = tuple(hidden_dims)
        self.n_outputs = n_outputs
        self.dropout = dropout

        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits

    # -- config (round-trips through JSON so to rebuild for inference) --
    def config(self) -> dict:
        return {
            "in_dim": self.in_dim,
            "hidden_dims": list(self.hidden_dims),
            "n_outputs": self.n_outputs,
            "dropout": self.dropout,
        }


# --------------------------------------------------------------------------- #
#  Inference pipeline — bundles standardization + net + (optional) Whisper
# --------------------------------------------------------------------------- #
class InferencePipeline:
    """Self-contained: standardize -> MLP -> probabilities -> binary calls.

    Two entry points:
      * predict_from_embedding(vec)  -> for a pre-pooled 512-d vector
      * predict_from_audio(path)     -> raw audio file, runs Whisper encoder

    Whisper is imported lazily so this object also works on machines where only
    embeddings are used.
    """

    def __init__(
        self,
        model: SymptomMLP,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        symptom_names: list[str],
        thresholds: np.ndarray | None = None,
        whisper_model_name: str = "base",
        denoised: bool = False,
    ):
        self.model = model.eval()
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
        self.symptom_names = list(symptom_names)
        # default decision threshold 0.5 per symptom unless overridden
        if thresholds is None:
            thresholds = np.full(len(symptom_names), 0.5, dtype=np.float64)
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        self.whisper_model_name = whisper_model_name
        # whether this model was trained on DENOISED audio; binds preprocessing
        self.denoised = bool(denoised)
        self._whisper = None  # lazy

    # ---- embedding path ---------------------------------------------------
    def predict_from_embedding(self, embedding: np.ndarray) -> dict:
        emb = np.asarray(embedding, dtype=np.float64).reshape(1, -1)
        if emb.shape[1] != self.scaler_mean.shape[0]:
            raise ValueError(
                f"Embedding dim {emb.shape[1]} != expected {self.scaler_mean.shape[0]}"
            )
        scaled = (emb - self.scaler_mean) / self.scaler_scale
        with torch.no_grad():
            logits = self.model(torch.tensor(scaled, dtype=torch.float32))
            probs = torch.sigmoid(logits).numpy().ravel()
        calls = (probs >= self.thresholds).astype(int)
        return {
            "symptoms": self.symptom_names,
            "probabilities": {n: float(p) for n, p in zip(self.symptom_names, probs)},
            "predictions": {n: int(c) for n, c in zip(self.symptom_names, calls)},
        }

    # ---- audio path -------------------------------------------------------
    def _load_whisper(self):
        if self._whisper is None:
            import whisper  # openai-whisper

            self._whisper = whisper.load_model(self.whisper_model_name)
        return self._whisper

    def embed_audio(self, audio_path: str | Path, denoised: bool | None = None) -> np.ndarray:
        """Audio file -> pooled 512-d embedding, matching training preprocessing.

        Chain: load_audio (16k mono) -> pad/trim 30s -> [denoise if denoised]
               -> log_mel -> encoder -> mean-pool.

        denoised: override the model's own flag (used by the app to compute a
                  raw and a denoised embedding from one file). Defaults to the
                  model's bound setting so a model never gets the wrong input.
        """
        import whisper

        use_denoise = self.denoised if denoised is None else denoised
        wmodel = self._load_whisper()
        audio = whisper.load_audio(str(audio_path))   # 16k mono float32
        audio = whisper.pad_or_trim(audio)            # 30s @ 16k  (BEFORE denoise)
        if use_denoise:
            audio = denoise_waveform(audio, sr=SR)    # dns64, matches training
            audio = whisper.pad_or_trim(audio)        # denoiser can change length
        mel = whisper.log_mel_spectrogram(audio).to(wmodel.device)
        with torch.no_grad():
            enc = wmodel.encoder(mel.unsqueeze(0))     # (1, n_frames, n_dim)
            pooled = pool_encoder_states(enc.squeeze(0))
        return pooled.cpu().numpy()

    def predict_from_audio(self, audio_path: str | Path) -> dict:
        emb = self.embed_audio(audio_path)
        out = self.predict_from_embedding(emb)
        out["audio_path"] = str(audio_path)
        return out

    # ---- persistence ------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_config": self.model.config(),
                "scaler_mean": self.scaler_mean,
                "scaler_scale": self.scaler_scale,
                "symptom_names": self.symptom_names,
                "thresholds": self.thresholds,
                "whisper_model_name": self.whisper_model_name,
                "denoised": self.denoised,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> "InferencePipeline":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = SymptomMLP(**ckpt["model_config"])
        model.load_state_dict(ckpt["model_state"])
        return cls(
            model=model,
            scaler_mean=ckpt["scaler_mean"],
            scaler_scale=ckpt["scaler_scale"],
            symptom_names=ckpt["symptom_names"],
            thresholds=ckpt.get("thresholds"),
            whisper_model_name=ckpt.get("whisper_model_name", "base"),
            denoised=ckpt.get("denoised", False),
        )
