from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import streamlit as st
import torch


def _load_audio_bytes_to_mono_wav(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode uploaded audio bytes into mono float32 array + sampling rate.

    Prefers soundfile (WAV/FLAC/OGG), falls back to librosa.
    """

    # Try soundfile first
    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(io.BytesIO(data))
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
        return audio, int(sr)
    except Exception:
        pass

    # Fallback: write temp file and let librosa decode (may require ffmpeg for mp3)
    import librosa  # type: ignore

    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        audio = audio.astype(np.float32)
        return audio, int(sr)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _resample(audio: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    if sr == target_sr:
        return audio, sr
    import librosa  # type: ignore

    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32), target_sr


@st.cache_resource(show_spinner=False)
def _load_model(model_source: str, local_only: bool, device: str):
    # We intentionally build the processor from the *slow* Whisper tokenizer.
    # This avoids some Windows environments failing to instantiate the "fast" tokenizer backend.
    from transformers import (
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )

    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_source, local_files_only=local_only)
        tokenizer = WhisperTokenizer.from_pretrained(model_source, local_files_only=local_only)
        processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    except Exception as e:
        raise RuntimeError(
            "Impossible de charger le processor/tokenizer depuis ce dossier.\n"
            "Vérifie que tu pointes vers le BON dossier (celui qui contient config.json, model.safetensors, tokenizer.json...).\n\n"
            f"Dossier: {model_source}\n"
            f"Erreur: {type(e).__name__}: {e}"
        ) from e

    model = WhisperForConditionalGeneration.from_pretrained(model_source, local_files_only=local_only)
    model.to(device)
    model.eval()
    return processor, model


def _pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    st.set_page_config(page_title="Anansi Wolof ASR Demo", layout="centered")

    st.title("Anansi — Démo ASR (Whisper)")
    st.write("Upload un fichier audio (WAV recommandé), puis lance la transcription.")

    device = _pick_device()
    st.caption(f"Device: {device}")

    with st.sidebar:
        st.header("Modèle")
        model_dir = st.text_input(
            "Chemin local du modèle (optionnel)",
            value="",
            help="Ex: outputs/whisper-small-wo ou un dossier téléchargé depuis Google Drive.",
        )
        hf_model = st.text_input("HF model id (fallback)", value="openai/whisper-small")
        task = st.selectbox("Task", options=["transcribe", "translate"], index=0)
        max_new_tokens = st.slider("max_new_tokens", min_value=32, max_value=256, value=128, step=16)

    uploaded = st.file_uploader("Audio", type=["wav", "flac", "ogg", "mp3", "m4a"]) 

    if uploaded is None:
        st.info("Choisis un fichier audio pour commencer.")
        return

    audio_bytes = uploaded.getvalue()

    if st.button("Transcrire"):
        with st.spinner("Chargement audio…"):
            audio, sr = _load_audio_bytes_to_mono_wav(audio_bytes)
            audio, sr = _resample(audio, sr, target_sr=16000)

        model_source = hf_model.strip() or "openai/whisper-small"
        local_only = False

        if model_dir.strip():
            p = Path(model_dir.strip())
            if not p.is_dir():
                st.error(f"Dossier modèle introuvable: {p}")
                return
            model_source = str(p)
            local_only = True

        with st.spinner("Chargement modèle…"):
            processor, model = _load_model(model_source=model_source, local_only=local_only, device=device)

        with st.spinner("Génération…"):
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            feats = inputs.input_features.to(device)
            with torch.no_grad():
                pred_ids = model.generate(feats, task=task, max_new_tokens=max_new_tokens)
            text = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()

        st.subheader("Transcription")
        st.text_area("", value=text, height=160)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
