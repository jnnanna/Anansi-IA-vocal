"""
Anansi — FastAPI backend for the push-to-talk voice assistant.

Endpoints
---------
GET  /           → serves the push-to-talk HTML page (web/static/index.html)
POST /voice      → receives audio file, returns JSON { transcription, response }
POST /transcribe → receives audio file, returns JSON { transcription }
GET  /health     → simple health check
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _ffmpeg_to_wav(input_path: str, output_path: str) -> bool:
    """Convert any audio format to WAV 16kHz mono using ffmpeg subprocess."""
    import shutil
    import subprocess

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        return False

    try:
        subprocess.run(
            [
                ffmpeg_bin, "-y",
                "-i", input_path,
                "-ac", "1",           # mono
                "-ar", "16000",       # 16 kHz
                "-f", "wav",
                "-sample_fmt", "s16", # PCM 16-bit
                output_path,
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _bytes_to_mono_16k(data: bytes) -> Tuple[np.ndarray, int]:
    """Decode raw audio bytes into mono float32 @ 16 kHz.

    Strategy:
    1. Try ffmpeg (handles webm, mp3, ogg, m4a, wav, etc.)
    2. Fall back to soundfile (WAV / FLAC)
    3. Fall back to librosa + audioread
    """

    # ── Strategy 1: ffmpeg (most reliable for browser WebM) ──
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
        tmp_in.write(data)
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path + ".wav"

    try:
        if _ffmpeg_to_wav(tmp_in_path, tmp_wav_path):
            import soundfile as sf
            audio, sr = sf.read(tmp_wav_path)
            if audio.ndim > 1:
                audio = audio[:, 0]
            return audio.astype(np.float32), int(sr)
    finally:
        for p in (tmp_in_path, tmp_wav_path):
            try:
                os.remove(p)
            except OSError:
                pass

    # ── Strategy 2: soundfile (WAV / FLAC in-memory) ──
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(data))
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
    except Exception:
        # ── Strategy 3: librosa / audioread ──
        import librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            audio, sr = librosa.load(tmp_path, sr=None, mono=True)
            audio = audio.astype(np.float32)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if sr != 16_000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000).astype(np.float32)
        sr = 16_000

    return audio, sr


# ---------------------------------------------------------------------------
# Simple rule-based response generator (placeholder for future LLM / NLP)
# ---------------------------------------------------------------------------

_RESPONSES = {
    "naka nga def": "Maa ngi fi rekk, jërëjëf! Na nga def?",
    "na nga def": "Maa ngi fi rekk, jërëjëf! Na nga def?",
    "jërëjëf": "Baal ma, am na la njàng!",
    "jerejef": "Baal ma, am na la njàng!",
    "waaw": "Waaw, dégg naa la.",
    "déedéet": "Baax na, dama la dégg.",
    "nan ga def": "Maa ngi fi rekk, jërëjëf!",
    "salam aleykum": "Aleykum salam!",
    "salaam aleekum": "Aleykum salam!",
}

def _generate_response(transcription: str) -> str:
    """Return a simple rule-based response.  Replace with LLM / NLU later."""
    text = transcription.strip().lower()
    for key, reply in _RESPONSES.items():
        if key in text:
            return reply
    return f"Dama la dégg: «{transcription}»"


# ---------------------------------------------------------------------------
# Model loading (singleton – loaded once at startup)
# ---------------------------------------------------------------------------

_processor = None
_model = None
_device = "cpu"


def _load_model(model_dir: str) -> None:
    global _processor, _model, _device

    from transformers import (
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )

    local_only = os.path.isdir(model_dir)
    source = model_dir if local_only else model_dir

    feature_extractor = WhisperFeatureExtractor.from_pretrained(source, local_files_only=local_only)
    tokenizer = WhisperTokenizer.from_pretrained(source, local_files_only=local_only)
    _processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = WhisperForConditionalGeneration.from_pretrained(source, local_files_only=local_only)
    _model.to(_device)
    _model.eval()
    print(f"[Anansi] Model loaded from {source} on {_device}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Anansi Voice API", version="0.1.0")

# Serve static files (index.html, etc.)
_static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/")
async def index():
    """Serve the push-to-talk page."""
    html = _static_dir / "index.html"
    if not html.is_file():
        raise HTTPException(500, "index.html not found in web/static/")
    return FileResponse(str(html), media_type="text/html")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None, "device": _device}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe uploaded audio and return text only."""
    if _model is None or _processor is None:
        raise HTTPException(503, "Model not loaded yet")

    raw = await audio.read()
    if len(raw) < 100:
        raise HTTPException(400, "Audio file too small or empty")

    try:
        arr, sr = _bytes_to_mono_16k(raw)
    except Exception as e:
        raise HTTPException(400, f"Cannot decode audio: {e}")

    inputs = _processor(arr, sampling_rate=sr, return_tensors="pt")
    feats = inputs.input_features.to(_device)

    with torch.no_grad():
        pred_ids = _model.generate(feats, max_new_tokens=128)

    text = _processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
    return JSONResponse({"transcription": text})


@app.post("/voice")
async def voice(audio: UploadFile = File(...)):
    """Transcribe audio + generate a response (push-to-talk endpoint)."""
    if _model is None or _processor is None:
        raise HTTPException(503, "Model not loaded yet")

    raw = await audio.read()
    if len(raw) < 100:
        raise HTTPException(400, "Audio file too small or empty")

    try:
        arr, sr = _bytes_to_mono_16k(raw)
    except Exception as e:
        raise HTTPException(400, f"Cannot decode audio: {e}")

    inputs = _processor(arr, sampling_rate=sr, return_tensors="pt")
    feats = inputs.input_features.to(_device)

    with torch.no_grad():
        pred_ids = _model.generate(feats, max_new_tokens=128)

    transcription = _processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
    response = _generate_response(transcription)

    return JSONResponse({
        "transcription": transcription,
        "response": response,
    })


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Anansi Voice API server")
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the fine-tuned Whisper model directory (contains config.json + model.safetensors + tokenizer.json).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    _load_model(args.model_dir)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
