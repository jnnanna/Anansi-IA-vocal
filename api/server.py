"""
Anansi IA Vocal — FastAPI backend
Endpoints:
  POST /transcribe   : reçoit un fichier audio, renvoie la transcription Whisper
  POST /chat         : reçoit du texte (wolof), renvoie une réponse simple (règles)
  GET  /             : sert la page push-to-talk (index.html)
  GET  /health       : health-check
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Anansi IA Vocal", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global model holder (loaded once at startup)
# ---------------------------------------------------------------------------
_processor = None
_model = None
_device: str = "cpu"

MODEL_DIR = os.environ.get(
    "ANANSI_MODEL_DIR",
    r"C:\models\anansi_wolof_outputs\checkpoint-1200",
)


def _load_model() -> None:
    global _processor, _model, _device

    from transformers import (
        WhisperFeatureExtractor,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        WhisperTokenizer,
    )

    model_source = MODEL_DIR
    if not Path(model_source).is_dir():
        raise RuntimeError(
            f"Model directory not found: {model_source}\n"
            "Set the ANANSI_MODEL_DIR env var to point to your checkpoint folder."
        )

    print(f"[Anansi] Loading model from {model_source} …")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_source, local_files_only=True)
    tokenizer = WhisperTokenizer.from_pretrained(model_source, local_files_only=True)
    _processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = WhisperForConditionalGeneration.from_pretrained(model_source, local_files_only=True)
    _model.to(_device)
    _model.eval()
    print(f"[Anansi] Model loaded on {_device}")


@app.on_event("startup")
async def startup() -> None:
    _load_model()


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _bytes_to_float32(data: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes → mono float32 + sampling rate."""
    # Try soundfile first (WAV, FLAC, OGG)
    try:
        import soundfile as sf
        audio, sr = sf.read(io.BytesIO(data))
        if audio.ndim > 1:
            audio = audio[:, 0]
        return audio.astype(np.float32), int(sr)
    except Exception:
        pass

    # Fallback: write to a temp file and use librosa (needs ffmpeg for mp3/m4a/webm)
    import librosa
    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        return audio.astype(np.float32), int(sr)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _resample_if_needed(audio: np.ndarray, sr: int, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return audio, sr
    import librosa
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32), target_sr


# ---------------------------------------------------------------------------
# Simple rule-based chatbot (wolof)
# ---------------------------------------------------------------------------

_RESPONSES: dict[str, str] = {
    "naka nga def": "Maa ngi fi rekk, jërëjëf. Yow naka?",
    "na nga def": "Maa ngi fi rekk, jërëjëf. Yow naka?",
    "nanga def": "Maa ngi fi rekk, jërëjëf. Yow naka?",
    "jërëjëf": "Amul solo!",
    "jerejef": "Amul solo!",
    "waaw": "Waaw, dégg naa.",
    "déedéet": "OK, baax na.",
    "deedeet": "OK, baax na.",
    "lan la tudd": "Maa tudd Anansi. Yow naka nga tudd?",
    "nan la tudd": "Maa tudd Anansi. Yow naka nga tudd?",
    "asalaamaalekum": "Maalekum salaam!",
    "salamalekum": "Maalekum salaam!",
}


def _generate_response(text: str) -> str:
    """Match input against known patterns; fallback to echo."""
    normalized = text.strip().lower()
    for pattern, reply in _RESPONSES.items():
        if pattern in normalized:
            return reply
    return f"Dégg naa: « {text.strip()} ». Maa ngi ci."


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "device": _device, "model_dir": MODEL_DIR}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Receive an audio file, return Whisper transcription."""
    if _processor is None or _model is None:
        raise HTTPException(503, "Model not loaded yet")

    data = await audio.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty audio file")

    try:
        arr, sr = _bytes_to_float32(data)
        arr, sr = _resample_if_needed(arr, sr)
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio: {e}")

    inputs = _processor(arr, sampling_rate=sr, return_tensors="pt")
    feats = inputs.input_features.to(_device)

    with torch.no_grad():
        pred_ids = _model.generate(feats, max_new_tokens=128)

    text = _processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()

    return {"transcription": text}


@app.post("/chat")
async def chat(text: str = Form(...)):
    """Receive a text string, return a simple rule-based response."""
    reply = _generate_response(text)
    return {"input": text, "response": reply}


@app.post("/voice")
async def voice(audio: UploadFile = File(...)):
    """All-in-one: transcribe audio → generate response → return both."""
    if _processor is None or _model is None:
        raise HTTPException(503, "Model not loaded yet")

    data = await audio.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty audio file")

    try:
        arr, sr = _bytes_to_float32(data)
        arr, sr = _resample_if_needed(arr, sr)
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio: {e}")

    inputs = _processor(arr, sampling_rate=sr, return_tensors="pt")
    feats = inputs.input_features.to(_device)

    with torch.no_grad():
        pred_ids = _model.generate(feats, max_new_tokens=128)

    text = _processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
    reply = _generate_response(text)

    return {"transcription": text, "response": reply}


# ---------------------------------------------------------------------------
# Serve the push-to-talk frontend
# ---------------------------------------------------------------------------
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "web" / "static"


@app.get("/")
async def index():
    index_path = _FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "Frontend index.html not found")
    return FileResponse(str(index_path), media_type="text/html")


# Mount static assets (CSS/JS) after the explicit route so "/" wins
if _FRONTEND_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model_dir", default=None, help="Override ANANSI_MODEL_DIR")
    args = p.parse_args()

    if args.model_dir:
        os.environ["ANANSI_MODEL_DIR"] = args.model_dir
        global MODEL_DIR
        MODEL_DIR = args.model_dir

    uvicorn.run("api.server:app", host=args.host, port=args.port, reload=False)
