"""
Anansi — FastAPI backend for the push-to-talk voice assistant.

Endpoints
---------
GET  /              → serves the push-to-talk HTML page
POST /voice         → mic audio → ASR → LLM response (JSON)
POST /transcribe    → mic audio → ASR only (JSON)
POST /tts           → text → edge-tts MP3 audio
GET  /tts_preview   → voice sample preview (cached MP3)
GET  /voices        → list available TTS voices
GET  /health        → health check
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
import uuid
from collections import defaultdict
from pathlib import Path
from threading import Lock
from typing import Optional, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
# Constants — System prompt, voices, TTS preview samples
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Tu es Anansi, un assistant poli qui parle Wolof par défaut. "
    "Réponds naturellement en Wolof. Si l'utilisateur parle en français, "
    "réponds en français. Réponses courtes (1–2 phrases), empathiques et utiles. "
    "N'écris pas la transcription mot\u2011à\u2011mot à moins que l'utilisateur le demande. "
    "Si tu n'es pas sûr, dis brièvement en Wolof que tu as besoin de précision, "
    "puis propose de chercher des ressources en français. Si tu ne peux pas "
    "répondre, excuse-toi en Wolof et propose des ressources en français."
)

AVAILABLE_VOICES = {
    "fr-FR-HenriNeural": "Henri (Homme)",
    "fr-FR-DeniseNeural": "Denise (Femme)",
    "fr-FR-BrigitteNeural": "Brigitte (Femme)",
    "fr-FR-GuillaumeNeural": "Guillaume (Homme)",
}

TTS_PREVIEW_SAMPLES = {
    "wol": "Nanga def, man maay Anansi. Nan laa mëna jàppale ?",
    "fr": "Bonjour, je suis Anansi. Comment puis\u2011je aider ?",
}

TTS_CACHE_DIR = Path(__file__).resolve().parent / "static" / "tts_cache"


# ---------------------------------------------------------------------------
# Session history (in-memory, keyed by session_id)
# ---------------------------------------------------------------------------

_sessions: dict[str, list] = defaultdict(list)
_session_lock = Lock()
MAX_HISTORY = 10  # keep last N exchanges


# ---------------------------------------------------------------------------
# OpenAI LLM response generator (with rule-based fallback)
# ---------------------------------------------------------------------------

_openai_client = None

_RULE_RESPONSES = {
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


def _generate_response_rules(transcription: str) -> str:
    """Fallback rule-based response when OpenAI is unavailable."""
    text = transcription.strip().lower()
    for key, reply in _RULE_RESPONSES.items():
        if key in text:
            return reply
    return f"Dama la dégg: «{transcription}»"


def _get_openai_client():
    """Lazy-init the OpenAI client (returns None if no API key)."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=api_key)
        return _openai_client
    except Exception as exc:
        print(f"[Anansi] Failed to initialise OpenAI client: {exc}")
        return None


def _generate_response(transcription: str, session_id: str = "default") -> str:
    """Generate a conversational response via OpenAI (fallback to rules)."""
    client = _get_openai_client()
    if client is None:
        return _generate_response_rules(transcription)

    # Build message list: system + history + current user turn
    with _session_lock:
        history = list(_sessions[session_id])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": transcription})

    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.6,
            max_tokens=200,
        )
        reply = resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[Anansi] OpenAI error: {exc}")
        return _generate_response_rules(transcription)

    # Update session history
    with _session_lock:
        _sessions[session_id].append({"role": "user", "content": transcription})
        _sessions[session_id].append({"role": "assistant", "content": reply})
        if len(_sessions[session_id]) > MAX_HISTORY * 2:
            _sessions[session_id] = _sessions[session_id][-MAX_HISTORY * 2:]

    return reply


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


# ---------------------------------------------------------------------------
# TTS endpoints (edge-tts)
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    text: str
    voice: str = "fr-FR-HenriNeural"


@app.get("/voices")
async def voices():
    """List available TTS voices."""
    return JSONResponse({
        "voices": AVAILABLE_VOICES,
        "default": "fr-FR-HenriNeural",
    })


@app.get("/tts_preview")
async def tts_preview(
    voice: str = Query("fr-FR-HenriNeural"),
    lang: str = Query("wol"),
):
    """Return a cached MP3 preview of a voice reading a sample sentence."""
    if voice not in AVAILABLE_VOICES:
        raise HTTPException(400, f"Unknown voice: {voice}")
    if lang not in TTS_PREVIEW_SAMPLES:
        raise HTTPException(400, f"Unknown lang (use 'wol' or 'fr'): {lang}")

    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = voice.replace("/", "_").replace("\\", "_")
    cache_path = TTS_CACHE_DIR / f"{safe_name}_{lang}.mp3"

    if not cache_path.is_file():
        try:
            import edge_tts
            communicate = edge_tts.Communicate(TTS_PREVIEW_SAMPLES[lang], voice)
            await communicate.save(str(cache_path))
        except Exception as exc:
            raise HTTPException(503, f"edge-tts synthesis failed: {exc}")

    return FileResponse(str(cache_path), media_type="audio/mpeg")


@app.post("/tts")
async def tts_synth(req: TTSRequest):
    """Synthesise text to MP3 via edge-tts and return the audio."""
    if req.voice not in AVAILABLE_VOICES:
        raise HTTPException(400, f"Unknown voice: {req.voice}")
    if not req.text.strip():
        raise HTTPException(400, "Text is empty")

    try:
        import edge_tts
        communicate = edge_tts.Communicate(req.text, req.voice)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        audio_data = b"".join(audio_chunks)
    except Exception as exc:
        raise HTTPException(503, f"edge-tts synthesis failed: {exc}")

    return Response(content=audio_data, media_type="audio/mpeg")


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
async def voice(
    audio: UploadFile = File(...),
    session_id: str = Form("default"),
):
    """Transcribe audio + generate a conversational response."""
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
    response = _generate_response(transcription, session_id=session_id)

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
        help="Path to the fine-tuned Whisper model directory.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    parser.add_argument(
        "--openai_model",
        default=None,
        help="Override OpenAI model name (default: env OPENAI_MODEL or gpt-4o-mini).",
    )
    args = parser.parse_args()

    # Load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("[Anansi] .env loaded")
    except ImportError:
        pass

    if args.openai_model:
        os.environ["OPENAI_MODEL"] = args.openai_model

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"[Anansi] OpenAI key found (…{api_key[-4:]}), LLM brain enabled")
    else:
        print("[Anansi] No OPENAI_API_KEY — using rule-based fallback responses")

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
