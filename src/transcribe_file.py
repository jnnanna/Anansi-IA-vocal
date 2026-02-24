from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


def _load_audio_mono(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio as mono float32, resampling to target_sr.

    Supports WAV/FLAC via soundfile; falls back to librosa for broader formats.
    """

    audio_file = str(audio_path)

    try:
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(audio_file)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = audio.astype(np.float32)
    except Exception:
        import librosa  # type: ignore

        audio, sr = librosa.load(audio_file, sr=None, mono=True)
        audio = audio.astype(np.float32)

    if sr != target_sr:
        import librosa  # type: ignore

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr


def _select_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe one audio file with Whisper (fine-tuned or base).")
    p.add_argument("--model_dir", default=None, help="Local model dir (OUTPUT_DIR) containing config + processor.")
    p.add_argument("--model", default="openai/whisper-small", help="HF model id fallback if --model_dir not provided.")
    p.add_argument("--audio", required=True, help="Path to audio file (.wav recommended).")
    p.add_argument("--device", default=None, help="Force device: 'cpu' or 'cuda'. Default: auto.")
    p.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])
    p.add_argument("--max_new_tokens", type=int, default=128)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    from transformers import WhisperForConditionalGeneration, WhisperProcessor  # lazy import

    model_source = args.model
    local_only = False

    if args.model_dir:
        model_path = Path(args.model_dir)
        if not model_path.is_dir():
            raise FileNotFoundError(f"--model_dir not found: {model_path}")
        model_source = str(model_path)
        local_only = True

    device = _select_device(args.device)

    processor = WhisperProcessor.from_pretrained(model_source, local_files_only=local_only)
    model = WhisperForConditionalGeneration.from_pretrained(model_source, local_files_only=local_only)
    model.to(device)
    model.eval()

    audio, sr = _load_audio_mono(args.audio, target_sr=16000)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    predicted_ids = model.generate(
        input_features,
        task=args.task,
        max_new_tokens=args.max_new_tokens,
    )
    text = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

    print(text.strip())


if __name__ == "__main__":
    # Avoid tokenizers parallelism warning noise on Windows.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
