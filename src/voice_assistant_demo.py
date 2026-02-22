from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import torch
import pyttsx3
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple voice assistant demo: mic → Whisper ASR → response → TTS.")
    p.add_argument("--model_dir", default="openai/whisper-small", help="HF model id or local fine-tuned dir")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--device", default=None, help="Force torch device: cuda/cpu")
    p.add_argument("--no_tts", action="store_true")
    return p.parse_args()


def record_audio(seconds: float, sample_rate: int) -> np.ndarray:
    frames = int(seconds * sample_rate)
    print(f"Recording {seconds:.1f}s… speak now.")
    audio = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return audio.squeeze(-1)


def wolof_rules(user_text: str) -> str:
    t = user_text.lower().strip()

    greetings = ["asalaamaalekum", "assalamu", "salaam", "salam"]
    if any(g in t for g in greetings):
        return "Maalekum salaam! Naka nga def?"

    if "naka" in t and ("def" in t or "nga" in t):
        return "Mangi fi rekk. Yow naka nga def?"

    if t == "":
        return "Damaa la déggul bu baax. Jëfandikoo baat bu gëna leer."

    return f"Damaa dégg: {user_text}. Waxal ma lu gëna bari."  # placeholder NLP


def tts_speak(text: str, engine: Optional[pyttsx3.Engine] = None) -> None:
    eng = engine or pyttsx3.init()
    eng.say(text)
    eng.runAndWait()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    engine = None if args.no_tts else pyttsx3.init()

    while True:
        try:
            audio = record_audio(args.seconds, args.sample_rate)
            inputs = processor(audio, sampling_rate=args.sample_rate, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            started = time.time()
            predicted_ids = model.generate(input_features)
            text = processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True).strip()
            dt = time.time() - started

            print(f"ASR ({dt:.2f}s): {text}")
            reply = wolof_rules(text)
            print(f"Assistant: {reply}")

            if not args.no_tts:
                tts_speak(reply, engine=engine)

            print("\nCtrl+C pour arrêter.\n")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
