from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import evaluate
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .dataset_utils import ensure_sampling_rate, infer_columns, load_hf_dataset, pick_splits, safe_get_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute WER for a Whisper checkpoint on a HF dataset.")
    p.add_argument("--model_dir", required=True, help="Path to fine-tuned model dir (contains config + processor).")
    p.add_argument("--dataset", default="KeraCare/Wolof-Kallaama")
    p.add_argument("--config", default=None)
    p.add_argument("--audio_col", default=None)
    p.add_argument("--text_col", default=None)
    p.add_argument("--sampling_rate", type=int, default=16000)
    p.add_argument("--split", default=None, help="Force eval split name (default: validation/test fallback).")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=-1)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    # If a local path is provided, ensure it exists before calling HF Hub logic.
    # When the directory doesn't exist (e.g., training crashed), older hub/transformers
    # versions may try to validate it as a repo id and raise a confusing error.
    is_local_dir = os.path.isdir(args.model_dir)
    if not is_local_dir:
        raise FileNotFoundError(
            "Model directory not found. Make sure training (Cellule 8) finished and saved files to --model_dir. "
            f"Got: {args.model_dir}"
        )

    required_files = ["config.json", "preprocessor_config.json"]
    missing = [f for f in required_files if not os.path.exists(os.path.join(args.model_dir, f))]
    if missing:
        raise FileNotFoundError(
            "Model directory exists but is missing required files. Training may have crashed before saving. "
            f"Missing: {missing} in {args.model_dir}"
        )

    processor = WhisperProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    ds_dict = load_hf_dataset(args.dataset, args.config)
    train_split, default_eval_split = pick_splits(ds_dict)
    eval_split = args.split or default_eval_split

    inferred = infer_columns(ds_dict[train_split])
    audio_col = args.audio_col or inferred.audio
    text_col = args.text_col or inferred.text

    ds_dict = ensure_sampling_rate(ds_dict, audio_col=audio_col, sampling_rate=args.sampling_rate)
    ds_eval = ds_dict[eval_split]

    if args.max_samples > 0:
        ds_eval = ds_eval.select(range(min(args.max_samples, len(ds_eval))))

    wer_metric = evaluate.load("wer")

    preds: List[str] = []
    refs: List[str] = []

    for start in tqdm(range(0, len(ds_eval), args.batch_size), desc="Evaluating"):
        batch = ds_eval[start : start + args.batch_size]

        audio_arrays = [ex[audio_col]["array"] for ex in batch]
        srs = [ex[audio_col]["sampling_rate"] for ex in batch]
        if len(set(srs)) != 1:
            raise ValueError(f"Mixed sampling rates in batch: {set(srs)}")

        inputs = processor(audio_arrays, sampling_rate=srs[0], return_tensors="pt")
        input_features = inputs.input_features.to(device)

        predicted_ids = model.generate(input_features)
        pred_text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

        ref_text = [safe_get_text(ex, text_col) for ex in batch]

        preds.extend([t.strip() for t in pred_text])
        refs.extend([t.strip() for t in ref_text])

    wer = wer_metric.compute(predictions=preds, references=refs)
    print({"wer": float(wer), "samples": len(refs), "split": eval_split, "device": device})


if __name__ == "__main__":
    main()
