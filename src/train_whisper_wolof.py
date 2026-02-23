from __future__ import annotations

import argparse
import os
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from .dataset_utils import ensure_sampling_rate, infer_columns, load_hf_dataset, pick_splits, safe_get_text


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If BOS is present, drop it (Whisper will add it)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on a Wolof HF dataset.")

    parser.add_argument("--dataset", default="KeraCare/Wolof-Kallaama")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument(
        "--language",
        default="",
        help="Optional Whisper language name for decoder prompt (e.g. 'english'). Leave empty to disable.",
    )
    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"])

    parser.add_argument("--audio_col", default=None)
    parser.add_argument("--text_col", default=None)
    parser.add_argument("--sampling_rate", type=int, default=16000)

    # Whisper has a hard limit on decoder labels length (typically 448 tokens).
    # If a transcript tokenizes longer than this, training crashes.
    parser.add_argument(
        "--max_label_length",
        type=int,
        default=448,
        help="Max target token length for labels (Whisper limit is usually 448).",
    )

    parser.add_argument("--output_dir", default="outputs/whisper-small-wo")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--max_steps", type=int, default=-1, help="Override epochs if > 0")

    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_proc", type=int, default=max(os.cpu_count() or 2, 2))
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import transformers  # type: ignore

        print("transformers:", getattr(transformers, "__version__", "unknown"))
    except Exception:
        pass

    ds_dict: DatasetDict = load_hf_dataset(args.dataset, args.config)
    train_split, eval_split = pick_splits(ds_dict)

    # Infer columns from train split unless explicitly given
    inferred = infer_columns(ds_dict[train_split])
    audio_col = args.audio_col or inferred.audio
    text_col = args.text_col or inferred.text

    ds_dict = ensure_sampling_rate(ds_dict, audio_col=audio_col, sampling_rate=args.sampling_rate)

    if args.max_train_samples > 0:
        ds_dict[train_split] = ds_dict[train_split].select(range(min(args.max_train_samples, len(ds_dict[train_split]))))
    if args.max_eval_samples > 0:
        ds_dict[eval_split] = ds_dict[eval_split].select(range(min(args.max_eval_samples, len(ds_dict[eval_split]))))

    # Load processor/model without forcing a language at load time because
    # some local/low-resource languages (e.g. "wolof") are not in the
    # tokenizer's supported-language list and will raise a ValueError.
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    # ── Generation config (transformers 5.x compatible) ──────────────
    # In transformers ≥5.0, generation-related settings MUST live on
    # `model.generation_config`, NOT on `model.config`.  Setting them on
    # model.config triggers:
    #   ValueError: You have modified the pretrained model configuration
    #   to control generation … This strategy is not supported anymore.
    #
    # We also clean up model.config to make sure no stale generation
    # keys remain there from the pretrained checkpoint.
    gen_cfg = model.generation_config

    # forced_decoder_ids — language/task prompt
    if args.language:
        try:
            gen_cfg.forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=args.language, task=args.task
            )
        except ValueError:
            print(
                f"Warning: language '{args.language}' not supported by the "
                "tokenizer — continuing without forced decoder ids."
            )
            gen_cfg.forced_decoder_ids = None
    else:
        gen_cfg.forced_decoder_ids = None

    gen_cfg.suppress_tokens = []
    gen_cfg.max_length = 448  # coherent with max_label_length

    # Remove generation keys from model.config so they don't conflict
    for key in ("forced_decoder_ids", "suppress_tokens"):
        if hasattr(model.config, key):
            delattr(model.config, key)

    def prepare_example(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = batch[audio_col]
        array = audio["array"]
        sr = audio["sampling_rate"]
        inputs = processor.feature_extractor(array, sampling_rate=sr)
        batch["input_features"] = inputs["input_features"][0]

        text = safe_get_text(batch, text_col)
        batch["labels"] = processor.tokenizer(
            text,
            truncation=True,
            max_length=args.max_label_length,
        ).input_ids
        return batch

    ds_train = ds_dict[train_split].map(
        prepare_example,
        remove_columns=ds_dict[train_split].column_names,
        num_proc=args.num_proc,
        desc="Preparing train set",
    )
    ds_eval = ds_dict[eval_split].map(
        prepare_example,
        remove_columns=ds_dict[eval_split].column_names,
        num_proc=args.num_proc,
        desc="Preparing eval set",
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred) -> Dict[str, float]:  # type: ignore[no-untyped-def]
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        # Replace -100 to pad_token_id so we can decode
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": float(wer)}

    # ── Build Seq2SeqTrainingArguments (version-tolerant) ──────────
    # We inspect the constructor signature and only pass supported kwargs,
    # because kwarg names change across transformers versions.
    sig_params = set(inspect.signature(Seq2SeqTrainingArguments.__init__).parameters.keys())

    kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "report_to": [],
        "fp16": args.fp16,
        "bf16": args.bf16,
        "seed": args.seed,
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
    }

    # Evaluation strategy key name changed across versions
    if "eval_strategy" in sig_params:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in sig_params:
        kwargs["evaluation_strategy"] = "steps"
    if "eval_steps" in sig_params:
        kwargs["eval_steps"] = args.eval_steps

    # predict_with_generate (seq2seq specific)
    if "predict_with_generate" in sig_params:
        kwargs["predict_with_generate"] = True
    # generation_max_length was removed in transformers 5.x
    # (generation config is on the model instead)
    if "generation_max_length" in sig_params:
        kwargs["generation_max_length"] = 448

    # Filter to only supported kwargs
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig_params}
    training_args = Seq2SeqTrainingArguments(**filtered_kwargs)

    # ── Build Seq2SeqTrainer (version-tolerant) ─────────────────────
    trainer_sig = set(inspect.signature(Seq2SeqTrainer.__init__).parameters.keys())

    trainer_kwargs: Dict[str, Any] = {
        "args": training_args,
        "model": model,
        "train_dataset": ds_train,
        "eval_dataset": ds_eval,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    # `tokenizer` kwarg was renamed/removed in some versions
    if "tokenizer" in trainer_sig:
        trainer_kwargs["tokenizer"] = processor.tokenizer
    elif "processing_class" in trainer_sig:
        trainer_kwargs["processing_class"] = processor

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train()

    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    metrics = trainer.evaluate()
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
