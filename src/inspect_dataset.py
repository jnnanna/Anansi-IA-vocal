from __future__ import annotations

import argparse

from datasets import Audio

from .dataset_utils import infer_columns, load_hf_dataset, pick_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Hugging Face dataset (columns/splits).")
    parser.add_argument("--dataset", required=True, help="HF dataset id, e.g. KeraCare/Wolof-Kallaama")
    parser.add_argument("--config", default=None, help="Optional dataset config name")
    args = parser.parse_args()

    ds = load_hf_dataset(args.dataset, args.config)
    print(f"Dataset: {args.dataset}")
    print(f"Splits: {list(ds.keys())}")

    train_key, eval_key = pick_splits(ds)
    print(f"Using train split: {train_key}")
    print(f"Using eval split: {eval_key}")

    sample_split = ds[train_key]
    print(f"Rows(train): {len(sample_split)}")
    print(f"Features: {sample_split.features}")

    try:
        cols = infer_columns(sample_split)
        print(f"Inferred audio column: {cols.audio}")
        print(f"Inferred text column: {cols.text}")
        feat = sample_split.features[cols.audio]
        print(f"Audio feature type: {type(feat)}")
        if isinstance(feat, Audio):
            print(f"Audio sampling_rate (declared): {feat.sampling_rate}")
    except Exception as exc:  # noqa: BLE001
        print(f"Column inference failed: {exc}")


if __name__ == "__main__":
    main()
