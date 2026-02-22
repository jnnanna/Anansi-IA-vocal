from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import datasets


@dataclass(frozen=True)
class DatasetColumns:
    audio: str
    text: str


def _find_audio_column(features: datasets.Features) -> Optional[str]:
    for name, feat in features.items():
        if isinstance(feat, datasets.Audio):
            return name
    for candidate in ("audio", "speech", "wav", "file"):
        if candidate in features:
            if isinstance(features[candidate], datasets.Audio):
                return candidate
    return None


def _find_text_column(features: datasets.Features) -> Optional[str]:
    for candidate in (
        "text",
        "sentence",
        "transcription",
        "transcript",
        "normalized_text",
        "target",
        "label",
    ):
        if candidate in features:
            return candidate
    return None


def infer_columns(dataset: datasets.Dataset) -> DatasetColumns:
    features = dataset.features
    audio_col = _find_audio_column(features)
    text_col = _find_text_column(features)

    if audio_col is None or text_col is None:
        raise ValueError(
            "Impossible d’inférer les colonnes audio/texte. "
            f"Colonnes trouvées: {list(features.keys())}. "
            "Passe `--audio_col` / `--text_col` explicitement."
        )

    return DatasetColumns(audio=audio_col, text=text_col)


def load_hf_dataset(dataset_id: str, dataset_config: Optional[str] = None) -> datasets.DatasetDict:
    if dataset_config:
        return datasets.load_dataset(dataset_id, dataset_config)
    return datasets.load_dataset(dataset_id)


def pick_splits(ds: datasets.DatasetDict) -> tuple[str, str]:
    keys = list(ds.keys())
    train_key = "train" if "train" in ds else keys[0]
    eval_key = "validation" if "validation" in ds else ("test" if "test" in ds else train_key)
    return train_key, eval_key


def ensure_sampling_rate(
    ds: datasets.DatasetDict,
    audio_col: str,
    sampling_rate: int = 16000,
) -> datasets.DatasetDict:
    return ds.cast_column(audio_col, datasets.Audio(sampling_rate=sampling_rate))


def safe_get_text(example: dict[str, Any], text_col: str) -> str:
    value = example.get(text_col)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)
