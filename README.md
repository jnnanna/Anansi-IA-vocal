# Anansi IA vocal (Wolof) — Whisper fine-tuning + démo micro

Objectif (MVP):
- ASR Wolof via fine-tuning de `openai/whisper-small` sur le dataset Hugging Face `KeraCare/Wolof-Kallaama`
- Évaluation WER
- Démo: micro → ASR → réponse (règles simples) → TTS (pyttsx3)

## 0) Pré-requis
- Python 3.10+ (3.11 OK)
- Pour entraîner: idéalement un GPU (Colab recommandé)

## 1) Installation (local Windows)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

> Note: `torch` n’est pas dans `requirements.txt` car l’installation dépend (CPU/GPU). Sur Colab, torch est déjà présent.

## 2) Inspecter le dataset (colonnes/splits)
```powershell
python -m src.inspect_dataset --dataset KeraCare/Wolof-Kallaama
```

## 3) Fine-tuning Whisper-small (Colab ou GPU)
Exemple (adaptable):
```powershell
python -m src.train_whisper_wolof \
  --dataset KeraCare/Wolof-Kallaama \
  --model openai/whisper-small \
  --language wolof \
  --output_dir outputs/whisper-small-wo \
  --max_steps 2000
```

## 4) Évaluer WER
```powershell
python -m src.evaluate_wer --model_dir outputs/whisper-small-wo
```

## 5) Démo assistant vocal (micro)
```powershell
python -m src.voice_assistant_demo --model_dir outputs/whisper-small-wo --seconds 5
```

## Colab
Voir `scripts/colab_setup.md`.
