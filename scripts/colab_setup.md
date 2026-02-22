# Colab — setup rapide

1. Ouvre un notebook Colab et active GPU: Runtime → Change runtime type → GPU.
2. Clone le repo ou upload les fichiers.

## Installation
```bash
pip install -U pip
pip install -r requirements.txt
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> Si ta version CUDA change, adapte l’URL PyTorch.

## Inspecter dataset
```bash
python -m src.inspect_dataset --dataset KeraCare/Wolof-Kallaama
```

## Lancer entraînement
```bash
python -m src.train_whisper_wolof \
  --dataset KeraCare/Wolof-Kallaama \
  --model openai/whisper-small \
  --language wolof \
  --output_dir outputs/whisper-small-wo \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --max_steps 2000 \
  --fp16
```

## Évaluer
```bash
python -m src.evaluate_wer --model_dir outputs/whisper-small-wo
```
