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

## 6) Démo web (upload audio → transcription)
```powershell
pip install -r requirements.txt
streamlit run web/streamlit_app.py
```

Tu peux utiliser un modèle fine-tuné en mettant le dossier local du modèle dans la barre latérale
("Chemin local du modèle").

## 7) CLI transcription (fichier audio)
```powershell
python -m src.transcribe_file --audio path\to\audio.wav --model openai/whisper-small

# ou modèle local
python -m src.transcribe_file --audio path\to\audio.wav --model_dir outputs\whisper-small-wo
```

## 8) Assistant vocal push-to-talk (micro → ASR → réponse → TTS)
Lance le serveur API + la page web push-to-talk :
```powershell
# Depuis le dossier du projet, venv activé
pip install -r requirements.txt

# Lancer le serveur (remplace le chemin modèle par le tien)
python web/api_server.py --model_dir "C:\models\anansi_wolof_outputs\checkpoint-1200"
```
Ouvre http://localhost:8000 dans ton navigateur.
- Clique sur le bouton micro 🎤 (ou appuie sur Espace)
- Parle en wolof
- Reclique (ou Espace) → la transcription s'affiche + Anansi répond
- La réponse est lue à voix haute (TTS navigateur, désactivable)

> Note : le TTS utilise la synthèse vocale du navigateur (Web Speech API).
> Pour le wolof, la voix sera approximative (fr-FR). Un vrai TTS wolof
> pourra être intégré plus tard.
> pourra être intégré plus tard.

## Colab
Voir `scripts/colab_setup.md`.
