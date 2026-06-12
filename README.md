# ArgueBot

A Gradio web app that lets you **upload text to train a chatbot's speaking style**, chat with it via your **local LLM** (Ollama), and hear replies spoken through your **local Chatterbox TTS Docker server**.

## Features

- **Style training** — upload `.txt`, `.md`, `.csv`, `.pdf` or paste text; more text = better style matching (RAG)
- **Chat** — bot avatar, conversation history, auto-spoken replies
- **Chatterbox TTS** — uses your Docker `chatterbox-tts-server` on port **8004**
  - Predefined voices (Abigail, Olivia, etc.)
  - **Clone voices** from your reference audio library (AlexJones, kryten2, etc.)
- **Settings** — LLM URL/model, Chatterbox URL, voice mode, TTS parameters, persona, avatar
- **Remote access** — `python app.py --share` creates a temporary `gradio.live` URL (same as podcaster2)

## Prerequisites

| Service | This machine |
|---------|--------------|
| **Ollama** | `http://127.0.0.1:11434` (default LLM) |
| **Chatterbox TTS** | Docker container `chatterbox-tts-server-cu128` on `http://127.0.0.1:8004` |
| **Python** | 3.11+ |

> Port 8080 on this machine is used by Frigate, not the LLM — ArgueBot defaults to Ollama on 11434.

## Quick Start

```bash
cd ArgueBot

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # optional — edit defaults

# Local only
python app.py

# Temporary public URL (gradio.live)
python app.py --share
```

Open **http://localhost:7860**

## Usage

### 1. Training tab
Upload writing samples (social posts, transcripts, articles). The bot retrieves relevant excerpts when answering to match your style.

### 2. Chat tab
Type a question. ArgueBot crafts a styled reply via Ollama and speaks it through Chatterbox.

### 3. Settings tab
- **LLM**: `http://127.0.0.1:11434`, model e.g. `qwen2.5:7b`
- **Chatterbox**: `http://127.0.0.1:8004`
- **Voice mode**:
  - `clone` — use a reference audio file from your Chatterbox library (e.g. `kryten2.mp3`)
  - `predefined` — built-in voice (e.g. `Olivia.wav`)
- Upload new reference voices directly to Chatterbox from Settings

### Voice sample prep (optional)

Use [yootoob](https://github.com/datagod/yootoob) to download Chatterbox-ready audio:

```bash
yootoob "https://youtube.com/..." -o voice.wav --chatterbox
```

Then upload the WAV via Settings → Upload reference voice.

## Chatterbox Docker

ArgueBot expects [Chatterbox-TTS-Server](https://github.com/devnen/Chatterbox-TTS-Server) running locally:

```bash
# Already running on this machine as:
docker ps | grep chatterbox
# chatterbox-tts-server-cu128   0.0.0.0:8004->8004/tcp
```

Manage voices and models via the Chatterbox UI at http://127.0.0.1:8004

## CLI options

```bash
python app.py                    # local
python app.py --share            # public gradio.live link
python app.py --server-port 7861
python app.py --server-name 0.0.0.0
```

## Project structure

```
ArgueBot/
├── app.py              # Gradio UI
├── prompts.py          # Style-aware system prompts
├── services/
│   ├── chatterbox.py   # Chatterbox-TTS-Server client
│   ├── corpus.py       # Text ingestion + chunking
│   ├── llm.py          # Ollama / OpenAI-compat client
│   ├── rag.py          # TF-IDF style retrieval
│   └── settings_store.py
└── data/               # corpus, avatar, SQLite (gitignored)
```

## License

MIT