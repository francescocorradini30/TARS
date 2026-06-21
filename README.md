# TARS

A local, always-listening voice assistant with the personality of TARS from *Interstellar* — dry, deadpan, and on your side. It hears you, thinks, and talks back through your speakers in real time, with a persistent memory that lets it actually remember you between sessions.

Built as a fully self-hosted pipeline: speech-to-text, language model, and text-to-speech can all run on your own GPU, with an optional cloud LLM backend when you want to free up VRAM.

> [!NOTE]
> A personal project / portfolio piece — a voice companion, not a product. Tested on Windows 11 with an NVIDIA GPU.

<!-- TODO: add a screenshot or short demo GIF here — it's the single biggest upgrade for a portfolio repo. -->
<!-- ![TARS demo](docs/demo.gif) -->

## Features

- **Always-listening, hands-free.** No push-to-talk. A lightweight energy-VAD watches the mic continuously and only spins up the heavy models when you actually speak.
- **Bilingual addressee detection.** Speak English and TARS answers; speak Italian and he ignores you. The gate works by transcribing each utterance twice (once forced to each language) and comparing acoustic confidence — so background chatter in your native language never triggers a response.
- **ChatGPT-style barge-in.** Talk over TARS mid-sentence with the wake word and he stops instantly. A rolling window of your in-progress speech is checked for the wake word every ~0.4s, so the interrupt fires the moment you say it — not at the end of his sentence.
- **Streaming pipeline.** The LLM streams tokens, sentences are cut from the stream and synthesized as they complete, and audio starts playing while the rest is still being generated — minimizing time-to-first-word.
- **Persistent memory.** Sessions are journaled and, at the end of each one, distilled into long-term episodic memories plus an evolving user profile. Memories **decay over time** unless re-recalled, so what matters stays sharp and trivia fades. TARS can bring things up on his own ("remember the other day when…").
- **Switchable LLM backend.** Run everything locally on Ollama, or flip one flag to use Groq's cloud (free tier) for a much larger model and ~2.8GB of freed VRAM.
- **GPU-accelerated throughout.** Whisper (STT) and Kokoro (TTS) both run on CUDA, with graceful CPU fallback for TTS.

## Architecture

```
   🎙  mic
    │  energy-VAD (continuous, cheap)
    ▼
┌─────────────────────────────────────────────────────────┐
│  STT  ·  faster-whisper (large-v3-turbo)                 │
│  • dual-pass en/it transcription → addressee gate        │
│  • separate base.en model for mid-speech barge-in checks │
└─────────────────────────────────────────────────────────┘
    │  accepted English utterance
    ▼
┌─────────────────────────────────────────────────────────┐
│  LLM  ·  TARSBrain (streaming)                           │
│  • Ollama (local GPU)  ──or──  Groq (cloud)              │
│  • TARS persona + injected memory context                │
└─────────────────────────────────────────────────────────┘
    │  token stream → sentences
    ▼
┌─────────────────────────────────────────────────────────┐
│  TTS  ·  Kokoro-ONNX (onnxruntime-gpu)                   │
│  • per-sentence synthesis, queued + streamed to the UI   │
└─────────────────────────────────────────────────────────┘
    │  WAV chunks (base64)
    ▼
   🔊  pywebview window (HTML/JS audio playback)

        ┌───────────────────────────────────────┐
        │  Memory · SQLite + JSON profile        │
        │  journals every turn, consolidates &   │
        │  decays at session close               │
        └───────────────────────────────────────┘
```

## Tech stack

| Layer | Tool |
|-------|------|
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) |
| Language model | [Ollama](https://ollama.com) (local) / [Groq](https://groq.com) (cloud) |
| Text-to-speech | [Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx) on `onnxruntime-gpu` |
| Audio I/O | `sounddevice`, `soundfile` |
| UI shell | `pywebview` (HTML/CSS/JS front-end) |
| Memory | SQLite (session ledger + episodic facts) + JSON (user profile) |

## Requirements

- **Windows 11** (the CUDA DLL bootstrapping in `main.py` is Windows-specific; other OSes may need tweaks).
- **Python 3.10+** (uses `X | Y` type-union syntax).
- **An NVIDIA GPU with CUDA** strongly recommended. ~3GB of VRAM covers Whisper turbo + the barge-in model + Kokoro; add ~2.5GB more if you also run the LLM locally on Ollama. A 6GB card is comfortable with the cloud LLM backend.
- **[Ollama](https://ollama.com)** installed, if you want the fully-local LLM backend.

## Installation

```bash
git clone https://github.com/francescocorradini30/TARS.git
cd TARS

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
```

> [!IMPORTANT]
> **onnxruntime conflict.** `kokoro-onnx` and `faster-whisper` both pull in CPU-only `onnxruntime`, which clashes with the GPU build. After installing, force the GPU build:
> ```bash
> pip uninstall -y onnxruntime
> pip install onnxruntime-gpu
> ```

The Whisper and Kokoro models download automatically on first run (Kokoro ~330MB) and are cached locally.

If you're using the local LLM backend, pull the model once:

```bash
ollama pull llama3.2:3b
```

## Configuration

Copy the example env file and edit it — your real keys stay out of git:

```bash
cp .env.example .env      # or: copy .env.example .env  (Windows)
```

Key settings (full list with comments in [`.env.example`](.env.example)):

| Variable | What it does |
|----------|--------------|
| `LLM_BACKEND` | `ollama` (local) or `groq` (cloud) |
| `GROQ_API_KEY` | required only for the Groq backend — [get one free](https://console.groq.com/keys) |
| `WHISPER_MODEL` | STT model size (`small` → `large-v3-turbo`) |
| `STT_INPUT_DEVICE` | mic index/name; empty = system default |
| `USER_NAME` | how TARS addresses you |
| `TARS_HUMOR` | personality humor dial (0–100) |
| `KOKORO_VOICE` | TTS voice |

## Usage

```bash
python main.py
```

A small window opens and TARS starts listening. Just talk to him in English. Say his name to interrupt him while he's speaking. Close the window to end the session — that's when he distills the conversation into long-term memory.

The console prints a detailed per-request timeline (VAD → transcription → LLM → first audio) plus the memory context injected at startup — handy for tuning latency.

## How it works

A few of the less obvious design decisions:

- **Why two transcriptions per utterance?** Whisper's acoustic language detection is unreliable on short clips, so instead of trusting it, TARS forces an English pass and an Italian pass and keeps whichever the audio scores higher confidence in. That comparison *is* the "are you talking to me?" gate. A wake-word tiebreaker accepts anything that clearly names TARS, regardless of the score.
- **Why a separate model for barge-in?** Mid-speech wake-word checks run on a small, fast `base.en` model on a rolling ~3s window, kept entirely separate from the main model so the two never contend for the GPU. Energy alone never triggers a barge-in (keyboards and breath are too noisy) — only a Whisper-confirmed wake word does.
- **Why does memory decay?** A memory's effective salience drops each session it isn't recalled; recalling it resets the clock. Important recurring things stay injected verbatim into the prompt, while one-off details fade out of the active context (but survive inside session recaps). After each session, an extra LLM pass merges near-duplicate profile entries and de-duplicates memories to keep the store compact.

## Project structure

```
main.py              orchestration: pipeline, barge-in wiring, session lifecycle
config.py            all settings, loaded from .env, + the TARS system prompt
core/
  stt.py             listener loop, VAD, dual-pass transcription, barge-in
  llm.py             TARSBrain — switchable Ollama/Groq streaming brain
  tts.py             Kokoro-ONNX synthesis (GPU/CPU)
  memory.py          SQLite + JSON persistent memory, consolidation, decay
  log.py             per-request timing/event logging
scripts/
  inspect_memory.py  inspect the local memory store
ui/                  pywebview front-end (HTML/CSS/JS audio playback)
```

## Roadmap

- True streaming Kokoro (emit audio while synthesis continues)
- Utterance queue to replace the transcribing flag (race fix on rapid back-to-back speech)
- Additional cloud LLM backends

## Acknowledgements

Built on the work of [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [Kokoro-ONNX](https://github.com/thewh1teagle/kokoro-onnx), [Ollama](https://ollama.com), and [Groq](https://groq.com). TARS's voice and persona are an affectionate homage to the TARS robot from Christopher Nolan's *Interstellar*.

## License

[MIT](LICENSE) © 2026 Francis

---

*This is a non-commercial fan/portfolio project and is not affiliated with, endorsed by, or sponsored by Warner Bros., Legendary Pictures, or the creators of* Interstellar. *The TARS name and character belong to their respective owners.*
