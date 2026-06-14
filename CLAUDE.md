# TARS — Assistente AI Personale

Desktop app con personalità TARS da Interstellar. Si avvia come un programma normale (come un videogioco/utility), niente browser, niente server esposto.

## Stack

| Componente | Tool | Note |
|---|---|---|
| Desktop | pywebview | `http_server=True`, bridge Python↔JS |
| STT | faster-whisper | modello `medium`, lingua `en`, CUDA o CPU |
| LLM | Ollama locale | default `llama3.2:3b`, streaming |
| TTS | ElevenLabs SDK | `eleven_multilingual_v2`, mp3_44100_128 |
| UI | HTML + CSS + vanilla JS | in `ui/`, servita da pywebview |

## Hardware utente
- CPU: Intel i5-13420H
- GPU: NVIDIA RTX 4050 Laptop (6GB VRAM, CUDA)
- RAM: 16GB
- OS: Windows 11

## Struttura progetto

```
TARS/
├── CLAUDE.md
├── main.py           ← entry point: pywebview window + TARSAPI (js_api)
├── config.py         ← variabili da .env + SYSTEM_PROMPT TARS
├── .env              ← chiavi API (non in git)
├── requirements.txt  ← pywebview, faster-whisper, ollama, elevenlabs, python-dotenv
├── core/
│   ├── llm.py        ← TARSBrain: storico + chat_stream() via Ollama
│   ├── stt.py        ← transcribe(bytes, mime) → str via faster-whisper
│   └── tts.py        ← synthesize(text) → bytes via ElevenLabs
└── ui/
    ├── index.html    ← layout: pannello TARS sx (vis + status), chat dx
    ├── style.css     ← dark monolitica, CRT scanline, stati colore
    └── app.js        ← bridge JS↔Python, MediaRecorder, audio queue
```

## Architettura pywebview

`main.py` crea la finestra con `js_api=TARSAPI()`. Il bridge funziona in due direzioni:

**JS → Python** (chiamate async dal browser):
- `window.pywebview.api.transcribe_audio(b64, mimeType)` → testo
- `window.pywebview.api.send_message(text)` → avvia streaming
- `window.pywebview.api.reset()` → resetta storia

**Python → JS** (push via `_window.evaluate_js()`):
- `startTarsStream()` — crea bubble di risposta nel chat
- `appendTarsText(chunk)` — appende token in streaming
- `queueAudio(b64)` — aggiunge chunk MP3 alla coda audio
- `endTarsStream()` — segnala fine stream; JS chiude quando coda audio è vuota

## Flusso dati

```
Browser (hold mic button / SPACE / text input)
  → MediaRecorder registra WebM/Opus
  → window.pywebview.api.transcribe_audio(b64, mime) → testo

Python (send_message)
  → startTarsStream() via evaluate_js
  → Ollama stream token per token
    → appendTarsText(chunk) via evaluate_js
    → _extract_sentence() accumula finché trova [.!?]
      → synthesize(sentence) → ElevenLabs → MP3 bytes
      → queueAudio(b64) via evaluate_js
  → endTarsStream() via evaluate_js

Browser (audio queue)
  → processQueue() suona MP3 in sequenza con AudioContext
  → finishProcessing() quando coda vuota + stream completo → status idle
```

## Stati UI

| Stato | Colore bordo TARS | Barre | Occhio |
|---|---|---|---|
| idle | grigio | ferme | grigio |
| listening | verde | animate | verde fisso |
| transcribing | ambra | ferme | ambra lampeggia |
| thinking | blu | ferme | blu lampeggia + scanline |
| speaking | ambra | animate veloci | ambra fisso |

## Personalità TARS
- Humor: 75%, Honesty: 90%
- Risponde **sempre in inglese** indipendentemente dalla lingua dell'utente
- Niente preamboli ("Certainly!", "Of course!" → vietati)
- Humor secco, deadpan, understatement
- Ack brevi per comandi: "Understood." / "Done." / "On it."
- Non rompe mai il personaggio

## Configurazione (.env)
```
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=...
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
WHISPER_MODEL=medium
WHISPER_DEVICE=cuda        # oppure cpu
USER_NAME=...
TARS_HUMOR=75
```

## Note tecniche importanti
- `ensure_ollama()` in main.py avvia `ollama serve` automaticamente se non è in esecuzione
- `_extract_sentence()` usa regex `[.!?][\"']?\s` per spezzare per frase → TTS per frase
- WHISPER_DEVICE=cuda richiede CUDA Toolkit 12.x installato; altrimenti usare `cpu` con `int8`
- Il resto del flusso funziona uguale — solo STT cambia compute type (float16 vs int8)

## Backlog
- Wake word ("Hey TARS") con OpenWakeWord
- Tool use: controllo PC (aprire app, browser, file)
- Memoria persistente tra sessioni (JSON o SQLite)
- Possibilità di switchare LLM/STT backend (locale ↔ cloud) per distribuire su PC deboli
