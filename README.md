Ho identificato i problemi. Eccoli in ordine di impatto:

  ---
  1. TTS blocca lo streaming LLM (problema principale)

  In send_message() riga 64, synthesize(sentence) è chiamata sincrona dentro il loop di Ollama. Ogni volta che viene
  trovata una frase, Ollama si ferma e aspetta che ElevenLabs risponda (~300-800ms per chiamata HTTP). Poi ricomincia a
  streammare.

  Il flusso reale è seriale: token... token... STOP → ElevenLabs → token... token... STOP → ElevenLabs...

  Fix: TTS va su un thread separato (ThreadPoolExecutor). Ollama continua a streammare mentre ElevenLabs processa la
  frase precedente in parallelo.

  ---
  2. Whisper su CPU

  Se WHISPER_DEVICE=cpu, trascrivere 5 secondi di audio con medium prende facilmente 4-8 secondi. Con CUDA sarebbe
  ~0.5s. Richiede CUDA Toolkit 12.x installato.

  Alternativa senza CUDA: scendere a small (ancora ottimo per inglese, molto più veloce).

  ---
  3. ElevenLabs usa il modello lento

  eleven_multilingual_v2 è il modello qualità alta. Esiste eleven_turbo_v2_5 che è ~60% più rapido con qualità
  leggermente inferiore.

  ---
  4. evaluate_js chiamata per ogni token

  Ogni singolo token Ollama fa un IPC call al renderer. Si può batchare (accumulare N caratteri o N ms prima di pushare)
   per ridurre overhead.

  ---
  Priorità consigliata

  ┌────────────────────────┬─────────┬────────────────────┐
  │          Fix           │ Impatto │     Difficoltà     │
  ├────────────────────────┼─────────┼────────────────────┤
  │ TTS su thread separato │ alto    │ medio (codice)     │
  ├────────────────────────┼─────────┼────────────────────┤
  │ CUDA per Whisper       │ alto    │ setup ambiente     │
  ├────────────────────────┼─────────┼────────────────────┤
  │ small invece di medium │ medio   │ cambia .env        │
  ├────────────────────────┼─────────┼────────────────────┤
  │ ElevenLabs turbo model │ medio   │ cambia una stringa │
  ├────────────────────────┼─────────┼────────────────────┤
  │ Batch evaluate_js      │ basso   │ codice             │
  └────────────────────────┴─────────┴────────────────────┘