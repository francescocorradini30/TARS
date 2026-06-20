import ollama
from typing import Iterator
from config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL, SYSTEM_PROMPT,
    LLM_BACKEND, GROQ_API_KEY, GROQ_MODEL,
)


class TARSBrain:
    """Conversation brain with a switchable backend.

    - "ollama": local model on the GPU. Free, offline, private, but shares VRAM
      with Whisper and tops out at the quality of a small local model.
    - "groq": cloud inference (free tier). Frees ~2.8GB of VRAM, runs a far
      bigger model (llama-3.3-70b) at ~200ms first-token. Needs internet + an
      API key; the conversation text leaves the machine.

    Both speak the same OpenAI-style message history, so switching is just a
    config flag — history, reset and the streaming contract stay identical.
    """

    def __init__(self):
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.backend = LLM_BACKEND
        if self.backend == "groq":
            if not GROQ_API_KEY:
                raise RuntimeError(
                    "LLM_BACKEND=groq but GROQ_API_KEY is empty. Get a free key at "
                    "https://console.groq.com/keys and put it in .env."
                )
            from groq import Groq
            self.client = Groq(api_key=GROQ_API_KEY)
        else:
            self.client = ollama.Client(host=OLLAMA_BASE_URL)

    def _stream_groq(self, messages: list[dict]) -> Iterator[str]:
        # max_tokens is a backstop against runaway monologues, not the primary
        # length control (that's the system prompt). Set generously so it only
        # ever trims a pathological ramble, never a genuine explanation (~250 tok).
        stream = self.client.chat.completions.create(
            model=GROQ_MODEL, messages=messages, stream=True, max_tokens=400,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _stream_ollama(self, messages: list[dict]) -> Iterator[str]:
        # keep_alive=-1 pins the model in VRAM between turns so we don't pay a
        # 2.5s reload tax. Harmless no-op semantics on the groq path (not used).
        stream = self.client.chat(
            model=OLLAMA_MODEL, messages=messages, stream=True, keep_alive=-1,
        )
        for chunk in stream:
            yield chunk["message"]["content"]

    def chat_stream(self, message: str) -> Iterator[str]:
        self.history.append({"role": "user", "content": message})
        full_text: list[str] = []
        stream = self._stream_groq if self.backend == "groq" else self._stream_ollama
        try:
            for delta in stream(self.history):
                full_text.append(delta)
                yield delta
        finally:
            content = "".join(full_text).strip()
            if content:
                self.history.append({"role": "assistant", "content": content})
            else:
                # Stream produced nothing (interrupted before first token, or
                # network/model error). Drop the dangling user turn so the next
                # request doesn't see two user messages in a row.
                self.history.pop()

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def warmup(self):
        """Force Ollama to load the model into GPU memory now, before Whisper
        grabs CUDA. No-op for the groq backend (nothing local to warm)."""
        if self.backend == "groq":
            return
        self.client.generate(
            model=OLLAMA_MODEL, prompt="hi", options={"num_predict": 1}, keep_alive=-1,
        )
