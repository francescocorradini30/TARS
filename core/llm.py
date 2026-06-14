import ollama
from typing import Iterator
from config import OLLAMA_MODEL, OLLAMA_BASE_URL, SYSTEM_PROMPT


class TARSBrain:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_BASE_URL)
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat_stream(self, message: str) -> Iterator[str]:
        self.history.append({"role": "user", "content": message})
        full_text: list[str] = []
        try:
            stream = self.client.chat(model=OLLAMA_MODEL, messages=self.history, stream=True)
            for chunk in stream:
                delta = chunk["message"]["content"]
                full_text.append(delta)
                yield delta
        finally:
            self.history.append({"role": "assistant", "content": "".join(full_text)})

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
