import base64
import json
import os
import re
import subprocess
import time
import urllib.request
import webview

from core.llm import TARSBrain
from core.stt import start_recording as stt_start, stop_and_transcribe as stt_stop
from core.tts import synthesize

_window = None
brain = TARSBrain()


def ensure_ollama():
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return
    except Exception:
        pass
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    for _ in range(20):
        try:
            urllib.request.urlopen("http://localhost:11434", timeout=1)
            return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Ollama did not start in time.")


def _extract_sentence(buffer: str) -> tuple[str | None, str]:
    match = re.search(r'[.!?]["\']?\s', buffer)
    if match:
        end = match.end()
        return buffer[:end].strip(), buffer[end:]
    return None, buffer


class TARSAPI:
    def start_recording(self) -> None:
        stt_start()

    def stop_and_process(self) -> None:
        t0 = time.perf_counter()
        text = stt_stop()
        if not text.strip():
            _window.evaluate_js("resetProcessing()")
            return
        print(f"[STT] {text}")
        _window.evaluate_js(f"onTranscript({json.dumps(text)})")
        self._run_pipeline(text, t0)

    def send_message(self, text: str) -> dict:
        self._run_pipeline(text)
        return {}

    def _run_pipeline(self, text: str, t0: float | None = None) -> None:
        sentence_buffer = ""
        first_audio = True
        _window.evaluate_js("startTarsStream()")

        for chunk in brain.chat_stream(text):
            sentence_buffer += chunk
            _window.evaluate_js(f"appendTarsText({json.dumps(chunk)})")

            while True:
                sentence, sentence_buffer = _extract_sentence(sentence_buffer)
                if sentence is None:
                    break
                audio_bytes = synthesize(sentence)
                if audio_bytes:
                    if first_audio and t0 is not None:
                        print(f"[LATENCY] {time.perf_counter() - t0:.2f}s")
                        first_audio = False
                    b64 = base64.b64encode(audio_bytes).decode()
                    _window.evaluate_js(f"queueAudio({json.dumps(b64)})")

        if sentence_buffer.strip():
            audio_bytes = synthesize(sentence_buffer.strip())
            if audio_bytes:
                if first_audio and t0 is not None:
                    print(f"[LATENCY] {time.perf_counter() - t0:.2f}s")
                    first_audio = False
                b64 = base64.b64encode(audio_bytes).decode()
                _window.evaluate_js(f"queueAudio({json.dumps(b64)})")

        _window.evaluate_js("endTarsStream()")

    def reset(self) -> None:
        brain.reset()


if __name__ == "__main__":
    ensure_ollama()
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")
    _window = webview.create_window(
        title="TARS",
        url=ui_path,
        js_api=TARSAPI(),
        width=960,
        height=640,
        resizable=True,
        min_size=(700, 480),
        background_color="#060608",
    )
    webview.start(
        http_server=True,
        storage_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".webview_data"),
    )
