import base64
import ctypes
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Pre-load CUDA 12 DLLs installed via pip so ctranslate2 finds them via LoadLibrary
_venv_root = os.path.dirname(os.path.dirname(sys.executable))
_nvidia_base = os.path.join(_venv_root, "Lib", "site-packages", "nvidia")
for _dll_path in [
    os.path.join(_nvidia_base, "cuda_runtime", "bin", "cudart64_12.dll"),
    os.path.join(_nvidia_base, "cublas",       "bin", "cublasLt64_12.dll"),
    os.path.join(_nvidia_base, "cublas",       "bin", "cublas64_12.dll"),
    os.path.join(_nvidia_base, "cudnn",        "bin", "cudnn64_9.dll"),
]:
    if os.path.exists(_dll_path):
        ctypes.WinDLL(_dll_path)

import webview
import core.stt as stt
import core.tts as tts
from core.llm import TARSBrain
from core.tts import synthesize

_window = None
brain = TARSBrain()
_tts_executor = ThreadPoolExecutor(max_workers=2)
_pipeline_lock = threading.Lock()


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


def _extract_sentence(buffer: str, allow_comma: bool = False) -> tuple[str | None, str]:
    """Pull the next TTS-able chunk out of the streaming LLM buffer.
    - Always splits on sentence-end punctuation (. ! ?).
    - When allow_comma is set (used for the first chunk of a response), also
      splits on , ; : once the buffer is long enough to make natural prosody.
      This halves time-to-first-audio for long responses."""
    match = re.search(r'[.!?]["\']?\s', buffer)
    if match:
        end = match.end()
        return buffer[:end].strip(), buffer[end:]
    if allow_comma and len(buffer) >= 25:
        match = re.search(r'[,;:]\s', buffer)
        if match:
            end = match.end()
            return buffer[:end].strip(), buffer[end:]
    return None, buffer


def _run_pipeline(text: str, t0: float | None = None) -> None:
    with _pipeline_lock:
        stt.interrupt_event.clear()
        stt.pipeline_active = True
        sentence_buffer = ""
        full_response = ""
        first_audio_t: float | None = None
        future_queue: queue.Queue = queue.Queue()
        _window.evaluate_js("startTarsStream()")

        def drain():
            nonlocal first_audio_t
            while True:
                future = future_queue.get()
                if future is None:
                    break
                audio_bytes = future.result()
                if audio_bytes and not stt.interrupt_event.is_set():
                    if first_audio_t is None and t0 is not None:
                        first_audio_t = time.perf_counter() - t0
                    b64 = base64.b64encode(audio_bytes).decode()
                    _window.evaluate_js(f"queueAudio({json.dumps(b64)})")

        drain_thread = threading.Thread(target=drain, daemon=True)
        drain_thread.start()

        first_chunk_pending = True
        try:
            for chunk in brain.chat_stream(text):
                if stt.interrupt_event.is_set():
                    break
                sentence_buffer += chunk
                full_response += chunk
                while True:
                    sentence, sentence_buffer = _extract_sentence(
                        sentence_buffer, allow_comma=first_chunk_pending,
                    )
                    if sentence is None:
                        break
                    future_queue.put(_tts_executor.submit(synthesize, sentence))
                    first_chunk_pending = False

            if not stt.interrupt_event.is_set() and sentence_buffer.strip():
                future_queue.put(_tts_executor.submit(synthesize, sentence_buffer.strip()))
        except Exception as e:
            msg = str(e)
            print(f"[PIPELINE-ERROR] {type(e).__name__}: {msg[:300]}")
            if 'CUDA error' in msg or 'cuda' in msg.lower():
                print("[HINT] Ollama GPU runner failed. Check: 1) `nvidia-smi` driver version "
                      "(need 545+), 2) `ollama --version` (update if old), 3) free VRAM with "
                      "`nvidia-smi` — Whisper medium uses ~1.5GB, llama3.2:3b ~2.5GB.")
        finally:
            # Release pipeline_active FIRST so a queued interruption can resume
            # immediately even if drain blocks. Then signal drain to stop and
            # wait with a timeout so a stuck TTS can't hold the lock forever.
            stt.pipeline_active = False
            future_queue.put(None)
            drain_thread.join(timeout=10.0)
            if drain_thread.is_alive():
                print("[PIPELINE] drain thread still alive after 10s — abandoning, may leak audio")
            if stt.interrupt_event.is_set():
                _window.evaluate_js("interruptStream()")
            else:
                _window.evaluate_js("endTarsStream()")

        response = full_response.strip()
        if response:
            if first_audio_t is not None:
                print(f"[TARS] ({first_audio_t:.2f}s) {response}")
            else:
                print(f"[TARS] {response}")


def _on_utterance(text: str) -> None:
    t0 = time.perf_counter()
    print(f"[YOU] {text}")
    _run_pipeline(text, t0)


def _on_window_loaded():
    stt.start_listener(_on_utterance)
    _window.evaluate_js("setStatus('idle')")


class TARSAPI:
    def send_message(self, text: str) -> dict:
        _run_pipeline(text)
        return {}

    def reset(self) -> None:
        brain.reset()


if __name__ == "__main__":
    ensure_ollama()
    # Load llama runner into GPU BEFORE Whisper takes the CUDA context.
    # Reverse order triggers "shared object initialization failed" on the
    # llama runner because Whisper holds the GPU first.
    print("[OLLAMA] warming up llama runner on GPU...")
    try:
        brain.warmup()
    except Exception as e:
        print(f"[OLLAMA-WARMUP] {type(e).__name__}: {e}")
    # Kokoro warmup in background — the model may need downloading (~350MB)
    # on first run; we don't want to block window startup behind that.
    threading.Thread(target=tts.warmup, daemon=True).start()
    ui_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "index.html")
    _window = webview.create_window(
        title="TARS",
        url=ui_path,
        js_api=TARSAPI(),
        width=300,
        height=560,
        resizable=True,
        min_size=(260, 480),
        background_color="#060608",
    )
    _window.events.loaded += _on_window_loaded
    webview.start(
        http_server=True,
        storage_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".webview_data"),
    )
