import base64
import ctypes
import io
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor

# Make the pip-installed CUDA 12 libraries discoverable so BOTH ctranslate2
# (faster-whisper) and onnxruntime-gpu (Kokoro TTS) find their dependencies.
# Three mechanisms are needed together — each provider resolves DLLs differently:
#   1) add every nvidia/*/bin to PATH    — cuDNN's loader (cudnn64_9.dll) finds its
#      split sub-DLLs (cudnn_ops/graph/...) only via PATH, not add_dll_directory.
#   2) os.add_dll_directory on each       — onnxruntime_providers_cuda.dll resolves
#      cublasLt/cufft/nvjitlink transitive deps.
#   3) WinDLL pre-load the top-level libs — forces them resident before ctranslate2
#      LoadLibrary's them (the original Whisper fix; order: runtime before the rest).
import glob as _glob
_venv_root = os.path.dirname(os.path.dirname(sys.executable))
_nvidia_base = os.path.join(_venv_root, "Lib", "site-packages", "nvidia")
_nv_bins = _glob.glob(os.path.join(_nvidia_base, "*", "bin"))
os.environ["PATH"] = os.pathsep.join(_nv_bins) + os.pathsep + os.environ.get("PATH", "")
for _d in _nv_bins:
    try:
        os.add_dll_directory(_d)
    except OSError:
        pass
for _dll_path in [
    os.path.join(_nvidia_base, "cuda_runtime", "bin", "cudart64_12.dll"),
    os.path.join(_nvidia_base, "nvjitlink",    "bin", "nvJitLink_120_0.dll"),
    os.path.join(_nvidia_base, "cublas",       "bin", "cublasLt64_12.dll"),
    os.path.join(_nvidia_base, "cublas",       "bin", "cublas64_12.dll"),
    os.path.join(_nvidia_base, "cufft",        "bin", "cufft64_11.dll"),
    os.path.join(_nvidia_base, "cudnn",        "bin", "cudnn64_9.dll"),
]:
    if os.path.exists(_dll_path):
        ctypes.WinDLL(_dll_path)

import webview
import core.stt as stt
import core.tts as tts
from config import LLM_BACKEND
from core.llm import TARSBrain
from core.log import Request, fmt_dur
from core.memory import MemoryStore
from core.tts import synthesize

_window = None
brain = TARSBrain()
memory = MemoryStore()
_tts_executor = ThreadPoolExecutor(max_workers=2)
_pipeline_lock = threading.Lock()

# Per-request cancellation. Each _run_pipeline owns its own Event and registers
# it as _current_cancel while it holds the pipeline lock. A barge-in cancels that
# specific pipeline — so the next request clearing ITS own token can never revive
# an older, still-draining response (the bug that made two answers overlap).
_cancel_lock = threading.Lock()
_current_cancel: threading.Event | None = None


def _timed_synthesize(text: str) -> tuple[bytes | None, float, str]:
    start = time.perf_counter()
    audio = synthesize(text)
    return audio, time.perf_counter() - start, text


def _wav_duration(audio_bytes: bytes) -> float:
    """Read frames + sample rate from the WAV header to know how long this clip
    will play. Feeds the speaker-busy timer so barge-in stays armed for the
    full playback window, not just while Python is queueing."""
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def _request_interrupt() -> None:
    """Barge-in: cancel the pipeline currently producing output (if any) and tell
    the browser to drop queued/playing audio. Registered with stt as the wake-word
    barge-in callback."""
    with _cancel_lock:
        ev = _current_cancel
    if ev is not None:
        ev.set()
    if _window is not None:
        _window.evaluate_js("interruptStream()")


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


def _run_pipeline(req: Request, text: str) -> None:
    global _current_cancel
    cancel = threading.Event()
    with req.phase("pipeline-wait-lock"):
        _pipeline_lock.acquire()
    try:
        with _cancel_lock:
            _current_cancel = cancel
        stt.pipeline_active = True
        # Journal the user turn the moment we start answering it. If the app dies
        # mid-response this line is already on disk, so the session is recoverable.
        memory.log_turn("user", text)
        sentence_buffer = ""
        full_response = ""
        first_audio_logged = False
        future_queue: queue.Queue = queue.Queue()
        _window.evaluate_js("startTarsStream()")

        def drain():
            nonlocal first_audio_logged
            while True:
                future = future_queue.get()
                if future is None:
                    break
                # Barge-in: drop the whole TTS backlog instantly instead of
                # synthesizing ~20 queued sentences while TARS should be silent.
                # This is what frees the pipeline lock fast enough to stop talking.
                if cancel.is_set():
                    future.cancel()
                    continue
                try:
                    audio_bytes, tts_dur, sent = future.result()
                except Exception:
                    continue
                preview = sent[:60].replace("\n", " ")
                if len(sent) > 60:
                    preview += "..."
                req.event("tts-sentence", f"{fmt_dur(tts_dur)} '{preview}'")
                if audio_bytes and not cancel.is_set():
                    if not first_audio_logged:
                        req.event("first-audio-ready", fmt_dur(req.t()))
                        first_audio_logged = True
                    wav_dur = _wav_duration(audio_bytes)
                    stt.extend_speaker_busy(wav_dur)
                    b64 = base64.b64encode(audio_bytes).decode()
                    _window.evaluate_js(f"queueAudio({json.dumps(b64)})")

        drain_thread = threading.Thread(target=drain, daemon=True)
        drain_thread.start()

        first_chunk_pending = True
        try:
            with req.phase("llm-stream") as llm_phase:
                ttft_start = time.perf_counter()
                ttft_logged = False
                for chunk in brain.chat_stream(text):
                    if not ttft_logged:
                        req.event("llm-ttft", fmt_dur(time.perf_counter() - ttft_start))
                        ttft_logged = True
                    if cancel.is_set():
                        req.event("interrupt-detected", "stopping LLM stream")
                        break
                    sentence_buffer += chunk
                    full_response += chunk
                    while True:
                        sentence, sentence_buffer = _extract_sentence(
                            sentence_buffer, allow_comma=first_chunk_pending,
                        )
                        if sentence is None:
                            break
                        future_queue.put(_tts_executor.submit(_timed_synthesize, sentence))
                        first_chunk_pending = False
                llm_phase.note(f"{len(full_response)} chars")

            if not cancel.is_set() and sentence_buffer.strip():
                future_queue.put(_tts_executor.submit(_timed_synthesize, sentence_buffer.strip()))
        except Exception as e:
            msg = str(e)
            req.event("pipeline-error", f"{type(e).__name__}: {msg[:200]}")
            if 'CUDA error' in msg or 'cuda' in msg.lower():
                print("[HINT] Ollama GPU runner failed. Check: 1) `nvidia-smi` driver version "
                      "(need 545+), 2) `ollama --version` (update if old), 3) free VRAM with "
                      "`nvidia-smi` — Whisper medium uses ~1.5GB, llama3.2:3b ~2.5GB.")
        finally:
            # Deregister our cancel token (only if it's still ours) and tell drain
            # to stop. With the cancel-aware drain above, a barged-in pipeline
            # drops its backlog and joins in milliseconds, so the lock frees fast.
            stt.pipeline_active = False
            with _cancel_lock:
                if _current_cancel is cancel:
                    _current_cancel = None
            future_queue.put(None)
            with req.phase("drain-wait") as p:
                drain_thread.join(timeout=10.0)
                p.note("leaked" if drain_thread.is_alive() else "clean")
            if cancel.is_set():
                _window.evaluate_js("interruptStream()")
            else:
                _window.evaluate_js("endTarsStream()")

        req.event("pipeline-total", fmt_dur(req.t()))
        response = full_response.strip()
        if response:
            print(f"[TARS req={req.id}] {response}")
            # Journal whatever TARS actually said — even a barged-in partial is a
            # real thing that happened, and consolidation should see it.
            memory.log_turn("assistant", response)
    finally:
        _pipeline_lock.release()


def _on_utterance(text: str, req: Request) -> None:
    print(f"[YOU req={req.id}] {text}")
    _run_pipeline(req, text)


def _on_window_loaded():
    stt.set_barge_in_callback(_request_interrupt)
    stt.start_listener(_on_utterance)
    _window.evaluate_js("setStatus('idle')")


class TARSAPI:
    def send_message(self, text: str) -> dict:
        req = Request()
        req.event("text-input", f"'{text[:60]}'")
        _run_pipeline(req, text)
        return {}

    def reset(self) -> None:
        brain.reset()


if __name__ == "__main__":
    # Cloud backend (groq) needs neither a local Ollama server nor a GPU warmup —
    # the LLM lives off-machine, which is exactly what frees the VRAM.
    if LLM_BACKEND == "ollama":
        ensure_ollama()
        # Load llama runner into GPU BEFORE Whisper takes the CUDA context.
        # Reverse order triggers "shared object initialization failed" on the
        # llama runner because Whisper holds the GPU first.
        print("[OLLAMA] warming up llama runner on GPU...")
        try:
            brain.warmup()
        except Exception as e:
            print(f"[OLLAMA-WARMUP] {type(e).__name__}: {e}")
    else:
        print(f"[LLM] backend={LLM_BACKEND} (cloud) — skipping Ollama + GPU warmup")
    # Kokoro warmup in background — the model may need downloading (~350MB)
    # on first run; we don't want to block window startup behind that.
    threading.Thread(target=tts.warmup, daemon=True).start()

    # Persistent memory. First mop up any session a previous crash left
    # un-distilled (the raw turns are safe in the journal), then open this
    # session — stamping its start time — and fold what we remember about the
    # user into TARS's system prompt so he walks in already knowing them.
    try:
        recovered = memory.consolidate_pending()
        if recovered:
            print(f"[MEMORY] recovered {recovered} un-consolidated session(s) from a previous run")
    except Exception as e:
        print(f"[MEMORY] startup consolidation skipped: {type(e).__name__}: {e}")
    sess = memory.start_session()
    print(f"[MEMORY] session {sess['id']} started at {sess['started_at']}")
    try:
        _ctx = memory.build_context_block()
        brain.set_memory_context(_ctx)
        # Log the exact memory TARS walks in with — the most direct proof that he
        # received what he learned in past sessions. (Test aid; cheap to keep.)
        print("[MEMORY] ----- injected context for this session -----")
        print(_ctx)
        print("[MEMORY] ----- end injected context -----")
    except Exception as e:
        print(f"[MEMORY] context build failed: {type(e).__name__}: {e}")
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

    # webview.start() blocks until the window is closed. Stop the mic listener
    # FIRST: otherwise it keeps capturing audio and, once the interpreter starts
    # tearing down the Whisper thread pool at exit, a late utterance hits a
    # shut-down pool ("cannot schedule new futures after shutdown"). Quieting the
    # listener here closes that window.
    stt.stop_listener()

    # Now finalize the session: stamp its end time, then distill it into long-term
    # memory (Groq by default). Doing it here — after the UI is gone — means the
    # LLM pass can take its time without freezing the close; a hard kill before
    # this is covered by the crash-recovery sweep at the next startup.
    memory.end_session()
    print(f"[MEMORY] session {memory.session_id} ended; consolidating...")
    try:
        memory.consolidate_pending()
        print("[MEMORY] consolidation done")
    except Exception as e:
        print(f"[MEMORY] shutdown consolidation failed: {type(e).__name__}: {e}")
    memory.close()
