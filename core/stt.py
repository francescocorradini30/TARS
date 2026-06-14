import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from config import WHISPER_MODEL, WHISPER_DEVICE

SAMPLE_RATE = 16000

_model: WhisperModel | None = None
_stream: sd.InputStream | None = None
_frames: list = []
_lock = threading.Lock()


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        compute = "float16" if WHISPER_DEVICE == "cuda" else "int8"
        _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=compute)
    return _model


def _callback(indata: np.ndarray, frames: int, time, status) -> None:
    with _lock:
        _frames.append(indata.copy())


def start_recording() -> None:
    global _stream, _frames
    with _lock:
        _frames = []
    _stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=_callback)
    _stream.start()


def stop_and_transcribe() -> str:
    global _stream
    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None

    with _lock:
        frames = list(_frames)

    if not frames:
        return ""

    audio = np.concatenate(frames).flatten()
    if len(audio) < SAMPLE_RATE * 0.3:
        return ""

    segments, _ = _get_model().transcribe(audio, language="en")
    return " ".join(s.text for s in segments).strip()
