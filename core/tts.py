import io
import threading
import urllib.request
from pathlib import Path
import soundfile as sf
from config import KOKORO_VOICE

# Kokoro-ONNX model files. Auto-downloaded on first run, then cached locally.
MODEL_DIR  = Path(__file__).resolve().parent.parent / "models" / "kokoro"
MODEL_URL  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

_kokoro = None
_download_lock = threading.Lock()


def _ensure_model() -> tuple[str, str]:
    # Lock so that parallel synthesize() calls (one per sentence) on the very
    # first run don't all race into urlretrieve and corrupt each other's writes.
    with _download_lock:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path  = MODEL_DIR / "kokoro-v1.0.onnx"
        voices_path = MODEL_DIR / "voices-v1.0.bin"
        # Detect partial/corrupted downloads from a prior crashed run and nuke.
        if model_path.exists() and model_path.stat().st_size < 100_000_000:
            print(f"[TTS] removing corrupted Kokoro model ({model_path.stat().st_size} bytes)")
            model_path.unlink()
        if voices_path.exists() and voices_path.stat().st_size < 1_000_000:
            print(f"[TTS] removing corrupted Kokoro voices ({voices_path.stat().st_size} bytes)")
            voices_path.unlink()
        if not model_path.exists():
            print(f"[TTS] downloading Kokoro model (~330MB) -> {model_path}")
            tmp = model_path.with_suffix(".onnx.part")
            urllib.request.urlretrieve(MODEL_URL, tmp)
            tmp.replace(model_path)
        if not voices_path.exists():
            print(f"[TTS] downloading Kokoro voices (~25MB) -> {voices_path}")
            tmp = voices_path.with_suffix(".bin.part")
            urllib.request.urlretrieve(VOICES_URL, tmp)
            tmp.replace(voices_path)
        return str(model_path), str(voices_path)


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        model_path, voices_path = _ensure_model()
        _kokoro = Kokoro(model_path, voices_path)
    return _kokoro


def warmup() -> None:
    """Force Kokoro to download/load the ONNX model and run a dummy synthesis
    so the first real request doesn't pay the cold-start tax (~1.5-2s)."""
    try:
        _get_kokoro().create("Ready.", voice=KOKORO_VOICE, lang="en-us")
        print("[TTS] Kokoro ready")
    except Exception as e:
        print(f"[TTS-WARMUP] {type(e).__name__}: {e}")


def synthesize(text: str) -> bytes | None:
    text = text.strip()
    if not text:
        return None
    try:
        samples, sample_rate = _get_kokoro().create(
            text, voice=KOKORO_VOICE, speed=1.0, lang="en-us",
        )
        if samples is None or len(samples) == 0:
            return None
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format='WAV')
        return buf.getvalue()
    except Exception as e:
        print(f"[TTS] {type(e).__name__}: {e}")
        return None
