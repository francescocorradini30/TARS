import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from config import WHISPER_MODEL, WHISPER_DEVICE

SAMPLE_RATE       = 16000
CHUNK_SAMPLES     = 512
ENERGY_THRESHOLD  = 0.012
SILENCE_CHUNKS    = 15            # ~0.48s of silence ends an utterance
PRE_ROLL_CHUNKS   = 10            # ~0.32s pre-roll
MIN_AUDIO_SAMPLES = int(SAMPLE_RATE * 0.35)

# Addressee gating: English words = directed at TARS, Italian words = ignore.
# Whisper's acoustic language detection mixes accent + content and fails in both
# directions on short utterances, so instead we transcribe twice (forcing each
# language) and pick whichever has higher avg_logprob — i.e. whichever language
# the audio is most coherent in. English must win by at least LANG_MARGIN to
# process: ties (within margin) on short accented utterances are often Italian
# that Whisper "made fit" as English (e.g. "Only fools rushing"), so we default
# to ignoring when scores are close.
LANG_MARGIN = 0.15

# Whisper hallucinates these on silence/noise, or transcribes filler voice
# sounds as these tokens — discard before gating.
_NOISE_TRANSCRIPTS = {
    # YouTube-style hallucinations
    "thanks for watching",
    "thank you for watching",
    "thank you so much for watching",
    "thank you",
    "thanks",
    "please subscribe",
    "subscribe",
    "bye",
    "okay",
    "ok",
    # filler voice sounds
    "hm", "hmm", "hmmm",
    "uh", "uhh", "uhhh",
    "uhm", "uhmm",
    "um", "umm",
    "mm", "mmm", "mmmm",
    "mhm", "mhmm",
    "ah", "ahh",
    "eh", "ehh",
    "er", "err",
    "oh", "ohh",
    "you",
    ".",
    "..",
    "...",
}


def _is_noise(text: str) -> bool:
    return text.lower().strip(" .,!?-") in _NOISE_TRANSCRIPTS


# interrupt_event: set by VAD when user speaks during the LLM/TTS pipeline.
# pipeline_active: toggled by main.py — True only while LLM+TTS is running.
# Interrupt is only signalled when pipeline_active, NOT during transcription.
interrupt_event  = threading.Event()
pipeline_active: bool = False

_model: WhisperModel | None = None
_running: bool     = False
_transcribing: bool = False   # True only during Whisper inference
_transcribe_pool = ThreadPoolExecutor(max_workers=2)


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        compute = "float16" if WHISPER_DEVICE == "cuda" else "int8"
        _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=compute)
    return _model


def _transcribe(audio: np.ndarray, language: str) -> tuple[str, float]:
    segments, _ = _get_model().transcribe(
        audio,
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        condition_on_previous_text=False,
        beam_size=5,
    )
    segs = list(segments)
    text = " ".join(s.text for s in segs).strip()
    score = sum(s.avg_logprob for s in segs) / len(segs) if segs else -10.0
    return text, score


def _process_utterance(audio: np.ndarray, callback) -> None:
    global _transcribing
    try:
        # Run both transcriptions in parallel — faster_whisper releases the GIL
        # during ctranslate2 inference, so this overlaps the en/it passes.
        en_future = _transcribe_pool.submit(_transcribe, audio, 'en')
        it_future = _transcribe_pool.submit(_transcribe, audio, 'it')
        en_text, en_score = en_future.result()
        it_text, it_score = it_future.result()

        # English must beat Italian by at least LANG_MARGIN. Ties (close scores)
        # are treated as Italian — common when Whisper "makes English fit"
        # short Italian utterances by accident.
        if en_score - it_score < LANG_MARGIN:
            print(f"[STT-IGNORED] (en {en_score:.2f} vs it {it_score:.2f}) {it_text}")
            return

        if not en_text or _is_noise(en_text):
            return
        print(f"[STT-MSG] (en {en_score:.2f} > it {it_score:.2f}) {en_text}")
        callback(en_text)
    finally:
        _transcribing = False


def _delayed_submit(audio: np.ndarray, callback) -> None:
    """Wait for the interrupted pipeline to finish, then transcribe and process."""
    global _transcribing
    deadline = time.time() + 3.0
    while pipeline_active and time.time() < deadline:
        time.sleep(0.02)
    if not pipeline_active:
        _transcribing = True
        _process_utterance(audio, callback)


def _listener_loop(callback) -> None:
    global _transcribing

    pre_buffer:    list = []
    utterance:     list = []
    in_speech            = False
    silence_count        = 0
    interrupted_this     = False

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        blocksize=CHUNK_SAMPLES) as stream:
        while _running:
            chunk, _ = stream.read(CHUNK_SAMPLES)
            chunk = chunk.flatten()
            is_speech = float(np.sqrt(np.mean(chunk ** 2))) > ENERGY_THRESHOLD

            if not in_speech:
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)
                if is_speech:
                    in_speech = True
                    utterance = list(pre_buffer) + [chunk]
                    silence_count = 0
                    # Only interrupt when LLM/TTS pipeline is actually running,
                    # not during Whisper transcription of a previous utterance.
                    if pipeline_active:
                        interrupt_event.set()
                        interrupted_this = True
                    else:
                        interrupted_this = False
            else:
                utterance.append(chunk)
                if not is_speech:
                    silence_count += 1
                    if silence_count >= SILENCE_CHUNKS:
                        in_speech = False
                        audio = np.concatenate(utterance)
                        utterance = []
                        pre_buffer = []
                        silence_count = 0
                        if len(audio) >= MIN_AUDIO_SAMPLES:
                            if interrupted_this:
                                threading.Thread(
                                    target=_delayed_submit,
                                    args=(audio, callback),
                                    daemon=True,
                                ).start()
                            elif not _transcribing:
                                _transcribing = True
                                threading.Thread(
                                    target=_process_utterance,
                                    args=(audio, callback),
                                    daemon=True,
                                ).start()
                            # else: transcription already running — discard this utterance
                        interrupted_this = False
                else:
                    silence_count = 0


def _warmup() -> None:
    """Run a dummy transcribe so the model is loaded + JIT warm before the
    first real utterance. Saves the cold-start cost on the first interaction."""
    dummy = np.zeros(SAMPLE_RATE, dtype=np.float32)
    try:
        _get_model().transcribe(dummy, language='en', beam_size=1)
    except Exception:
        pass


def start_listener(callback) -> None:
    global _running
    _running = True
    threading.Thread(target=_warmup, daemon=True).start()
    threading.Thread(target=_listener_loop, args=(callback,), daemon=True).start()


def stop_listener() -> None:
    global _running
    _running = False
