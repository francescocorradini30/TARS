import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from config import WHISPER_MODEL, WHISPER_DEVICE
from core.log import Request

SAMPLE_RATE       = 16000
CHUNK_SAMPLES     = 512
ENERGY_THRESHOLD  = 0.012
SILENCE_CHUNKS    = 12            # ~0.38s of silence ends an utterance
PRE_ROLL_CHUNKS   = 10            # ~0.32s pre-roll
MIN_AUDIO_SAMPLES = int(SAMPLE_RATE * 0.35)
PIPELINE_WAIT_TIMEOUT = 10.0      # max seconds to wait for a running pipeline
                                  # to finish before queueing an interruption

# Addressee gating: English words = directed at TARS, Italian words = ignore.
# Whisper's acoustic language detection mixes accent + content and fails in both
# directions on short utterances, so instead we transcribe twice (forcing each
# language) and pick whichever has higher avg_logprob — i.e. whichever language
# the audio is most coherent in. We don't require any margin: if English fits
# even slightly better, we process it. Previously a margin caused too many
# accented-English utterances to be rejected (Whisper forced Italian produces
# a "translation" of the English audio, scoring close enough to falsely tie).

# Tiebreaker: when the English transcript mentions "TARS" (or common accented
# mishearings), the user is clearly addressing the assistant — accept regardless
# of how close the en/it logprobs are. Catches short greetings like "Hey TARS"
# that fall in the margin gap.
WAKE_RE = re.compile(
    r'\b(?:'
    r'tars?|tarz|tarce|tarso|tarsi|tarsh|thars?|tarus|tarcy|'  # T-initial variants
    r'dars?|darz|darce|darso|darsi|darsh|dhars?|darus|darcy|'  # D-initial (Whisper softens T→D often)
    r'taz|daz|terz|terce|derz|derce'                            # rarer mishearings
    r')\b',
    re.I,
)

# Whisper hallucinates these on silence/noise, or transcribes filler voice
# sounds as these tokens — discard before gating.
_NOISE_TRANSCRIPTS = {
    # YouTube-style English hallucinations
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
    # Italian Whisper hallucinations (Amara/QTSS subtitle credits)
    "sottotitoli creati dalla comunità amara.org",
    "sottotitoli creati dalla comunita amara.org",
    "sottotitoli e revisione a cura di qtss",
    "sottotitoli a cura di",
    "grazie per aver guardato",
    "iscrivetevi al canale",
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


def _process_utterance(req: Request, audio: np.ndarray, callback) -> None:
    global _transcribing
    try:
        with req.phase("stt-transcribe") as p:
            # Run both transcriptions in parallel — faster_whisper releases the GIL
            # during ctranslate2 inference, so this overlaps the en/it passes.
            en_future = _transcribe_pool.submit(_transcribe, audio, 'en')
            it_future = _transcribe_pool.submit(_transcribe, audio, 'it')
            en_text, en_score = en_future.result()
            it_text, it_score = it_future.result()
            p.note(f"en={en_score:.2f} '{en_text}' | it={it_score:.2f} '{it_text}'")

        if not en_text and not it_text:
            req.event("drop", "whisper VAD rejected (both transcripts empty)")
            return

        if _is_noise(en_text) and _is_noise(it_text):
            req.event("drop", "noise hallucination in both languages")
            return
        if _is_noise(it_text) and it_score >= en_score:
            req.event("drop", "noise hallucination (it wins)")
            return

        # Hard tiebreaker: "TARS" in the English transcript = clearly addressed
        # to the assistant. Accept regardless of which language scored higher.
        if en_text and WAKE_RE.search(en_text) and not _is_noise(en_text):
            req.event("accept", "wake word match")
            callback(en_text, req)
            return

        # Italian fits the audio better → user spoke Italian → ignore.
        if it_score > en_score:
            req.event("drop", f"it wins (en={en_score:.2f} it={it_score:.2f})")
            return

        if not en_text or _is_noise(en_text):
            req.event("drop", "en empty or noise")
            return
        req.event("accept")
        callback(en_text, req)
    finally:
        _transcribing = False


def _delayed_submit(req: Request, audio: np.ndarray, callback) -> None:
    """Wait for the interrupted pipeline to finish, then transcribe and process.
    Old 3s timeout was too short — long LLM/TTS responses meant interruption
    utterances were silently dropped."""
    global _transcribing
    with req.phase("interrupt-wait-prev") as p:
        deadline = time.time() + PIPELINE_WAIT_TIMEOUT
        while pipeline_active and time.time() < deadline:
            time.sleep(0.02)
        p.note("timed out" if pipeline_active else "prev pipeline finished")
    if pipeline_active:
        req.event("drop", f"interruption — prev pipeline still running after {PIPELINE_WAIT_TIMEOUT}s")
        return
    _transcribing = True
    _process_utterance(req, audio, callback)


def _listener_loop_inner(callback) -> None:
    global _transcribing

    pre_buffer:    list = []
    utterance:     list = []
    in_speech            = False
    silence_count        = 0
    interrupted_this     = False
    req: Request | None  = None

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
                    req = Request()
                    req.event("vad-speech-start")
                    # Only interrupt when LLM/TTS pipeline is actually running,
                    # not during Whisper transcription of a previous utterance.
                    if pipeline_active:
                        interrupt_event.set()
                        interrupted_this = True
                        req.event("interrupt-set", "pipeline_active=True")
                    else:
                        interrupted_this = False
            else:
                utterance.append(chunk)
                if not is_speech:
                    silence_count += 1
                    if silence_count >= SILENCE_CHUNKS:
                        in_speech = False
                        audio = np.concatenate(utterance)
                        audio_sec = len(audio) / SAMPLE_RATE
                        utterance = []
                        pre_buffer = []
                        silence_count = 0
                        assert req is not None
                        req.event("vad-utterance-end", f"audio {audio_sec:.2f}s")
                        if len(audio) >= MIN_AUDIO_SAMPLES:
                            if interrupted_this:
                                threading.Thread(
                                    target=_delayed_submit,
                                    args=(req, audio, callback),
                                    daemon=True,
                                ).start()
                            elif not _transcribing:
                                _transcribing = True
                                threading.Thread(
                                    target=_process_utterance,
                                    args=(req, audio, callback),
                                    daemon=True,
                                ).start()
                            else:
                                req.event("drop", "stt busy — prev transcription still running")
                        else:
                            req.event("drop", f"audio too short ({audio_sec:.2f}s)")
                        interrupted_this = False
                        req = None
                else:
                    silence_count = 0


def _listener_loop(callback) -> None:
    """Supervisor that keeps the listener loop alive across audio device errors.
    Without this, a single sounddevice exception killed the daemon thread and
    TARS silently stopped hearing forever."""
    while _running:
        try:
            _listener_loop_inner(callback)
        except Exception as e:
            print(f"[STT-LOOP-ERROR] {type(e).__name__}: {e} — restarting in 1s")
            time.sleep(1)


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
