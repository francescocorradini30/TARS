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
SILENCE_CHUNKS    = 22            # ~0.70s of silence ends an utterance — long
                                  # enough to ride out natural mid-sentence
                                  # pauses (thinking, breath, soft phonemes)
                                  # so we don't ship a fragment and cut the user
                                  # off. Lower = snappier turns but more cutoffs.
PRE_ROLL_CHUNKS   = 10            # ~0.32s pre-roll
MIN_AUDIO_SAMPLES = int(SAMPLE_RATE * 0.35)

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


# interrupt_event: set by _process_utterance when a new utterance is accepted
# while TARS is still speaking. The LLM loop and drain thread in main.py poll
# this to abort.
# _speaker_busy_until: monotonic timestamp until which TARS is expected to still
# be producing sound (LLM streaming or queued WAVs playing on JS side). Used to
# decide whether an accepted utterance should barge in on the current response.
# pipeline_active is kept for log diagnostics only — not used for gating.
interrupt_event  = threading.Event()
pipeline_active: bool = False
_speaker_busy_until: float = 0.0
_speaker_lock = threading.Lock()
_barge_in_cb = None

_model: WhisperModel | None = None
_running: bool     = False
_transcribing: bool = False   # True only during Whisper inference
_transcribe_pool = ThreadPoolExecutor(max_workers=2)

# Latest-wins STT queue. While Whisper is busy on one utterance, a newly finished
# utterance isn't dropped — it's stashed here and run as soon as the current
# transcription frees the gate. Only ONE pending slot: if you speak a third time
# before the second runs, the newest wins (older queued speech is stale anyway).
# _stt_lock guards both _transcribing and _pending_utterance together.
_stt_lock = threading.Lock()
_pending_utterance: tuple | None = None   # (Request, np.ndarray) or None

# 300ms tail buffer past the computed end of the last queued WAV — covers JS
# audio start latency, speaker reverb, and small clock drift between Python and
# the browser's AudioContext. Empirically large enough to suppress echo without
# making barge-in feel laggy.
SPEAKER_TAIL_BUFFER = 0.3


def speaker_active() -> bool:
    return time.monotonic() < _speaker_busy_until


def extend_speaker_busy(audio_duration: float) -> None:
    """Push the speaker-busy deadline forward by audio_duration seconds, measured
    from whichever is later: now or the existing deadline. Called when a WAV is
    queued to JS — multiple back-to-back WAVs chain correctly."""
    global _speaker_busy_until
    with _speaker_lock:
        now = time.monotonic()
        start = now if now > _speaker_busy_until else _speaker_busy_until
        _speaker_busy_until = start + audio_duration + SPEAKER_TAIL_BUFFER


def reset_speaker_busy() -> None:
    """Zero out the speaker-busy deadline. Called on wake-word barge-in once JS
    has been told to clear its audio queue."""
    global _speaker_busy_until
    with _speaker_lock:
        _speaker_busy_until = 0.0


def set_barge_in_callback(cb) -> None:
    """Register a callable invoked when a wake-word interrupt fires during
    active speaker output. main.py uses this to call interruptStream() JS so
    the browser silences any audio still playing or queued."""
    global _barge_in_cb
    _barge_in_cb = cb


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
        # beam 5 kept for accuracy (accented English + reliable en/it gate
        # logprobs). The 4-6s transcribes were VRAM spill from the medium model,
        # not beam search — switching to the small model (fits in VRAM) is what
        # fixes the latency, so we can afford the accurate beam here.
        beam_size=5,
    )
    segs = list(segments)
    text = " ".join(s.text for s in segs).strip()
    score = sum(s.avg_logprob for s in segs) / len(segs) if segs else -10.0
    return text, score


def _dispatch_or_queue(req: Request, audio: np.ndarray, callback) -> None:
    """Start a transcription if Whisper is free, otherwise stash this utterance
    as the pending 'latest' one (superseding any older pending). Called from the
    listener at every utterance boundary."""
    global _transcribing, _pending_utterance
    with _stt_lock:
        if _transcribing:
            if _pending_utterance is not None:
                _pending_utterance[0].event("drop", "superseded by newer utterance")
            _pending_utterance = (req, audio)
            req.event("queued", "stt busy — will run after current transcription")
            return
        _transcribing = True
    threading.Thread(target=_process_utterance,
                     args=(req, audio, callback), daemon=True).start()


def _release_or_chain(callback) -> None:
    """Called the instant a transcription's Whisper phase finishes. If a newer
    utterance was queued while we were busy, run it now (keeping the gate held);
    otherwise free the gate. This is what turns 'stt busy' drops into a queue."""
    global _transcribing, _pending_utterance
    with _stt_lock:
        nxt = _pending_utterance
        _pending_utterance = None
        if nxt is None:
            _transcribing = False
            return
        # keep _transcribing True and hand the gate to the queued utterance
    nreq, naudio = nxt
    threading.Thread(target=_process_utterance,
                     args=(nreq, naudio, callback), daemon=True).start()


def _process_utterance(req: Request, audio: np.ndarray, callback) -> None:
    try:
        with req.phase("stt-transcribe") as p:
            # Run both transcriptions in parallel — faster_whisper releases the GIL
            # during ctranslate2 inference, so this overlaps the en/it passes.
            en_future = _transcribe_pool.submit(_transcribe, audio, 'en')
            it_future = _transcribe_pool.submit(_transcribe, audio, 'it')
            en_text, en_score = en_future.result()
            it_text, it_score = it_future.result()
            p.note(f"en={en_score:.2f} '{en_text}' | it={it_score:.2f} '{it_text}'")
    finally:
        # Free / hand off the Whisper gate the instant inference is done — BEFORE
        # running the (potentially multi-second) LLM+TTS pipeline via callback().
        # The callback runs synchronously in this thread, so holding the gate
        # until then would drop every utterance spoken during a response (incl.
        # the "tars" wake word) before it could be transcribed — no barge-in.
        # _release_or_chain also picks up any utterance queued while we were busy.
        _release_or_chain(callback)

    if not en_text and not it_text:
        req.event("drop", "whisper VAD rejected (both transcripts empty)")
        return

    if _is_noise(en_text) and _is_noise(it_text):
        req.event("drop", "noise hallucination in both languages")
        return
    if _is_noise(it_text) and it_score >= en_score:
        req.event("drop", "noise hallucination (it wins)")
        return

    has_wake = bool(en_text and WAKE_RE.search(en_text) and not _is_noise(en_text))
    busy = speaker_active()

    # No echo gate: the user runs TARS on headphones, so the mic never hears
    # TARS's own output. That assumption is what makes mid-speech barge-in work
    # — without echo, a real interrupting utterance closes normally (the old
    # gate waited for silence that never came while TARS echoed into the mic).
    def _maybe_interrupt(reason: str) -> None:
        # If TARS is still producing sound when we accept an utterance, abort the
        # running LLM loop and tell JS to dump its audio queue so the new turn
        # starts immediately.
        if busy:
            interrupt_event.set()
            reset_speaker_busy()
            if _barge_in_cb is not None:
                _barge_in_cb()
            req.event("interrupt-set", reason)

    # Wake word path: always accept, interrupting if TARS is mid-response.
    if has_wake:
        _maybe_interrupt("wake word during speech")
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
    # English directed at TARS while it's speaking also barges in — you don't
    # have to say "tars" to cut it off, just talk to it.
    _maybe_interrupt("english during speech")
    req.event("accept")
    callback(en_text, req)


def _listener_loop_inner(callback) -> None:
    pre_buffer:    list = []
    utterance:     list = []
    in_speech            = False
    silence_count        = 0
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
                    # Note: we do NOT set interrupt_event here. VAD energy alone
                    # is a noisy signal (echo, keyboard, breath all trigger it);
                    # interrupting on raw energy kills real responses with false
                    # positives. The interrupt fires later, in _process_utterance,
                    # only when Whisper has confirmed real wake-word speech.
                    req.event("vad-speech-start", f"speaker_active={speaker_active()}")
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
                            _dispatch_or_queue(req, audio, callback)
                        else:
                            req.event("drop", f"audio too short ({audio_sec:.2f}s)")
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
