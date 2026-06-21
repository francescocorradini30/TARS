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
MAX_UTTERANCE_CHUNKS = int(SAMPLE_RATE * 12 / CHUNK_SAMPLES)  # force-end after
                                  # ~12s even without silence, so a runaway blob
                                  # (mic never goes quiet) still gets processed.

# Mid-speech barge-in: while TARS is talking, periodically transcribe a short
# rolling window of the in-progress utterance and fire the interrupt the instant
# the wake word appears — instead of waiting for the whole sentence to finish.
BARGE_CHECK_CHUNKS   = 12         # ~0.38s between checks (throttle)
BARGE_WINDOW_CHUNKS  = 94         # ~3.0s of recent audio fed to each check. The
                                  # wake word almost always sits at the START of a
                                  # barge-in ("TARS, let's talk about..."); a 1.5s
                                  # window let it scroll out of view before a check
                                  # caught it, so the interrupt missed and TARS kept
                                  # talking over you. 3s keeps it in frame for ~8
                                  # checks → the stop fires reliably. base.en greedy
                                  # on 3s is still well under BARGE_CHECK_TIMEOUT.
BARGE_CHECK_TIMEOUT  = 3.0        # if a check runs longer than this, assume it
                                  # hung and allow new checks (self-heal) so a
                                  # single stuck transcribe can't permanently
                                  # disable mid-speech detection.
# Dedicated lightweight model for barge checks. Kept SEPARATE from the main en/it
# model: running the rolling check concurrently on the shared ctranslate2 model
# hung it and killed mid-speech detection for the rest of the session. base.en is
# small + fast and only has to spot the wake word.
BARGE_WHISPER_MODEL  = "base.en"

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
# Hotwords bias the decoder toward these tokens so Whisper stops rendering the
# wake word and common address forms as similar-sounding everyday words
# ("hey TARS" -> "Cheers"). Passed ONLY to the English pass: biasing the Italian
# pass would distort the en/it addressee logprob comparison the gate relies on.
HOTWORDS = "TARS, hey TARS, TARS are you there"

# Same bias for the mid-speech barge check. The check runs on the small base.en
# model, which without this hint mis-transcribes the wake word and silently fails
# to interrupt — so TARS only stops at end-of-utterance (the big model's pass)
# instead of the instant you say "TARS". MUST stay in sync with the wake word.
BARGE_HOTWORDS = "TARS"

WAKE_RE = re.compile(
    r'\b(?:'
    r'tars?|tarz|tarce|tarso|tarsi|tarsh|thars?|tarus|tarcy|'  # T-initial variants
    r'dars?|darz|darce|darso|darsi|darsh|dhars?|darus|darcy|'  # D-initial (Whisper softens T→D often)
    r'taz|daz|terz|terce|derz|derce|'                           # rarer mishearings
    r'cheers?|chars?|chers?|tchars?'                            # "hey TARS" → "cheers/chars/chers"
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


# _speaker_busy_until: monotonic timestamp until which TARS is expected to still
# be producing sound (LLM streaming or queued WAVs playing on JS side). Used to
# decide whether the wake word should barge in on the current response.
# pipeline_active is kept for log diagnostics only — not used for gating.
# Barge-in itself is driven by _barge_in_cb (registered by main.py), which cancels
# the running pipeline and tells JS to drop its audio — no shared event here.
pipeline_active: bool = False
_speaker_busy_until: float = 0.0
_speaker_lock = threading.Lock()
_barge_in_cb = None
_barge_checking: bool = False     # True while a mid-speech wake-word check runs
_barge_check_started: float = 0.0  # monotonic start of the in-flight check

_model: WhisperModel | None = None
_barge_model: WhisperModel | None = None  # separate instance for barge checks
# Serializes model construction. Without it the warmup thread + the en and it
# transcribe futures of the very first utterance all race into WhisperModel(),
# each allocating the full model on the GPU at once (3x ~1.6GB for turbo) -> CUDA
# OOM. The lock makes construction happen exactly once; later callers reuse it.
# Shared by both models so turbo and base.en never allocate on CUDA simultaneously.
_model_load_lock = threading.Lock()
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
    """Register a callable invoked when the wake word fires during active speaker
    output. main.py uses this to cancel the running pipeline and call
    interruptStream() so the browser silences any audio still playing or queued."""
    global _barge_in_cb
    _barge_in_cb = cb


def _do_barge_in(req: Request, reason: str) -> None:
    """Stop the in-flight response right now: clear our speaker-busy timer and
    invoke the registered callback (which cancels the pipeline + dumps JS audio)."""
    reset_speaker_busy()
    if _barge_in_cb is not None:
        _barge_in_cb()
    req.event("barge-in", reason)


def _strip_wake(text: str) -> str:
    """Remove the wake word so we can tell a bare 'tars' (just stop & listen)
    from a real command like 'tars, tell me a joke'."""
    return WAKE_RE.sub("", text).strip(" ,.!?-").strip()


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        with _model_load_lock:
            if _model is None:  # re-check: another thread may have loaded it while we waited
                compute = "float16" if WHISPER_DEVICE == "cuda" else "int8"
                _model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=compute)
    return _model


def _get_barge_model() -> WhisperModel:
    global _barge_model
    if _barge_model is None:
        with _model_load_lock:
            if _barge_model is None:
                compute = "float16" if WHISPER_DEVICE == "cuda" else "int8"
                _barge_model = WhisperModel(BARGE_WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=compute)
    return _barge_model


def _transcribe(audio: np.ndarray, language: str,
                hotwords: str | None = None) -> tuple[str, float]:
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
        hotwords=hotwords,
    )
    segs = list(segments)
    text = " ".join(s.text for s in segs).strip()
    score = sum(s.avg_logprob for s in segs) / len(segs) if segs else -10.0
    return text, score


def _transcribe_fast(audio: np.ndarray) -> str:
    """Single-pass, greedy English transcription used only for mid-speech
    wake-word detection. Runs on the dedicated lightweight model so it never
    contends with the main en/it transcribes; beam_size=1 keeps it fast enough
    to run every ~0.4s without starving the audio loop."""
    segments, _ = _get_barge_model().transcribe(
        audio,
        language='en',
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=200),
        condition_on_previous_text=False,
        beam_size=1,
        hotwords=BARGE_HOTWORDS,
    )
    return " ".join(s.text for s in segments).strip()


def _barge_check(window: np.ndarray, req: Request) -> None:
    """Worker thread: if the rolling window contains the wake word and TARS is
    still speaking, interrupt immediately. The full utterance keeps accumulating
    in the listener and is transcribed + processed normally at its end, so no
    spoken word is lost."""
    global _barge_checking
    try:
        if not speaker_active():
            return
        text = _transcribe_fast(window)
        if text and WAKE_RE.search(text) and not _is_noise(text) and speaker_active():
            req.barged_in = True
            _do_barge_in(req, f"wake word mid-speech '{text[:40]}'")
    finally:
        _barge_checking = False


def _start_barge_check(window: np.ndarray, req: Request) -> None:
    global _barge_checking, _barge_check_started
    _barge_checking = True
    _barge_check_started = time.monotonic()
    threading.Thread(target=_barge_check, args=(window, req), daemon=True).start()


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
    # Is this utterance aimed at TARS by *context* rather than by the language gate?
    # Two cases: a confirmed mid-speech barge-in, or speech captured while TARS was
    # still talking. In both, the WAKE WORD decides — not the en/it comparison — so
    # the Italian pass is dead weight. Skipping it halves transcription latency on
    # exactly the path that hurts most: cutting TARS off to say something new.
    directed = req.barged_in or speaker_active()
    try:
        with req.phase("stt-transcribe") as p:
            # faster_whisper releases the GIL during ctranslate2 inference, so the
            # en/it passes overlap when we actually need both.
            en_future = _transcribe_pool.submit(_transcribe, audio, 'en', HOTWORDS)
            if directed:
                # English-only: the addressee is already settled, so don't pay for
                # the Italian pass (which on a single GPU serializes behind en).
                en_text, en_score = en_future.result()
                it_text, it_score = "", -10.0
                p.note(f"en={en_score:.2f} '{en_text}' | it=skipped (directed)")
            else:
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
    # Reuse the up-front context decision: if we treated this as directed-at-TARS
    # (single English pass), the accept/drop branch must agree. Recomputing
    # speaker_active() here could flip False mid-call (TARS's audio just ended) and
    # send a wake-word-less, English-only utterance into the idle gate with a bogus
    # it_score — so anchor to the same flag instead.
    busy = directed

    def _accept_barge(reason: str) -> None:
        # ChatGPT-style: a barge-in answers the WHOLE prompt we just captured.
        # If the rolling check hasn't already stopped TARS, stop it now.
        if speaker_active():
            _do_barge_in(req, reason)
        # Bare "tars" with no command → just stop and keep listening, no reply.
        if not _strip_wake(en_text):
            req.event("drop", "bare wake word — stopped, listening")
            return
        req.event("accept", reason)
        callback(en_text, req)

    # The rolling check already barged in mid-utterance → always answer it.
    if req.barged_in:
        _accept_barge("barge-in prompt")
        return

    # While TARS is speaking, ONLY the wake word "tars" interrupts. Anything else
    # the mic catches during playback is ignored so TARS isn't cut off by random
    # talk. (Headphones: the mic never hears TARS, so this only ever sees you.)
    if busy:
        if has_wake:
            _accept_barge("wake word during speech")
        else:
            req.event("drop", "speaker active without wake word")
        return

    # Idle: normal addressee gate — English = talking to TARS, Italian = ignore.
    # But if the English transcript carries the wake word, the user is clearly
    # addressing TARS — accept regardless of how the en/it logprobs landed. This is
    # the documented WAKE_RE tiebreaker, which was previously only wired into the
    # busy/barge-in path: a short "hey TARS" that scored marginally better as
    # Italian got wrongly dropped here.
    if has_wake:
        req.event("accept", "wake word (idle tiebreaker)")
        callback(en_text, req)
        return
    if it_score > en_score:
        req.event("drop", f"it wins (en={en_score:.2f} it={it_score:.2f})")
        return
    if not en_text or _is_noise(en_text):
        req.event("drop", "en empty or noise")
        return
    req.event("accept")
    callback(en_text, req)


def _listener_loop_inner(callback) -> None:
    pre_buffer:    list = []
    utterance:     list = []
    in_speech            = False
    silence_count        = 0
    chunks_since_check   = 0
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
                    chunks_since_check = 0
                    req = Request()
                    # Note: we do NOT barge in on raw VAD energy. Energy alone is
                    # noisy (keyboard, breath all trigger it); interrupting on it
                    # kills real responses with false positives. Barge-in fires
                    # only once Whisper confirms the wake word — instantly via the
                    # rolling _barge_check, or at utterance end in _process_utterance.
                    req.event("vad-speech-start", f"speaker_active={speaker_active()}")
            else:
                utterance.append(chunk)
                if not is_speech:
                    silence_count += 1
                else:
                    silence_count = 0

                ended_on_silence = silence_count >= SILENCE_CHUNKS
                if ended_on_silence or len(utterance) >= MAX_UTTERANCE_CHUNKS:
                    in_speech = False
                    audio = np.concatenate(utterance)
                    audio_sec = len(audio) / SAMPLE_RATE
                    utterance = []
                    pre_buffer = []
                    silence_count = 0
                    assert req is not None
                    tag = "audio" if ended_on_silence else "max-dur"
                    req.event("vad-utterance-end", f"{tag} {audio_sec:.2f}s")
                    if len(audio) >= MIN_AUDIO_SAMPLES:
                        _dispatch_or_queue(req, audio, callback)
                    else:
                        req.event("drop", f"audio too short ({audio_sec:.2f}s)")
                    req = None
                    continue

                # Still mid-utterance: if TARS is talking over us, poll a rolling
                # window for the wake word so "tars" cuts it off instantly instead
                # of at end-of-sentence. The utterance keeps growing for the normal
                # end-of-turn transcription. A check that runs past BARGE_CHECK_TIMEOUT
                # is treated as hung so it can't permanently disable detection.
                checking = _barge_checking and (
                    time.monotonic() - _barge_check_started < BARGE_CHECK_TIMEOUT)
                if speaker_active() and not checking:
                    chunks_since_check += 1
                    if chunks_since_check >= BARGE_CHECK_CHUNKS:
                        chunks_since_check = 0
                        window = np.concatenate(utterance[-BARGE_WINDOW_CHUNKS:])
                        _start_barge_check(window, req)
                elif not speaker_active():
                    chunks_since_check = 0


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
    try:
        _get_barge_model().transcribe(dummy, language='en', beam_size=1)
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
