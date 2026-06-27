"""Per-request phase timers for diagnosing TARS pipeline latency.

Every utterance gets a Request id and a t0 anchored at speech start. All log
lines include the id and the relative timestamp, so a single conversation turn
can be reconstructed from interleaved STT/LLM/TTS output. Use event() for
instantaneous markers (decisions, drops, interrupts) and phase() as a context
manager around work with measurable duration."""
import sys
import threading
import time
from contextlib import contextmanager
from itertools import count

_id_counter = count(1)
_id_lock = threading.Lock()
_print_lock = threading.Lock()


def _next_id() -> int:
    with _id_lock:
        return next(_id_counter)


def fmt_dur(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


def _emit(line: str) -> None:
    with _print_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


class Phase:
    __slots__ = ("_note",)

    def __init__(self) -> None:
        self._note = ""

    def note(self, text: str) -> None:
        self._note = text


class Request:
    __slots__ = ("id", "t0", "barged_in", "speech_end")

    def __init__(self) -> None:
        self.id = _next_id()
        self.t0 = time.perf_counter()
        # Set True by the rolling wake-word check when this utterance interrupts
        # TARS mid-response, so _process_utterance always answers the full prompt.
        self.barged_in = False
        # Stamped when the user STOPS talking (vad-utterance-end). t() measures from
        # speech START — so for a long utterance it mostly counts how long the user
        # spoke, not how responsive TARS was. reply_t() is the latency they actually
        # feel: from when they stopped to now.
        self.speech_end: float | None = None

    def t(self) -> float:
        return time.perf_counter() - self.t0

    def mark_speech_end(self) -> None:
        self.speech_end = time.perf_counter()

    def reply_t(self) -> float:
        base = self.speech_end if self.speech_end is not None else self.t0
        return time.perf_counter() - base

    def event(self, name: str, msg: str = "") -> None:
        tail = f" — {msg}" if msg else ""
        _emit(f"[req={self.id} t+{self.t():5.2f}s] {name}{tail}")

    @contextmanager
    def phase(self, name: str):
        p = Phase()
        start = time.perf_counter()
        try:
            yield p
        finally:
            dur = time.perf_counter() - start
            tail = f" — {p._note}" if p._note else ""
            _emit(f"[req={self.id} t+{self.t():5.2f}s] {name} {fmt_dur(dur)}{tail}")
