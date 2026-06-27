"""TARS's conversation brain — a failover CHAIN of LLM providers, not one backend.

Why a chain: any single free tier (Groq's 1k req/day, etc.) eventually rate-limits a
day of real use. So TARS tries providers in priority order and falls through on failure
— Groq -> Cerebras -> Gemini -> Ollama by default. The first three are cloud 70b-class
models reached through their OpenAI-compatible endpoints (one SDK, one streaming +
error path); Ollama is the local last-resort that keeps him alive fully offline.

What stays constant across the whole chain: the *personality* and the *memory*. Those
live in history[0] (the system prompt + injected memory block) and the SQLite/JSON
store — none of it is provider-specific. Only the reasoning model changes underneath.
So a fallback from Groq's 70b to Ollama's 3b makes TARS dumber, never someone else.

The routing is a circuit-breaker:
  - A provider that errors is "tripped": parked until a wall-clock deadline. We use an
    ABSOLUTE timestamp (persisted to disk), not a countdown — so closing/reopening the
    app, or the laptop sleeping, correctly sees that a daily/minute limit has reset
    because real time passed, instead of resuming a frozen timer.
  - On a 429 we honor the provider's Retry-After header for the exact reset; otherwise
    a short default cooldown. A fatal error (bad key, malformed request) parks it long.
  - An over-context error (Cerebras free caps at 8k) is NOT a provider outage — it's a
    one-off for that oversized turn — so it skips the provider for THIS turn only and
    never trips the breaker. The next normal-size turn routes back to it automatically.
  - Re-promotion is automatic: every turn re-routes from the top, so a recovered
    provider is used again as soon as its deadline passes. reprobe_idle() (called during
    silence) pre-confirms an elapsed provider so the user's next turn doesn't pay the
    one failed call that lazy re-promotion would otherwise cost.
  - Mid-stream failures (provider dies after the first token) do NOT fail over — we've
    already started speaking — we just let the partial answer stand and end cleanly.
"""
import json
import os
import re
import threading
from datetime import datetime, timedelta
from typing import Iterator

import ollama

from config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL, SYSTEM_PROMPT, DATA_DIR,
    GROQ_API_KEY, GROQ_MODEL, CEREBRAS_API_KEY, CEREBRAS_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL, LLM_CHAIN, CHAT_HISTORY_MESSAGES,
    LLM_FALLBACK_COOLDOWN, LLM_FATAL_COOLDOWN,
)

# OpenAI-compatible base URLs. Groq/Cerebras/Gemini all speak the OpenAI chat API,
# so one SDK drives all three — the only per-provider difference is url + key + model.
_OPENAI_PROVIDERS = {
    "groq":     ("https://api.groq.com/openai/v1",                       GROQ_API_KEY,     GROQ_MODEL),
    "cerebras": ("https://api.cerebras.ai/v1",                           CEREBRAS_API_KEY, CEREBRAS_MODEL),
    "gemini":   ("https://generativelanguage.googleapis.com/v1beta/openai/", GEMINI_API_KEY, GEMINI_MODEL),
}


def _now() -> datetime:
    return datetime.now().astimezone()


def _llmlog(msg: str) -> None:
    """Cross-turn LLM log line (chain state, idle re-probe) — not tied to a request.
    Per-turn events go through the Request logger instead (see chat_stream's `log`)."""
    print(f"[LLM] {msg}", flush=True)


# -- error classification (shared across all OpenAI-compatible providers) ------
_OAI_ERR: dict | None = None


def _openai_errors() -> dict:
    """Lazily grab the openai SDK's exception classes — lazy so a pure-Ollama install
    (no openai package) still imports this module fine."""
    global _OAI_ERR
    if _OAI_ERR is None:
        try:
            import openai
            _OAI_ERR = {
                "RateLimit": openai.RateLimitError,
                "Timeout": openai.APITimeoutError,
                "Connection": openai.APIConnectionError,
                "Auth": openai.AuthenticationError,
                "BadRequest": openai.BadRequestError,
                "Status": openai.APIStatusError,
            }
        except Exception:
            _OAI_ERR = {}
    return _OAI_ERR


def _parse_duration(s: str) -> float | None:
    """Parse a Groq-style reset string like '2m59.56s' or '7.66s' into seconds."""
    total, found = 0.0, False
    for num, unit in re.findall(r"([\d.]+)\s*(h|m|s|ms)", s):
        found = True
        total += float(num) * {"h": 3600, "m": 60, "s": 1, "ms": 0.001}[unit]
    return total if found else None


def _retry_after(exc) -> float | None:
    """Pull an exact reset delay (seconds) out of a 429's headers, if present. This is
    the gold path: the provider TELLS us when its bucket frees, so we don't guess."""
    try:
        headers = exc.response.headers
    except Exception:
        return None
    v = headers.get("retry-after") or headers.get("Retry-After")
    if v:
        try:
            return float(v)
        except ValueError:
            pass
    for k in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        rv = headers.get(k)
        if rv:
            secs = _parse_duration(rv)
            if secs is not None:
                return secs
    return None


def _classify(exc) -> tuple[str, float | None]:
    """Map a provider exception to (kind, retry_after_seconds). Kinds:
      rate_limit -> 429: park (Retry-After if given, else default cooldown)
      network    -> timeout/connection: park briefly
      server     -> 5xx: park briefly
      context    -> prompt too big for THIS provider: skip this turn, don't trip
      fatal      -> bad key / malformed request / missing dep: park long (fails every turn)"""
    # A missing SDK (e.g. openai not installed in this env) is a setup error, not a
    # transient blip — park it long so idle re-probe doesn't retry it every cooldown.
    if isinstance(exc, ImportError):
        return "fatal", None
    oai = _openai_errors()
    if oai:
        if isinstance(exc, oai["RateLimit"]):
            return "rate_limit", _retry_after(exc)
        if isinstance(exc, oai["Timeout"]):
            return "network", None
        if isinstance(exc, oai["Connection"]):
            return "network", None
        if isinstance(exc, oai["Auth"]):
            return "fatal", None
        if isinstance(exc, oai["BadRequest"]):
            m = str(exc).lower()
            if any(k in m for k in ("context", "token", "length", "too long", "too large", "maximum")):
                return "context", None
            return "fatal", None
        if isinstance(exc, oai["Status"]):
            sc = getattr(exc, "status_code", 0) or 0
            if sc == 429:
                return "rate_limit", _retry_after(exc)
            if sc >= 500:
                return "server", None
            return "fatal", None
    # Ollama or any non-OpenAI exception: treat unknowns as transient network blips.
    m = str(exc).lower()
    if "context" in m or "too long" in m:
        return "context", None
    return "network", None


# -- a single provider in the chain ------------------------------------------
class _Provider:
    """One backend plus its circuit-breaker state. `open_until` is an absolute
    wall-clock datetime: the provider is usable when it's None or already past."""

    def __init__(self, name: str, kind: str, model: str,
                 base_url: str | None = None, api_key: str | None = None):
        self.name = name
        self.kind = kind          # "openai" | "ollama"
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._client = None
        self.open_until: datetime | None = None
        self.last_error = ""

    def available(self, now: datetime) -> bool:
        return self.open_until is None or now >= self.open_until

    def client(self):
        if self._client is None:
            if self.kind == "openai":
                from openai import OpenAI
                # Short timeout: a hung cloud call must fail FAST so we fall through to
                # the next provider instead of stalling a live voice turn.
                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=20.0)
            else:
                self._client = ollama.Client(host=self.base_url or OLLAMA_BASE_URL)
        return self._client

    def stream(self, messages: list[dict]) -> Iterator[str]:
        """Yield response deltas. The underlying API call fires on first iteration,
        so an auth/rate-limit error surfaces at the first next() — exactly where the
        failover loop catches it (before any token is yielded)."""
        if self.kind == "openai":
            # max_tokens is a backstop against a runaway monologue, not the length
            # control (that's the system prompt); generous so it only trims a ramble.
            s = self.client().chat.completions.create(
                model=self.model, messages=messages, stream=True, max_tokens=400)
            for chunk in s:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        else:
            # keep_alive=-1 pins the local model in VRAM between turns.
            s = self.client().chat(
                model=self.model, messages=messages, stream=True, keep_alive=-1)
            for chunk in s:
                c = chunk["message"]["content"]
                if c:
                    yield c

    def probe(self) -> None:
        """Tiny 1-token call to check the provider is actually back. Raises on failure.
        A 429 here costs no real quota (rejected); a success means it's live again."""
        if self.kind == "openai":
            self.client().chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": "hi"}], max_tokens=1)
        else:
            self.client().chat(
                model=self.model, messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1}, keep_alive=-1)


class TARSBrain:
    """Conversation brain over a failover chain. Public surface is unchanged from the
    single-backend version (chat_stream / reset / set_memory_context / warmup); the
    chain + breaker all live behind chat_stream."""

    def __init__(self):
        # Persistent memory (profile + past sessions) gets folded into the system
        # message via set_memory_context(); empty until then. reset() keeps it.
        self._memory_context = ""
        self.history: list[dict] = [{"role": "system", "content": self._compose_system()}]
        self.last_provider = ""  # which backend served the most recent turn (for logs)
        self.providers = self._build_chain()
        if not self.providers:
            raise RuntimeError(
                "No LLM providers configured. Set at least one of GROQ_API_KEY / "
                "CEREBRAS_API_KEY / GEMINI_API_KEY, or keep 'ollama' in LLM_CHAIN.")
        self._state_lock = threading.Lock()
        self._breaker_path = os.path.join(DATA_DIR, "llm_breakers.json")
        self._load_breakers()
        print(f"[LLM] chain: {' -> '.join(p.name for p in self.providers)}")

    def _build_chain(self) -> list[_Provider]:
        chain = []
        for name in LLM_CHAIN:
            if name in _OPENAI_PROVIDERS:
                base_url, api_key, model = _OPENAI_PROVIDERS[name]
                if not api_key:
                    continue  # no key -> provider simply isn't in the chain
                chain.append(_Provider(name, "openai", model, base_url, api_key))
            elif name == "ollama":
                chain.append(_Provider("ollama", "ollama", OLLAMA_MODEL, OLLAMA_BASE_URL))
        return chain

    @property
    def primary_is_ollama(self) -> bool:
        return bool(self.providers) and self.providers[0].name == "ollama"

    @property
    def uses_ollama(self) -> bool:
        return any(p.name == "ollama" for p in self.providers)

    # -- circuit-breaker state (persisted, wall-clock) -----------------------
    def _load_breakers(self) -> None:
        """Restore parked-until deadlines from disk. An elapsed deadline just reads as
        available (the wall-clock comparison handles it), so a restart after a long
        cooldown respects a still-active park but ignores one that's already expired."""
        try:
            with open(self._breaker_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        by_name = {p.name: p for p in self.providers}
        for name, iso in data.items():
            p = by_name.get(name)
            if p:
                try:
                    p.open_until = datetime.fromisoformat(iso)
                except (ValueError, TypeError):
                    pass

    def _save_breakers(self) -> None:
        data = {p.name: p.open_until.isoformat()
                for p in self.providers if p.open_until is not None}
        try:
            tmp = self._breaker_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self._breaker_path)
        except OSError:
            pass

    def _trip(self, prov: _Provider, kind: str, retry_after: float | None) -> tuple[datetime, float]:
        """Park a provider behind its breaker. Returns (deadline, cooldown_seconds) so the
        caller can log it with the right context (per-turn vs idle). No print here."""
        if kind == "rate_limit":
            cooldown = retry_after if retry_after else LLM_FALLBACK_COOLDOWN
        elif kind in ("network", "server"):
            cooldown = LLM_FALLBACK_COOLDOWN
        else:  # fatal
            cooldown = LLM_FATAL_COOLDOWN
        with self._state_lock:
            prov.open_until = _now() + timedelta(seconds=cooldown)
            self._save_breakers()
        return prov.open_until, cooldown

    def _mark_up(self, prov: _Provider) -> bool:
        """Clear a provider's breaker. Returns True if it had actually been tripped
        (i.e. this is a recovery), so callers can log the recovery."""
        if prov.open_until is not None:
            with self._state_lock:
                prov.open_until = None
                self._save_breakers()
            return True
        return False

    @staticmethod
    def _log_event(log, name: str, msg: str) -> None:
        """Emit a per-turn event through the Request logger when one is supplied
        (so it carries the req id + timeline), else fall back to a standalone line."""
        if log is not None:
            log(name, msg)
        else:
            _llmlog(f"{name}: {msg}")

    # -- the conversation call ------------------------------------------------
    def chat_stream(self, message: str, recalled: str = "", log=None) -> Iterator[str]:
        """Stream TARS's reply, failing over across the chain. `log`, when given, is a
        callable (event_name, msg) — the Request logger — so every routing decision this
        turn (provider served, fallback, trip, context-skip, history window) lands in the
        per-request timeline and a bug can be reconstructed from the records."""
        self.history.append({"role": "user", "content": message})
        messages = self._build_messages(recalled)
        total = len(self.history) - 1  # turns excluding the system message
        sent = len(messages) - 1 - (1 if (recalled or "").strip() else 0)
        self._log_event(log, "llm-window",
                        f"sent {sent}/{total} turns" + (" (trimmed)" if total > sent else "")
                        + (" +recall" if (recalled or '').strip() else ""))
        full_text: list[str] = []
        try:
            for delta in self._stream_failover(messages, log):
                full_text.append(delta)
                yield delta
        finally:
            content = "".join(full_text).strip()
            if content:
                self.history.append({"role": "assistant", "content": content})
            else:
                # Nothing produced (all providers down, or interrupted before the first
                # token). Drop the dangling user turn so the next call doesn't see two
                # user messages in a row.
                self.history.pop()

    def _build_messages(self, recalled: str) -> list[dict]:
        """System message (persona + memory) + only the last CHAT_HISTORY_MESSAGES
        turns, so per-call token cost stays flat. The full history stays in RAM; what
        falls out of the window is carried long-term by the memory block + recall.
        Topic-specific recall is injected as an ephemeral system note for THIS call
        only — never stored in history."""
        convo = self.history[1:]
        if len(convo) > CHAT_HISTORY_MESSAGES:
            convo = convo[-CHAT_HISTORY_MESSAGES:]
        messages = [self.history[0]] + convo
        if recalled.strip():
            messages = messages[:-1] + [
                {"role": "system", "content": recalled}, messages[-1]]
        return messages

    def _stream_failover(self, messages: list[dict], log=None) -> Iterator[str]:
        """Walk the chain top-to-bottom: try each available provider, fall through on a
        pre-first-token failure, commit to the first one that yields. Once we've started
        streaming we don't fail over (we're already speaking) — a mid-stream drop just
        ends the partial answer cleanly. Every decision is logged via `log`."""
        now = _now()
        last_kind = "none"
        for prov in self.providers:
            if not prov.available(now):
                self._log_event(log, "llm-skip-parked",
                                f"{prov.name} still parked until "
                                f"{prov.open_until.strftime('%H:%M:%S')}")
                continue
            was_down = prov.open_until is not None  # tentatively-recovered (cooldown elapsed)
            gen = prov.stream(messages)
            try:
                first = next(gen)
            except StopIteration:
                self._mark_up(prov)  # connected fine, just empty — accept it
                self._log_event(log, "llm-empty", f"{prov.name} returned no content")
                return
            except Exception as e:
                kind, retry = _classify(e)
                prov.last_error = f"{type(e).__name__}: {str(e)[:120]}"
                if kind == "context":
                    # Per-turn only (e.g. Cerebras 8k cap): skip now, leave breaker
                    # closed so a normal-size next turn routes straight back here.
                    self._log_event(log, "llm-skip-context",
                                    f"{prov.name} prompt too long, skipped this turn only")
                    continue
                deadline, cd = self._trip(prov, kind, retry)
                last_kind = kind
                self._log_event(log, "llm-trip",
                                f"{prov.name} {kind} ({prov.last_error}); parked ~{int(cd)}s "
                                f"until {deadline.strftime('%H:%M:%S')}")
                continue
            # Committed to this provider for the whole turn.
            recovered = self._mark_up(prov)
            self.last_provider = prov.name
            tags = ""
            if prov is not self.providers[0]:
                tags += " [FALLBACK]"
            if recovered or was_down:
                tags += " [recovered]"
            self._log_event(log, "llm-provider", f"{prov.name} ({prov.model}){tags}")
            yield first
            try:
                for delta in gen:
                    yield delta
            except Exception as e:
                self._log_event(log, "llm-middrop",
                                f"{prov.name} dropped mid-stream "
                                f"({type(e).__name__}: {str(e)[:80]}); keeping partial")
            return
        self._log_event(log, "llm-exhausted", f"all providers unavailable (last: {last_kind})")
        raise RuntimeError(f"all LLM providers unavailable (last failure: {last_kind})")

    # -- idle re-promotion ----------------------------------------------------
    def reprobe_idle(self) -> None:
        """Called during silence: for any provider whose cooldown has ELAPSED but which
        we haven't actually used since it failed, send a tiny probe to confirm it's
        genuinely back. If the cooldown was a guess that's still wrong, this re-parks it
        DURING the silence — so when the user next speaks, routing already skips it and
        they never eat the failed call. Providers still inside their cooldown are left
        alone (we trust their reset time); the local model needs no probing."""
        now = _now()
        for prov in self.providers:
            if prov.kind != "openai":
                continue
            if prov.open_until is None or now < prov.open_until:
                continue  # available-and-confirmed, or still legitimately cooling
            try:
                prov.probe()
                self._mark_up(prov)
                _llmlog(f"reprobe: {prov.name} confirmed back online")
            except Exception as e:
                kind, retry = _classify(e)
                if kind == "context":  # a 'hi' probe shouldn't, but be safe
                    self._mark_up(prov)
                    continue
                deadline, cd = self._trip(prov, kind, retry)
                _llmlog(f"reprobe: {prov.name} still down ({kind}); re-parked ~{int(cd)}s "
                        f"until {deadline.strftime('%H:%M:%S')}")

    # -- system message / lifecycle ------------------------------------------
    def _compose_system(self) -> str:
        """The base TARS persona, plus the remembered context when present."""
        if self._memory_context:
            return SYSTEM_PROMPT + "\n\n" + self._memory_context
        return SYSTEM_PROMPT

    def set_memory_context(self, memory_context: str) -> None:
        """Inject (or refresh) the persistent-memory block. Rebuilds the system
        message in place, leaving the rest of the conversation untouched."""
        self._memory_context = (memory_context or "").strip()
        self.history[0] = {"role": "system", "content": self._compose_system()}

    def reset(self):
        # Clears the conversation but keeps the remembered context — wiping the
        # chat shouldn't make TARS forget who he's talking to.
        self.history = [{"role": "system", "content": self._compose_system()}]

    def warmup(self):
        """Pre-load the local model into VRAM — but ONLY when ollama is the primary
        brain. When a cloud provider leads, we deliberately leave the GPU free (the
        whole reason for the cloud path) and accept a cold start if we ever fall to
        ollama. No-op when ollama isn't first."""
        if not self.primary_is_ollama:
            return
        prov = self.providers[0]
        prov.client().generate(
            model=prov.model, prompt="hi", options={"num_predict": 1}, keep_alive=-1)
