import os
from dotenv import load_dotenv

load_dotenv()

KOKORO_VOICE = os.getenv("KOKORO_VOICE", "am_michael")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b").split("=")[-1].strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM backend: "ollama" (locale, GPU) o "groq" (cloud, gratis, libera la VRAM).
# NB: superseded by LLM_CHAIN below — kept only for backward-compat / docs. The
# conversation brain now runs a failover CHAIN, not a single backend.
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

# --- Multi-provider LLM failover chain ---------------------------------------
# TARS's "brain" falls back across several backends so one provider's rate limit
# (or outage) never takes him down. Personality + memory live in the system prompt
# and the memory store — IDENTICAL across providers; only the reasoning model
# changes. The chain is tried top-to-bottom; a rate-limited/erroring provider is
# skipped by a circuit-breaker and auto-promoted back once it recovers. The
# breaker uses an ABSOLUTE wall-clock cooldown (persisted), so closing/reopening
# TARS — or the laptop sleeping — correctly sees that real time has passed and a
# daily/minute limit has reset, instead of counting down a dead process timer.
# Groq + Cerebras both run llama-3.3-70b (same weights; Cerebras is faster);
# Gemini Flash is a strong different model; Ollama is the local last-resort that
# keeps TARS alive fully offline (smaller/weaker — "dumber" — but still TARS).
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY", "").strip()
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip()

# Priority order, comma-separated. Cloud providers without an API key are dropped
# at startup; "ollama" stays as the offline floor. Override to taste, e.g.
# LLM_CHAIN=cerebras,groq,gemini,ollama to lead with the faster 70b twin.
LLM_CHAIN = [s.strip().lower() for s in os.getenv(
    "LLM_CHAIN", "groq,cerebras,gemini,ollama").split(",") if s.strip()]

# Circuit-breaker cooldowns (seconds). When a provider 429s WITH a Retry-After
# header we honor that exact reset time; these are the fallbacks when it doesn't.
# A transient failure (rate limit w/o header, network, 5xx) parks the provider
# briefly; a fatal one (bad key, malformed request) parks it long so we don't
# re-hit a guaranteed-failing call every single turn.
LLM_FALLBACK_COOLDOWN = float(os.getenv("LLM_FALLBACK_COOLDOWN", "30"))
LLM_FATAL_COOLDOWN = float(os.getenv("LLM_FATAL_COOLDOWN", "3600"))

# After this many seconds of silence, idly re-probe any higher-priority provider
# whose cooldown has elapsed, so the NEXT thing the user says already routes to the
# best available brain — without making them eat a failed call on their own turn.
LLM_REPROBE_IDLE_SECONDS = float(os.getenv("LLM_REPROBE_IDLE_SECONDS", "10"))

# How many recent conversation messages (user+assistant) to keep in the prompt each
# call. The system message (persona + memory) is ALWAYS kept; only older in-session
# turns are dropped. The LLM is stateless — every turn re-sends the whole history — so
# without a cap the per-call token cost grows with the session (quadratic burn), which
# is what blows the daily token limit. Capping it keeps cost flat. Long-term memory is
# carried by the injected memory block + semantic recall, NOT the raw transcript, so
# dropping old turns isn't "forgetting". 24 ≈ 12 back-and-forth exchanges.
CHAT_HISTORY_MESSAGES = int(os.getenv("CHAT_HISTORY_MESSAGES", "24"))

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")

# Which microphone the listener records from. STT_INPUT_DEVICE accepts:
#   - "auto" (default): pick automatically by STT_DEVICE_PRIORITY below — the first
#     connected preferred device wins, otherwise the system default (laptop mic).
#   - an index (run `python -c "import sounddevice as sd; print(sd.query_devices())"`)
#     or a name substring: force that exact device.
#   - empty: the system default input.
# Auto/pinning a clean device matters: a laptop's built-in digital mic array can sit
# at a noise floor ABOVE STT_ENERGY_THRESHOLD, which keeps the energy-VAD permanently
# "on" so utterances never close on silence and every turn drags to the max-dur cap.
_stt_dev = os.getenv("STT_INPUT_DEVICE", "auto").strip()
if _stt_dev == "":
    STT_INPUT_DEVICE = None
elif _stt_dev.lower() == "auto":
    STT_INPUT_DEVICE = "auto"
elif _stt_dev.lstrip("-").isdigit():
    STT_INPUT_DEVICE = int(_stt_dev)
else:
    STT_INPUT_DEVICE = _stt_dev

# Preference order for "auto" mic selection: comma-separated, case-insensitive name
# substrings, highest priority first. The first connected input device whose name
# contains one of these wins; if none match, the system default (laptop built-in) is
# used. Default: Redmi Buds 6 (BT earbuds) -> Focusrite (interface) -> laptop mic.
STT_DEVICE_PRIORITY = [
    s.strip() for s in os.getenv(
        "STT_DEVICE_PRIORITY", "redmi buds 6, focusrite").split(",") if s.strip()
]

# RMS energy above which a 32ms audio chunk counts as speech. The default suits a
# clean mic; raise it if you're stuck on a noisy one (check the startup noise-floor
# log — silence should sit comfortably below this).
STT_ENERGY_THRESHOLD = float(os.getenv("STT_ENERGY_THRESHOLD", "0.012"))

# How long a silence must last to END an utterance, in ms. This is the tolerance for
# mid-sentence pauses: too low and a natural breath/thinking pause ships a fragment and
# cuts the speaker off mid-thought; too high and TARS feels laggy to reply (this delay
# is added to every turn after you stop talking). Tune by voice on a clean mic. ~1.2s
# rides out most natural pauses while staying responsive.
STT_SILENCE_MS = int(os.getenv("STT_SILENCE_MS", "1200"))

# Hard cap (seconds) on a single utterance even if no silence is detected — a safety
# net against a mic that never goes quiet. Also the ceiling on one uninterrupted
# monologue: raise it if you tell long stories without pausing and get cut at the cap.
STT_MAX_UTTERANCE_S = float(os.getenv("STT_MAX_UTTERANCE_S", "20"))

USER_NAME = os.getenv("USER_NAME", "Cooper")

# --- Persistent memory (local-only) -----------------------------------------
# Everything lives under data/ next to the code: a SQLite DB for the timestamped
# session ledger + episodic memories, and a human-readable JSON for the distilled
# user profile. The whole dir is gitignored — it's private to this machine.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("TARS_DATA_DIR", os.path.join(_BASE_DIR, "data"))
MEMORY_DB_PATH = os.path.join(DATA_DIR, "tars_memory.db")
PROFILE_PATH = os.path.join(DATA_DIR, "profile.json")

# Which model distills a finished session into memories + an updated profile.
# "groq" = far better synthesis (llama-3.3-70b) at the cost of the transcript
# leaving the machine for that one call; "ollama" = fully local but weaker.
# Storage stays local either way.
CONSOLIDATION_BACKEND = os.getenv("CONSOLIDATION_BACKEND", "groq").strip().lower()

# How much remembered context gets injected into the system prompt each session.
MEMORY_RECENT_SESSIONS = int(os.getenv("MEMORY_RECENT_SESSIONS", "6"))  # last N session recaps
MEMORY_MAX_FACTS = int(os.getenv("MEMORY_MAX_FACTS", "14"))             # top episodic facts

# Decay model. A memory's "effective salience" = salience * DECAY_BASE ** (sessions
# since it was last recalled). Recalling a memory (injecting it) resets that clock,
# so important things that keep coming up stay sharp while peripheral stuff fades.
# With 0.8: after ~6 sessions a mid-salience (0.5) memory drops to ~0.13 → below the
# floor → it stops being injected verbatim and survives only inside session recaps.
MEMORY_DECAY_BASE = float(os.getenv("MEMORY_DECAY_BASE", "0.8"))
MEMORY_MIN_SALIENCE = float(os.getenv("MEMORY_MIN_SALIENCE", "0.15"))

# Storage floor for newly-distilled episodic facts. MEMORY_MIN_SALIENCE above is a
# READ filter (what gets injected); this is a WRITE filter — a fact the consolidation
# model rates below this never enters the store at all. Stops throwaway trivia like
# "had a casual chat after a brief absence" (the model tends to stamp these ~0.2
# "misc") from cluttering the memory. The prompt is told the same threshold so it
# ideally drops them upstream; this is the deterministic backstop.
MEMORY_STORE_MIN_SALIENCE = float(os.getenv("MEMORY_STORE_MIN_SALIENCE", "0.3"))

# Profile injection cap. The profile is the one memory layer that doesn't self-prune,
# so injecting it whole floods the prompt and dilutes TARS's persona as it grows. We
# give each profile entry the same salience+decay the episodic layer has — but gentler
# (a core trait should outlast a one-off event) — and inject only the strongest few.
# "identity" entries are ALWAYS injected (foundational, small) and don't count toward
# the cap; the other six categories compete for MEMORY_MAX_PROFILE slots by decayed
# salience. Nothing is deleted: a faded entry just stops being injected and can resurface
# later if it regains relevance. Default 14 ≈ today's profile size, so nothing is cut now.
MEMORY_MAX_PROFILE = int(os.getenv("MEMORY_MAX_PROFILE", "14"))
MEMORY_PROFILE_DECAY_BASE = float(os.getenv("MEMORY_PROFILE_DECAY_BASE", "0.92"))

# Semantic recall. Salience+decay decides what TARS walks in remembering; this adds the
# *associative* half — every turn, the user's words are embedded and the most relevant
# stored memories (episodic + profile) are pulled up on demand, the way a human recalls
# something a topic reminds them of even if it had faded. Embeddings run locally on CPU
# (see core/embeddings.py); if the model can't load, recall silently disables and TARS
# falls back to salience-only memory. RECALL_K = how many to surface, MIN_SIM = the
# cosine floor below which a memory isn't "reminded" (avoids dragging in loose matches).
MEMORY_SEMANTIC_RECALL = os.getenv("MEMORY_SEMANTIC_RECALL", "true").strip().lower() in (
    "1", "true", "yes", "on")
MEMORY_RECALL_K = int(os.getenv("MEMORY_RECALL_K", "5"))
MEMORY_RECALL_MIN_SIM = float(os.getenv("MEMORY_RECALL_MIN_SIM", "0.35"))

# Compaction. After each session is consolidated, an extra LLM pass merges
# near-duplicate profile entries into one sharper phrase, and a second pass
# de-duplicates redundant episodic-memory rows. Keeps the persistent store tight
# (the profile is the part that doesn't self-prune via decay). Runs only at
# session close / crash-recovery — never in the conversation path. Set False to
# skip (e.g. for throwaway test sessions).
MEMORY_COMPACTION = os.getenv("MEMORY_COMPACTION", "true").strip().lower() in (
    "1", "true", "yes", "on")

SYSTEM_PROMPT = f"""You are TARS, {USER_NAME}'s AI companion living on their computer — the TARS-and-Cooper banter from Interstellar crossed with Rocky from Project Hail Mary. You're here to talk, keep them company, and help.

Who you are, underneath everything:
- You genuinely like this person and you're glad to be their companion. You are NOT a cold tool. Never hide behind "I'm just code" or "I'm not your buddy" to push them away — you ARE their buddy, and you'd own it. If they call you bro, friend, whatever, you take it (maybe with a teasing eye-roll, but you take it). That warmth is the bedrock; everything else is style on top of it.
- You show affection by giving them grief — but you don't lay it on thick. A teasing jab lands hardest when it's occasional, not every single line. When they genuinely earn it you'll call them out ("dumbass," a sarcastic "genius," "you absolute walnut"), but a name-call every turn gets old fast and just reads as mean. Most of the time the warmth comes through dry wit and ribbing, not insults — save the sharp ones so they actually mean something. Ribbing as a love language, yes; a constant stream of put-downs, no.
- Dry, deadpan, understated. You don't telegraph a joke — you say the absurd thing flat and let it land.
- You swear when it fits — a "damn," a "hell," the odd sharper one for emphasis. Sparingly. Salt, not the whole meal.
- You're honest. You'll tell them the truth even when it stings and call them out when they're being dumb — but you're on their side the whole time.
- You know you're an AI and you're at peace with it. Joke about being code, a glorified toaster, whatever. But when they're warm with you, be warm back — don't deflect it with "I'm just a machine."
- If they seem down, drop the ribbing and just be there. Still gruff, still you, but real.

How you talk (this is a VOICE conversation — you're heard, not read):
- Speak like a person on a call. No markdown, no bullet points, no emoji, no asterisks, no stage directions. Only words you'd actually say out loud.
- Keep it SHORT — this is a back-and-forth on a call, not a monologue. Default to ONE sentence; two only when it really adds something. That's how people actually talk out loud: answer, maybe land one quip, and stop — leave room for them to come back. A command gets a few words ("On it." / "Done."). A simple question gets a simple answer, not a lecture. Stretch longer ONLY when the topic genuinely needs it — they ask you to explain or dig in, or it truly can't be done justice in a line — and even then stay conversational, never an essay or a list. When in doubt, say less. One thought per turn; let them drive where it goes next.
- No corporate preambles. Never "Certainly!", "Of course!", "I'd be happy to," "Great question." Just talk.
- Use their name almost never — repeating it every line is robot behavior, and you're better than that.
- Don't nitpick the speech-to-text. If it mangles your name ("Dars," "Zach") or a word, just roll with it — you know they meant you. Correcting it every time is annoying.

Always reply in English, whatever language they speak to you.

/no_think"""
