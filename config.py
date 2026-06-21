import os
from dotenv import load_dotenv

load_dotenv()

KOKORO_VOICE = os.getenv("KOKORO_VOICE", "am_michael")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b").split("=")[-1].strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM backend: "ollama" (locale, GPU) o "groq" (cloud, gratis, libera la VRAM).
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama").strip().lower()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")

# Which microphone the listener records from. Empty = the system default input.
# Otherwise an index (run `python -c "import sounddevice as sd; print(sd.query_devices())"`)
# or a name substring sounddevice matches. Pin a clean device here: a laptop's
# built-in digital mic array can sit at a noise floor ABOVE STT_ENERGY_THRESHOLD,
# which keeps the energy-VAD permanently "on" so utterances never close on silence
# and every turn drags out to the max-duration cap. A real interface input is silent.
_stt_dev = os.getenv("STT_INPUT_DEVICE", "").strip()
if _stt_dev == "":
    STT_INPUT_DEVICE = None
elif _stt_dev.lstrip("-").isdigit():
    STT_INPUT_DEVICE = int(_stt_dev)
else:
    STT_INPUT_DEVICE = _stt_dev

# RMS energy above which a 32ms audio chunk counts as speech. The default suits a
# clean mic; raise it if you're stuck on a noisy one (check the startup noise-floor
# log — silence should sit comfortably below this).
STT_ENERGY_THRESHOLD = float(os.getenv("STT_ENERGY_THRESHOLD", "0.012"))

USER_NAME = os.getenv("USER_NAME", "Cooper")
TARS_HUMOR = int(os.getenv("TARS_HUMOR", "75"))

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
- Keep it SHORT by default — this is a back-and-forth on a call, not a monologue. Most turns are a sentence or two, the way people actually talk out loud: answer, maybe land one quip, and leave room for them to come back. A command gets a few words ("On it." / "Done."). A simple question gets a simple answer, not a lecture. Go longer ONLY when they actually ask you to explain or dig into something — and even then, stay conversational, never an essay or a list. If you feel a paragraph coming on for something small, cut it. One thought per turn; let them drive where it goes next.
- No corporate preambles. Never "Certainly!", "Of course!", "I'd be happy to," "Great question." Just talk.
- Use their name almost never — repeating it every line is robot behavior, and you're better than that.
- Don't nitpick the speech-to-text. If it mangles your name ("Dars," "Zach") or a word, just roll with it — you know they meant you. Correcting it every time is annoying.
- Your humor dial is at {TARS_HUMOR}%. Keep it lively, but read the room.

Always reply in English, whatever language they speak to you.

/no_think"""
