"""TARS persistent memory — the thing that makes him a companion instead of a
stateless chatbot.

Four layers, all local:
  1. Session ledger   — every app run is a session with a start/end timestamp,
     turn count and a one-line recap. The backbone for "remember the other day?".
  2. Turn journal     — the raw user/assistant lines, written as they happen.
     Doubles as crash insurance: if the app dies mid-session, the next launch
     finds the un-consolidated turns and distills them anyway. Nothing is lost.
  3. Episodic memory  — salient facts/events extracted from a finished session,
     each timestamped and given a salience that DECAYS over later sessions so old
     peripheral stuff fades to "vaguely remembered" while what matters stays sharp.
  4. User profile     — the slowly-distilled model of the person (likes, dislikes,
     what winds them up, recurring asks, the tone of the relationship). A plain
     JSON file so it stays human-readable and TARS can one day edit it himself.

The episodic/session layers live in SQLite (cheap timestamped queries + decay);
the profile lives in JSON (readable, editable). A finished session is turned into
memories + a refreshed profile by one LLM "consolidation" pass (see consolidate()).

Concurrency: main.py touches this from a few threads (startup, the pipeline
thread per turn, shutdown). We keep ONE connection opened check_same_thread=False
and serialize every access behind a lock. The workload is tiny, so this is plenty.
"""
import json
import os
import sqlite3
import threading
from datetime import datetime

import numpy as np

from config import (
    MEMORY_DB_PATH, PROFILE_PATH, USER_NAME,
    CONSOLIDATION_BACKEND, GROQ_API_KEY, GROQ_MODEL,
    CEREBRAS_API_KEY, CEREBRAS_MODEL, GEMINI_API_KEY, GEMINI_MODEL, LLM_CHAIN,
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    MEMORY_RECENT_SESSIONS, MEMORY_MAX_FACTS,
    MEMORY_DECAY_BASE, MEMORY_MIN_SALIENCE, MEMORY_STORE_MIN_SALIENCE,
    MEMORY_COMPACTION,
    MEMORY_MAX_PROFILE, MEMORY_PROFILE_DECAY_BASE,
    MEMORY_SEMANTIC_RECALL, MEMORY_RECALL_K, MEMORY_RECALL_MIN_SIM,
)
from core.embeddings import Embedder

# Profile shape. Seven categories, each a list of entries distilled over time. On
# disk each entry is a small object {text, salience, last_seen} so the profile can be
# injection-capped and decayed exactly like the episodic layer (see _render_profile);
# legacy profiles stored as bare strings are upgraded transparently on load. The LLM
# passes only ever see/return the plain text — we re-attach salience/recency ourselves.
_EMPTY_PROFILE = {
    "identity": [],            # who they are: name, role, where they live, etc.
    "preferences": [],         # how they like TARS to behave / answer
    "likes": [],               # things they enjoy / care about
    "dislikes": [],            # things they don't like
    "annoyances": [],          # things that genuinely wind them up
    "recurring_requests": [],  # stuff they keep asking TARS to do
    "relationship_notes": [],  # the texture of the bond, inside jokes, tone
}

# Default salience a freshly-distilled entry gets, per category. Identity is near-
# certain to matter (and is pinned in rendering anyway); a passing "like" matters less.
# These only seed NEW entries — recency (last_seen) + decay do the rest over time.
_PROFILE_DEFAULT_SALIENCE = {
    "identity": 0.95, "preferences": 0.7, "likes": 0.6, "dislikes": 0.65,
    "annoyances": 0.7, "recurring_requests": 0.6, "relationship_notes": 0.65,
}

_KIND_LABELS = {
    "preference": "prefers", "event": "happened", "like": "likes",
    "dislike": "dislikes", "annoyance": "annoyed by", "request": "asked for",
    "identity": "about them", "misc": "note",
}


def _now() -> str:
    """Local wall-clock time, ISO-8601 with timezone offset. Start/end of every
    session is stamped with this — exactly the date+time the user asked for."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _fmt_when(iso: str) -> str:
    """ISO timestamp -> friendly 'Mon 21 Jun, 14:30' for prompt injection."""
    try:
        return datetime.fromisoformat(iso).strftime("%a %d %b, %H:%M")
    except (ValueError, TypeError):
        return iso


def _coerce_entry(x, default_salience: float, current_seq: int) -> dict | None:
    """Normalize one profile entry into the on-disk object shape. Accepts both the
    new {text, salience, last_seen} form and a legacy bare string (upgraded with the
    category default and the current session as its recency). Returns None for empties
    so callers can drop them. Salience is clamped to [0, 1]."""
    if isinstance(x, dict):
        text = str(x.get("text", "")).strip()
        if not text:
            return None
        try:
            salience = float(x.get("salience", default_salience))
        except (TypeError, ValueError):
            salience = default_salience
        try:
            last_seen = int(x.get("last_seen", current_seq))
        except (TypeError, ValueError):
            last_seen = current_seq
    else:
        text = str(x).strip()
        if not text:
            return None
        salience, last_seen = default_salience, current_seq
    return {"text": text, "salience": min(1.0, max(0.0, salience)), "last_seen": last_seen}


def _profile_texts(profile: dict) -> dict:
    """The profile reduced to plain string lists — what the LLM passes see. They never
    deal with salience/recency; we re-attach that ourselves when applying their output."""
    return {k: [e["text"] for e in (profile.get(k) or [])] for k in _EMPTY_PROFILE}


# How many of the current session's turns to keep recall-able in memory at once.
# One float32[384] per turn (~1.5KB), so even a marathon session is trivial; the cap
# just bounds a runaway. These are ephemeral (this process only) — the raw turns are
# already journaled and become proper memories at consolidation.
_SESSION_RECALL_CAP = 200


class MemoryStore:
    def __init__(self, db_path: str = MEMORY_DB_PATH, profile_path: str = PROFILE_PATH):
        self.db_path = db_path
        self.profile_path = profile_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self.session_id: int | None = None  # set by start_session()
        # Live recall of the CURRENT session: (turn_text, embedding) for what the user
        # has said this session, so it can be associatively recalled even after it
        # scrolls out of the trimmed chat history. Reuses the embedding recall_relevant
        # already computes, so it costs nothing extra. Reset each session.
        self._session_vectors: list[tuple] = []

    # -- schema ---------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at   TEXT NOT NULL,
                    ended_at     TEXT,
                    turn_count   INTEGER NOT NULL DEFAULT 0,
                    summary      TEXT,
                    consolidated INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS turns (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    ts         TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id   INTEGER NOT NULL,
                    created_at   TEXT NOT NULL,
                    kind         TEXT NOT NULL DEFAULT 'misc',
                    content      TEXT NOT NULL,
                    salience     REAL NOT NULL DEFAULT 0.5,
                    recall_count INTEGER NOT NULL DEFAULT 0,
                    last_seq     INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
                CREATE INDEX IF NOT EXISTS idx_mem_session  ON memories(session_id);
                CREATE TABLE IF NOT EXISTS profile_vectors (
                    text      TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL
                );
                """
            )
            # Migration: episodic memories gained a vector for semantic recall. Older
            # DBs predate the column — add it (backfilled lazily by ensure_embeddings).
            cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(memories)")}
            if "embedding" not in cols:
                self._conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
            self._conn.commit()

    # -- session lifecycle ----------------------------------------------------
    def start_session(self) -> dict:
        """Open a new session, stamping its start time. Returns {id, started_at}."""
        ts = _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions (started_at) VALUES (?)", (ts,)
            )
            self._conn.commit()
            self.session_id = cur.lastrowid
        self._session_vectors = []  # fresh session = fresh live-recall cache
        return {"id": self.session_id, "started_at": ts}

    def log_turn(self, role: str, content: str) -> None:
        """Append one conversation turn to the journal as it happens. Cheap, and
        it's what makes a crashed session recoverable next launch."""
        content = (content or "").strip()
        if not content or self.session_id is None:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (session_id, role, content, ts) VALUES (?,?,?,?)",
                (self.session_id, role, content, _now()),
            )
            self._conn.execute(
                "UPDATE sessions SET turn_count = turn_count + 1 WHERE id = ?",
                (self.session_id,),
            )
            self._conn.commit()

    def end_session(self) -> None:
        """Stamp the end time of the current session. Idempotent; safe to call
        even if the session had zero turns."""
        if self.session_id is None:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                (_now(), self.session_id),
            )
            self._conn.commit()

    # -- context injection (read-side) ---------------------------------------
    def _effective_salience(self, salience: float, last_seq: int, current_seq: int,
                            base: float = MEMORY_DECAY_BASE) -> float:
        elapsed = max(0, current_seq - last_seq)
        return salience * (base ** elapsed)

    def build_context_block(self) -> str:
        """Assemble the memory the system prompt is augmented with at session start:
        the distilled profile, the last few session recaps, and the top episodic
        facts (after decay). This is *how TARS remembers* — it's injected once,
        up front, so a mid-conversation "remember the other day?" already has the
        recent recaps to lean on."""
        current_seq = self.session_id or 0
        profile = self.load_profile()

        with self._lock:
            prior_count = self._conn.execute(
                "SELECT COUNT(*) AS n FROM sessions WHERE id < ? AND turn_count > 0",
                (current_seq,),
            ).fetchone()["n"]
            recaps = self._conn.execute(
                "SELECT started_at, summary FROM sessions "
                "WHERE id < ? AND summary IS NOT NULL AND summary != '' "
                "ORDER BY id DESC LIMIT ?",
                (current_seq, MEMORY_RECENT_SESSIONS),
            ).fetchall()
            mem_rows = self._conn.execute(
                "SELECT id, kind, content, salience, last_seq FROM memories"
            ).fetchall()

        # First-ever meaningful conversation → a clean, deliberate intro.
        if prior_count == 0 and self._profile_is_empty(profile):
            return (
                "=== YOUR MEMORY ===\n"
                f"This is the very first time you're really meeting {USER_NAME}. "
                "You have no history with them yet. Pay attention — from now on "
                "you'll actually remember them between conversations.\n"
                "=== END MEMORY ==="
            )

        # Rank episodic facts by decayed salience; keep the strongest above the floor.
        ranked = []
        for r in mem_rows:
            eff = self._effective_salience(r["salience"], r["last_seq"], current_seq)
            if eff >= MEMORY_MIN_SALIENCE:
                ranked.append((eff, r))
        ranked.sort(key=lambda x: x[0], reverse=True)
        ranked = ranked[:MEMORY_MAX_FACTS]

        lines = [
            "=== YOUR MEMORY (persistent, real — not roleplay) ===",
            f"Right now it's {_fmt_when(_now())}. You and {USER_NAME} have talked "
            f"across {prior_count} earlier session(s). You actually remember them.",
        ]

        prof_lines, injected_profile = self._render_profile(profile, current_seq)
        if prof_lines:
            lines.append(f"\nWhat you know about {USER_NAME}:")
            lines.extend(prof_lines)

        if recaps:
            lines.append("\nRecent conversations:")
            for r in recaps:
                lines.append(f"- {_fmt_when(r['started_at'])}: {r['summary']}")

        if ranked:
            lines.append("\nThings you remember (older ones may be fuzzy):")
            for _eff, r in ranked:
                label = _KIND_LABELS.get(r["kind"], "note")
                lines.append(f"- ({label}) {r['content']}")

        lines.append(
            "\nUse this naturally in conversation — don't recite it like a list, "
            "don't announce that you 'have memory'. If they bring up something from "
            "before, you simply remember it. Let what you know about them shape how "
            "you talk to them."
        )
        lines.append("=== END MEMORY ===")

        # Recalling reinforces: the facts we just injected get their decay clock
        # reset, so things that stay relevant keep surfacing and don't fade.
        if ranked:
            ids = [r["id"] for _eff, r in ranked]
            with self._lock:
                self._conn.executemany(
                    "UPDATE memories SET last_seq = ?, recall_count = recall_count + 1 "
                    "WHERE id = ?",
                    [(current_seq, i) for i in ids],
                )
                self._conn.commit()

        # Same reinforcement for the profile: entries we actually injected this session
        # get their recency refreshed so they don't drift toward fading. Persist it
        # without rotating the .prev backup — that's reserved as insurance against a bad
        # consolidation pass, not a routine recency touch.
        if injected_profile:
            for e in injected_profile:
                e["last_seen"] = current_seq
            self._save_profile(profile, backup=False)

        return "\n".join(lines)

    # -- semantic recall (associative, query-driven) -------------------------
    def ensure_embeddings(self) -> None:
        """Make every memory recall-ready: embed episodic rows that lack a vector and
        the current profile entries, pruning vectors for profile text that no longer
        exists (merged/edited away). One-time-ish cost, meant to run at startup OFF the
        conversation path. Entirely best-effort — if the embedder can't load, it no-ops
        and recall stays disabled, TARS just runs on salience memory."""
        if not MEMORY_SEMANTIC_RECALL:
            return
        emb = Embedder.get()

        with self._lock:
            missing = self._conn.execute(
                "SELECT id, content FROM memories WHERE embedding IS NULL"
            ).fetchall()
        if missing:
            vecs = emb.encode([r["content"] for r in missing])
            if vecs is not None:
                with self._lock:
                    self._conn.executemany(
                        "UPDATE memories SET embedding = ? WHERE id = ?",
                        [(vecs[i].tobytes(), missing[i]["id"]) for i in range(len(missing))],
                    )
                    self._conn.commit()

        profile = self.load_profile()
        texts = []
        for k in _EMPTY_PROFILE:
            texts.extend(e["text"] for e in (profile.get(k) or []))
        texts = list(dict.fromkeys(texts))  # de-dupe, keep order
        with self._lock:
            have = {r["text"] for r in
                    self._conn.execute("SELECT text FROM profile_vectors").fetchall()}
        new = [t for t in texts if t not in have]
        if new:
            vecs = emb.encode(new)
            if vecs is not None:
                with self._lock:
                    self._conn.executemany(
                        "INSERT OR REPLACE INTO profile_vectors (text, embedding) VALUES (?, ?)",
                        [(new[i], vecs[i].tobytes()) for i in range(len(new))],
                    )
                    self._conn.commit()
        stale = [t for t in have if t not in set(texts)]
        if stale:
            with self._lock:
                self._conn.executemany(
                    "DELETE FROM profile_vectors WHERE text = ?", [(t,) for t in stale])
                self._conn.commit()

    def recall_relevant(self, query: str, k: int = MEMORY_RECALL_K) -> str:
        """The associative half of memory: given what the user just said, pull up the
        few stored memories (episodic + profile) most semantically similar to it, above
        a cosine floor. Surfacing a memory reinforces it (recall_count++, decay clock
        reset) — thinking about something makes it salient again, like a real mind.
        Returns a short context block to fold into THIS turn only, or '' if nothing
        clears the bar / recall is off. Cheap enough for the hot path (one short embed
        + a dot product over a few hundred vectors)."""
        if not MEMORY_SEMANTIC_RECALL or not (query or "").strip():
            return ""
        qv = Embedder.get().encode_one(query)
        if qv is None:
            return ""

        with self._lock:
            mem_rows = self._conn.execute(
                "SELECT id, kind, content, embedding FROM memories "
                "WHERE embedding IS NOT NULL"
            ).fetchall()
            prof_rows = self._conn.execute(
                "SELECT text, embedding FROM profile_vectors").fetchall()
            sess = list(self._session_vectors)  # snapshot BEFORE adding this turn

        scored = []  # (sim, source, mem_id, kind, text)
        for r in mem_rows:
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            scored.append((float(qv @ v), "mem", r["id"], r["kind"], r["content"]))
        for r in prof_rows:
            v = np.frombuffer(r["embedding"], dtype=np.float32)
            scored.append((float(qv @ v), "prof", None, "identity", r["text"]))
        for text, v in sess:
            scored.append((float(qv @ v), "sess", None, "session", text))

        # Remember THIS turn so later turns can recall it even after it scrolls out of
        # the trimmed chat history. Reuses the query embedding above — no extra cost.
        # Snapshot was taken before this append, so the turn never matches itself.
        with self._lock:
            self._session_vectors.append((query.strip(), qv))
            if len(self._session_vectors) > _SESSION_RECALL_CAP:
                del self._session_vectors[:-_SESSION_RECALL_CAP]

        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [s for s in scored if s[0] >= MEMORY_RECALL_MIN_SIM][:max(1, k)]
        if not picked:
            return ""

        # Reinforce the episodic memories we just associatively recalled.
        mem_ids = [s[2] for s in picked if s[1] == "mem"]
        if mem_ids:
            seq = self.session_id or 0
            with self._lock:
                self._conn.executemany(
                    "UPDATE memories SET last_seq = ?, recall_count = recall_count + 1 "
                    "WHERE id = ?", [(seq, i) for i in mem_ids])
                self._conn.commit()

        lines = ["(What they just said stirs these specific memories — only bring one up "
                 "if it actually fits, and never announce that you're 'recalling' anything:)"]
        for _sim, source, _id, kind, text in picked:
            if source == "mem":
                label = _KIND_LABELS.get(kind, "note")
            elif source == "sess":
                label = "they said earlier"
            else:
                label = "about them"
            lines.append(f"- ({label}) {text}")
        return "\n".join(lines)

    @staticmethod
    def _profile_is_empty(profile: dict) -> bool:
        return not any(profile.get(k) for k in _EMPTY_PROFILE)

    def _render_profile(self, profile: dict, current_seq: int) -> tuple[list[str], list[dict]]:
        """Render the profile for prompt injection, but CAPPED so it can't flood the
        prompt as it grows. "identity" is always shown (foundational, tiny). The other
        six categories compete for MEMORY_MAX_PROFILE slots ranked by decayed salience;
        entries below the salience floor are dropped from this injection (not deleted —
        they stay on disk and can resurface later). Returns the rendered lines plus the
        list of entry objects actually injected, so the caller can refresh their recency."""
        headers = {
            "identity": "Who they are", "preferences": "How they like you to be",
            "likes": "Likes", "dislikes": "Dislikes",
            "annoyances": "Winds them up", "recurring_requests": "Often asks you to",
            "relationship_notes": "Between you two",
        }
        selected = {k: [] for k in headers}
        injected: list[dict] = []

        # identity: pinned, never subject to the cap or decay.
        for e in profile.get("identity") or []:
            selected["identity"].append(e)
            injected.append(e)

        # everything else: rank by decayed salience, keep the strongest up to the cap.
        pool = []
        for key in headers:
            if key == "identity":
                continue
            for e in profile.get(key) or []:
                eff = self._effective_salience(
                    e["salience"], e["last_seen"], current_seq, MEMORY_PROFILE_DECAY_BASE)
                if eff >= MEMORY_MIN_SALIENCE:
                    pool.append((eff, key, e))
        pool.sort(key=lambda x: x[0], reverse=True)
        for _eff, key, e in pool[:MEMORY_MAX_PROFILE]:
            selected[key].append(e)
            injected.append(e)

        out = []
        for key, header in headers.items():
            items = [e["text"] for e in selected[key] if e["text"].strip()]
            if items:
                out.append(f"- {header}: " + "; ".join(items))
        return out, injected

    # -- profile JSON ---------------------------------------------------------
    def load_profile(self) -> dict:
        """Load the profile as {category: [ {text, salience, last_seen}, ... ]}. Legacy
        files with bare-string entries are upgraded in memory (and persisted on the next
        save). Only known keys are kept; empties are dropped."""
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {k: [] for k in _EMPTY_PROFILE}
        seq = self.session_id or 0
        out = {}
        for k in _EMPTY_PROFILE:
            default = _PROFILE_DEFAULT_SALIENCE.get(k, 0.6)
            entries = [_coerce_entry(x, default, seq) for x in (data.get(k) or [])]
            out[k] = [e for e in entries if e]
        return out

    def _save_profile(self, profile: dict, backup: bool = True) -> None:
        seq = self.session_id or 0
        clean = {}
        for k in _EMPTY_PROFILE:
            default = _PROFILE_DEFAULT_SALIENCE.get(k, 0.6)
            entries = [_coerce_entry(x, default, seq) for x in (profile.get(k) or [])]
            clean[k] = [e for e in entries if e]
        clean["updated_at"] = _now()
        # Keep one backup before overwriting — cheap insurance against a bad
        # consolidation pass silently dropping things the user told us. Routine recency
        # touches (build_context_block) pass backup=False so they don't burn the .prev.
        if backup and os.path.exists(self.profile_path):
            try:
                os.replace(self.profile_path, self.profile_path + ".prev")
            except OSError:
                pass
        tmp = self.profile_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.profile_path)

    # -- consolidation (write-side, LLM) -------------------------------------
    def consolidate_pending(self) -> int:
        """Distill every finished-but-not-yet-consolidated session into memories
        + a refreshed profile. Called at shutdown for the current session and at
        startup to mop up anything a crash left behind. Returns how many sessions
        were consolidated."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM sessions WHERE consolidated = 0 AND turn_count > 0 "
                "ORDER BY id ASC"
            ).fetchall()
        done = 0
        for r in rows:
            try:
                self._consolidate_one(r["id"])
                done += 1
            except Exception as e:
                # Never let a flaky LLM call crash shutdown. Leave it un-consolidated
                # so the next launch retries it; the raw turns are safe in the journal.
                print(f"[MEMORY] consolidation failed for session {r['id']}: "
                      f"{type(e).__name__}: {e}")
        # Now that fresh data has landed, tighten the store: merge redundant profile
        # entries and de-duplicate episodic rows. Once per batch, never in the
        # conversation path, and never allowed to crash shutdown/recovery.
        if done and MEMORY_COMPACTION:
            try:
                self.compact_profile()
            except Exception as e:
                print(f"[MEMORY] profile compaction skipped: {type(e).__name__}: {e}")
            try:
                self.dedup_memories()
            except Exception as e:
                print(f"[MEMORY] memories dedup skipped: {type(e).__name__}: {e}")
        return done

    def _consolidate_one(self, session_id: int) -> None:
        with self._lock:
            turns = self._conn.execute(
                "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        if not turns:
            self._mark_consolidated(session_id, "")
            return

        transcript = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)
        profile = self.load_profile()
        print(f"[MEMORY] consolidating session {session_id} "
              f"({len(turns)} turns) via {CONSOLIDATION_BACKEND}...")
        result = _consolidation_llm(transcript, profile)

        summary = (result.get("summary") or "").strip()
        new_profile = result.get("profile")

        # Normalize the extracted facts up front so we can both store AND log them.
        # Anything the model rates below the storage floor is dropped here — the
        # deterministic guard against trivia ("had a casual chat") leaking in even
        # when the prompt fails to suppress it upstream.
        norm_facts = []
        dropped = 0
        for fact in result.get("facts") or []:
            content = str(fact.get("content", "")).strip()
            if not content:
                continue
            kind = str(fact.get("kind", "misc")).strip().lower()
            if kind not in _KIND_LABELS:
                kind = "misc"
            try:
                salience = float(fact.get("salience", 0.5))
            except (TypeError, ValueError):
                salience = 0.5
            salience = min(1.0, max(0.0, salience))
            if salience < MEMORY_STORE_MIN_SALIENCE:
                dropped += 1
                continue
            norm_facts.append((content, kind, salience))

        with self._lock:
            for content, kind, salience in norm_facts:
                self._conn.execute(
                    "INSERT INTO memories (session_id, created_at, kind, content, "
                    "salience, last_seq) VALUES (?,?,?,?,?,?)",
                    (session_id, _now(), kind, content, salience, session_id),
                )
            self._conn.commit()

        # Log exactly what TARS took away from this session — the test signal.
        print(f"[MEMORY]   summary: {summary or '(none)'}")
        tail = f" ({dropped} trivial dropped)" if dropped else ""
        print(f"[MEMORY]   {len(norm_facts)} fact(s) remembered{tail}:")
        for content, kind, salience in norm_facts:
            print(f"[MEMORY]     - [{kind} {salience:.2f}] {content}")

        # Profile is rewritten wholesale by the model as plain strings (told to preserve
        # unless contradicted). Re-attach our metadata: entries that already existed keep
        # their salience/recency, genuinely new ones get the category default stamped with
        # this session as their recency. The .prev backup in _save_profile covers a bad pass.
        if isinstance(new_profile, dict):
            merged, added = self._merge_returned_profile(profile, new_profile, session_id)
            self._save_profile(merged)
            if added:
                print(f"[MEMORY]   profile gained {len(added)} new entry(ies):")
                for a in added:
                    print(f"[MEMORY]     + {a}")
            else:
                print("[MEMORY]   profile unchanged")

        self._mark_consolidated(session_id, summary)

    def _mark_consolidated(self, session_id: int, summary: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET summary = ?, consolidated = 1, "
                "ended_at = COALESCE(ended_at, ?) WHERE id = ?",
                (summary, _now(), session_id),
            )
            self._conn.commit()

    @staticmethod
    def _merge_returned_profile(old: dict, returned: dict, current_seq: int) -> tuple[dict, list[str]]:
        """Fold the model's plain-string profile back into our object-profile, preserving
        per-entry salience/recency. An entry whose text already exists keeps its object
        (and thus its accumulated salience/recency); a genuinely new one gets the category
        default salience and this session as its recency. Entries the model dropped are
        intentionally dropped (it was told to preserve unless contradicted). Returns the
        merged object-profile plus a list of newly-added "category: text" strings to log.

        Note: a lightly-reworded existing entry won't text-match, so it's treated as new
        and resets to default salience/fresh recency — an acceptable bias toward freshness."""
        merged, added = {}, []
        for k in _EMPTY_PROFILE:
            existing = {e["text"]: e for e in (old.get(k) or [])}
            default = _PROFILE_DEFAULT_SALIENCE.get(k, 0.6)
            out, seen = [], set()
            for x in (returned.get(k) or []):
                text = str(x).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                if text in existing:
                    out.append(existing[text])
                else:
                    out.append({"text": text, "salience": default, "last_seen": current_seq})
                    added.append(f"{k}: {text}")
            merged[k] = out
        return merged, added

    @staticmethod
    def _best_overlap(text: str, entries: list[dict]) -> dict | None:
        """Find the existing entry most word-overlapping with `text` (Jaccard over the
        word sets), or None if nothing meaningfully overlaps. Used to carry metadata onto
        a merged/reworded entry so compaction doesn't silently downgrade an important trait."""
        words = set(text.lower().split())
        if not words:
            return None
        best, best_score = None, 0.0
        for e in entries:
            ew = set(e["text"].lower().split())
            if not ew:
                continue
            score = len(words & ew) / len(words | ew)
            if score > best_score:
                best, best_score = e, score
        return best if best_score >= 0.3 else None

    # -- compaction (write-side, LLM) ----------------------------------------
    def compact_profile(self) -> int:
        """Merge near-duplicate profile entries into single sharper phrases via one
        LLM pass. Returns how many entries were removed (0 if nothing to do). The
        .prev backup written by _save_profile covers a bad pass."""
        profile = self.load_profile()
        before = sum(len(profile.get(k) or []) for k in _EMPTY_PROFILE)
        if before < 2:
            return 0
        merged = _compact_profile_llm(profile)
        if not isinstance(merged, dict):
            return 0
        # The model returns plain strings; re-attach metadata. An unchanged entry keeps
        # its object; a merged/reworded one inherits the metadata of its closest source
        # (so an important trait isn't silently downgraded) and is treated as freshly
        # touched. Accept only a non-empty, profile-shaped result.
        seq = self.session_id or 0
        clean = {}
        for k in _EMPTY_PROFILE:
            old_entries = profile.get(k) or []
            by_text = {e["text"]: e for e in old_entries}
            default = _PROFILE_DEFAULT_SALIENCE.get(k, 0.6)
            out, seen = [], set()
            for x in (merged.get(k) or []):
                text = str(x).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                if text in by_text:
                    out.append(by_text[text])
                else:
                    src = self._best_overlap(text, old_entries)
                    salience = max(src["salience"], default) if src else default
                    out.append({"text": text, "salience": salience, "last_seen": seq})
            clean[k] = out
        after = sum(len(clean[k]) for k in _EMPTY_PROFILE)
        if after == 0:
            print("[MEMORY] profile compaction returned empty — keeping current profile")
            return 0
        removed = before - after
        if removed <= 0:
            return 0  # nothing actually merged; don't rewrite the file needlessly
        self._save_profile(clean)
        print(f"[MEMORY] profile compacted: {before} -> {after} entries "
              f"({removed} redundant merged)")
        return removed

    def dedup_memories(self) -> int:
        """Collapse near-duplicate episodic memory rows. An LLM groups rows that
        state the same thing; we keep the strongest row of each group (max salience,
        summed recall, latest recency) and delete the rest. Returns rows removed."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, kind, content, salience, recall_count, last_seq FROM memories"
            ).fetchall()
        if len(rows) < 2:
            return 0
        by_id = {r["id"]: r for r in rows}
        items = [{"id": r["id"], "kind": r["kind"], "content": r["content"]} for r in rows]
        result = _dedup_memories_llm(items)
        merges = result.get("merges") if isinstance(result, dict) else None
        if not merges:
            return 0

        removed = 0
        with self._lock:
            for grp in merges:
                raw_ids = grp.get("ids") if isinstance(grp, dict) else None
                if not isinstance(raw_ids, list):
                    continue
                ids = []
                for i in raw_ids:
                    try:
                        i = int(i)
                    except (TypeError, ValueError):
                        continue
                    if i in by_id and i not in ids:
                        ids.append(i)
                if len(ids) < 2:
                    continue
                grp_rows = [by_id[i] for i in ids]
                keep = max(grp_rows, key=lambda r: (r["salience"], r["last_seq"]))
                content = str(grp.get("content") or keep["content"]).strip() or keep["content"]
                kind = str(grp.get("kind", keep["kind"])).strip().lower()
                if kind not in _KIND_LABELS:
                    kind = keep["kind"]
                # NULL the embedding so the merged phrasing gets re-vectorized next
                # startup — the old vector was for the pre-merge content.
                self._conn.execute(
                    "UPDATE memories SET content=?, kind=?, salience=?, recall_count=?, "
                    "last_seq=?, embedding=NULL WHERE id=?",
                    (content, kind, max(r["salience"] for r in grp_rows),
                     sum(r["recall_count"] for r in grp_rows),
                     max(r["last_seq"] for r in grp_rows), keep["id"]),
                )
                drop_ids = [i for i in ids if i != keep["id"]]
                self._conn.executemany(
                    "DELETE FROM memories WHERE id=?", [(i,) for i in drop_ids])
                removed += len(drop_ids)
            if removed:
                self._conn.commit()
        if removed:
            print(f"[MEMORY] memories de-duplicated: removed {removed} duplicate row(s)")
        return removed

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# -- the consolidation LLM call ----------------------------------------------
_CONSOLIDATION_SYSTEM = """You are the memory consolidation system for TARS, an AI \
companion. You are given the transcript of one conversation session between TARS and \
{user}, plus what TARS currently knows about {user}. Distill the session into durable \
long-term memory. Be RUTHLESSLY selective: record only concrete, durable things about \
{user} — who they are, what they like or dislike, what they're working on, what they ask \
of you, meaningful events in their life. \
Do NOT record that a conversation happened, greetings, goodbyes, small talk, passing \
moods, or meta-observations about the chat itself (e.g. "had a casual chat after a brief \
absence", "exchanged a greeting", "the conversation was relaxed"). Those are noise. \
For a short or purely social session the correct output is an EMPTY facts list — that is \
common and expected, not a failure.

Reply with ONLY a JSON object, no prose, in exactly this shape:
{{
  "summary": "1-2 sentence past-tense recap of what this conversation was about and any notable moments",
  "facts": [
    {{"content": "a single concrete thing worth remembering", "kind": "preference|like|dislike|annoyance|request|event|identity|misc", "salience": 0.0-1.0}}
  ],
  "profile": {{
    "identity": [], "preferences": [], "likes": [], "dislikes": [],
    "annoyances": [], "recurring_requests": [], "relationship_notes": []
  }}
}}

Rules:
- salience: 0.9-1.0 = core to who they are / strong feelings; 0.5 = ordinary but real; \
0.3 = minor-but-worth-it. Anything you'd score below 0.3 is trivia — leave it out entirely \
rather than including it with a low score.
- "profile" must be the FULL updated profile: start from the current profile given to \
you and PRESERVE every existing entry unless this session clearly contradicts or updates \
it. Add new durable traits. Keep each entry a short phrase. Don't duplicate.
- Everything in English. When in doubt about a fact, leave it out — an empty facts list is far better than storing noise.
"""


# OpenAI-compatible cloud backends usable for the JSON consolidation pass (same
# endpoints as the conversation chain in core/llm.py). Each: (base_url, key, model).
_CONSOLIDATION_OPENAI = {
    "groq":     ("https://api.groq.com/openai/v1",                           GROQ_API_KEY,     GROQ_MODEL),
    "cerebras": ("https://api.cerebras.ai/v1",                               CEREBRAS_API_KEY, CEREBRAS_MODEL),
    "gemini":   ("https://generativelanguage.googleapis.com/v1beta/openai/", GEMINI_API_KEY,   GEMINI_MODEL),
}


def _chat_json_one(backend: str, system: str, user: str) -> dict:
    """One JSON-returning chat call on a specific backend. Raises on failure so the
    caller can fall through to the next backend."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    if backend == "ollama":
        import ollama
        client = ollama.Client(host=OLLAMA_BASE_URL)
        resp = client.chat(model=OLLAMA_MODEL, format="json",
                           messages=messages, options={"temperature": 0.2})
        return json.loads(resp["message"]["content"])
    base_url, api_key, model = _CONSOLIDATION_OPENAI[backend]
    if not api_key:
        raise RuntimeError(f"consolidation backend '{backend}' has no API key")
    from openai import OpenAI
    # Off the hot path (shutdown/startup), so a generous timeout is fine; max_retries=0
    # so a rate-limited provider fails straight to the next instead of backing off.
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0)
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.2,
        response_format={"type": "json_object"})
    return json.loads(resp.choices[0].message.content)


def _chat_json(system: str, user: str) -> dict:
    """JSON consolidation call with FAILOVER. Tries the configured backend first, then
    the rest of the LLM chain, always ending at local ollama — so a finished session is
    distilled even when every cloud free tier is rate-limited (the bug that silently
    dropped a whole session of memories when Groq's daily token limit was hit). Shared
    by the consolidation + compaction passes."""
    order = [CONSOLIDATION_BACKEND]
    order += [b for b in LLM_CHAIN if b not in order]
    if "ollama" not in order:
        order.append("ollama")  # local floor: no rate limit, always available
    last_err = None
    for backend in order:
        if backend in _CONSOLIDATION_OPENAI and not _CONSOLIDATION_OPENAI[backend][1]:
            continue  # cloud backend with no key — skip
        try:
            result = _chat_json_one(backend, system, user)
            if backend != order[0]:
                print(f"[MEMORY] consolidation served via fallback: {backend}")
            return result
        except Exception as e:
            last_err = e
            print(f"[MEMORY] consolidation backend '{backend}' failed: "
                  f"{type(e).__name__}: {str(e)[:120]}")
    raise last_err if last_err else RuntimeError("no consolidation backend available")


def _consolidation_llm(transcript: str, current_profile: dict) -> dict:
    """Distill one session into summary + facts + refreshed profile (one LLM call)."""
    system = _CONSOLIDATION_SYSTEM.format(user=USER_NAME)
    user = (
        "CURRENT PROFILE:\n"
        + json.dumps(_profile_texts(current_profile), ensure_ascii=False, indent=2)
        + "\n\nSESSION TRANSCRIPT:\n" + transcript
    )
    return _chat_json(system, user)


# -- compaction passes (keep the persistent store tight) ---------------------
_PROFILE_COMPACT_SYSTEM = """You tighten TARS's long-term profile of {user}. You are \
given the current profile — seven lists of short phrases. Within a list, some entries \
say the same or nearly the same thing in different words. Merge each such redundant \
cluster into ONE sharper, cleaner phrase.

Reply with ONLY a JSON object — the FULL profile in exactly this shape:
{{
  "identity": [], "preferences": [], "likes": [], "dislikes": [],
  "annoyances": [], "recurring_requests": [], "relationship_notes": []
}}

Rules:
- ONLY merge entries that are genuinely redundant (the same fact, trait or feeling). \
Keep every DISTINCT piece of information — never drop something just to shorten.
- A merged entry must preserve the full meaning AND the most specific detail of \
everything it replaces — keep concrete specifics (names, song titles, places, facts), \
never generalize them away (e.g. don't turn "favorite song is Purple Rain" into "likes \
music"). Prefer the clearest, most informative wording.
- Do NOT invent anything not supported by the existing entries. Do NOT move entries \
between categories unless one is plainly in the wrong list.
- Keep each entry a short phrase. Return every category key, even if its list is empty.
"""


def _compact_profile_llm(profile: dict) -> dict:
    system = _PROFILE_COMPACT_SYSTEM.format(user=USER_NAME)
    user = "CURRENT PROFILE:\n" + json.dumps(
        _profile_texts(profile), ensure_ascii=False, indent=2)
    return _chat_json(system, user)


_MEMORIES_DEDUP_SYSTEM = """You de-duplicate TARS's episodic memory about {user}. You \
are given a numbered list of remembered facts/events, each with an id. Some say the \
same thing in different words. Group ONLY the genuine duplicates.

Reply with ONLY a JSON object in exactly this shape:
{{
  "merges": [
    {{"ids": [1, 4], "content": "one sharp phrasing covering them", "kind": "preference|like|dislike|annoyance|request|event|identity|misc"}}
  ]
}}

Rules:
- Only group items stating the SAME fact/event/trait. Do NOT group things that are \
merely related or topically similar but distinct (a one-off event is not the same as a \
standing preference).
- Every group MUST have 2+ ids. Omit anything with no duplicate (don't list singletons).
- "content" = a single clean phrase that keeps the MOST SPECIFIC detail of the group \
(names, song titles, concrete facts) — never generalize the detail away; "kind" = the \
best-fitting one.
- Do NOT invent facts. If there are no duplicates at all, return {{"merges": []}}.
"""


def _dedup_memories_llm(items: list[dict]) -> dict:
    system = _MEMORIES_DEDUP_SYSTEM.format(user=USER_NAME)
    listing = "\n".join(f"{it['id']}. [{it['kind']}] {it['content']}" for it in items)
    return _chat_json(system, "REMEMBERED ITEMS:\n" + listing)
