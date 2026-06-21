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

from config import (
    MEMORY_DB_PATH, PROFILE_PATH, USER_NAME,
    CONSOLIDATION_BACKEND, GROQ_API_KEY, GROQ_MODEL,
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    MEMORY_RECENT_SESSIONS, MEMORY_MAX_FACTS,
    MEMORY_DECAY_BASE, MEMORY_MIN_SALIENCE, MEMORY_COMPACTION,
)

# Profile shape. Lists of short strings, distilled over time. Kept deliberately
# simple and additive so the consolidation model can return the whole thing back.
_EMPTY_PROFILE = {
    "identity": [],            # who they are: name, role, where they live, etc.
    "preferences": [],         # how they like TARS to behave / answer
    "likes": [],               # things they enjoy / care about
    "dislikes": [],            # things they don't like
    "annoyances": [],          # things that genuinely wind them up
    "recurring_requests": [],  # stuff they keep asking TARS to do
    "relationship_notes": [],  # the texture of the bond, inside jokes, tone
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
                """
            )
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
    def _effective_salience(self, salience: float, last_seq: int, current_seq: int) -> float:
        elapsed = max(0, current_seq - last_seq)
        return salience * (MEMORY_DECAY_BASE ** elapsed)

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

        prof_lines = self._render_profile(profile)
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

        return "\n".join(lines)

    @staticmethod
    def _profile_is_empty(profile: dict) -> bool:
        return not any(profile.get(k) for k in _EMPTY_PROFILE)

    @staticmethod
    def _render_profile(profile: dict) -> list[str]:
        headers = {
            "identity": "Who they are", "preferences": "How they like you to be",
            "likes": "Likes", "dislikes": "Dislikes",
            "annoyances": "Winds them up", "recurring_requests": "Often asks you to",
            "relationship_notes": "Between you two",
        }
        out = []
        for key, header in headers.items():
            items = profile.get(key) or []
            if items:
                out.append(f"- {header}: " + "; ".join(items))
        return out

    # -- profile JSON ---------------------------------------------------------
    def load_profile(self) -> dict:
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {k: list(v) for k, v in _EMPTY_PROFILE.items()}
        # Normalize: keep only known keys, ensure they're lists of strings.
        return {
            k: [str(x) for x in (data.get(k) or []) if str(x).strip()]
            for k in _EMPTY_PROFILE
        }

    def _save_profile(self, profile: dict) -> None:
        clean = {
            k: [str(x) for x in (profile.get(k) or []) if str(x).strip()]
            for k in _EMPTY_PROFILE
        }
        clean["updated_at"] = _now()
        # Keep one backup before overwriting — cheap insurance against a bad
        # consolidation pass silently dropping things the user told us.
        if os.path.exists(self.profile_path):
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
        norm_facts = []
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
        print(f"[MEMORY]   {len(norm_facts)} fact(s) remembered:")
        for content, kind, salience in norm_facts:
            print(f"[MEMORY]     - [{kind} {salience:.2f}] {content}")

        # Profile is rewritten wholesale by the model (told to preserve unless
        # contradicted). Only accept it if it looks like our shape; the .prev
        # backup in _save_profile covers a bad pass. Log what newly landed.
        if isinstance(new_profile, dict):
            added = []
            for k in _EMPTY_PROFILE:
                before = set(profile.get(k) or [])
                for x in (new_profile.get(k) or []):
                    if str(x).strip() and str(x) not in before:
                        added.append(f"{k}: {x}")
            self._save_profile(new_profile)
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
        # Accept only something shaped like our profile that didn't wipe everything.
        clean = {k: [str(x).strip() for x in (merged.get(k) or []) if str(x).strip()]
                 for k in _EMPTY_PROFILE}
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
                self._conn.execute(
                    "UPDATE memories SET content=?, kind=?, salience=?, recall_count=?, "
                    "last_seq=? WHERE id=?",
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
long-term memory. Be selective: capture what genuinely matters for knowing this person \
and continuing the relationship — skip small talk and one-off trivia.

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
- salience: 0.9-1.0 = core to who they are / strong feelings; 0.5 = ordinary; 0.2 = minor.
- "profile" must be the FULL updated profile: start from the current profile given to \
you and PRESERVE every existing entry unless this session clearly contradicts or updates \
it. Add new durable traits. Keep each entry a short phrase. Don't duplicate.
- Everything in English. If the session was trivial, it's fine to return few or no facts.
"""


def _chat_json(system: str, user: str) -> dict:
    """One JSON-returning chat call on the configured consolidation backend (groq
    default, ollama local). Shared by the consolidation + compaction passes."""
    if CONSOLIDATION_BACKEND == "ollama":
        import ollama
        client = ollama.Client(host=OLLAMA_BASE_URL)
        resp = client.chat(
            model=OLLAMA_MODEL, format="json",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            options={"temperature": 0.2},
        )
        raw = resp["message"]["content"]
    else:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "CONSOLIDATION_BACKEND=groq but GROQ_API_KEY is empty. Add it to .env "
                "or set CONSOLIDATION_BACKEND=ollama for fully-local consolidation."
            )
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
    return json.loads(raw)


def _consolidation_llm(transcript: str, current_profile: dict) -> dict:
    """Distill one session into summary + facts + refreshed profile (one LLM call)."""
    system = _CONSOLIDATION_SYSTEM.format(user=USER_NAME)
    user = (
        "CURRENT PROFILE:\n"
        + json.dumps({k: current_profile.get(k, []) for k in _EMPTY_PROFILE},
                     ensure_ascii=False, indent=2)
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
        {k: profile.get(k, []) for k in _EMPTY_PROFILE}, ensure_ascii=False, indent=2)
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
