"""Read-only window into TARS's persistent memory. Run it any time (especially
right after closing a session) to see, as ground truth, what actually got stored:

    .venv/Scripts/python.exe scripts/inspect_memory.py

Touches nothing — pure SELECT + a file read. Safe to run while the app is closed.
"""
import json
import os
import sqlite3
import sys

# Make the project root importable when run as `python scripts/inspect_memory.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (  # noqa: E402
    MEMORY_DB_PATH, PROFILE_PATH, MEMORY_DECAY_BASE, MEMORY_MIN_SALIENCE,
)


def _rule(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    if not os.path.exists(MEMORY_DB_PATH):
        print(f"No memory DB yet at {MEMORY_DB_PATH} — run the app at least once.")
        return

    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row

    # Current session ordinal drives the decay math (sessions.id is the ordinal).
    max_id = conn.execute("SELECT COALESCE(MAX(id), 0) AS m FROM sessions").fetchone()["m"]

    _rule("SESSIONS (start / end timestamps)")
    sessions = conn.execute(
        "SELECT id, started_at, ended_at, turn_count, consolidated, summary "
        "FROM sessions ORDER BY id ASC"
    ).fetchall()
    if not sessions:
        print("(none)")
    for s in sessions:
        flag = "ok" if s["consolidated"] else "PENDING"
        print(f"\n#{s['id']}  [{flag}]  turns={s['turn_count']}")
        print(f"    start: {s['started_at']}")
        print(f"    end:   {s['ended_at'] or '(still open / not closed cleanly)'}")
        print(f"    recap: {s['summary'] or '(not consolidated yet)'}")

    _rule(f"EPISODIC MEMORIES  (decay base={MEMORY_DECAY_BASE}, floor={MEMORY_MIN_SALIENCE})")
    mems = conn.execute(
        "SELECT id, session_id, created_at, kind, content, salience, "
        "recall_count, last_seq FROM memories ORDER BY id ASC"
    ).fetchall()
    if not mems:
        print("(none yet — facts appear after a session is consolidated on close)")
    for m in mems:
        elapsed = max(0, max_id - m["last_seq"])
        eff = m["salience"] * (MEMORY_DECAY_BASE ** elapsed)
        faded = "  <faded>" if eff < MEMORY_MIN_SALIENCE else ""
        print(f"\n#{m['id']}  [{m['kind']}]  (from session {m['session_id']})")
        print(f"    {m['content']}")
        print(f"    salience={m['salience']:.2f}  effective={eff:.2f}  "
              f"recalls={m['recall_count']}{faded}")

    conn.close()

    _rule("USER PROFILE (profile.json)")
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            print(json.dumps(json.load(f), indent=2, ensure_ascii=False))
    except FileNotFoundError:
        print("(no profile.json yet — appears after the first session is consolidated)")
    except json.JSONDecodeError as e:
        print(f"(profile.json is corrupt: {e})")


if __name__ == "__main__":
    main()
