"""TARS's action rail — the reusable "binario" every TARS app/tool rides on.

Why this lives apart from the LLM: the model only ever *decides* to call a tool
(it emits a name + JSON arguments). The actual work happens here, in plain Python,
completely decoupled from WHICH provider in the failover chain produced the call.
So a tool behaves identically whether Groq, Cerebras, Gemini, or the local Ollama
3b is the brain that turn — you define the action once and it runs on any model.
(Ollama is the exception by choice: the local 3b isn't reliable at tool calls, so
offline the tools simply don't fire and TARS just talks — graceful degrade.)

Two design choices keep the rail cheap and safe:

  - CONDITIONAL exposure. A tool's schema is sent to the model ONLY on turns whose
    text trips one of the tool's trigger words. A normal chat turn ships NO tool
    schemas at all, so the baseline prompt — and the daily free-tier token budget —
    stays exactly as lean as it was before tools existed. `relevant()` is that gate.
    A false trigger only means the schema is *offered* that turn (a few extra tokens);
    the model still decides whether to actually call it, so generous triggers are fine.

  - WHITELIST dispatch. The model can only ever invoke a name registered here; an
    unknown or garbled name returns an error string and executes nothing. There is no
    eval/shell — every tool is an explicit Python function. This is the seam where any
    future "control the PC" action must add its own guard rails (confirmation, sandbox).
"""
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable


@dataclass
class Tool:
    """One action TARS can take. `func` takes the parsed arguments dict and returns a
    short string the model reads back to phrase its spoken reply. `triggers` are the
    words that make this tool's schema eligible to be offered that turn."""
    name: str
    description: str
    parameters: dict                      # JSON-Schema for the arguments object
    func: Callable[[dict], str]
    triggers: tuple[str, ...] = ()
    _trigger_re: re.Pattern | None = field(default=None, init=False, repr=False)

    def matches(self, text: str) -> bool:
        if not self.triggers:
            return False
        if self._trigger_re is None:
            # word-boundary alternation so "day" matches "what day" but not "payday"
            self._trigger_re = re.compile(
                r"\b(" + "|".join(re.escape(t) for t in self.triggers) + r")\b",
                re.IGNORECASE)
        return bool(self._trigger_re.search(text))

    def schema(self) -> dict:
        """The OpenAI-compatible function-tool spec sent to the provider."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def relevant(self, text: str) -> list[dict]:
        """The tool schemas to offer for a turn with this user text. Empty list ==
        ship no tools (identical to a pre-tools turn)."""
        return [t.schema() for t in self._tools.values() if t.matches(text)]

    def dispatch(self, name: str, arguments: dict) -> str:
        """Run a registered tool by name (whitelist). Never raises into the pipeline:
        an unknown name or a tool that throws comes back as a string the model can
        speak around, so a bad call degrades to a sentence, never a crash."""
        tool = self._tools.get(name)
        if tool is None:
            return f"(error: no such tool '{name}')"
        try:
            return tool.func(arguments or {})
        except Exception as e:  # a tool bug must not take down the voice turn
            return f"(error running {name}: {type(e).__name__}: {e})"


registry = ToolRegistry()


# -- the first tool: the simplest possible one, to prove the rail end-to-end ----
# TARS knowing the real date/time is genuinely useful on its own AND is the date
# context the calendar app will build on next. No arguments, no side effects, no
# risk — exactly the trivial tool you want to validate the streaming + TTS plumbing
# with before anything that actually touches the machine.
def _get_current_datetime(_args: dict) -> str:
    now = datetime.now().astimezone()
    return now.strftime("Current local date and time: %A, %d %B %Y, %H:%M (%Z).")


registry.register(Tool(
    name="get_current_datetime",
    description="Get the current local date and time. Call this whenever the user "
                "asks what time/day/date it is, or you need today's date to answer.",
    parameters={"type": "object", "properties": {}, "required": []},
    func=_get_current_datetime,
    triggers=(
        "time", "what time", "o'clock", "clock", "hour", "date", "day", "today",
        "tonight", "tomorrow", "yesterday", "weekday", "month", "year", "what day",
    ),
))
