from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from .command_resolver import CommandResolutionError, prepare_command, render_command
from .models import AppConfig
from .process_utils import process_group_kwargs, terminate_process_tree

T = TypeVar("T", bound=BaseModel)
StatusCallback = Callable[[str, str], None]
CancelCheck = Callable[[], str | None]
OpenCodeProfile = Literal["default", "analysis", "judge", "executor", "refinement"]

JSON_START = "<GOAL_AGENT_JSON>"
JSON_END = "</GOAL_AGENT_JSON>"

_STREAM_CHUNK_BYTES = 64 * 1024
_MAX_STREAM_LINE_BYTES = 32 * 1024 * 1024

_AUTO_APPROVE_FLAG_CACHE: dict[tuple[str, ...], str] = {}

# Keep enough room for OpenCode's system prompt, tool schemas, and a final response.
# This is injected only into Goal Agent child processes and does not modify the
# user's OpenCode configuration files.
_DEFAULT_MODEL_CONTEXT_TOKENS = 65_536
_GOAL_AGENT_MAX_COMPACTION_RESERVED_TOKENS = 24_000
_DEFAULT_CONTEXT_OVERFLOW_RETRIES = 2
_COMPACT_RECOVERY_PROMPT_CHARS = 28_000
_DEFAULT_RESPONSE_ONLY_STALL_SECONDS = 300

# OpenCode compaction is reactive. A single tool-heavy turn can add more context
# than the remaining window before OpenCode gets a chance to compact it. Goal
# Agent therefore applies per-call step limits and watches the streamed tool
# payload itself. The character budgets are deliberately conservative for a
# 65,536-token local model while still leaving useful room for executor work.
_PROFILE_PROMPT_CHAR_LIMITS: dict[OpenCodeProfile, int] = {
    "default": 96_000,
    "analysis": 72_000,
    "judge": 72_000,
    "executor": 80_000,
    "refinement": 64_000,
}
_PROFILE_STEPS: dict[OpenCodeProfile, int] = {
    "default": 8,
    # Tool-free roles do not need an agentic step cap.  In OpenCode, reaching
    # ``steps`` activates a special last-step/assistant-prefill path.  Some
    # OpenAI-compatible local providers answer that prefill with an immediate
    # EOS token, producing a clean step_start/step_finish stream but no text.
    # Leave these profiles uncapped and deny their tools instead.
    "analysis": 0,
    "judge": 0,
    "executor": 6,
    "refinement": 0,
}
_PROFILE_TOOL_OUTPUT_BUDGET_CHARS: dict[OpenCodeProfile, int | None] = {
    "default": None,
    "analysis": 0,
    "judge": 0,
    "executor": 120_000,
    "refinement": 64_000,
}


def _safe_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None
    return None


def _context_window_from_inline_config(config: dict, model: str | None) -> int | None:
    """Best-effort context window lookup from OPENCODE_CONFIG_CONTENT."""

    selected_model = (model or config.get("model") or "").strip()
    if not selected_model or "/" not in selected_model:
        return None

    provider_name, model_name = selected_model.split("/", 1)
    providers = config.get("provider")
    if not isinstance(providers, dict):
        return None
    provider = providers.get(provider_name)
    if not isinstance(provider, dict):
        return None
    models = provider.get("models")
    if not isinstance(models, dict):
        return None
    model_data = models.get(model_name)
    if not isinstance(model_data, dict):
        return None
    limit = model_data.get("limit")
    if not isinstance(limit, dict):
        return None
    return _safe_positive_int(limit.get("context"))


def _compaction_reserved_tokens(context_window_tokens: int | None) -> int:
    if context_window_tokens is None:
        return _GOAL_AGENT_MAX_COMPACTION_RESERVED_TOKENS
    # Reserve ~28% for system/tool schema/final response while keeping room for
    # task input on smaller windows like 32k.
    adaptive = int(context_window_tokens * 0.28)
    return max(4_096, min(_GOAL_AGENT_MAX_COMPACTION_RESERVED_TOKENS, adaptive))


def _profile_prompt_char_limit(
    profile: OpenCodeProfile,
    *,
    context_window_tokens: int | None,
    recovery_level: int,
) -> int:
    base = _PROFILE_PROMPT_CHAR_LIMITS[profile]
    if context_window_tokens is None:
        scaled = base
    else:
        scale = max(0.2, min(1.0, context_window_tokens / _DEFAULT_MODEL_CONTEXT_TOKENS))
        scaled = int(base * scale)
    if recovery_level:
        scaled = max(8_000, scaled // (2 ** recovery_level))
    return max(8_000, scaled)


_TOOL_FREE_AGENT_PROMPTS: dict[OpenCodeProfile, str] = {
    "analysis": (
        "You are Goal Agent's response-only strategist. Use only the task and "
        "evidence supplied in the user message. Do not call tools or inspect the "
        "workspace. Always produce a substantive final text response in the exact "
        "structured format requested; never end with an empty response."
    ),
    "judge": (
        "You are Goal Agent's response-only evaluator. Use only the supplied "
        "evidence. Do not call tools or inspect the workspace. Always produce a "
        "substantive final text response in the exact structured format requested; "
        "never end with an empty response."
    ),
    "refinement": (
        "You are Goal Agent's response-only goal and success-criteria refiner. "
        "Goal Agent already supplied a bounded project snapshot and conversation. "
        "Do not call tools, ask through a tool, or inspect the workspace. Respond "
        "directly with the requested SetupProposal JSON and never end with an empty "
        "response."
    ),
}


@dataclass(slots=True)
class OpenCodeResult:
    text: str
    raw_stdout: str
    stderr: str
    exit_code: int
    session_id: str | None = None
    interrupted: bool = False
    interrupt_reason: str | None = None
    duration_seconds: float = 0.0
    events: list[dict] = field(default_factory=list)


class OpenCodeError(RuntimeError):
    pass


class OpenCodeContextOverflowError(OpenCodeError):
    def __init__(
        self,
        message: str,
        *,
        requested_tokens: int | None = None,
        context_size: int | None = None,
    ) -> None:
        super().__init__(message)
        self.requested_tokens = requested_tokens
        self.context_size = context_size


class OpenCodeContextBudgetExceededError(OpenCodeContextOverflowError):
    """Raised before the next model call when streamed tool output is too large."""

    def __init__(self, *, observed_chars: int, budget_chars: int) -> None:
        self.observed_chars = observed_chars
        self.budget_chars = budget_chars
        super().__init__(
            "Goal Agent stopped this OpenCode session before the next model request because "
            f"tool output grew to {observed_chars:,} characters, above the safe budget of "
            f"{budget_chars:,}. A fresh compact retry will continue from the workspace state."
        )


class OpenCodeInterrupted(OpenCodeError):
    def __init__(self, reason: str):
        super().__init__(f"OpenCode was interrupted: {reason}")
        self.reason = reason


def _context_overflow_error(text: str) -> OpenCodeContextOverflowError | None:
    if not text:
        return None
    lowered = text.lower()
    markers = (
        "contextoverflowerror",
        "exceed_context_size_error",
        "exceeds the available context size",
        "context length exceeded",
        "maximum context length",
    )
    if not any(marker in lowered for marker in markers):
        return None

    requested: int | None = None
    context_size: int | None = None
    patterns = [
        r"request \((\d+) tokens\) exceeds the available context size \((\d+) tokens\)",
        r'"n_prompt_tokens"\s*:\s*(\d+).*?"n_ctx"\s*:\s*(\d+)',
        r"requested[^0-9]*(\d+).*?(?:context|maximum)[^0-9]*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            requested = int(match.group(1))
            context_size = int(match.group(2))
            break

    if requested is not None and context_size is not None:
        message = (
            f"OpenCode exceeded the model context window: {requested:,} prompt tokens "
            f"for a {context_size:,}-token context. Goal Agent will retry in a fresh "
            "OpenCode session with a compact task brief and restricted project inspection."
        )
    else:
        message = (
            "OpenCode exceeded the model context window. Goal Agent will retry in a "
            "fresh OpenCode session with a compact task brief and restricted project inspection."
        )
    return OpenCodeContextOverflowError(
        message, requested_tokens=requested, context_size=context_size
    )




def _event_part(event: dict) -> dict:
    part = event.get("part")
    if isinstance(part, dict):
        return part
    properties = event.get("properties")
    if isinstance(properties, dict) and isinstance(properties.get("part"), dict):
        return properties["part"]
    return {}


def _event_text(event: dict) -> str | None:
    """Extract completed assistant text from known OpenCode event shapes."""

    event_type = str(event.get("type", ""))
    part = _event_part(event)
    part_type = str(part.get("type", ""))

    is_text_event = event_type == "text" or part_type == "text"
    if not is_text_event:
        return None

    # Raw SDK part updates can include an in-progress full-text snapshot.  Prefer
    # completed parts when timing metadata is present; the CLI's normalized
    # ``type=text`` events are already emitted only after completion.
    if event_type != "text" and part_type == "text":
        timing = part.get("time")
        if isinstance(timing, dict) and timing and not timing.get("end"):
            return None

    candidates = (
        part.get("text"),
        event.get("text"),
        event.get("delta"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _parse_ndjson_events(text: str) -> list[dict] | None:
    """Parse OpenCode's ``--format json`` newline-delimited event stream.

    Returns ``None`` when the input is not recognizably an event stream.  This
    distinction prevents normal pretty-printed proposal JSON from being treated
    as a sequence of events.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    events: list[dict] = []
    invalid = 0
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            invalid += 1
            continue
        if isinstance(value, dict) and (
            "type" in value
            or "sessionID" in value
            or isinstance(value.get("properties"), dict)
        ):
            events.append(value)
        else:
            invalid += 1

    if not events:
        return None
    # Permit a small number of launcher/log lines around a valid event stream,
    # but do not classify arbitrary prose containing one JSON object as NDJSON.
    if len(events) == 1 and invalid:
        return None
    if invalid > max(2, len(events) // 3):
        return None
    return events


def _event_stream_summary(events: list[dict]) -> str:
    counts: dict[str, int] = {}
    tools: list[str] = []
    for event in events:
        event_type = str(event.get("type", "event"))
        counts[event_type] = counts.get(event_type, 0) + 1
        part = _event_part(event)
        if event_type in {"tool_use", "tool"} or part.get("type") == "tool":
            tool = str(part.get("tool") or part.get("name") or event.get("tool") or "tool")
            if tool not in tools:
                tools.append(tool)
    count_text = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    tool_text = f"; tools used: {', '.join(tools[:8])}" if tools else ""
    return count_text + tool_text


def _diagnostic_output(result: OpenCodeResult, *, max_chars: int = 6000) -> str:
    """Return a bounded diagnostic without dumping whole project tool results."""

    if result.text.strip():
        return _bounded_text(result.text.strip(), max_chars)
    if result.events:
        summary = _event_stream_summary(result.events)
        tail = _bounded_text(result.raw_stdout[-max_chars:], max_chars)
        return f"OpenCode event stream summary: {summary}\nLast events (bounded):\n{tail}"
    return _bounded_text((result.raw_stdout or result.stderr).strip(), max_chars)



def _bounded_text(text: str, max_chars: int) -> str:
    """Keep both the task definition and the latest instructions when bounding text."""

    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 5
    tail = max_chars - head
    return (
        text[:head].rstrip()
        + "\n\n… [middle omitted by Goal Agent context recovery] …\n\n"
        + text[-tail:].lstrip()
    )


def _context_recovery_prompt(
    prompt: str,
    *,
    retry_number: int,
    profile: OpenCodeProfile,
    context_window_tokens: int | None,
) -> str:
    """Build a fresh-session task brief that discourages context-expanding tool use."""

    adaptive_cap = _profile_prompt_char_limit(
        profile,
        context_window_tokens=context_window_tokens,
        recovery_level=retry_number,
    )
    budget = max(8_000, min(adaptive_cap, _COMPACT_RECOVERY_PROMPT_CHARS // retry_number))
    bounded = _bounded_text(prompt, budget)
    file_limit = 8 if retry_number == 1 else 4
    if profile in {"analysis", "judge", "refinement"}:
        recovery_rules = """- Do not call tools, inspect files, search the project, or delegate to subagents.
- Use only the bounded task brief and embedded evidence.
- Return the required structured response immediately."""
    else:
        recovery_rules = f"""- Do not delegate to subagents or use broad/exhaustive project exploration.
- Do not recursively read the repository or inspect dependency, vendor, build, cache, log,
  generated, binary, media, .git, node_modules, dist, target, coverage, or .goal-agent trees.
- Inspect at most {file_limit} small, directly relevant text files. Read bounded sections rather
  than entire large files. Prefer exact paths named in the task, current hypothesis, tests, or errors.
- Run only focused commands whose output is bounded. Redirect or filter noisy output.
- Complete one bounded decision/change and return the required final structured response promptly.
- If more work is needed than fits this attempt, report the remaining work as blockers instead of
  expanding context until it overflows again."""
    return f"""
CONTEXT-OVERFLOW RECOVERY MODE — FRESH SESSION {retry_number}

A previous OpenCode process exceeded the model context window. This is a new session; do not
continue, compact, or reconstruct the failed session. The task brief below is authoritative.
The previous process may already have changed workspace files.

Rules for this recovery attempt:
{recovery_rules}

BOUNDED TASK BRIEF
{bounded}
""".strip()


def _profile_permissions(profile: OpenCodeProfile, *, recovery_level: int) -> dict[str, str]:
    """Return a least-privilege permission set for one Goal Agent call."""

    deny_all = {"*": "deny"}
    if profile in {"analysis", "judge"}:
        return deny_all
    if profile == "refinement":
        # Goal Agent supplies a bounded project snapshot.  Keeping refinement
        # tool-free prevents OpenCode from consuming every step on repository
        # reads and then ending without the required proposal response.
        return deny_all
    if profile == "executor":
        permissions = {
            "*": "deny",
            "read": "allow",
            "edit": "allow",
            "glob": "allow",
            "grep": "allow",
            "list": "allow",
            "bash": "allow",
            "lsp": "allow",
            "todowrite": "allow",
        }
        if recovery_level >= 2:
            # Exact paths and focused commands only on the strict retry. This
            # prevents another broad exploration from rebuilding the same huge turn.
            permissions["glob"] = "deny"
            permissions["list"] = "deny"
            permissions["grep"] = "deny"
        return permissions
    return {
        "*": "deny",
        "read": "allow",
        "edit": "allow",
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "bash": "allow",
        "lsp": "allow",
        "todowrite": "allow",
    }


def _profile_steps(
    profile: OpenCodeProfile, *, recovery_level: int
) -> int | None:
    if profile in _TOOL_FREE_AGENT_PROMPTS:
        return None
    steps = _PROFILE_STEPS[profile]
    if recovery_level:
        if profile == "executor":
            return 3 if recovery_level == 1 else 2
        return 1
    return steps


def _profile_tool_budget(
    profile: OpenCodeProfile,
    *,
    context_window_tokens: int | None,
    recovery_level: int,
) -> int | None:
    budget = _PROFILE_TOOL_OUTPUT_BUDGET_CHARS[profile]
    if budget is None:
        return None
    if context_window_tokens is not None:
        scale = max(0.2, min(1.0, context_window_tokens / _DEFAULT_MODEL_CONTEXT_TOKENS))
        budget = max(24_000, int(budget * scale))
    # Recovery sessions start fresh with no accumulated prior turns, so there is
    # no prior context to conserve.  Apply at most a halving on the first retry
    # and keep it constant thereafter; avoid the previous exponential shrink that
    # made even modest file reads exceed the budget at level >= 2.
    if recovery_level >= 1:
        budget = max(48_000, budget // 2)
    return budget


def _opencode_environment(
    *,
    agent: str | None,
    profile: OpenCodeProfile,
    context_window_tokens: int | None,
    recovery_level: int = 0,
) -> dict[str, str]:
    """Return a child environment with compaction and per-call safety controls.

    Inline config has the highest OpenCode precedence. The override applies only
    to the spawned Goal Agent child process; it does not rewrite the user's
    opencode.json or permanently change the selected agent.
    """

    env = dict(os.environ)
    raw = env.get("OPENCODE_CONFIG_CONTENT", "").strip()
    config: dict = {}
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                config = parsed
        except json.JSONDecodeError:
            # Preserve non-JSON/JSONC user content. We cannot safely merge it,
            # but fresh-session recovery and prompt bounding still remain active.
            return env

    existing = config.get("compaction")
    compaction = dict(existing) if isinstance(existing, dict) else {}
    compaction["auto"] = True
    compaction["prune"] = True
    current_reserved = _safe_positive_int(compaction.get("reserved"))
    minimum_reserved = _compaction_reserved_tokens(context_window_tokens)
    if current_reserved is None or current_reserved < minimum_reserved:
        compaction["reserved"] = minimum_reserved
    config["compaction"] = compaction

    if agent:
        agents = config.get("agent")
        agents = dict(agents) if isinstance(agents, dict) else {}
        selected = agents.get(agent)
        selected = dict(selected) if isinstance(selected, dict) else {}
        steps = _profile_steps(profile, recovery_level=recovery_level)
        if steps is None:
            # Do not trigger OpenCode's max-step assistant-prefill path for
            # response-only calls.  Remove both the current and legacy fields in
            # case they came from OPENCODE_CONFIG_CONTENT supplied by the user.
            selected.pop("steps", None)
            selected.pop("maxSteps", None)
        else:
            selected["steps"] = steps
        selected["permission"] = _profile_permissions(
            profile, recovery_level=recovery_level
        )
        response_only_prompt = _TOOL_FREE_AGENT_PROMPTS.get(profile)
        if response_only_prompt:
            # Override the selected agent's normal coding/planning prompt only in
            # this child process.  Built-in plan prompts can otherwise encourage
            # repository exploration or terminate silently when every tool is
            # denied.  The user's persistent OpenCode configuration is untouched.
            selected["description"] = f"Goal Agent {profile} response-only worker"
            selected["mode"] = "primary"
            selected["prompt"] = response_only_prompt
        agents[agent] = selected
        config["agent"] = agents

    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config, separators=(",", ":"))
    return env


class OpenCodeRunner:
    def __init__(self, config: AppConfig):
        self.config = config

    def context_window_tokens(self, model: str | None) -> int | None:
        if self.config.model_context_tokens:
            return self.config.model_context_tokens
        raw = os.environ.get("OPENCODE_CONFIG_CONTENT", "").strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return _context_window_from_inline_config(parsed, model)

    def build_command(
        self,
        *,
        model: str | None,
        agent: str | None,
        title: str,
        output_format: str = "json",
        auto_approve_flag: str | None = None,
    ) -> list[str]:
        command = [*self.config.opencode_command, "run"]
        if output_format:
            command.extend(["--format", output_format])
        command.extend(["--dir", str(self.config.project_path)])
        command.extend(["--title", title])
        if model:
            command.extend(["--model", model])
        if agent:
            command.extend(["--agent", agent])
        if self.config.auto_approve and auto_approve_flag:
            command.append(auto_approve_flag)
        if self.config.attach_url:
            command.extend(["--attach", self.config.attach_url])
            if self.config.attach_username:
                command.extend(["--username", self.config.attach_username])
            if self.config.attach_password_env:
                password = os.getenv(self.config.attach_password_env)
                if password:
                    command.extend(["--password", password])
        return command

    async def detect_auto_approve_flag(
        self, *, cancel_check: CancelCheck | None = None
    ) -> str | None:
        """Return the permission bypass flag supported by this OpenCode CLI.

        OpenCode has shipped both ``--auto`` and
        ``--dangerously-skip-permissions`` for non-interactive runs. Probe the
        installed CLI once and cache the result so projects keep working across
        OpenCode releases without paying the probe cost on every agent step.
        """

        if not self.config.auto_approve:
            return None

        cache_key = tuple(str(part) for part in self.config.opencode_command)
        cached = _AUTO_APPROVE_FLAG_CACHE.get(cache_key)
        if cached:
            return cached

        help_command = [*self.config.opencode_command, "run", "--help"]
        process: asyncio.subprocess.Process | None = None
        help_text = ""
        try:
            invocation = prepare_command(help_command)
            process = await asyncio.create_subprocess_exec(
                *invocation,
                cwd=str(self.config.project_path),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **process_group_kwargs(),
            )
            communication = asyncio.create_task(process.communicate())
            probe_started = time.monotonic()
            while not communication.done():
                if cancel_check:
                    reason = cancel_check()
                    if reason:
                        await terminate_process_tree(process)
                        await asyncio.gather(communication, return_exceptions=True)
                        raise OpenCodeInterrupted(reason)
                if time.monotonic() - probe_started > 5:
                    await terminate_process_tree(process)
                    await asyncio.gather(communication, return_exceptions=True)
                    break
                await asyncio.sleep(min(self.config.poll_interval_seconds, 0.1))
            if communication.done() and not communication.cancelled():
                result = communication.result()
                stdout, stderr = result
                help_text = (stdout + b"\n" + stderr).decode("utf-8", errors="replace")
        except asyncio.CancelledError:
            if process and process.returncode is None:
                await terminate_process_tree(process)
            raise
        except (FileNotFoundError, CommandResolutionError, OSError):
            # The normal launch path will produce the detailed command error.
            pass

        if "--dangerously-skip-permissions" in help_text:
            flag = "--dangerously-skip-permissions"
        elif re.search(r"(?<![A-Za-z0-9_-])--auto(?![A-Za-z0-9_-])", help_text):
            flag = "--auto"
        else:
            # Prefer the current explicit flag when probing is unavailable.
            # Older CLIs that advertise --auto are handled by the branch above.
            flag = "--dangerously-skip-permissions"

        _AUTO_APPROVE_FLAG_CACHE[cache_key] = flag
        return flag

    async def run(
        self,
        prompt: str,
        *,
        model: str | None = None,
        agent: str | None = None,
        title: str = "Goal agent",
        status_callback: StatusCallback | None = None,
        cancel_check: CancelCheck | None = None,
        timeout_seconds: int | None = None,
        profile: OpenCodeProfile = "default",
        recovery_level: int = 0,
        context_window_tokens: int | None = None,
    ) -> OpenCodeResult:
        selected_model = model or self.config.model
        active_context_window = context_window_tokens or self.context_window_tokens(selected_model)
        auto_approve_flag = await self.detect_auto_approve_flag(cancel_check=cancel_check)
        command = self.build_command(
            model=selected_model,
            agent=agent,
            title=title,
            output_format="json",
            auto_approve_flag=auto_approve_flag,
        )
        started = time.monotonic()
        try:
            invocation = prepare_command(command)
            process = await asyncio.create_subprocess_exec(
                *invocation,
                cwd=str(self.config.project_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_opencode_environment(
                    agent=agent,
                    profile=profile,
                    context_window_tokens=active_context_window,
                    recovery_level=recovery_level,
                ),
                **process_group_kwargs(),
            )
        except (FileNotFoundError, CommandResolutionError) as exc:
            rendered = render_command(command)
            raise OpenCodeError(
                f"Could not start OpenCode. {exc}\nConfigured command: {rendered}"
            ) from exc

        assert process.stdin is not None
        process.stdin.write(prompt.encode("utf-8"))
        await process.stdin.drain()
        process.stdin.close()

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        text_parts: list[str] = []
        events: list[dict] = []
        reported_errors: list[str] = []
        session_id: str | None = None
        tool_output_chars = 0
        tool_budget = _profile_tool_budget(
            profile,
            context_window_tokens=active_context_window,
            recovery_level=recovery_level,
        )
        budget_error: OpenCodeContextBudgetExceededError | None = None
        step_events_seen = 0
        response_only_stall_error: OpenCodeError | None = None

        async def iter_stream_lines(
            stream: asyncio.StreamReader,
            *,
            stream_name: str,
        ):
            """Yield arbitrarily long newline-delimited records without readline limits.

            OpenCode JSON events can contain complete tool results and may exceed
            asyncio's default 64 KiB StreamReader line limit.  Using readline()
            allowed the reader task to fail silently, after which OpenCode could
            block forever on a full stdout pipe.
            """

            pending = bytearray()
            while True:
                chunk = await stream.read(_STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                pending.extend(chunk)
                records = pending.split(b"\n")
                pending = bytearray(records.pop())
                for raw_record in records:
                    yield raw_record.rstrip(b"\r")
                if len(pending) > _MAX_STREAM_LINE_BYTES:
                    raise OpenCodeError(
                        f"OpenCode {stream_name} produced a single record larger than "
                        f"{_MAX_STREAM_LINE_BYTES // (1024 * 1024)} MiB. "
                        "The output stream was stopped to prevent an indefinite hang."
                    )
            if pending:
                yield bytes(pending).rstrip(b"\r")

        async def read_stdout() -> None:
            nonlocal session_id, tool_output_chars, budget_error, step_events_seen
            assert process.stdout is not None
            async for raw in iter_stream_lines(process.stdout, stream_name="stdout"):
                line = raw.decode("utf-8", errors="replace")
                stdout_lines.append(line)
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if line.strip():
                        text_parts.append(line)
                    continue
                if not isinstance(event, dict):
                    continue
                events.append(event)
                properties = event.get("properties") if isinstance(event.get("properties"), dict) else {}
                session_id = (
                    event.get("sessionID")
                    or properties.get("sessionID")
                    or session_id
                )
                event_type = str(event.get("type", "event"))
                part = _event_part(event)
                part_type = str(part.get("type", ""))
                if event_type == "text" or part_type == "text":
                    text = _event_text(event)
                    if text:
                        text_parts.append(text)
                        if status_callback:
                            status_callback("text", text[-500:])
                elif event_type in {"tool_use", "tool"} or part_type in {
                    "tool",
                    "tool-use",
                }:
                    tool = part.get("tool") or part.get("name") or event.get("tool") or "tool"
                    part_state = part.get("state") if isinstance(part.get("state"), dict) else {}
                    state = (
                        part_state.get("status")
                        or part.get("state")
                        or event.get("state")
                        or "running"
                    )
                    # The NDJSON record includes the returned tool payload. Stop
                    # before OpenCode can feed an unbounded accumulation into the
                    # next model request. Workspace edits already made are kept.
                    tool_output_chars += len(line)
                    if (
                        tool_budget is not None
                        and tool_output_chars > tool_budget
                        and budget_error is None
                    ):
                        budget_error = OpenCodeContextBudgetExceededError(
                            observed_chars=tool_output_chars,
                            budget_chars=tool_budget,
                        )
                        if status_callback:
                            status_callback("context_recovery", str(budget_error))
                        return
                    if status_callback:
                        status_callback("tool", f"{tool}: {state}")
                elif event_type == "error":
                    detail = event.get("error") or properties.get("error") or "OpenCode reported an error"
                    if isinstance(detail, (dict, list)):
                        rendered_detail = json.dumps(detail, ensure_ascii=False)
                    else:
                        rendered_detail = str(detail)
                    reported_errors.append(rendered_detail)
                    if status_callback:
                        status_callback("error", rendered_detail[-500:])
                elif event_type in {"step_start", "step_finish"}:
                    step_events_seen += 1
                    if status_callback:
                        status_callback(event_type, event_type)
                elif status_callback:
                    status_callback(event_type, event_type)

        async def read_stderr() -> None:
            assert process.stderr is not None
            async for raw in iter_stream_lines(process.stderr, stream_name="stderr"):
                line = raw.decode("utf-8", errors="replace")
                stderr_lines.append(line)
                if status_callback and line.strip():
                    status_callback("stderr", line[-500:])

        stdout_task = asyncio.create_task(read_stdout(), name="opencode-stdout-reader")
        stderr_task = asyncio.create_task(read_stderr(), name="opencode-stderr-reader")
        wait_task = asyncio.create_task(process.wait(), name="opencode-process-wait")
        timeout = timeout_seconds or self.config.opencode_timeout_seconds
        interrupted = False
        interrupt_reason: str | None = None
        reader_error: BaseException | None = None

        if status_callback:
            status_callback("started", f"OpenCode started (timeout: {timeout}s)")

        try:
            while not wait_task.done():
                if (
                    profile in _TOOL_FREE_AGENT_PROMPTS
                    and not text_parts
                    and step_events_seen
                ):
                    stall_seconds = _safe_positive_int(
                        os.getenv("GOAL_AGENT_RESPONSE_ONLY_STALL_SECONDS")
                    ) or _DEFAULT_RESPONSE_ONLY_STALL_SECONDS
                    if time.monotonic() - started > stall_seconds:
                        if process.returncode is None:
                            await terminate_process_tree(process)
                        response_only_stall_error = OpenCodeError(
                            "OpenCode stalled in response-only mode and did not emit assistant text. "
                            "Goal Agent terminated the stuck process and will retry this structured step."
                        )
                        break
                if budget_error is not None:
                    if process.returncode is None:
                        await terminate_process_tree(process)
                    break
                for task in (stdout_task, stderr_task):
                    if task.done() and not task.cancelled():
                        exc = task.exception()
                        if exc is not None:
                            reader_error = exc
                            await terminate_process_tree(process)
                            break
                if reader_error is not None:
                    break
                if cancel_check:
                    reason = cancel_check()
                    if reason:
                        interrupted = True
                        interrupt_reason = reason
                        await terminate_process_tree(process)
                        break
                if time.monotonic() - started > timeout:
                    await terminate_process_tree(process)
                    interrupt_reason = "timeout"
                    interrupted = True
                    break
                await asyncio.sleep(self.config.poll_interval_seconds)
            await wait_task
        except asyncio.CancelledError:
            if process.returncode is None:
                await terminate_process_tree(process)
            raise
        finally:
            reader_results = await asyncio.gather(
                stdout_task, stderr_task, return_exceptions=True
            )
            for value in reader_results:
                if isinstance(value, BaseException) and not isinstance(
                    value, asyncio.CancelledError
                ):
                    reader_error = reader_error or value

        if budget_error is not None:
            raise budget_error

        if response_only_stall_error is not None:
            raise response_only_stall_error

        if reader_error is not None:
            raise OpenCodeError(
                "Failed while reading OpenCode output. The process was terminated "
                f"instead of leaving the request stuck: {reader_error}"
            ) from reader_error

        result = OpenCodeResult(
            text="\n".join(text_parts).strip(),
            raw_stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
            exit_code=process.returncode if process.returncode is not None else -1,
            session_id=session_id,
            interrupted=interrupted,
            interrupt_reason=interrupt_reason,
            duration_seconds=time.monotonic() - started,
            events=events,
        )
        if interrupted:
            raise OpenCodeInterrupted(interrupt_reason or "interrupted")
        overflow_parts = [*reported_errors, result.stderr]
        # Inspect unstructured stdout only on a failed process. A successful
        # assistant response may legitimately discuss a prior context error.
        if result.exit_code != 0:
            overflow_parts.append(result.raw_stdout)
        overflow = _context_overflow_error("\n".join(overflow_parts))
        if overflow is not None:
            raise overflow
        if result.exit_code != 0:
            raise OpenCodeError(
                f"OpenCode exited with code {result.exit_code}:\n{result.stderr or result.raw_stdout}"
            )
        return result

    async def run_structured(
        self,
        prompt: str,
        response_model: type[T],
        *,
        model: str | None = None,
        agent: str | None = None,
        title: str = "Goal agent structured step",
        status_callback: StatusCallback | None = None,
        cancel_check: CancelCheck | None = None,
        attempts: int = 2,
        context_overflow_retries: int = _DEFAULT_CONTEXT_OVERFLOW_RETRIES,
        profile: OpenCodeProfile = "default",
    ) -> tuple[T, OpenCodeResult]:
        schema = json.dumps(response_model.model_json_schema(), indent=2)

        def structured_prompt(task_prompt: str) -> str:
            return (
                f"{task_prompt.rstrip()}\n\n"
                "Your final response MUST contain exactly one JSON object between these markers:\n"
                f"{JSON_START}\n{{...}}\n{JSON_END}\n"
                "Do not place commentary inside the markers. The JSON must satisfy this schema:\n"
                f"{schema}\n"
            )

        def format_repair_prompt(raw_response: str) -> str:
            """Build a narrow conversion prompt from prose to required structured JSON."""

            bounded_response = _bounded_text(raw_response.strip(), 28_000)
            return (
                "FORMAT-REPAIR MODE\n\n"
                "Your previous response did not contain valid structured JSON. "
                "Do NOT add commentary. Convert the existing response content into exactly one "
                "JSON object between the required markers, preserving intent where possible.\n\n"
                "Output format requirement:\n"
                f"{JSON_START}\n{{...}}\n{JSON_END}\n\n"
                "The JSON must satisfy this schema:\n"
                f"{schema}\n\n"
                "If information is missing, use conservative defaults and explain gaps in "
                "assistant_message or readiness_reason fields rather than adding prose outside JSON.\n\n"
                "CONTENT TO CONVERT\n"
                f"{bounded_response}\n"
            )

        selected_model = model or self.config.model
        detected_context_window = self.context_window_tokens(selected_model)
        active_context_window = detected_context_window
        base_task_prompt = _bounded_text(
            prompt,
            _profile_prompt_char_limit(
                profile,
                context_window_tokens=active_context_window,
                recovery_level=0,
            ),
        )
        active_task_prompt = base_task_prompt
        last_error: Exception | None = None
        last_result: OpenCodeResult | None = None
        parse_attempt = 1
        overflow_attempt = 0

        while parse_attempt <= attempts:
            current_prompt = structured_prompt(active_task_prompt)
            if parse_attempt > 1 and last_error:
                current_prompt += (
                    "\nThe previous response could not be parsed. Correct the output format. "
                    f"Parser error: {last_error}\n"
                )
            try:
                result = await self.run(
                    current_prompt,
                    model=model,
                    agent=agent,
                    title=(
                        title
                        if overflow_attempt == 0
                        else f"{title} (context recovery {overflow_attempt})"
                    ),
                    status_callback=status_callback,
                    cancel_check=cancel_check,
                    profile=profile,
                    recovery_level=overflow_attempt,
                    context_window_tokens=active_context_window,
                )
            except OpenCodeContextOverflowError as exc:
                if overflow_attempt >= context_overflow_retries:
                    if status_callback:
                        status_callback(
                            "error",
                            f"Context recovery exhausted after {overflow_attempt} fresh-session "
                            f"retries: {exc}",
                        )
                    raise
                overflow_attempt += 1
                if exc.context_size is not None:
                    active_context_window = exc.context_size
                active_task_prompt = _context_recovery_prompt(
                    base_task_prompt,
                    retry_number=overflow_attempt,
                    profile=profile,
                    context_window_tokens=active_context_window,
                )
                last_error = None
                parse_attempt = 1
                if status_callback:
                    requested = (
                        f"{exc.requested_tokens:,}"
                        if exc.requested_tokens is not None
                        else "too many"
                    )
                    limit = (
                        f"{exc.context_size:,}"
                        if exc.context_size is not None
                        else "the available"
                    )
                    status_callback(
                        "context_recovery",
                        f"OpenCode used {requested} prompt tokens for a {limit}-token window. "
                        f"Starting fresh compact retry {overflow_attempt}/{context_overflow_retries} "
                        "with pruned tool history and restricted project inspection.",
                    )
                continue
            except OpenCodeError as exc:
                # Some OpenCode/model combinations can loop forever emitting only
                # step events in response-only mode. Retry the structured call
                # once or twice before surfacing the error.
                if (
                    profile in _TOOL_FREE_AGENT_PROMPTS
                    and "did not emit assistant text" in str(exc).lower()
                    and parse_attempt < attempts
                ):
                    last_error = exc
                    parse_attempt += 1
                    if status_callback:
                        status_callback(
                            "retry",
                            "OpenCode produced no assistant text in response-only mode; retrying in a fresh process.",
                        )
                    continue
                raise

            last_result = result
            try:
                if status_callback:
                    status_callback("parsing", "Validating OpenCode response")
                payload = extract_json_payload(result.text or result.raw_stdout)
                validated = response_model.model_validate(payload)
                if status_callback:
                    suffix = (
                        f" after {overflow_attempt} context-recovery "
                        f"{'retry' if overflow_attempt == 1 else 'retries'}"
                        if overflow_attempt
                        else ""
                    )
                    status_callback("complete", f"Response received and validated{suffix}")
                return validated, result
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                # Some providers can answer with a prose analysis even when the
                # prompt requires marker-delimited JSON. Attempt one bounded
                # conversion pass before consuming another full retry attempt.
                raw_response = (result.text or result.raw_stdout).strip()
                if raw_response:
                    try:
                        if status_callback:
                            status_callback(
                                "retry",
                                "Response was prose instead of JSON; attempting format repair.",
                            )
                        repair_result = await self.run(
                            format_repair_prompt(raw_response),
                            model=model,
                            agent=agent,
                            title=f"{title} (format repair)",
                            status_callback=status_callback,
                            cancel_check=cancel_check,
                            profile=profile,
                            recovery_level=overflow_attempt,
                            context_window_tokens=active_context_window,
                        )
                        repair_payload = extract_json_payload(
                            repair_result.text or repair_result.raw_stdout
                        )
                        repaired = response_model.model_validate(repair_payload)
                        if status_callback:
                            status_callback(
                                "complete",
                                "Response repaired and validated as structured JSON.",
                            )
                        return repaired, repair_result
                    except (OpenCodeError, ValueError, ValidationError, json.JSONDecodeError):
                        pass

                last_error = exc
                parse_attempt += 1
                if parse_attempt <= attempts and status_callback:
                    status_callback(
                        "retry",
                        f"Response format was invalid; retrying: {exc}",
                    )

        assert last_result is not None
        raise OpenCodeError(
            f"OpenCode did not return valid {response_model.__name__} JSON after {attempts} attempts: "
            f"{last_error}\nDiagnostic output:\n{_diagnostic_output(last_result)}"
        )

    async def list_models(self) -> list[str]:
        command = [*self.config.opencode_command, "models"]
        try:
            invocation = prepare_command(command)
            process = await asyncio.create_subprocess_exec(
                *invocation,
                cwd=str(self.config.project_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, CommandResolutionError) as exc:
            raise OpenCodeError(
                f"Could not start OpenCode. {exc}\nConfigured command: {render_command(command)}"
            ) from exc
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise OpenCodeError(stderr.decode("utf-8", errors="replace"))
        models: list[str] = []
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            candidate = line.strip()
            if candidate and "/" in candidate and not candidate.startswith(("Provider", "-")):
                models.append(candidate.split()[0])
        return list(dict.fromkeys(models))


def extract_json_payload(text: str) -> dict:
    """Extract the model's structured payload without confusing NDJSON events.

    ``opencode run --format json`` writes one event object per line.  Parsing the
    whole stream as one JSON document produces the misleading ``Extra data``
    error seen when OpenCode emitted tool events but no final assistant message.
    """

    marker_pattern = re.compile(
        re.escape(JSON_START) + r"\s*(\{.*?\})\s*" + re.escape(JSON_END),
        re.DOTALL,
    )
    matches = marker_pattern.findall(text)
    if matches:
        return json.loads(matches[-1])

    fence_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_matches:
        return json.loads(fence_matches[-1])

    stripped = text.strip()
    if not stripped:
        raise ValueError("OpenCode completed without an assistant text response")

    # A model may return a bare single JSON object without markers.
    try:
        direct = json.loads(stripped)
    except json.JSONDecodeError:
        direct = None
    if isinstance(direct, dict) and not (
        "type" in direct and ("sessionID" in direct or "part" in direct)
    ):
        return direct

    events = _parse_ndjson_events(text)
    if events is not None:
        assistant_parts = [value for event in events if (value := _event_text(event))]
        if assistant_parts:
            return extract_json_payload("\n".join(assistant_parts))
        summary = _event_stream_summary(events)
        raise ValueError(
            "OpenCode completed without a final assistant text response. "
            f"It emitted only event records ({summary}). This usually means the agent "
            "used its step budget on tools or stopped before returning the required JSON."
        )

    # Last-resort support for prose surrounding one unmarked JSON object. Use
    # raw_decode from each opening brace rather than slicing first-to-last, which
    # would again combine multiple JSON documents and produce Extra data.
    decoder = json.JSONDecoder()
    candidates: list[dict] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    for candidate in reversed(candidates):
        if not ("type" in candidate and ("sessionID" in candidate or "part" in candidate)):
            return candidate

    raise ValueError("No proposal JSON object was found in OpenCode's assistant response")

