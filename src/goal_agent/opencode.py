from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .command_resolver import CommandResolutionError, prepare_command, render_command
from .models import AppConfig
from .process_utils import process_group_kwargs, terminate_process_tree

T = TypeVar("T", bound=BaseModel)
StatusCallback = Callable[[str, str], None]
CancelCheck = Callable[[], str | None]

JSON_START = "<GOAL_AGENT_JSON>"
JSON_END = "</GOAL_AGENT_JSON>"

_STREAM_CHUNK_BYTES = 64 * 1024
_MAX_STREAM_LINE_BYTES = 32 * 1024 * 1024

_AUTO_APPROVE_FLAG_CACHE: dict[tuple[str, ...], str] = {}

# Keep enough room for OpenCode's system prompt, tool schemas, and a final response.
# This is injected only into Goal Agent child processes and does not modify the
# user's OpenCode configuration files.
_GOAL_AGENT_COMPACTION_RESERVED_TOKENS = 12_000
_DEFAULT_CONTEXT_OVERFLOW_RETRIES = 2
_COMPACT_RECOVERY_PROMPT_CHARS = 28_000


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


def _context_recovery_prompt(prompt: str, *, retry_number: int) -> str:
    """Build a fresh-session task brief that discourages context-expanding tool use."""

    budget = max(12_000, _COMPACT_RECOVERY_PROMPT_CHARS // retry_number)
    bounded = _bounded_text(prompt, budget)
    file_limit = 8 if retry_number == 1 else 4
    return f"""
CONTEXT-OVERFLOW RECOVERY MODE — FRESH SESSION {retry_number}

A previous OpenCode process exceeded the model context window. This is a new session; do not
continue, compact, or reconstruct the failed session. The task brief below is authoritative.
The previous process may already have changed workspace files, so inspect current state before
acting, but keep inspection strictly bounded.

Rules for this recovery attempt:
- Do not delegate to subagents or use broad/exhaustive project exploration.
- Do not recursively read the repository or inspect dependency, vendor, build, cache, log,
  generated, binary, media, .git, node_modules, dist, target, coverage, or .goal-agent trees.
- Inspect at most {file_limit} small, directly relevant text files. Read bounded sections rather
  than entire large files. Prefer exact paths named in the task, current hypothesis, tests, or errors.
- Run only focused commands whose output is bounded. Redirect or filter noisy output.
- Complete one bounded decision/change and return the required final structured response promptly.
- If more work is needed than fits this attempt, report the remaining work as blockers instead of
  expanding context until it overflows again.

BOUNDED TASK BRIEF
{bounded}
""".strip()


def _opencode_environment() -> dict[str, str]:
    """Return an environment with safer OpenCode compaction for agent-loop child runs.

    OpenCode supports runtime overrides through OPENCODE_CONFIG_CONTENT.  Enabling
    pruning and reserving headroom reduces the chance that large tool results push the
    next model request over its context limit. Existing inline settings are preserved.
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
            # Do not replace a user-supplied value that OpenCode may understand even if
            # it is JSONC or otherwise not parseable by Python. Recovery retries still apply.
            return env

    existing = config.get("compaction")
    compaction = dict(existing) if isinstance(existing, dict) else {}
    compaction["auto"] = True
    compaction["prune"] = True
    current_reserved = compaction.get("reserved")
    if not isinstance(current_reserved, int) or current_reserved < _GOAL_AGENT_COMPACTION_RESERVED_TOKENS:
        compaction["reserved"] = _GOAL_AGENT_COMPACTION_RESERVED_TOKENS
    config["compaction"] = compaction
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config, separators=(",", ":"))
    return env


class OpenCodeRunner:
    def __init__(self, config: AppConfig):
        self.config = config

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
    ) -> OpenCodeResult:
        selected_model = model or self.config.model
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
                env=_opencode_environment(),
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

        def event_part(event: dict) -> dict:
            part = event.get("part")
            if isinstance(part, dict):
                return part
            properties = event.get("properties")
            if isinstance(properties, dict) and isinstance(properties.get("part"), dict):
                return properties["part"]
            return {}

        async def read_stdout() -> None:
            nonlocal session_id
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
                part = event_part(event)
                part_type = str(part.get("type", ""))
                if event_type == "text" or part_type == "text":
                    text = part.get("text") or event.get("text") or event.get("delta")
                    if isinstance(text, str) and text:
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

        base_task_prompt = prompt
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
                active_task_prompt = _context_recovery_prompt(
                    base_task_prompt, retry_number=overflow_attempt
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
            f"{last_error}\nRaw output:\n{last_result.text or last_result.raw_stdout}"
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

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in OpenCode output")
