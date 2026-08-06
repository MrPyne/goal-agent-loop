from __future__ import annotations

import asyncio
import glob
import os
import re
import time
from pathlib import Path
from typing import Callable

from .models import (
    AppConfig,
    CriterionDefinition,
    CriterionKind,
    CriterionResult,
    JudgeDecision,
    OverrideMode,
    utc_now,
)
from .opencode import CancelCheck, OpenCodeRunner
from .process_utils import process_group_kwargs, terminate_process_tree

ResultCallback = Callable[[CriterionResult], None]
StatusCallback = Callable[[str, str], None]

_AI_EVIDENCE_MAX_FILES = 8
_AI_EVIDENCE_MAX_FILE_CHARS = 12_000
_AI_EVIDENCE_MAX_TOTAL_CHARS = 48_000
_COMMAND_OUTPUT_JUDGE_MAX_CHARS = 10_000
_EXCLUDED_EVIDENCE_DIRS = {
    ".git",
    ".goal-agent",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "target",
    "coverage",
    "vendor",
    "__pycache__",
}
_TEXT_SUFFIXES = {
    ".txt", ".md", ".rst", ".json", ".jsonl", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".csv", ".tsv", ".py", ".js", ".jsx", ".ts", ".tsx",
    ".html", ".css", ".scss", ".java", ".kt", ".c", ".h", ".cpp", ".hpp",
    ".cs", ".go", ".rs", ".sh", ".ps1", ".bat", ".cmd", ".xml", ".sql",
    ".log",
}


class EvaluationInterrupted(RuntimeError):
    pass


class CriteriaEvaluator:
    def __init__(self, config: AppConfig, runner: OpenCodeRunner):
        self.config = config
        self.runner = runner

    async def evaluate_all(
        self,
        criteria: list[CriterionDefinition],
        *,
        goal: str,
        steering: str,
        model: str | None,
        cancel_check: CancelCheck | None = None,
        result_callback: ResultCallback | None = None,
        status_callback: StatusCallback | None = None,
    ) -> dict[str, CriterionResult]:
        results: dict[str, CriterionResult] = {}
        for criterion in criteria:
            if cancel_check and cancel_check():
                raise EvaluationInterrupted(cancel_check() or "interrupted")
            if status_callback:
                status_callback("criterion", f"Checking {criterion.id}: {criterion.description}")
            result = await self.evaluate_one(
                criterion,
                goal=goal,
                steering=steering,
                model=model,
                cancel_check=cancel_check,
                status_callback=status_callback,
            )
            results[criterion.id] = result
            if result_callback:
                result_callback(result)
        return results

    async def evaluate_one(
        self,
        criterion: CriterionDefinition,
        *,
        goal: str,
        steering: str,
        model: str | None,
        cancel_check: CancelCheck | None = None,
        status_callback: StatusCallback | None = None,
    ) -> CriterionResult:
        started = time.monotonic()
        if criterion.override != OverrideMode.AUTO:
            passed = criterion.override == OverrideMode.PASS
            return CriterionResult(
                criterion_id=criterion.id,
                passed=passed,
                status="pass" if passed else "fail",
                summary=f"User override: {criterion.override.value}",
                evaluation_method="human_override",
                checked_at=utc_now(),
                duration_seconds=time.monotonic() - started,
            )

        try:
            if criterion.kind == CriterionKind.COMMAND:
                result = await self._command(criterion, model=model, cancel_check=cancel_check)
            elif criterion.kind == CriterionKind.FILE_EXISTS:
                result = self._file_exists(criterion)
            elif criterion.kind == CriterionKind.FILE_CONTAINS:
                result = self._file_contains(criterion)
            elif criterion.kind == CriterionKind.AI_JUDGE:
                result = await self._ai_judge(
                    criterion,
                    goal=goal,
                    steering=steering,
                    model=model,
                    cancel_check=cancel_check,
                    status_callback=status_callback,
                )
            elif criterion.kind == CriterionKind.MANUAL:
                result = CriterionResult(
                    criterion_id=criterion.id,
                    passed=False,
                    status="fail",
                    summary="Human approval is required; AI and automated checks cannot pass this criterion",
                    evidence=[
                        "Use an AI evidence review criterion when the outcome can be judged from project evidence, "
                        "or set a human override after personal verification."
                    ],
                    evaluation_method="human_required",
                )
            else:
                raise ValueError(f"Unsupported criterion kind: {criterion.kind}")
        except EvaluationInterrupted:
            raise
        except Exception as exc:
            result = CriterionResult(
                criterion_id=criterion.id,
                passed=False,
                status="error",
                summary="Criterion check failed",
                error=str(exc),
                evidence=[str(exc)],
            )
        if result.evaluation_method is None:
            result.evaluation_method = {
                CriterionKind.COMMAND: "command",
                CriterionKind.FILE_EXISTS: "file_exists",
                CriterionKind.FILE_CONTAINS: "file_contains",
                CriterionKind.AI_JUDGE: "ai_judge",
                CriterionKind.MANUAL: "human_required",
            }[criterion.kind]
        result.checked_at = utc_now()
        result.duration_seconds = time.monotonic() - started
        return result

    async def _command(
        self,
        criterion: CriterionDefinition,
        *,
        model: str | None,
        cancel_check: CancelCheck | None,
    ) -> CriterionResult:
        assert criterion.command is not None
        process = await asyncio.create_subprocess_shell(
            criterion.command,
            cwd=str(self.config.project_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **process_group_kwargs(),
        )
        started = time.monotonic()
        timeout = criterion.timeout_seconds or self.config.criterion_timeout_seconds
        communicate_task = asyncio.create_task(process.communicate())
        while not communicate_task.done():
            if cancel_check:
                reason = cancel_check()
                if reason:
                    await terminate_process_tree(process)
                    communicate_task.cancel()
                    raise EvaluationInterrupted(reason)
            if time.monotonic() - started > timeout:
                await terminate_process_tree(process)
                communicate_task.cancel()
                return CriterionResult(
                    criterion_id=criterion.id,
                    passed=False,
                    status="error",
                    summary=f"Command timed out after {timeout}s",
                    evidence=[criterion.command],
                )
            await asyncio.sleep(self.config.poll_interval_seconds)
        stdout, stderr = await communicate_task
        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        passed = process.returncode == criterion.expected_exit_code
        checks: list[str] = []

        def _assert_output(
            *,
            label: str,
            pattern: str,
            value: str,
            regex: bool,
            case_sensitive: bool,
        ) -> bool:
            if regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                matched = re.search(pattern, value, flags=flags) is not None
            else:
                haystack = value if case_sensitive else value.lower()
                needle = pattern if case_sensitive else pattern.lower()
                matched = needle in haystack
            checks.append(
                f"{label} {'matched' if matched else 'did not match'} "
                + (f"regex /{pattern}/" if regex else f"text '{pattern}'")
            )
            return matched

        if criterion.stdout_contains is not None:
            passed = passed and _assert_output(
                label="stdout",
                pattern=criterion.stdout_contains,
                value=stdout_text,
                regex=False,
                case_sensitive=criterion.output_case_sensitive,
            )
        if criterion.stderr_contains is not None:
            passed = passed and _assert_output(
                label="stderr",
                pattern=criterion.stderr_contains,
                value=stderr_text,
                regex=False,
                case_sensitive=criterion.output_case_sensitive,
            )
        if criterion.stdout_regex is not None:
            passed = passed and _assert_output(
                label="stdout",
                pattern=criterion.stdout_regex,
                value=stdout_text,
                regex=True,
                case_sensitive=criterion.output_case_sensitive,
            )
        if criterion.stderr_regex is not None:
            passed = passed and _assert_output(
                label="stderr",
                pattern=criterion.stderr_regex,
                value=stderr_text,
                regex=True,
                case_sensitive=criterion.output_case_sensitive,
            )

        ai_judge_summary: str | None = None
        ai_judge_evidence: list[str] = []
        ai_judge_confidence: float | None = None
        if criterion.output_judge_prompt:
            decision = await self._judge_command_output(
                criterion,
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                exit_code=process.returncode,
                model=model,
                cancel_check=cancel_check,
            )
            ai_judge_confidence = decision.confidence
            ai_judge_passed = (
                decision.passed and decision.confidence >= criterion.output_confidence_threshold
            )
            passed = passed and ai_judge_passed
            ai_judge_summary = decision.summary
            ai_judge_evidence = [*decision.evidence]
            if decision.missing:
                ai_judge_evidence.append("Missing: " + "; ".join(decision.missing))

        evidence = [f"$ {criterion.command}", f"exit code: {process.returncode}"]
        if checks:
            evidence.extend(checks)
        if stdout_text:
            evidence.append("stdout:\n" + _tail(stdout_text))
        if stderr_text:
            evidence.append("stderr:\n" + _tail(stderr_text))
        evidence.extend(ai_judge_evidence)

        if ai_judge_summary:
            summary = ai_judge_summary
        elif checks:
            summary = (
                "Command output satisfied all configured checks"
                if passed
                else "Command output did not satisfy one or more configured checks"
            )
        else:
            summary = (
                f"Command exited with expected code {criterion.expected_exit_code}"
                if passed
                else f"Expected exit code {criterion.expected_exit_code}, got {process.returncode}"
            )

        return CriterionResult(
            criterion_id=criterion.id,
            passed=passed,
            status="pass" if passed else "fail",
            summary=summary,
            evidence=evidence,
            confidence=ai_judge_confidence,
        )

    async def _judge_command_output(
        self,
        criterion: CriterionDefinition,
        *,
        stdout_text: str,
        stderr_text: str,
        exit_code: int,
        model: str | None,
        cancel_check: CancelCheck | None,
    ) -> JudgeDecision:
        prompt = f"""
You are validating whether a command criterion is satisfied from command output.
Use only the supplied command, exit code, stdout, stderr, and rubric. Do not call tools.

CRITERION
ID: {criterion.id}
Description: {criterion.description[:4000]}
Rubric: {(criterion.output_judge_prompt or '')[:8000]}
Minimum confidence required: {criterion.output_confidence_threshold}

COMMAND EXECUTION
Command: {criterion.command}
Exit code: {exit_code}
Expected exit code: {criterion.expected_exit_code}

STDOUT
{self._clip_evidence(stdout_text, _COMMAND_OUTPUT_JUDGE_MAX_CHARS)}

STDERR
{self._clip_evidence(stderr_text, _COMMAND_OUTPUT_JUDGE_MAX_CHARS)}

Decide PASS only when the output satisfies the rubric with concrete evidence. If output is
ambiguous or incomplete, fail and explain what is missing.
"""
        decision, _ = await self.runner.run_structured(
            prompt,
            JudgeDecision,
            model=model,
            agent=self.config.evaluator_agent,
            title=f"Judge command output {criterion.id}",
            cancel_check=cancel_check,
            profile="judge",
        )
        return decision

    def _file_exists(self, criterion: CriterionDefinition) -> CriterionResult:
        assert criterion.path is not None
        path = self._resolve_path(criterion.path)
        passed = path.exists()
        return CriterionResult(
            criterion_id=criterion.id,
            passed=passed,
            status="pass" if passed else "fail",
            summary=f"{path.relative_to(self.config.project_path)} {'exists' if passed else 'does not exist'}",
            evidence=[str(path)],
        )

    def _file_contains(self, criterion: CriterionDefinition) -> CriterionResult:
        assert criterion.path is not None
        assert criterion.contains is not None
        path = self._resolve_path(criterion.path)
        if not path.is_file():
            return CriterionResult(
                criterion_id=criterion.id,
                passed=False,
                status="fail",
                summary="File does not exist",
                evidence=[str(path)],
            )
        content = path.read_text(encoding="utf-8", errors="replace")
        needle = criterion.contains
        if criterion.regex:
            flags = 0 if criterion.case_sensitive else re.IGNORECASE
            passed = re.search(needle, content, flags=flags) is not None
        else:
            haystack = content if criterion.case_sensitive else content.lower()
            expected = needle if criterion.case_sensitive else needle.lower()
            passed = expected in haystack
        return CriterionResult(
            criterion_id=criterion.id,
            passed=passed,
            status="pass" if passed else "fail",
            summary=("Required content found" if passed else "Required content not found"),
            evidence=[f"path: {criterion.path}", f"pattern: {needle}"],
        )

    async def _ai_judge(
        self,
        criterion: CriterionDefinition,
        *,
        goal: str,
        steering: str,
        model: str | None,
        cancel_check: CancelCheck | None,
        status_callback: StatusCallback | None,
    ) -> CriterionResult:
        evidence_paths = "\n".join(f"- {path}" for path in criterion.evidence_paths) or "- none specified"
        evidence_snapshot = self._collect_ai_evidence(criterion)
        prompt = f"""
You are the evaluator in a persistent autonomous goal loop. Goal Agent has already collected a bounded, read-only
snapshot of the requested workspace evidence below. Do not inspect the project or call tools; judge only from the
criterion, the supplied evidence, and the explicit pass/fail rubric.

OVERALL GOAL
{goal[:8000]}

CRITERION
ID: {criterion.id}
Description: {criterion.description[:4000]}
Judging instruction: {(criterion.judge_prompt or '')[:12000]}
Minimum confidence required: {criterion.confidence_threshold}
Suggested evidence paths:
{evidence_paths}

CURRENT USER STEERING
{steering[:6000]}

BOUNDED WORKSPACE EVIDENCE
{evidence_snapshot}

Decide whether the criterion is actually satisfied now. Require concrete evidence from the supplied snapshot.
Do not pass it based only on another agent's claim. If evidence is missing, truncated at a decisive point, stale,
or ambiguous, fail it and identify exactly what evidence is still needed. Return the decision promptly without tools.
"""
        decision, _ = await self.runner.run_structured(
            prompt,
            JudgeDecision,
            model=model,
            agent=self.config.evaluator_agent,
            title=f"Evaluate criterion {criterion.id}",
            status_callback=status_callback,
            cancel_check=cancel_check,
            profile="judge",
        )
        passed = decision.passed and decision.confidence >= criterion.confidence_threshold
        evidence = [*decision.evidence]
        if decision.missing:
            evidence.append("Missing: " + "; ".join(decision.missing))
        return CriterionResult(
            criterion_id=criterion.id,
            passed=passed,
            status="pass" if passed else "fail",
            summary=decision.summary,
            evidence=evidence,
            confidence=decision.confidence,
        )

    def _collect_ai_evidence(self, criterion: CriterionDefinition) -> str:
        """Collect bounded project evidence so the judge does not need OpenCode tools.

        This makes qualitative checks deterministic with respect to what the model
        sees and prevents one large read/glob result from overflowing the model
        context before OpenCode can compact it.
        """

        candidates: list[Path] = []
        notes: list[str] = []
        patterns = criterion.evidence_paths or []
        for raw_pattern in patterns:
            if len(candidates) >= _AI_EVIDENCE_MAX_FILES:
                break
            try:
                base = self._resolve_path(raw_pattern) if not glob.has_magic(raw_pattern) else None
            except ValueError as exc:
                notes.append(f"- {raw_pattern}: rejected ({exc})")
                continue

            matches: list[Path]
            if base is not None:
                matches = [base]
            else:
                absolute_pattern = str(self.config.project_path / raw_pattern)
                matches = []
                for value in glob.iglob(absolute_pattern, recursive=True):
                    matches.append(Path(value).resolve())
                    if len(matches) >= _AI_EVIDENCE_MAX_FILES:
                        break

            for match in sorted(matches, key=lambda value: str(value).lower()):
                if len(candidates) >= _AI_EVIDENCE_MAX_FILES:
                    break
                try:
                    relative_match = match.relative_to(self.config.project_path)
                except ValueError:
                    notes.append(f"- {raw_pattern}: ignored match outside project ({match})")
                    continue
                if any(part in _EXCLUDED_EVIDENCE_DIRS for part in relative_match.parts):
                    continue
                if match.is_file():
                    candidates.append(match)
                elif match.is_dir():
                    for root, dirs, files in os.walk(match):
                        dirs[:] = [name for name in dirs if name not in _EXCLUDED_EVIDENCE_DIRS]
                        for filename in sorted(files):
                            path = Path(root) / filename
                            if self._is_text_evidence(path):
                                candidates.append(path)
                                if len(candidates) >= _AI_EVIDENCE_MAX_FILES:
                                    break
                        if len(candidates) >= _AI_EVIDENCE_MAX_FILES:
                            break
                else:
                    notes.append(f"- {raw_pattern}: does not exist")

        if not patterns:
            notes.append("- No evidence_paths were specified; no workspace files were supplied to the AI judge.")
        elif not candidates:
            notes.append("- No readable text evidence matched the configured evidence_paths.")

        chunks: list[str] = []
        used = 0
        seen: set[Path] = set()
        for path in candidates:
            if path in seen or used >= _AI_EVIDENCE_MAX_TOTAL_CHARS:
                continue
            seen.add(path)
            relative = path.relative_to(self.config.project_path)
            if not self._is_text_evidence(path):
                notes.append(f"- {relative}: binary or unsupported text format; metadata only")
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                notes.append(f"- {relative}: could not read ({exc})")
                continue
            available = min(
                _AI_EVIDENCE_MAX_FILE_CHARS,
                _AI_EVIDENCE_MAX_TOTAL_CHARS - used,
            )
            clipped = self._clip_evidence(content, available)
            used += len(clipped)
            chunks.append(
                f"--- {relative.as_posix()} ({path.stat().st_size} bytes) ---\n{clipped}"
            )

        header = "Evidence collection notes:\n" + ("\n".join(notes) if notes else "- none")
        if not chunks:
            return header + "\n\nNo file content was collected."
        return header + "\n\n" + "\n\n".join(chunks)

    @staticmethod
    def _is_text_evidence(path: Path) -> bool:
        if path.suffix.lower() in _TEXT_SUFFIXES:
            return True
        return path.name.lower() in {
            "dockerfile", "makefile", "readme", "license", "agents.md", "context.md"
        }

    @staticmethod
    def _clip_evidence(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        head = max_chars * 3 // 5
        tail = max_chars - head
        return (
            text[:head].rstrip()
            + "\n\n… [middle omitted by Goal Agent evidence budget] …\n\n"
            + text[-tail:].lstrip()
        )

    def _resolve_path(self, relative_path: str) -> Path:
        path = (self.config.project_path / relative_path).resolve()
        try:
            path.relative_to(self.config.project_path)
        except ValueError as exc:
            raise ValueError(f"Criterion path must stay inside project: {relative_path}") from exc
        return path


def all_required_pass(
    criteria: list[CriterionDefinition], results: dict[str, CriterionResult]
) -> bool:
    required = [criterion for criterion in criteria if criterion.required]
    return bool(required) and all(results.get(item.id) and results[item.id].passed for item in required)


def passed_required_count(
    criteria: list[CriterionDefinition], results: dict[str, CriterionResult]
) -> int:
    return sum(
        1
        for criterion in criteria
        if criterion.required and results.get(criterion.id) and results[criterion.id].passed
    )


def _tail(text: str, max_chars: int = 4000) -> str:
    return text if len(text) <= max_chars else "…" + text[-max_chars:]
