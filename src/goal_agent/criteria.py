from __future__ import annotations

import asyncio
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
                result = await self._command(criterion, cancel_check)
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
        self, criterion: CriterionDefinition, cancel_check: CancelCheck | None
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
        evidence = [f"$ {criterion.command}", f"exit code: {process.returncode}"]
        if stdout_text:
            evidence.append("stdout:\n" + _tail(stdout_text))
        if stderr_text:
            evidence.append("stderr:\n" + _tail(stderr_text))
        return CriterionResult(
            criterion_id=criterion.id,
            passed=passed,
            status="pass" if passed else "fail",
            summary=(
                f"Command exited with expected code {criterion.expected_exit_code}"
                if passed
                else f"Expected exit code {criterion.expected_exit_code}, got {process.returncode}"
            ),
            evidence=evidence,
        )

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
        prompt = f"""
You are the evaluator in a persistent autonomous goal loop. Inspect the project in the current working directory.
You are read-only: do not modify project files.

OVERALL GOAL
{goal}

CRITERION
ID: {criterion.id}
Description: {criterion.description}
Judging instruction: {criterion.judge_prompt}
Minimum confidence required: {criterion.confidence_threshold}
Suggested evidence paths:
{evidence_paths}

CURRENT USER STEERING
{steering[:6000]}

CONTEXT DISCIPLINE
Start with the suggested evidence paths. Do not delegate to subagents or perform broad recursive searches. Inspect at
most eight small, directly relevant text files and bounded command output. Avoid dependency, vendor, build, cache, log,
generated, media, binary, .git, and .goal-agent trees. Return a decision promptly.

Decide whether the criterion is actually satisfied now. Require concrete evidence from the workspace.
Do not pass it based only on another agent's claim. If evidence is missing or ambiguous, fail it.
"""
        decision, _ = await self.runner.run_structured(
            prompt,
            JudgeDecision,
            model=model,
            agent=self.config.evaluator_agent,
            title=f"Evaluate criterion {criterion.id}",
            status_callback=status_callback,
            cancel_check=cancel_check,
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
