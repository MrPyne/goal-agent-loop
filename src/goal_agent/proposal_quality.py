from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

from .command_resolver import resolve_executable
from .models import (
    CriteriaQualityIssue,
    CriterionDefinition,
    CriterionKind,
    SetupProposal,
)

_VAGUE_WORDS = re.compile(
    r"\b(good|great|high[- ]quality|clear|complete|correct|proper|properly|"
    r"user[- ]friendly|robust|reasonable|appropriate|works?|working|polished|"
    r"professional|efficient|fast|easy|usable|well[- ]documented)\b",
    re.IGNORECASE,
)
_EXPLICIT_SIGNAL = re.compile(
    r"\b(pass only if|fail if|must|at least|at most|exactly|no more than|"
    r"exit code|contains|exists|all of|each of|every|without|\d+(?:\.\d+)?%?)\b",
    re.IGNORECASE,
)
_EVIDENCE_SIGNAL = re.compile(
    r"\b(evidence|inspect|verify|read|run|output|test|file|report|log|screenshot|"
    r"response|result|artifact|command)\b",
    re.IGNORECASE,
)
_TRIVIAL_COMMAND = re.compile(
    r"^\s*(true|:\s*|echo\b.*|exit\s+0|python\s+-c\s+[\"']?pass[\"']?)\s*$",
    re.IGNORECASE,
)
_SHELL_BUILTINS = {
    "cd",
    "set",
    "export",
    "alias",
    "source",
    ".",
    "pushd",
    "popd",
}
_PYTHON_LAUNCHERS = {"python", "python3", "py"}
_METRIC_THRESHOLD = re.compile(
    r"\b(?:score|accuracy|pass[ -]?rate|success[ -]?rate|percentage)\b[\s\S]{0,100}?"
    r"\b(?:at least|at most|no more than|>=|<=|above|below)\s*\d+(?:\.\d+)?\s*%?",
    re.IGNORECASE,
)


def _command_executable(command: str) -> str | None:
    text = command.strip()
    if not text:
        return None
    try:
        parts = shlex.split(text, posix=os.name != "nt")
    except ValueError:
        # Fall back to first token when shell quoting is malformed.
        parts = text.split()
    if not parts:
        return None
    return parts[0].strip().strip('"').strip("'")


def _command_parts(command: str) -> list[str]:
    text = command.strip()
    if not text:
        return []
    try:
        parts = shlex.split(text, posix=os.name != "nt")
    except ValueError:
        parts = text.split()
    return [part.strip() for part in parts if part.strip()]


def _project_relative_path(path_text: str, *, project_path: Path) -> Path | None:
    candidate = (project_path / path_text).resolve()
    try:
        candidate.relative_to(project_path)
    except ValueError:
        return None
    return candidate


def _criterion_signature(criterion: CriterionDefinition) -> tuple:
    return (
        criterion.kind.value,
        (criterion.command or "").strip(),
        (criterion.path or "").strip(),
        (criterion.contains or "").strip(),
        (criterion.judge_prompt or "").strip(),
        (criterion.output_judge_prompt or "").strip(),
    )


def assess_setup_proposal(
    proposal: SetupProposal,
    *,
    project_path: Path | None = None,
) -> SetupProposal:
    """Add deterministic quality findings and prevent premature finalization.

    The AI still designs the goal and criteria, but this guard catches common vague or
    non-verifiable stopping gates before the UI presents them as final.
    """

    checked = proposal.model_copy(deep=True)
    issues: list[CriteriaQualityIssue] = list(checked.criteria_quality_issues)
    seen: set[tuple[str | None, str]] = {
        (item.criterion_id, item.issue.strip().lower()) for item in issues
    }

    def add(
        criterion_id: str | None,
        issue: str,
        suggested_fix: str,
        severity: str = "blocking",
    ) -> None:
        key = (criterion_id, issue.strip().lower())
        if key in seen:
            return
        seen.add(key)
        issues.append(
            CriteriaQualityIssue(
                criterion_id=criterion_id,
                issue=issue,
                suggested_fix=suggested_fix,
                severity=severity,
            )
        )

    required = [item for item in checked.criteria if item.required]
    if not required:
        add(
            None,
            "No required success criterion exists.",
            "Add at least one required criterion that directly proves the goal outcome.",
        )

    ids: set[str] = set()
    seen_signatures: dict[tuple, str] = {}
    for criterion in checked.criteria:
        cid = criterion.id
        if cid in ids:
            add(cid, "Criterion ID is duplicated.", "Use a unique stable ID.")
        ids.add(cid)

        description = criterion.description.strip()
        if len(description) < 18:
            add(
                cid,
                "The criterion description is too short to define an observable result.",
                "State the exact artifact, behavior, threshold, or output that must be observed.",
            )
        if _VAGUE_WORDS.search(description) and not _EXPLICIT_SIGNAL.search(description):
            add(
                cid,
                "The description relies on vague quality language without an operational threshold.",
                "Replace vague words with a checklist, numeric threshold, exact required behavior, or named evidence.",
            )

        signature = _criterion_signature(criterion)
        duplicate_of = seen_signatures.get(signature)
        if duplicate_of is not None:
            add(
                cid,
                f"This criterion appears redundant with '{duplicate_of}'.",
                "Merge overlapping checks or make each criterion prove a distinct requirement.",
                severity="warning",
            )
        else:
            seen_signatures[signature] = cid

        if criterion.kind == CriterionKind.AI_JUDGE:
            prompt = (criterion.judge_prompt or "").strip()
            if len(prompt) < 100:
                add(
                    cid,
                    "AI review instructions are not detailed enough for repeatable pass/fail decisions.",
                    "Write a strict checklist with explicit PASS-only-if and FAIL-if conditions and the evidence to inspect.",
                )
            lowered = prompt.lower()
            if "pass" not in lowered or "fail" not in lowered:
                add(
                    cid,
                    "AI review does not explicitly define both pass and fail conditions.",
                    "Include 'PASS only if …' and 'FAIL if …' rules.",
                )
            if not _EVIDENCE_SIGNAL.search(prompt):
                add(
                    cid,
                    "AI review does not identify concrete evidence to inspect.",
                    "Name files, commands, outputs, artifacts, logs, or observable behavior that provide evidence.",
                )
            if not criterion.evidence_paths:
                add(
                    cid,
                    "No preferred evidence paths are supplied for this AI review.",
                    "Add likely project-relative files or directories so evaluation is grounded and efficient.",
                    severity="warning",
                )
        elif criterion.kind == CriterionKind.COMMAND and criterion.output_judge_prompt:
            prompt = criterion.output_judge_prompt.strip()
            if len(prompt) < 80:
                add(
                    cid,
                    "Command output AI review instructions are too short for repeatable judging.",
                    "Write explicit PASS-only-if and FAIL-if rules tied to stdout/stderr output evidence.",
                )
            lowered = prompt.lower()
            if "pass" not in lowered or "fail" not in lowered:
                add(
                    cid,
                    "Command output AI review does not explicitly define both pass and fail conditions.",
                    "Include 'PASS only if ...' and 'FAIL if ...' rules based on the command output.",
                )
        elif criterion.kind == CriterionKind.MANUAL and criterion.required:
            add(
                cid,
                "This required criterion needs human approval and cannot finish autonomously.",
                "Use AI evidence review unless personal approval is intentionally part of the goal.",
                severity="warning",
            )

        if criterion.kind == CriterionKind.COMMAND and criterion.command:
            command = criterion.command.strip()
            if _METRIC_THRESHOLD.search(description) and not criterion.output_judge_prompt:
                add(
                    cid,
                    "A numeric score threshold is described, but this command criterion only checks an exit code.",
                    "Add an output_judge_prompt that explicitly checks the reported metric and its threshold, "
                    "or use a verifier command that exits non-zero whenever the metric is below the threshold. "
                    "For external benchmarks, also require official-run provenance and raw result artifacts.",
                )
            if _TRIVIAL_COMMAND.match(command):
                add(
                    cid,
                    "Command check is trivially passable and does not prove goal completion.",
                    "Replace it with a real verification command that validates required behavior or outputs.",
                )
            if "|| true" in command.lower() or "; true" in command.lower():
                add(
                    cid,
                    "Command appears to suppress failures, which can hide unmet goal conditions.",
                    "Use a command that fails on unmet conditions and keep expected_exit_code aligned with real failure semantics.",
                )

            command_parts = _command_parts(command)
            executable = _command_executable(command)
            if executable:
                lowered = executable.lower()
                if lowered in _SHELL_BUILTINS:
                    add(
                        cid,
                        "Command starts with a shell builtin and is unlikely to be a standalone verification step.",
                        "Use an executable verification command (for example `python -m pytest`, `npm run test`, or a project script).",
                    )
                elif project_path is not None and os.path.isdir(project_path):
                    resolution = resolve_executable(executable)
                    if not resolution.found:
                        add(
                            cid,
                            f"Command executable '{executable}' could not be resolved on this system.",
                            "Use a resolvable executable or a project-local command path; for Python checks prefer `python -m ...`.",
                            severity="blocking" if criterion.required else "warning",
                        )

                if project_path is not None and command_parts:
                    launcher = command_parts[0].lower()
                    if launcher in _PYTHON_LAUNCHERS:
                        if len(command_parts) >= 2 and command_parts[1] == "-m":
                            # Module execution is environment-dependent; skip file existence check.
                            pass
                        elif len(command_parts) >= 2 and command_parts[1] not in {
                            "-c",
                            "-V",
                            "--version",
                        }:
                            script_path = command_parts[1]
                            resolved_script = _project_relative_path(
                                script_path,
                                project_path=project_path,
                            )
                            if resolved_script is None:
                                add(
                                    cid,
                                    f"Python script path '{script_path}' escapes the project directory.",
                                    "Use a project-relative script path for verification commands.",
                                )
                            elif not resolved_script.exists():
                                add(
                                    cid,
                                    f"Python script '{script_path}' does not exist in the project.",
                                    "Create the script first or switch to an existing verification command.",
                                    severity="blocking" if criterion.required else "warning",
                                )

        if project_path is not None and criterion.path:
            resolved = _project_relative_path(criterion.path, project_path=project_path)
            if resolved is None:
                add(
                    cid,
                    f"Criterion path '{criterion.path}' escapes the project directory.",
                    "Use a project-relative path within the workspace.",
                )

    required_kinds = {item.kind for item in required}
    if required and required_kinds == {CriterionKind.FILE_EXISTS}:
        add(
            None,
            "All required criteria only check that files exist, which is easy to game and may not prove behavior.",
            "Add at least one required behavioral check (command, file_contains, or strict AI evidence review).",
        )

    behavior_kinds = {CriterionKind.COMMAND, CriterionKind.FILE_CONTAINS, CriterionKind.AI_JUDGE}
    if required and not any(item.kind in behavior_kinds for item in required):
        add(
            None,
            "Required criteria do not include a behavioral verification step.",
            "Add a required command or content/rubric check that verifies the goal outcome, not only artifact presence.",
        )

    checked.criteria_quality_issues = issues
    blockers = [item for item in issues if item.severity == "blocking"]
    if checked.clarifying_questions or blockers:
        checked.ready_to_finalize = False
        reasons: list[str] = []
        if checked.clarifying_questions:
            reasons.append(f"{len(checked.clarifying_questions)} clarification(s) remain")
        if blockers:
            reasons.append(f"{len(blockers)} blocking criteria quality issue(s) remain")
        checked.readiness_reason = "; ".join(reasons) + "."
    elif checked.ready_to_finalize:
        checked.readiness_reason = checked.readiness_reason or (
            "No material clarification remains and every required criterion has a concrete verification method."
        )
    else:
        checked.readiness_reason = checked.readiness_reason or (
            "The proposal is still a draft. Continue refinement or ask the AI to perform a final readiness review."
        )
    return checked
