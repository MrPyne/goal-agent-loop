from __future__ import annotations

import re

from .models import (
    CriteriaQualityIssue,
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


def assess_setup_proposal(proposal: SetupProposal) -> SetupProposal:
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
        elif criterion.kind == CriterionKind.MANUAL and criterion.required:
            add(
                cid,
                "This required criterion needs human approval and cannot finish autonomously.",
                "Use AI evidence review unless personal approval is intentionally part of the goal.",
                severity="warning",
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
