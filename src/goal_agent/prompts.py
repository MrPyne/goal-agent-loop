from __future__ import annotations

import json
import re

from .models import CriteriaDocument, Hypothesis, RunState


def _clip(value: object, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 5
    tail = max_chars - head
    return text[:head].rstrip() + " … [omitted] … " + text[-tail:].lstrip()


def _criterion_payload(criteria: CriteriaDocument) -> list[dict]:
    """Compact criterion definitions without removing verification semantics."""

    payload: list[dict] = []
    for item in criteria.criteria:
        row = {
            "id": item.id,
            "description": _clip(item.description, 1800),
            "kind": item.kind.value,
            "required": item.required,
            "override": item.override.value,
        }
        if item.command:
            row["command"] = _clip(item.command, 1800)
            row["expected_exit_code"] = item.expected_exit_code
            if item.stdout_contains is not None:
                row["stdout_contains"] = _clip(item.stdout_contains, 1000)
            if item.stderr_contains is not None:
                row["stderr_contains"] = _clip(item.stderr_contains, 1000)
            if item.stdout_regex is not None:
                row["stdout_regex"] = _clip(item.stdout_regex, 1000)
            if item.stderr_regex is not None:
                row["stderr_regex"] = _clip(item.stderr_regex, 1000)
            if (
                item.stdout_regex is not None
                or item.stderr_regex is not None
                or item.stdout_contains is not None
                or item.stderr_contains is not None
            ):
                row["output_case_sensitive"] = item.output_case_sensitive
            if item.output_judge_prompt:
                row["output_judge_prompt"] = _clip(item.output_judge_prompt, 2500)
                row["output_confidence_threshold"] = item.output_confidence_threshold
        if item.path:
            row["path"] = item.path
        if item.contains is not None:
            row["contains"] = _clip(item.contains, 1200)
            row["regex"] = item.regex
            row["case_sensitive"] = item.case_sensitive
        if item.judge_prompt:
            row["judge_prompt"] = _clip(item.judge_prompt, 3500)
            row["evidence_paths"] = item.evidence_paths[:12]
            row["confidence_threshold"] = item.confidence_threshold
        payload.append(row)
    return payload


def _implementation_diagnosis_payload(diagnosis: object) -> dict[str, object]:
    """Keep a serial executor focused on the one actionable repair."""

    plan = list(getattr(diagnosis, "executor_plan", None) or [])
    direct_action = ""
    for step in plan:
        candidate = re.sub(
            r"^(?:step\s*)?\d+\s*[:.)-]\s*", "", str(step).strip(), flags=re.IGNORECASE
        )
        if candidate.lower().startswith(
            ("create ", "implement ", "write ", "add ", "fix ", "edit ", "patch ")
        ):
            direct_action = candidate
            break
    return {
        "classification": getattr(diagnosis, "classification", "unknown"),
        "root_cause": _clip(getattr(diagnosis, "root_cause", ""), 1000),
        "recommended_project_change": _clip(
            getattr(diagnosis, "recommended_project_change", ""), 1800
        ),
        "first_direct_implementation_action": _clip(direct_action, 1400),
    }


def _analysis_payload(state: RunState) -> dict | None:
    analysis = state.evaluation_analysis
    if analysis is None:
        return None
    return {
        "summary": _clip(analysis.summary, 1800),
        "material_progress": analysis.material_progress,
        "progress_assessment": _clip(analysis.progress_assessment, 1800),
        "progress_evidence": [_clip(item, 1000) for item in analysis.progress_evidence[-6:]],
        "cross_criterion_findings": [
            _clip(item, 1200) for item in analysis.cross_criterion_findings[-6:]
        ],
        "recommended_next_focus": [
            _clip(item, 1200) for item in analysis.recommended_next_focus[:8]
        ],
        "criterion_analyses": [
            {
                "criterion_id": item.criterion_id,
                "observed_status": item.observed_status,
                "interpretation": _clip(item.interpretation, 1400),
                "likely_causes": [_clip(v, 800) for v in item.likely_causes[:5]],
                "useful_evidence": [_clip(v, 800) for v in item.useful_evidence[:5]],
                "recommended_actions": [_clip(v, 900) for v in item.recommended_actions[:5]],
            }
            for item in analysis.criterion_analyses
        ],
    }


def setup_prompt(
    rough_goal: str,
    user_answers: str = "",
    *,
    low_context: bool = False,
    project_snapshot: str = "",
) -> str:
    inspection_policy = (
        "LOW-CONTEXT RECOVERY MODE: Use only the supplied bounded project snapshot, goal, conversation, and draft. "
        "Do not call tools, inspect files, search the project, or delegate to subagents. Return the complete proposal promptly."
        if low_context
        else
        "PROJECT INSPECTION: Goal Agent has already supplied a bounded read-only project snapshot. Use only that snapshot. "
        "Do not call tools, inspect additional files, search the project, or delegate to subagents."
    )
    rough_goal = _clip(rough_goal, 8000)
    user_answers = _clip(user_answers, 24_000)
    project_snapshot = _clip(project_snapshot, 48_000)
    return f"""
You are an interactive goal-definition partner for a persistent autonomous AI agent loop. This is one turn in an
ongoing refinement conversation, not a one-shot form. Read the supplied bounded conversation context, answer the
user's latest message, revise the goal and criteria when useful, and ask only the questions that materially affect
what success means. Older discussion may be compacted, but the current draft and recent user corrections are authoritative.

{inspection_policy}

CURRENT ROUGH GOAL OR SAVED GOAL
{rough_goal}

REFINEMENT CONVERSATION SO FAR
{user_answers or '(none yet)'}

BOUNDED PROJECT SNAPSHOT
{project_snapshot or '(No project snapshot was available. Base the proposal on the saved goal and conversation.)'}

RESEARCH-FIRST CRITERIA DISCOVERY
Derive criteria from evidence in the snapshot (verification signals, docs, manifests, root layout), not from generic templates.
When proposing a command criterion, prefer commands that are explicitly discoverable from the snapshot (for example
existing test/build/check scripts, CI-aligned commands, or documented verification flows).

Return a complete current proposal on every turn, including:
1. refined_goal: a durable outcome statement that is specific about what must exist or work, while avoiding an
   unnecessarily narrow implementation.
2. assistant_message: a concise conversational reply explaining what changed, what is still uncertain, and what you
   need from the user next. Do not merely repeat the JSON fields.
3. clarifying_questions: only unanswered questions whose answers could materially change the goal or success tests.
   Prefer 1-3 focused questions. Return an empty list when no material ambiguity remains.
4. assumptions: explicit assumptions currently being used so the user can correct them.
5. a complete replacement criteria list.
6. criteria_quality_issues: any remaining vagueness, weak proof, overlap, false-positive risk, or unverifiable wording.
7. ready_to_finalize: true only when no material questions remain and every required criterion is concrete, binary,
   repeatable, and collectively sufficient to prove the goal.
8. readiness_reason: explain why it is or is not ready.

CONCRETE SUCCESS-CRITERIA STANDARD
Every required criterion must be atomic, observable, and independently verifiable during every loop. It must say
exactly what evidence causes PASS and what causes FAIL. Criteria must prove the outcome, not merely that work was
attempted. Avoid vague words such as good, clear, complete, user-friendly, robust, professional, correct, or works
unless they are operationalized with an explicit checklist, threshold, required behavior, named artifact, or test.

Choose the strongest practical verification method:
- command: an automated test/check exits with the expected code. Include the exact command.
    Add stdout/stderr text or regex checks when output is part of success.
    For qualitative command output, include output_judge_prompt with explicit PASS/FAIL rules.
- file_exists: a specific required artifact exists at a project-relative path.
- file_contains: a specific file contains exact text or a regex.
- ai_judge: only for genuinely qualitative outcomes. The judge_prompt must be a strict, repeatable rubric containing
  explicit 'PASS only if ...' and 'FAIL if ...' rules, the exact evidence to inspect, and no reliance on unsupported
  claims. Populate evidence_paths with likely project-relative evidence whenever possible.
- manual: only when the user explicitly requires personal human approval that cannot reasonably be inferred from
  project evidence.

Prefer deterministic checks whenever they truly prove the result. Combine deterministic and AI-reviewed criteria when
needed. Include regression or preservation criteria when success could otherwise be achieved by breaking existing
behavior. Use paths relative to the target project directory. Keep criterion IDs stable when their meaning is unchanged.

Before setting ready_to_finalize=true, perform a final adversarial review: could a weak, incomplete, or cosmetic
implementation still pass? If yes, tighten the criteria and report the issue instead of finalizing.
"""


def criteria_refinement_prompt(goal: str, criteria: CriteriaDocument, feedback: str) -> str:
    return f"""
You are refining success criteria for a persistent autonomous goal loop.

GOAL
{_clip(goal, 8000)}

CURRENT CRITERIA
{json.dumps(_criterion_payload(criteria), indent=2)}

USER FEEDBACK
{_clip(feedback, 8000)}

RESEARCH-FIRST RULE
If feedback requests different criteria, infer revisions from concrete project evidence and verification signals,
not from generic criterion templates.

Return a complete replacement criteria list. Keep useful deterministic checks, remove redundant or gameable checks,
and ensure all required criteria together prove that the goal is achieved. Every criterion must be atomic, binary, and
observable, with an exact verification mechanism. Replace vague terms such as good, clear, complete, correct, robust,
professional, user-friendly, or works with measurable thresholds, required behaviors, named artifacts, or explicit
checklists. Prefer command/file checks whenever they truly prove the outcome. For ai_judge, write a strict judge_prompt
that identifies the evidence to inspect and contains explicit PASS-only-if and FAIL-if rules; populate evidence_paths
where practical. For command criteria, prefer deterministic output checks and add output_judge_prompt only when
the output requires qualitative interpretation. Use manual only when the user explicitly asks for a human-only approval gate. Include regression or
preservation checks when a weak implementation could otherwise pass by breaking existing behavior. Criterion IDs must
remain stable when the meaning is unchanged. Use paths relative to the project directory.
"""


def evaluation_analysis_prompt(
    *,
    goal: str,
    criteria: CriteriaDocument,
    results: dict,
    previous_results: dict | None,
    iteration: int,
    label: str,
    steering: str,
    active_hypothesis: Hypothesis | None = None,
    execution_report: dict | None = None,
) -> str:
    result_payload = []
    for criterion in criteria.criteria:
        result = results.get(criterion.id)
        result_payload.append(
            {
                "criterion": next(
                    item for item in _criterion_payload(CriteriaDocument(criteria=[criterion]))
                ),
                "result": (
                    {
                        "criterion_id": result.criterion_id,
                        "passed": result.passed,
                        "status": result.status,
                        "summary": _clip(result.summary, 1800),
                        "evidence": [_clip(item, 1800) for item in result.evidence[-5:]],
                        "error": _clip(result.error, 1800) if result.error else None,
                        "confidence": result.confidence,
                    }
                    if result
                    else None
                ),
            }
        )
    previous_payload = []
    if previous_results:
        for criterion in criteria.criteria:
            result = previous_results.get(criterion.id)
            if result:
                previous_payload.append(
                    {
                        "criterion_id": criterion.id,
                        "status": result.status,
                        "summary": _clip(result.summary, 1200),
                        "evidence": [_clip(item, 1200) for item in result.evidence[-4:]],
                    }
                )
    return f"""
You are the diagnostic evaluator in a persistent autonomous goal loop. The individual criteria have already been
checked. Analyze their actual pass, fail, and error outputs so the next strategist can choose a strong root-cause
hypothesis. All evidence you may use is embedded below. Do not call tools, inspect files, or modify the project.

OVERALL GOAL
{_clip(goal, 8000)}

ITERATION AND EVALUATION
Iteration: {iteration}
Evaluation: {label}

CRITERIA, RESULTS, AND EVIDENCE
{json.dumps(result_payload, indent=2)}

PREVIOUS RESULTS FROM THIS ITERATION, WHEN AVAILABLE
{json.dumps(previous_payload, indent=2)}

ACTIVE HYPOTHESIS, IF ANY
{json.dumps(active_hypothesis.model_dump(mode='json') if active_hypothesis else None, indent=2)[:8000]}

EXECUTION REPORT, IF ANY
{_clip(json.dumps(execution_report, indent=2), 8000)}

LIVE USER STEERING
{_clip(steering, 6000)}

For every criterion, explain what its observed output means. Analyze passing criteria too: identify reliable evidence,
possible regressions, dependencies, and what should be preserved. For failed or errored criteria, diagnose likely root
causes and propose concrete next checks/actions. Look for cross-criterion patterns and causal links. Do not overturn a
deterministic result. An ai_judge result is already the pass/fail decision for that qualitative criterion; analyze its
evidence and missing items. A human-only criterion remains blocked until a human override is supplied.

When previous results are available, set material_progress true only when concrete output or evidence improved even if a
criterion still fails. Put that proof in progress_evidence. Do not call cosmetic activity or an executor claim progress.
Return a concise structured analysis. recommended_next_focus should prioritize the most useful next work, not merely
repeat the criterion descriptions.
"""


def strategy_prompt(
    *,
    goal: str,
    criteria: CriteriaDocument,
    state: RunState,
    steering: str,
    recent_hypotheses: list[Hypothesis],
) -> str:
    criteria_status = []
    for criterion in criteria.criteria:
        result = state.criteria_results.get(criterion.id)
        criteria_status.append(
            {
                "id": criterion.id,
                "description": _clip(criterion.description, 1600),
                "kind": criterion.kind.value,
                "required": criterion.required,
                "status": result.status if result else "unchecked",
                "summary": _clip(result.summary, 1400) if result else "Not checked",
                "evidence": [_clip(item, 1000) for item in result.evidence[-3:]] if result else [],
            }
        )
    history = []
    for item in recent_hypotheses[-8:]:
        history.append(
            {
                "id": item.id,
                "iteration": item.iteration,
                "hypothesis": _clip(item.statement, 1600),
                "rationale": _clip(item.rationale, 1200),
                "target_criteria": item.target_criteria,
                "plan": [_clip(step, 900) for step in item.plan[:8]],
                "status": item.status,
                "outcome": _clip(item.outcome, 1600),
                "evidence": [_clip(value, 700) for value in item.evidence[-5:]],
            }
        )
    evaluation_analysis = _analysis_payload(state)
    return f"""
You are the strategist in a persistent autonomous agent loop. Your job is to choose the next falsifiable hypothesis
that is most likely to move the project from its current state to the goal.

GOAL
{_clip(goal, 8000)}

CRITERIA AND CURRENT RESULTS
{json.dumps(criteria_status, indent=2)}

AI DIAGNOSIS OF THE LATEST CRITERIA OUTPUTS
{json.dumps(evaluation_analysis, indent=2)}

RECENT HYPOTHESES AND OUTCOMES
{json.dumps(history, indent=2)}

CONSECUTIVE ITERATIONS WITHOUT MEASURABLE PROGRESS
{state.consecutive_no_progress}

LIVE USER STEERING
{_clip(steering, 6000)}

CONTEXT DISCIPLINE
All information available to the strategist is embedded above. Do not call tools, inspect project files, delegate to
subagents, or perform searches. Return one hypothesis promptly from the supplied criterion evidence and diagnosis.

Use both the raw criterion outputs and the diagnostic analysis.
Propose one root-cause hypothesis, why it is plausible, which failed criteria it targets, which passing behavior must be
preserved, and a concrete short plan for the executor. The hypothesis must be testable this iteration.
Avoid repeating failed approaches unless there is new evidence or a materially different implementation.
When progress has stalled, step back and challenge assumptions rather than making cosmetic changes.

TIMEOUT SELF-HEALING RULE
If any criterion has status "timeout" or if the executor has been blocked by timeouts for multiple iterations,
the hypothesis MUST address the timeout as a first-class problem. Propose one of:
- Adding progress output to the long-running script so the executor can see it is alive (e.g. print per-sample progress)
- Reducing the eval scope temporarily (smaller max_rows or a subset flag) to verify the command works
- Caching or pre-computing results to avoid re-running the full eval every iteration
Do NOT propose re-running the same timed-out command unchanged.
"""


def criterion_fix_prompt(
    *,
    goal: str,
    criterion,  # CriterionDefinition
    result,     # CriterionResult
    criteria: CriteriaDocument,
    steering: str,
    repair_diagnosis=None,
    execution_directive: str = "",
) -> str:
    """Single-criterion executor prompt for serial fix mode.

    Unlike the free-form executor_prompt, this gives the executor one concrete,
    unambiguous mission: make THIS criterion pass.  No hypothesis needed — the
    criterion definition and its current failure evidence are the full spec.
    """
    evidence_lines = "\n".join(f"  - {e}" for e in (result.evidence or [])[:8]) or "  (none)"
    command_hint = ""
    if criterion.command:
        command_hint = f"\nCRITERION COMMAND (run this to verify after your fix):\n  {criterion.command}"
    judge_hint = ""
    if criterion.judge_prompt:
        judge_hint = f"\nJUDGE RUBRIC:\n{_clip(criterion.judge_prompt, 2000)}"
    elif criterion.output_judge_prompt:
        judge_hint = f"\nOUTPUT JUDGE RUBRIC:\n{_clip(criterion.output_judge_prompt, 2000)}"
    required_by = [c.id for c in criteria.criteria if c.required and not c.id == criterion.id]
    diagnosis_hint = "(No independent diagnosis was available; use the failure evidence.)"
    first_direct_action = ""
    if repair_diagnosis is not None:
        diagnosis_payload = _implementation_diagnosis_payload(repair_diagnosis)
        diagnosis_hint = json.dumps(diagnosis_payload, indent=2)
        first_direct_action = str(
            diagnosis_payload.get("first_direct_implementation_action") or ""
        )
    implementation_requirement = ""
    if repair_diagnosis is not None and (
        repair_diagnosis.recommended_project_change or repair_diagnosis.executor_plan
    ):
        implementation_requirement = """

MANDATORY IMPLEMENTATION BEHAVIOR
The independent diagnosis already identifies a project change that can move this criterion.
Make that change now. If it calls for a new script or structured evidence pipeline, creating the
script is the minimum relevant change; do not substitute project reconnaissance for implementation.
Choose the first plan item that directly serves this criterion (skip work that belongs only to another
criterion). Read only the directly relevant existing script/config files needed to implement it.
Do not list the project root, recursively enumerate logs/ or outputs/, or look for criteria.yaml in
the project root: the criterion and failure evidence are already supplied above.
Never fabricate a checkpoint, promotion pass, replay score, or historical log. If a required artifact
is genuinely absent, implement and run the code path that produces it (for example the missing stage
training or evaluation), then record the actual result.
""".strip()
    return f"""
You are the executor in a persistent autonomous goal loop operating in single-criterion fix mode.
Your ONLY task this iteration is to make the one criterion below pass.
Do NOT touch unrelated files, run long-running evals, or attempt to fix other criteria.

OVERALL GOAL
{_clip(goal, 4000)}

TARGET CRITERION TO FIX
ID: {criterion.id}
Kind: {criterion.kind.value}
Required: {criterion.required}
Description: {criterion.description}
{command_hint}
{judge_hint}

CURRENT FAILURE EVIDENCE
Status: {result.status}
Summary: {_clip(result.summary, 1200)}
Evidence:
{evidence_lines}

LIVE USER STEERING
{_clip(steering, 1800)}

INDEPENDENT REPAIR DIAGNOSIS
{diagnosis_hint}

SYSTEM EXECUTION DIRECTIVE
{_clip(execution_directive, 2500) if execution_directive else '(No extra directive.)'}
{implementation_requirement}

FIRST REQUIRED ACTION
{("Use edit/write now to perform this action: " + first_direct_action) if first_direct_action else "Make the first source edit identified by the recommended project change now."}
Do not use bash, read, list, glob, or grep before that first edit. If the named file is absent, create it with
the edit/write tool; do not use a shell command to construct it.

INSTRUCTIONS
1. Inspect the failure evidence above — that is your full diagnosis.
2. Make the MINIMUM change required to satisfy the criterion.  Prefer editing an
   existing file over creating a new one; prefer a small targeted fix over a rewrite.
   This workspace runs commands through Windows PowerShell. Use PowerShell syntax
   (for example `;`, `Test-Path`, and `Get-Content`) even if the tool is labelled
   `bash`; never use Unix-only shell operators such as `||` or `&&`, `head`, or
   shell-specific path expansion.
3. If the criterion requires a COMMAND, run the command after your fix and include the
   output in your report so the evaluator can confirm without re-running everything.
4. If the fix requires creating a NEW SCRIPT, create it, run it, and verify it exits
   with the expected code before reporting.
5. Do NOT run quality_gate.py, promotion_check.py, or any eval that takes > 5 minutes —
   those are handled by the automated criterion evaluator, not the executor.
6. A criteria_revision in the diagnosis is advisory; never silently edit
   `.goal-agent/control/criteria.yaml`. You MAY change a project verification script when doing
   so makes it faithfully measure the existing criterion. Do not merely lower a threshold or remove
   a check to force a pass; split unrelated checks into separate measurements when warranted.
7. Missing test or benchmark infrastructure is implementation work. When a criterion needs a runner,
   dataset, model adapter, scorer, raw-result artifact, provenance record, or validation script, create or
   repair those project components and validate each piece. Do not ask to change the criterion merely because
   those components do not exist yet.
8. For a named external benchmark, first use its official repository/docs to discover the real installation
   and execution interface. Do not guess a Python API or CLI. Never substitute fixed task outcomes, random
   outcomes, a fixture, a diagnostic subset, or copied results for an official run, and never label such output
   as official. If installation or model-adapter integration fails, preserve the failure, report the exact command
   and error, and implement the next missing integration component; keep the criterion failing.
9. Report: files changed, commands run, exact output, and whether the criterion now passes.
""".strip()


def serial_criterion_diagnosis_prompt(
    *,
    goal: str,
    criterion,
    result,
    criteria: CriteriaDocument,
    criteria_results: dict[str, object],
    steering: str,
) -> str:
    """Ask for a contract-level diagnosis before serial mode asks the executor to act."""

    evidence = [_clip(item, 4000) for item in (result.evidence or [])[-6:]]
    related_results = [
        {
            "criterion_id": item.id,
            "status": getattr(criteria_results.get(item.id), "status", "unchecked"),
            "passed": bool(getattr(criteria_results.get(item.id), "passed", False)),
            "summary": _clip(getattr(criteria_results.get(item.id), "summary", ""), 700),
        }
        for item in criteria.criteria
        if item.id != criterion.id and criteria_results.get(item.id) is not None
    ]
    return f"""
You are the diagnostic criterion-repair strategist for a persistent autonomous goal loop.
You do not edit files or run tools. Diagnose whether the failed verification command is actually
measuring the success criterion it is being used to prove, then give the executor a concrete plan.

OVERALL GOAL
{_clip(goal, 6000)}

TARGET CRITERION
{json.dumps(_criterion_payload(CriteriaDocument(criteria=[criterion]))[0], indent=2)}

FAILURE RESULT
{json.dumps({
    "status": result.status,
    "summary": _clip(result.summary, 1800),
    "error": _clip(result.error, 1200) if result.error else None,
    "evidence": evidence,
}, indent=2)}

OTHER CRITERIA
{json.dumps(_criterion_payload(criteria), indent=2)}

OTHER CRITERIA RESULTS
{json.dumps(related_results, indent=2)}

LIVE USER STEERING
{_clip(steering, 4000)}

Return a concise structured diagnosis. Classify the primary cause as one of implementation_defect,
model_capability, criterion_measurement_mismatch, missing_evidence, environment, or unknown.

EVIDENCE-PATH AUDIT
Before classifying a missing-evidence failure, compare every named required artifact in the failure result,
steering, and rubric with the criterion's evidence_paths. If a structured artifact (for example a JSON/JSONL
result file) is named as required but is absent from evidence_paths, classify this as
criterion_measurement_mismatch and propose a review-required replacement that adds the artifact. Do not send
the executor to recreate an artifact merely because the judge was not given a path to inspect. A genuine
missing_evidence classification is appropriate only when the required artifact is already in evidence_paths
or the result contains direct proof that its production command failed.
When steering specifies an artifact schema, preserve that schema in any proposed replacement; adding an
evidence path must not make a malformed artifact pass.

If the command bundles several independent properties, identify the mismatch explicitly. Recommend
the project-code change that would make the command faithfully measure the CURRENT criterion; this
can include changing a gate script or splitting its output into independent metrics. Do not recommend
lowering a threshold or deleting a check merely to obtain a pass.

CRITERIA-REVISION ADMISSION GATE
criteria_revision MUST be null by default. Set it only when, and only when, the current criterion is:
1. not_meaningful — it cannot express a coherent observable outcome;
2. unachievable — direct evidence proves no legal in-scope implementation can satisfy it, not merely that
   code, a dataset, a runner, or an integration is currently missing;
3. contradictory — it is logically incompatible with a named current criterion; or
4. duplicate — another independent criterion directly verifies the same required behavior.
When non-null, set criteria_revision.revision_reason to exactly one of not_meaningful, unachievable,
contradictory, or duplicate, and give concrete evidence for that reason in the rationale and safeguards.
For every other failure — including missing scripts, datasets, official benchmark packages, adapters, scorers,
raw traces, validation code, configuration, or output artifacts — criteria_revision MUST remain null. These are
project implementation tasks. Include the component-by-component build and validation plan in executor_plan.

Only include criteria_revision when the current criteria themselves should change. It must be a complete,
review-required replacement for the affected criterion(s), preserve the goal's intended standard, and use atomic
independently verifiable checks. A suggestion is advisory: the user must approve it before criteria.yaml is
changed. If the existing criterion is sound but its command is the wrong proxy, leave criteria_revision null and
recommend a project-code fix instead.

Before returning criteria_revision, compare every proposed criterion with CURRENT CRITERIA. Return
criteria_revision null when the intended replacement/split is already present. If splitting the target
criterion into several atomic criteria, proposed_criteria is the complete replacement set: do not retain
or re-propose the broad original criterion unless it remains independently necessary.
For a command criterion, adding evidence_paths or explanatory safeguards without changing the command,
its expected output/exit conditions, threshold, or success semantics is not a criteria revision. Return
criteria_revision null and direct the executor to implement or repair the command's producer instead.

DUPLICATE-CRITERION RULE
Compare TARGET CRITERION against OTHER CRITERIA RESULTS. If another passing criterion already verifies the
same required behavior using live/direct evidence, while the target relies on a stub, replay, fixture, or
non-live proxy, the target is redundant rather than a reason to repair the project. In that case, return a
review-required criteria_revision with remove_target=true and proposed_criteria=[]; explain the overlap and
name the stronger passing criterion. Do not send the executor to make the weaker proxy pass.

EXTERNAL-BENCHMARK NON-REMOVAL RULE
A named external benchmark, score threshold, or comparison requirement in the goal is an independent required
outcome. Never propose removing its criterion merely because a comparison report, aggregate score, fixture, or
another criterion claims to cover it. This applies especially when the failing command reports a fallback,
diagnostic subset, placeholder, generated score, missing official package, missing raw traces, or unverified
provenance. Those are implementation/evidence failures, not redundancy. Leave criteria_revision null and direct
the executor to implement the official harness, model adapter, scorer, and traceable raw-result artifacts.
Only consider a benchmark criterion redundant if the goal itself explicitly makes it optional AND another direct
official run of the identical benchmark already passes with independently inspectable provenance and raw results.

CHECKER-TO-PRODUCER AUDIT
For a failed command that parses or scores another program's transcript, treat the checker and the producer as
separate components. Do not classify this as model_capability merely because the checker reports zero calls or
zero passes: that conclusion requires raw producer output showing the model did not perform the action. When the
failure evidence contains only checker summaries, classify it as implementation_defect or unknown and direct the
executor to run one small, equivalent producer command (not the full gate), compare its exact event markers and
values against the checker's predicates, and patch the checker if it rejects a conforming transcript. Likewise,
identify stub or replay evaluators that only inspect recorded dataset metadata; their pass rates are not evidence
of live model behavior. Never lower the requested behavioral threshold to conceal a parser defect.
""".strip()


def executor_prompt(
    *,
    goal: str,
    criteria: CriteriaDocument,
    hypothesis: Hypothesis,
    steering: str,
    baseline_results: dict | None = None,
    execution_directive: str = "",
    consecutive_no_progress: int = 0,
) -> str:
    failing_baseline: list[dict] = []
    if baseline_results:
        for criterion in criteria.criteria:
            result = baseline_results.get(criterion.id)
            if not result:
                continue
            status = getattr(result, "status", "unchecked")
            passed = bool(getattr(result, "passed", False))
            if status == "pass" or passed:
                continue
            evidence = getattr(result, "evidence", []) or []
            failing_baseline.append(
                {
                    "id": criterion.id,
                    "kind": criterion.kind.value,
                    "required": criterion.required,
                    "summary": _clip(getattr(result, "summary", ""), 1200),
                    "error": _clip(getattr(result, "error", ""), 800) if getattr(result, "error", None) else None,
                    "evidence": [_clip(item, 500) for item in evidence[:3]],
                }
            )

    candidate_commands = [
        {
            "id": criterion.id,
            "required": criterion.required,
            "command": _clip(criterion.command, 1200),
            "expected_exit_code": criterion.expected_exit_code,
        }
        for criterion in criteria.criteria
        if criterion.command
    ]

    # When the loop has been stuck, derive a forced first action from the hypothesis plan or failing
    # command criteria so the model cannot fall back to pure reconnaissance.
    forced_first_action = ""
    if consecutive_no_progress >= 2:
        # Prefer the first concrete command criterion that is currently failing AND is not a
        # long-running gate (timeout_seconds > 300 means it can't complete inside a single
        # executor OpenCode session without triggering the stall detector).
        first_command: str | None = None
        for criterion in criteria.criteria:
            cmd_timeout = criterion.timeout_seconds or 0
            if criterion.command and criterion.required and cmd_timeout <= 300:
                first_command = criterion.command
                break
        # Also try to pull the first non-inspection plan step (contains 'run', 'create', 'fix', 'execute').
        first_plan_action: str | None = None
        action_keywords = ("run ", "create ", "fix ", "execute ", "write ", "add ", "implement ")
        for step in (hypothesis.plan or []):
            if any(step.lower().startswith(kw) or f" {kw}" in step.lower() for kw in action_keywords):
                first_plan_action = step
                break

        parts: list[str] = []
        if first_plan_action:
            parts.append(f"Plan action: {first_plan_action}")
        if first_command:
            parts.append(f"Criterion command to run: {first_command}")
        if parts:
            forced_first_action = (
                f"\n\nFORCED FIRST ACTION (the loop has had no measurable progress for "
                f"{consecutive_no_progress} consecutive iterations because the executor "
                "only performed file/directory reconnaissance without making changes).\n"
                "Your first tool call MUST be one of these concrete actions — not a directory listing "
                "or project-root file read:\n"
                + "\n".join(f"  {i+1}. {p}" for i, p in enumerate(parts))
                + "\nDo NOT output the final report JSON until you have executed at least one of the above."
            )

    return f"""
You are the executor in a persistent autonomous goal loop. Work directly in the current project directory.
You may inspect files, edit files, create files, and run commands needed to pursue the active hypothesis.{forced_first_action}

GOAL
{_clip(goal, 8000)}

SUCCESS CRITERIA
{json.dumps(_criterion_payload(criteria), indent=2)}

ACTIVE HYPOTHESIS
{_clip(json.dumps(hypothesis.model_dump(mode='json'), indent=2), 9000)}

BASELINE FAILURES TO RESOLVE THIS ITERATION
{json.dumps(failing_baseline, indent=2) if failing_baseline else '(No failing baseline criteria were supplied.)'}

CRITERION COMMANDS AVAILABLE
{json.dumps(candidate_commands, indent=2) if candidate_commands else '(No command criteria are defined.)'}

LIVE USER STEERING
{_clip(steering, 6000)}

SYSTEM EXECUTION DIRECTIVE
{_clip(execution_directive, 3000) if execution_directive else '(No extra directive.)'}

CONTEXT DISCIPLINE
Work from the active hypothesis and exact evidence paths first. Do not delegate to subagents or broadly inventory the
repository. Avoid dependency, vendor, build, cache, log, generated, media, binary, .git, and .goal-agent trees. Read
bounded sections of directly relevant files and keep command output focused. Make one bounded, testable change set.

Execute the plan rather than merely describing it. Preserve useful existing work. Run relevant tests/checks before
finishing. Do not weaken tests or criteria simply to obtain a pass. Do not edit anything under .goal-agent unless the
user's goal explicitly requires it. If the hypothesis is wrong, gather evidence and make the safest useful progress
possible. End with a factual report of actions, changed files, commands, evidence, and blockers.

ACTION POLICY
- Do not end the iteration with reconnaissance-only work when the baseline failures already identify a missing
    build artifact or executable command path.
- If baseline failures show missing artifacts (for example file-not-found checkpoints, binaries, generated outputs),
    attempt at least one artifact-producing command this iteration unless a concrete blocker prevents execution.
- Prefer commands already present in ACTIVE HYPOTHESIS.plan or CRITERION COMMANDS AVAILABLE when they can directly
    resolve a failing required criterion.
- If execution is blocked, record the exact failing command, error, and the minimum patch needed to unblock the next
    run. Do not stop at file listing and static inspection alone.
"""
