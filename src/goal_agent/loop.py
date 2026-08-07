from __future__ import annotations

import asyncio
import re
import time
import uuid

from filelock import FileLock, Timeout

from .criteria import (
    CriteriaEvaluator,
    EvaluationInterrupted,
    all_required_pass,
    passed_required_count,
)
from .models import (
    AgentPhase,
    CriteriaDocument,
    CriterionAnalysis,
    EvaluationAnalysis,
    EventRecord,
    ExecutionReport,
    Hypothesis,
    RunPhase,
    RunState,
    StrategyDecision,
    utc_now,
)
from .opencode import (
    OpenCodeContextOverflowError,
    OpenCodeError,
    OpenCodeInterrupted,
    OpenCodeRunner,
)
from .prompts import criterion_fix_prompt, evaluation_analysis_prompt, executor_prompt, strategy_prompt
from .storage import ProjectStore


class LoopAlreadyRunning(RuntimeError):
    pass


class GoalAgentLoop:
    def __init__(self, store: ProjectStore):
        self.store = store
        self.loop_lock = FileLock(str(store.loop_lock_path))
        self.state = store.load_state()
        self._last_status_write = 0.0

    async def run_forever(self) -> RunState:
        try:
            self.loop_lock.acquire(timeout=0)
        except Timeout as exc:
            raise LoopAlreadyRunning(
                f"Another goal-agent loop already owns {self.store.loop_lock_path}"
            ) from exc

        try:
            self.state.started_at = self.state.started_at or utc_now()
            self.state.ended_at = None
            self.state.last_error = None
            self.store.append_event(EventRecord(type="loop_started", message="Agent loop started"))
            while True:
                control = self._read_control_safely()
                if control is None:
                    await asyncio.sleep(1)
                    continue
                if control.desired_state.value == "stopped":
                    self._set_phase(RunPhase.STOPPED, "Stopped by user")
                    break
                if control.desired_state.value == "paused":
                    self._set_phase(RunPhase.PAUSED, "Paused; waiting for desired_state: running")
                    await asyncio.sleep(1)
                    continue

                try:
                    await self._run_iteration()
                    if self.state.phase == RunPhase.ACHIEVED:
                        break
                except (OpenCodeInterrupted, EvaluationInterrupted) as exc:
                    reason = getattr(exc, "reason", str(exc))
                    self.store.append_event(
                        EventRecord(type="step_interrupted", message=f"Step interrupted: {reason}")
                    )
                    if reason == "stopped":
                        self._set_phase(RunPhase.STOPPED, "Stopped by user")
                        break
                    self._reset_agents()
                    self._set_phase(RunPhase.PAUSED, "Paused during active work")
                except OpenCodeContextOverflowError as exc:
                    # Repeating the same iteration indefinitely only creates more failed
                    # OpenCode sessions. run_structured already exhausted fresh compact
                    # retries, so pause and let the user increase context or narrow scope.
                    self.state.last_error = str(exc)
                    self._mark_active_agent_error(str(exc))
                    note = (
                        "Auto-paused after OpenCode exhausted fresh-session context recovery. "
                        "Increase the model context, narrow the goal/evidence scope, or resume "
                        "after updating OpenCode compaction settings."
                    )
                    self.store.update_control(desired_state="paused", note=note)
                    self._set_phase(RunPhase.PAUSED, note)
                    self.store.append_event(
                        EventRecord(
                            type="context_recovery_exhausted",
                            message=str(exc),
                            data={
                                "iteration": self.state.iteration,
                                "requested_tokens": exc.requested_tokens,
                                "context_size": exc.context_size,
                            },
                        )
                    )
                except Exception as exc:  # keep the persistent loop recoverable
                    self.state.last_error = str(exc)
                    self._mark_active_agent_error(str(exc))
                    self._set_phase(RunPhase.ERROR, f"Iteration failed: {exc}")
                    self.store.append_event(
                        EventRecord(
                            type="iteration_error",
                            message=str(exc),
                            data={"iteration": self.state.iteration},
                        )
                    )
                    await asyncio.sleep(2)

                control = self._read_control_safely()
                if control and control.desired_state.value == "stopped":
                    self._set_phase(RunPhase.STOPPED, "Stopped by user")
                    break

                config = self.store.read_config()
                if config.max_iterations is not None and self.state.iteration >= config.max_iterations:
                    self.store.update_control(
                        desired_state="paused",
                        note=f"Paused after configured max_iterations={config.max_iterations}",
                    )
                    self._set_phase(
                        RunPhase.PAUSED,
                        f"Reached max_iterations={config.max_iterations}; edit config or resume",
                    )
                    continue
                await asyncio.sleep(config.iteration_delay_seconds)

            self.state.ended_at = utc_now()
            self.store.save_state(self.state)
            self.store.append_event(EventRecord(type="loop_exited", message=self.state.message))
            return self.state
        finally:
            self.loop_lock.release()

    # ------------------------------------------------------------------
    # Serial criterion mode
    # ------------------------------------------------------------------

    async def _run_iteration_serial(self, config) -> None:
        """Fix criteria one at a time, in declaration order.

        Flow per iteration:
          1. Determine the current target criterion (state.serial_target_criterion).
          2. If target is None  → find the first failing required criterion.
             If all pass        → set target to '__final_check__'.
          3. If target == '__final_check__' → run full combined evaluation.
             All pass → mark achieved.  Any fail → reset target to None (restart serial).
          4. Otherwise → evaluate the single target criterion.
             Pass  → advance (set target to None so next iteration finds the next failing one).
             Fail  → run executor with a single-criterion fix directive, then re-evaluate once.
        """
        runner = OpenCodeRunner(config)
        evaluator = CriteriaEvaluator(config, runner)
        control = self.store.read_control()
        model = control.model_override or config.model
        goal = self.store.read_goal()
        criteria = self.store.read_criteria()
        steering = self.store.read_steering()

        if not goal or goal.startswith("Describe the single outcome"):
            self._set_phase(RunPhase.ERROR, "No usable goal.")
            await asyncio.sleep(2)
            return
        if not criteria.criteria:
            self._set_phase(RunPhase.ERROR, "No criteria.")
            await asyncio.sleep(2)
            return

        self.state.iteration += 1
        iteration = self.state.iteration
        self._reset_agents()
        self._set_phase(RunPhase.RUNNING, f"Iteration {iteration}: serial criterion mode")
        self.store.append_event(EventRecord(
            type="iteration_started",
            message=f"Iteration {iteration} started (serial mode)",
            data={"model": model},
        ))

        target_id = self.state.serial_target_criterion

        # ── Phase A: decide / look up the target ──────────────────────
        if target_id is None:
            # Find the first required criterion that is not yet passing.
            current = self.state.criteria_results
            target_id = next(
                (c.id for c in criteria.criteria if c.required and not (current.get(c.id) and current[c.id].passed)),
                None,
            )
            if target_id is None:
                # Every required criterion is already passing → final check.
                self.state.serial_target_criterion = "__final_check__"
                target_id = "__final_check__"
            else:
                self.state.serial_target_criterion = target_id
            self.store.save_state(self.state)

        # ── Phase B: final combined check ─────────────────────────────
        if target_id == "__final_check__":
            self._update_agent("evaluator", AgentPhase.WORKING, "Final combined check", "Running all criteria together")
            before = dict(self.state.criteria_results)
            results = await self._evaluate(
                evaluator, criteria,
                goal=goal, steering=steering, model=model,
                label="Final combined check",
                artifact_prefix="serial-final",
                previous_results=before,
            )
            passing = passed_required_count(criteria.criteria, results)
            required = sum(1 for c in criteria.criteria if c.required)
            if all_required_pass(criteria.criteria, results):
                self._mark_achieved(criteria)
            else:
                self.state.serial_target_criterion = None  # restart serial
                self.store.append_event(EventRecord(
                    type="serial_final_check_failed",
                    message=f"Final combined check: {passing}/{required} required pass — restarting serial fix loop",
                    data={k: v.model_dump(mode="json") for k, v in results.items()},
                ))
                self.store.save_state(self.state)
            return

        # ── Phase C: evaluate the single target criterion ─────────────
        target_def = next((c for c in criteria.criteria if c.id == target_id), None)
        if target_def is None:
            self.state.serial_target_criterion = None
            self.store.save_state(self.state)
            return

        self._update_agent("evaluator", AgentPhase.WORKING, f"Checking {target_id}", target_def.description[:120])
        result = await evaluator.evaluate_one(
            target_def,
            goal=goal,
            steering=steering,
            model=model,
            cancel_check=self._cancel_check,
            status_callback=self._agent_callback("evaluator"),
        )
        self.state.criteria_results[target_id] = result
        self.store.save_state(self.state)

        if result.passed:
            self.store.append_event(EventRecord(
                type="serial_criterion_passed",
                message=f"{target_id} now passes — advancing to next criterion",
                data=result.model_dump(mode="json"),
            ))
            self.state.serial_target_criterion = None  # advance on next iteration
            self.state.consecutive_no_progress = 0
            self.store.save_state(self.state)
            return

        # ── Phase D: run executor to fix the failing criterion ────────
        self.store.append_event(EventRecord(
            type="serial_criterion_failed",
            message=f"{target_id} fails: {result.summary[:120]}",
            data=result.model_dump(mode="json"),
        ))

        fix_p = criterion_fix_prompt(
            goal=goal,
            criterion=target_def,
            result=result,
            criteria=criteria,
            steering=steering,
        )
        self.store.save_run_artifact(iteration, f"serial-fix-{target_id}-prompt.md", fix_p)
        self._update_agent("executor", AgentPhase.WORKING, f"Fixing {target_id}", result.summary[:120])

        report: ExecutionReport
        try:
            report, exec_result = await runner.run_structured(
                fix_p,
                ExecutionReport,
                model=model,
                agent=config.executor_agent,
                title=f"Iteration {iteration}: fix {target_id}",
                status_callback=self._agent_callback("executor"),
                cancel_check=self._cancel_check,
                attempts=2,
                profile="executor",
            )
            self.store.save_run_artifact(iteration, f"serial-fix-{target_id}-output.txt", exec_result.text)
        except OpenCodeInterrupted:
            raise
        except OpenCodeError as exc:
            report = ExecutionReport(
                summary=f"Executor error while fixing {target_id}",
                blockers=[str(exc)],
            )
            self.store.save_run_artifact(iteration, f"serial-fix-{target_id}-error.txt", str(exc))

        self._update_agent("executor", AgentPhase.COMPLETE, f"Fix attempt for {target_id}", report.summary)
        self.store.append_event(EventRecord(
            type="serial_fix_executed",
            message=report.summary,
            data=report.model_dump(mode="json"),
        ))

        # ── Phase E: re-evaluate the same criterion after the fix ─────
        self._update_agent("evaluator", AgentPhase.WORKING, f"Re-checking {target_id} after fix", "")
        result_after = await evaluator.evaluate_one(
            target_def,
            goal=goal,
            steering=steering,
            model=model,
            cancel_check=self._cancel_check,
            status_callback=self._agent_callback("evaluator"),
        )
        self.state.criteria_results[target_id] = result_after

        if result_after.passed:
            self.store.append_event(EventRecord(
                type="serial_criterion_passed",
                message=f"{target_id} passes after fix — advancing",
                data=result_after.model_dump(mode="json"),
            ))
            self.state.serial_target_criterion = None
            self.state.consecutive_no_progress = 0
        else:
            self.store.append_event(EventRecord(
                type="serial_criterion_still_failing",
                message=f"{target_id} still fails after fix attempt: {result_after.summary[:120]}",
                data=result_after.model_dump(mode="json"),
            ))
            self.state.consecutive_no_progress += 1

        self.store.save_state(self.state)

    # ------------------------------------------------------------------
    # Standard hypothesis-driven iteration
    # ------------------------------------------------------------------

    async def _run_iteration(self) -> None:
        config = self.store.read_config()
        if config.criterion_serial_mode:
            await self._run_iteration_serial(config)
            return
        runner = OpenCodeRunner(config)
        evaluator = CriteriaEvaluator(config, runner)
        control = self.store.read_control()
        model = control.model_override or config.model
        goal = self.store.read_goal()
        criteria = self.store.read_criteria()
        steering = self.store.read_steering()

        if not goal or goal.startswith("Describe the single outcome"):
            self._set_phase(RunPhase.ERROR, "No usable goal. Run goal-agent setup or edit control/goal.md")
            await asyncio.sleep(2)
            return
        if not criteria.criteria:
            self._set_phase(
                RunPhase.ERROR,
                "No criteria. Run goal-agent setup or edit control/criteria.yaml",
            )
            await asyncio.sleep(2)
            return

        self.state.iteration += 1
        iteration = self.state.iteration
        self._reset_agents()
        self._set_phase(RunPhase.RUNNING, f"Iteration {iteration}: checking current state")
        self.store.append_event(
            EventRecord(
                type="iteration_started",
                message=f"Iteration {iteration} started",
                data={"model": model},
            )
        )

        before = await self._evaluate(
            evaluator,
            criteria,
            goal=goal,
            steering=steering,
            model=model,
            label="Baseline evaluation",
            artifact_prefix="baseline",
        )
        if all_required_pass(criteria.criteria, before):
            self._mark_achieved(criteria)
            return

        # Re-read live files after evaluation so user edits steer the very next decision.
        config = self.store.read_config()
        runner = OpenCodeRunner(config)
        evaluator = CriteriaEvaluator(config, runner)
        control = self.store.read_control()
        model = control.model_override or config.model
        goal = self.store.read_goal()
        criteria = self.store.read_criteria()
        steering = self.store.read_steering()
        if self.state.consecutive_no_progress >= config.no_progress_rethink_after:
            steering += (
                "\n\nSYSTEM STALL NOTICE: The loop has made no measurable progress for "
                f"{self.state.consecutive_no_progress} iterations. Choose a materially different root-cause "
                "hypothesis, inspect assumptions, and do not repeat cosmetic variants of failed work."
            )

        self._update_agent(
            "strategist",
            AgentPhase.WORKING,
            "Forming the next falsifiable hypothesis",
            "Reviewing failed criteria and previous outcomes",
        )
        recent = self.state.hypotheses[-config.max_recent_hypotheses :]
        strategist_prompt = strategy_prompt(
            goal=goal,
            criteria=criteria,
            state=self.state,
            steering=steering,
            recent_hypotheses=recent,
        )
        self.store.save_run_artifact(iteration, "strategist-prompt.md", strategist_prompt)
        decision, strategy_result = await runner.run_structured(
            strategist_prompt,
            StrategyDecision,
            model=model,
            agent=config.strategist_agent,
            title=f"Goal loop {iteration}: strategy",
            status_callback=self._agent_callback("strategist"),
            cancel_check=self._cancel_check,
            profile="analysis",
        )
        self.store.save_run_artifact(iteration, "strategist-output.txt", strategy_result.text)
        hypothesis = Hypothesis(
            id=f"H{iteration:05d}-{uuid.uuid4().hex[:6]}",
            iteration=iteration,
            statement=decision.hypothesis,
            rationale=decision.rationale,
            expected_impact=decision.expected_impact,
            target_criteria=decision.target_criteria,
            plan=decision.plan,
            status="active",
        )
        self.state.hypotheses.append(hypothesis)
        self.state.active_hypothesis_id = hypothesis.id
        self._update_agent(
            "strategist",
            AgentPhase.COMPLETE,
            f"Selected {hypothesis.id}",
            hypothesis.statement,
        )
        self.store.append_event(
            EventRecord(
                type="hypothesis_selected",
                message=hypothesis.statement,
                data=hypothesis.model_dump(mode="json"),
            )
        )

        # Re-read steering and success definitions immediately before execution.
        goal = self.store.read_goal()
        criteria = self.store.read_criteria()
        steering = self.store.read_steering()
        self._update_agent(
            "executor",
            AgentPhase.WORKING,
            f"Executing {hypothesis.id}",
            hypothesis.plan[0] if hypothesis.plan else hypothesis.statement,
        )
        execute_prompt = executor_prompt(
            goal=goal,
            criteria=criteria,
            hypothesis=hypothesis,
            steering=steering,
            baseline_results=before,
            consecutive_no_progress=self.state.consecutive_no_progress,
        )
        self.store.save_run_artifact(iteration, "executor-prompt.md", execute_prompt)
        report: ExecutionReport
        try:
            report, execution_result = await runner.run_structured(
                execute_prompt,
                ExecutionReport,
                model=model,
                agent=config.executor_agent,
                title=f"Goal loop {iteration}: execute {hypothesis.id}",
                status_callback=self._agent_callback("executor"),
                cancel_check=self._cancel_check,
                attempts=2,
                profile="executor",
            )
            self.store.save_run_artifact(iteration, "executor-output.txt", execution_result.text)
        except OpenCodeInterrupted:
            raise
        except OpenCodeError as exc:
            # Work may already have been applied even if the final report was malformed.
            report = ExecutionReport(
                summary="Executor ended without a parseable report; evaluation will inspect actual workspace state.",
                blockers=[str(exc)],
            )
            self.store.save_run_artifact(iteration, "executor-output-error.txt", str(exc))

        if self._should_escalate_executor_action(
            criteria=criteria,
            baseline_results=before,
            report=report,
        ):
            escalation_directive = (
                "The prior execution report was reconnaissance-only while required criteria still fail "
                "because missing artifacts were identified. In this retry, run at least one artifact-producing "
                "command from the active hypothesis plan or criterion commands (for example training/build). "
                "If the command fails, capture the exact failing command, stderr/error, and minimum code/config "
                "patch to unblock the next attempt. Do not stop at static inspection."
            )
            self.store.append_event(
                EventRecord(
                    type="execution_action_escalation",
                    message="Executor retry triggered after reconnaissance-only attempt despite missing-artifact failures",
                    data={"hypothesis_id": hypothesis.id},
                )
            )
            retry_prompt = executor_prompt(
                goal=goal,
                criteria=criteria,
                hypothesis=hypothesis,
                steering=steering,
                baseline_results=before,
                execution_directive=escalation_directive,
                consecutive_no_progress=self.state.consecutive_no_progress,
            )
            self.store.save_run_artifact(iteration, "executor-retry-prompt.md", retry_prompt)
            self._update_agent(
                "executor",
                AgentPhase.WORKING,
                f"Retrying {hypothesis.id} with action escalation",
                "Previous execution was reconnaissance-only; forcing artifact-producing attempt",
            )
            try:
                retry_report, retry_result = await runner.run_structured(
                    retry_prompt,
                    ExecutionReport,
                    model=model,
                    agent=config.executor_agent,
                    title=f"Goal loop {iteration}: execute {hypothesis.id} (action retry)",
                    status_callback=self._agent_callback("executor"),
                    cancel_check=self._cancel_check,
                    attempts=1,
                    profile="executor",
                )
                self.store.save_run_artifact(iteration, "executor-retry-output.txt", retry_result.text)
                report = retry_report
            except OpenCodeInterrupted:
                raise
            except OpenCodeError as exc:
                self.store.save_run_artifact(iteration, "executor-retry-output-error.txt", str(exc))
                report.blockers.append(f"Action-escalation retry failed: {exc}")

        # Verification retry: when the executor applied a fix but explicitly did not
        # re-run the script to verify it, retry immediately so the fix can be confirmed
        # without waiting for a full criteria evaluation cycle (~45 min).
        _UNVERIFIED_MARKERS = (
            "not yet re-executed",
            "fix applied but not",
            "script fix applied",
            "applied but not run",
            "not re-run",
            "needs re-run",
            "not verified",
            "not yet verified",
        )
        if report.files_changed and report.blockers and not self._is_report_format_blocker(report):
            if any(
                any(marker in (b or "").lower() for marker in _UNVERIFIED_MARKERS)
                for b in report.blockers
            ):
                verify_directive = (
                    "The previous execution applied a code fix but did NOT re-run the script to verify it. "
                    "Your ONLY task this round is to run the script that was just fixed and report the output. "
                    "Do not make any additional changes — just execute and report. "
                    f"Files changed in previous round: {', '.join(report.files_changed[:5])}"
                )
                self.store.append_event(
                    EventRecord(
                        type="execution_verification_retry",
                        message="Executor verification retry: fix applied but not re-executed",
                        data={"hypothesis_id": hypothesis.id, "files": report.files_changed[:5]},
                    )
                )
                verify_prompt = executor_prompt(
                    goal=goal,
                    criteria=criteria,
                    hypothesis=hypothesis,
                    steering=steering,
                    baseline_results=before,
                    execution_directive=verify_directive,
                    consecutive_no_progress=self.state.consecutive_no_progress,
                )
                self.store.save_run_artifact(iteration, "executor-verify-prompt.md", verify_prompt)
                self._update_agent(
                    "executor",
                    AgentPhase.WORKING,
                    f"Verifying {hypothesis.id} fix",
                    "Re-running the fixed script to confirm it works",
                )
                try:
                    verify_report, verify_result = await runner.run_structured(
                        verify_prompt,
                        ExecutionReport,
                        model=model,
                        agent=config.executor_agent,
                        title=f"Goal loop {iteration}: verify {hypothesis.id}",
                        status_callback=self._agent_callback("executor"),
                        cancel_check=self._cancel_check,
                        attempts=1,
                        profile="executor",
                    )
                    self.store.save_run_artifact(iteration, "executor-verify-output.txt", verify_result.text)
                    report = verify_report
                except OpenCodeInterrupted:
                    raise
                except OpenCodeError as exc:
                    self.store.save_run_artifact(iteration, "executor-verify-output-error.txt", str(exc))
                    report.blockers.append(f"Verification retry failed: {exc}")
        self._update_agent(
            "executor",
            AgentPhase.COMPLETE,
            f"Finished {hypothesis.id}",
            report.summary,
        )
        self.store.append_event(
            EventRecord(
                type="execution_complete",
                message=report.summary,
                data=report.model_dump(mode="json"),
            )
        )

        # The user may have changed the criteria during execution; evaluate the latest definition.
        config = self.store.read_config()
        runner = OpenCodeRunner(config)
        evaluator = CriteriaEvaluator(config, runner)
        control = self.store.read_control()
        model = control.model_override or config.model
        goal = self.store.read_goal()
        criteria = self.store.read_criteria()
        steering = self.store.read_steering()
        after = await self._evaluate(
            evaluator,
            criteria,
            goal=goal,
            steering=steering,
            model=model,
            label=f"Evaluating {hypothesis.id}",
            artifact_prefix="post-execution",
            active_hypothesis=hypothesis,
            execution_report=report,
            previous_results=before,
        )

        before_count = passed_required_count(criteria.criteria, before)
        after_count = passed_required_count(criteria.criteria, after)
        if after_count > before_count:
            hypothesis.status = "supported"
            hypothesis.outcome = f"Required criteria passing increased from {before_count} to {after_count}."
            hypothesis.evidence.extend(report.evidence)
            self.state.consecutive_no_progress = 0
        else:
            target_improved = any(
                after.get(criterion_id)
                and after[criterion_id].passed
                and not (before.get(criterion_id) and before[criterion_id].passed)
                for criterion_id in hypothesis.target_criteria
            )
            if target_improved:
                hypothesis.status = "supported"
                hypothesis.outcome = "A targeted criterion improved, while total required passes stayed level."
                self.state.consecutive_no_progress = 0
            elif self.state.evaluation_analysis and self.state.evaluation_analysis.material_progress:
                hypothesis.status = "supported"
                evidence = "; ".join(self.state.evaluation_analysis.progress_evidence[:3])
                hypothesis.outcome = "Concrete partial progress was observed while criteria remain failing."
                if evidence:
                    hypothesis.outcome += " " + evidence
                self.state.consecutive_no_progress = 0
            elif report.blockers:
                hypothesis.status = "inconclusive"
                hypothesis.outcome = "Execution encountered blockers: " + "; ".join(report.blockers)
                self.state.consecutive_no_progress += 1
            else:
                hypothesis.status = "refuted"
                hypothesis.outcome = "No measurable required-criterion improvement was observed."
                self.state.consecutive_no_progress += 1
        if self.state.evaluation_analysis:
            diagnostic = (
                self.state.evaluation_analysis.progress_assessment
                or self.state.evaluation_analysis.summary
            ).strip()
            if diagnostic:
                hypothesis.outcome += f" AI diagnosis: {diagnostic}"
            hypothesis.evidence.extend(
                self.state.evaluation_analysis.recommended_next_focus[:3]
            )
        hypothesis.updated_at = utc_now()
        self.state.active_hypothesis_id = None
        self._reset_agents()
        self.store.append_event(
            EventRecord(
                type="hypothesis_evaluated",
                message=hypothesis.outcome,
                data={"hypothesis_id": hypothesis.id, "status": hypothesis.status},
            )
        )

        if all_required_pass(criteria.criteria, after):
            self._mark_achieved(criteria)
            return

        self._set_phase(
            RunPhase.RUNNING,
            f"Iteration {iteration} complete; {after_count} required criteria passing",
        )

    async def _evaluate(
        self,
        evaluator: CriteriaEvaluator,
        criteria: CriteriaDocument,
        *,
        goal: str,
        steering: str,
        model: str | None,
        label: str,
        artifact_prefix: str,
        active_hypothesis: Hypothesis | None = None,
        execution_report: ExecutionReport | None = None,
        previous_results: dict | None = None,
    ) -> dict:
        self._update_agent("evaluator", AgentPhase.WORKING, label, "Checking each success criterion")
        # Use a new dictionary so a later evaluation cannot mutate the saved baseline by alias.
        self.state.criteria_results = {}

        def on_result(result) -> None:
            self.state.criteria_results[result.criterion_id] = result
            self._save_state_throttled(force=True)

        results = await evaluator.evaluate_all(
            criteria.criteria,
            goal=goal,
            steering=steering,
            model=model,
            cancel_check=self._cancel_check,
            result_callback=on_result,
            status_callback=self._agent_callback("evaluator"),
        )
        self.state.criteria_results = results
        passing = passed_required_count(criteria.criteria, results)
        required = sum(1 for criterion in criteria.criteria if criterion.required)
        self.store.append_event(
            EventRecord(
                type="criteria_evaluated",
                message=f"{passing}/{required} required criteria pass",
                data={
                    key: value.model_dump(mode="json") for key, value in results.items()
                },
            )
        )

        self._update_agent(
            "evaluator",
            AgentPhase.WORKING,
            f"Analyzing {label.lower()}",
            "Diagnosing passing, failing, and errored criterion outputs",
        )
        analysis_prompt = evaluation_analysis_prompt(
            goal=goal,
            criteria=criteria,
            results=results,
            previous_results=previous_results,
            iteration=self.state.iteration,
            label=label,
            steering=steering,
            active_hypothesis=active_hypothesis,
            execution_report=(
                execution_report.model_dump(mode="json") if execution_report else None
            ),
        )
        self.store.save_run_artifact(
            self.state.iteration,
            f"{artifact_prefix}-analysis-prompt.md",
            analysis_prompt,
        )
        try:
            analysis, analysis_result = await evaluator.runner.run_structured(
                analysis_prompt,
                EvaluationAnalysis,
                model=model,
                agent=self.store.read_config().evaluator_agent,
                title=f"Goal loop {self.state.iteration}: analyze {label}",
                status_callback=self._agent_callback("evaluator"),
                cancel_check=self._cancel_check,
                profile="analysis",
            )
            analysis.iteration = self.state.iteration
            analysis.label = label
            analysis.source = "ai"
            analysis = self._normalize_evaluation_analysis(
                analysis=analysis,
                criteria=criteria,
                results=results,
            )
            self.store.save_run_artifact(
                self.state.iteration,
                f"{artifact_prefix}-analysis-output.txt",
                analysis_result.text,
            )
        except OpenCodeInterrupted:
            raise
        except OpenCodeError as exc:
            analysis = self._fallback_evaluation_analysis(
                criteria=criteria,
                results=results,
                label=label,
                error=str(exc),
            )
            self.store.save_run_artifact(
                self.state.iteration,
                f"{artifact_prefix}-analysis-error.txt",
                str(exc),
            )
            self.store.append_event(
                EventRecord(
                    type="criteria_analysis_fallback",
                    message=f"AI diagnosis unavailable; using result-based fallback: {exc}",
                    data={"label": label},
                )
            )

        self.state.evaluation_analysis = analysis
        self._save_state_throttled(force=True)
        self.store.save_run_artifact(
            self.state.iteration,
            f"{artifact_prefix}-analysis.json",
            analysis.model_dump_json(indent=2),
        )
        self.store.append_event(
            EventRecord(
                type="criteria_analyzed",
                message=analysis.summary,
                data=analysis.model_dump(mode="json"),
            )
        )
        self._update_agent(
            "evaluator",
            AgentPhase.COMPLETE,
            label,
            f"{passing}/{required} required criteria pass · {analysis.summary}",
        )
        return results

    def _normalize_evaluation_analysis(
        self,
        *,
        analysis: EvaluationAnalysis,
        criteria: CriteriaDocument,
        results: dict,
    ) -> EvaluationAnalysis:
        """Keep AI diagnosis grounded in the actual criterion evaluator outputs."""

        supplied = {item.criterion_id: item for item in analysis.criterion_analyses}
        normalized: list[CriterionAnalysis] = []
        for criterion in criteria.criteria:
            result = results.get(criterion.id)
            actual_status = result.status if result else "unchecked"
            item = supplied.get(criterion.id)
            if item is None:
                item = CriterionAnalysis(
                    criterion_id=criterion.id,
                    observed_status=actual_status,
                    interpretation=result.summary if result else "Not checked",
                    useful_evidence=result.evidence[-3:] if result else [],
                    recommended_actions=(
                        ["Preserve this passing behavior and rerun the check after changes."]
                        if actual_status == "pass"
                        else ["Inspect the recorded evidence and address the direct failure."]
                    ),
                    confidence=0.35,
                )
            else:
                # The raw evaluator remains the stopping authority. AI may interpret the
                # evidence but cannot rewrite a deterministic or prior AI-judge result.
                item.observed_status = actual_status
                if not item.useful_evidence and result:
                    item.useful_evidence = result.evidence[-3:]
            normalized.append(item)
        analysis.criterion_analyses = normalized
        return analysis

    def _fallback_evaluation_analysis(
        self,
        *,
        criteria: CriteriaDocument,
        results: dict,
        label: str,
        error: str,
    ) -> EvaluationAnalysis:
        analyses: list[CriterionAnalysis] = []
        failed: list[str] = []
        passing: list[str] = []
        for criterion in criteria.criteria:
            result = results.get(criterion.id)
            status = result.status if result else "unchecked"
            summary = result.summary if result else "Not checked"
            actions: list[str] = []
            causes: list[str] = []
            if status == "pass":
                passing.append(criterion.id)
                actions.append("Preserve the behavior and rerun this check after changes.")
            else:
                failed.append(criterion.id)
                causes.append(summary)
                actions.append("Inspect the recorded evidence and address the direct failure before rerunning.")
            analyses.append(
                CriterionAnalysis(
                    criterion_id=criterion.id,
                    observed_status=status,
                    interpretation=summary,
                    likely_causes=causes,
                    useful_evidence=(result.evidence[-3:] if result else []),
                    recommended_actions=actions,
                    confidence=0.35,
                )
            )
        summary = (
            f"{len(passing)} criteria pass and {len(failed)} need attention; "
            "AI diagnostic analysis was unavailable."
        )
        return EvaluationAnalysis(
            iteration=self.state.iteration,
            label=label,
            summary=summary,
            progress_assessment=f"Fallback analysis generated because OpenCode analysis failed: {error}",
            material_progress=False,
            progress_evidence=[],
            criterion_analyses=analyses,
            cross_criterion_findings=[],
            recommended_next_focus=failed[:5],
            source="fallback",
        )

    def _mark_achieved(self, criteria: CriteriaDocument) -> None:
        required = sum(1 for criterion in criteria.criteria if criterion.required)
        self._reset_agents(complete=True)
        self.state.phase = RunPhase.ACHIEVED
        self.state.message = f"Goal achieved: all {required} required criteria pass"
        self.state.ended_at = utc_now()
        self.state.active_hypothesis_id = None
        self.store.save_state(self.state)
        self.store.update_control(
            desired_state="stopped",
            note=f"Automatically stopped because all {required} required criteria passed.",
        )
        self.store.append_event(
            EventRecord(
                type="goal_achieved",
                message=self.state.message,
                data={"iteration": self.state.iteration},
            )
        )

    def _cancel_check(self) -> str | None:
        try:
            desired = self.store.read_control().desired_state.value
        except Exception:
            # A text editor can briefly expose an incomplete save. Do not kill work for that.
            return None
        return None if desired == "running" else desired

    def _read_control_safely(self):
        try:
            return self.store.read_control()
        except Exception as exc:
            self.state.last_error = f"Invalid control.yaml: {exc}"
            self._set_phase(RunPhase.ERROR, self.state.last_error)
            return None

    def _set_phase(self, phase: RunPhase, message: str) -> None:
        self.state.phase = phase
        self.state.message = message
        self.store.save_state(self.state)

    def _should_escalate_executor_action(
        self,
        *,
        criteria: CriteriaDocument,
        baseline_results: dict,
        report: ExecutionReport,
    ) -> bool:
        if report.blockers and not self._is_report_format_blocker(report):
            return False
        if not self._has_missing_artifact_failures(criteria, baseline_results):
            return False
        return self._report_is_reconnaissance_only(report)

    def _is_report_format_blocker(self, report: ExecutionReport) -> bool:
        if not report.blockers:
            return False
        format_markers = (
            "valid executionreport json",
            "no proposal json object",
            "parseable report",
        )
        return all(
            any(marker in blocker.lower() for marker in format_markers)
            for blocker in report.blockers
        )

    def _has_missing_artifact_failures(
        self,
        criteria: CriteriaDocument,
        baseline_results: dict,
    ) -> bool:
        required_ids = {item.id for item in criteria.criteria if item.required}
        signals = (
            "filenotfound",
            "not found",
            "no such file",
            "does not exist",
            "cannot find",
            "missing",
            "checkpoint",
            "directory not found",
        )
        for criterion_id, result in baseline_results.items():
            if criterion_id not in required_ids:
                continue
            status = str(getattr(result, "status", "unchecked"))
            if status == "pass":
                continue
            chunks = [
                str(getattr(result, "summary", "")),
                str(getattr(result, "error", "")),
                " ".join(str(item) for item in (getattr(result, "evidence", None) or [])),
            ]
            corpus = " ".join(chunks).lower()
            if any(token in corpus for token in signals):
                return True
        return False

    def _report_is_reconnaissance_only(self, report: ExecutionReport) -> bool:
        if report.files_changed:
            return False
        commands = [item.strip() for item in report.commands_run if str(item).strip()]
        if not commands:
            return True
        return all(self._is_read_only_command(item) for item in commands)

    def _is_read_only_command(self, command: str) -> bool:
        lowered = command.strip().lower()
        read_only_prefixes = (
            "read ",
            "cat ",
            "type ",
            "get-content ",
            "get-childitem",
            "dir ",
            "ls ",
            "test-path ",
            "findstr ",
            "select-string ",
            "rg ",
            "grep ",
            "where ",
            "which ",
            "stat ",
        )
        if lowered.startswith(read_only_prefixes):
            return True
        # Treat simple listing/check pipelines as reconnaissance.
        if re.search(r"\b(get-childitem|test-path|select-string|findstr|rg|grep|cat|type|get-content)\b", lowered):
            if not re.search(r"\b(python|pip|pytest|npm|pnpm|yarn|cargo|go\s+test|make|cmake|dotnet|mvn|gradle|train|build|run)\b", lowered):
                return True
        return False

    def _update_agent(self, name: str, phase: AgentPhase, task: str, detail: str = "") -> None:
        agent = self.state.agents[name]
        if phase == AgentPhase.WORKING and agent.phase != AgentPhase.WORKING:
            agent.started_at = utc_now()
        agent.phase = phase
        agent.task = task
        agent.detail = detail[-1000:]
        agent.updated_at = utc_now()
        self._save_state_throttled(force=phase != AgentPhase.WORKING)

    def _agent_callback(self, name: str):
        def callback(event_type: str, detail: str) -> None:
            agent = self.state.agents[name]
            if event_type == "context_recovery":
                agent.phase = AgentPhase.WAITING
                agent.task = "Recovering from context overflow"
            elif event_type in {"started", "text", "tool", "parsing", "retry"}:
                if agent.phase in {AgentPhase.WAITING, AgentPhase.ERROR}:
                    agent.phase = AgentPhase.WORKING
                if event_type == "started" and agent.task == "Recovering from context overflow":
                    agent.task = "Running compact fresh-session retry"
            elif event_type == "error":
                # OpenCode can emit an overflow error immediately before Goal Agent
                # starts a recovery retry. Keep it transient until retries are exhausted.
                agent.phase = AgentPhase.WAITING
                agent.task = "OpenCode reported an error; checking recovery"
            elif event_type == "complete":
                agent.phase = AgentPhase.WORKING
            agent.detail = detail[-1000:]
            agent.updated_at = utc_now()
            self._save_state_throttled(force=event_type in {"context_recovery", "error"})

        return callback

    def _save_state_throttled(self, force: bool = False) -> None:
        try:
            refresh = self.store.read_config().status_refresh_seconds
        except Exception:
            refresh = 1.0
        now = time.monotonic()
        if force or now - self._last_status_write >= refresh:
            self.store.save_state(self.state)
            self._last_status_write = now

    def _mark_active_agent_error(self, error: str) -> None:
        active = [
            agent
            for agent in self.state.agents.values()
            if agent.phase in {AgentPhase.WORKING, AgentPhase.WAITING}
        ]
        targets = active or list(self.state.agents.values())[:1]
        for agent in targets:
            agent.phase = AgentPhase.ERROR
            agent.task = "Error"
            agent.detail = error[-1000:]
            agent.updated_at = utc_now()
        self.store.save_state(self.state)

    def _reset_agents(self, error: str | None = None, complete: bool = False) -> None:
        for agent in self.state.agents.values():
            if error and agent.phase in {AgentPhase.WORKING, AgentPhase.WAITING}:
                agent.phase = AgentPhase.ERROR
                agent.detail = error[-1000:]
                agent.task = "Error"
            elif error:
                agent.phase = AgentPhase.IDLE
                agent.task = "Idle"
                agent.detail = ""
            elif complete:
                agent.phase = AgentPhase.COMPLETE
                agent.task = "Goal achieved"
                agent.detail = "All required criteria pass"
            else:
                agent.phase = AgentPhase.IDLE
                agent.task = "Idle"
                agent.detail = ""
            agent.updated_at = utc_now()
        self.store.save_state(self.state)
