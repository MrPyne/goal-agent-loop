import asyncio
import json
import sys
from pathlib import Path

from goal_agent.loop import GoalAgentLoop
from goal_agent.models import (
    CriteriaDocument,
    CriterionDefinition,
    CriterionKind,
    CriteriaRevisionSuggestion,
    ExecutionReport,
    RunPhase,
    SerialCriterionDiagnosis,
)
from goal_agent.storage import ProjectStore


def test_loop_reaches_goal_and_stops(tmp_path: Path) -> None:
    fake = tmp_path / "fake_opencode.py"
    fake.write_text(
        r'''
import json
import pathlib
import sys

args = sys.argv[1:]
if args and args[0] == "models":
    print("fake/model")
    raise SystemExit(0)

prompt = sys.stdin.read()
if "You are the strategist" in prompt:
    payload = {
        "hypothesis": "Creating done.txt will satisfy the remaining criterion.",
        "rationale": "The only failed criterion requires that file.",
        "expected_impact": "The required file_exists criterion will pass.",
        "target_criteria": ["done-file"],
        "plan": ["Create done.txt"],
        "avoid_repeating": []
    }
elif "You are the executor" in prompt:
    pathlib.Path("done.txt").write_text("complete\n", encoding="utf-8")
    payload = {
        "summary": "Created done.txt",
        "actions": ["Created done.txt"],
        "files_changed": ["done.txt"],
        "commands_run": [],
        "evidence": ["done.txt exists"],
        "blockers": []
    }
elif "You are the diagnostic evaluator" in prompt:
    passing = pathlib.Path("done.txt").exists()
    payload = {
        "summary": "The completion artifact now passes." if passing else "The completion artifact is still missing.",
        "progress_assessment": "The hypothesis solved the failed criterion." if passing else "The file must be created next.",
        "criterion_analyses": [{
            "criterion_id": "done-file",
            "observed_status": "pass",
            "interpretation": "done.txt exists" if passing else "done.txt is absent",
            "likely_causes": [] if passing else ["The artifact has not been created"],
            "useful_evidence": ["done.txt"],
            "recommended_actions": ["Preserve the file"] if passing else ["Create done.txt"],
            "confidence": 0.99
        }],
        "cross_criterion_findings": [],
        "recommended_next_focus": [] if passing else ["Create done.txt"]
    }
else:
    payload = {
        "passed": True,
        "confidence": 1.0,
        "summary": "pass",
        "evidence": [],
        "missing": []
    }
text = "<GOAL_AGENT_JSON>\n" + json.dumps(payload) + "\n</GOAL_AGENT_JSON>"
event = {"type": "text", "sessionID": "fake-session", "part": {"type": "text", "text": text}}
print(json.dumps(event), flush=True)
''',
        encoding="utf-8",
    )

    store = ProjectStore(tmp_path)
    store.initialize(model="fake/model")
    config = store.read_config()
    config.opencode_command = [sys.executable, str(fake)]
    config.poll_interval_seconds = 0.01
    config.iteration_delay_seconds = 0.01
    config.opencode_timeout_seconds = 10
    store.write_config(config)
    store.write_goal("Create the completion artifact")
    store.write_criteria(
        CriteriaDocument(
            criteria=[
                CriterionDefinition(
                    id="done-file",
                    description="done.txt exists",
                    kind=CriterionKind.FILE_EXISTS,
                    path="done.txt",
                )
            ]
        )
    )
    store.update_control(desired_state="running")

    state = asyncio.run(GoalAgentLoop(store).run_forever())

    assert state.phase == RunPhase.ACHIEVED
    assert (tmp_path / "done.txt").exists()
    assert store.read_control().desired_state.value == "stopped"
    assert state.hypotheses[-1].status == "supported"
    assert state.evaluation_analysis is not None
    assert state.evaluation_analysis.source == "ai"
    assert state.evaluation_analysis.criterion_analyses[0].observed_status == "pass"
    assert "AI diagnosis" in state.hypotheses[-1].outcome
    baseline_analysis_path = store.iteration_dir(1) / "baseline-analysis.json"
    assert baseline_analysis_path.exists()
    baseline_analysis = json.loads(baseline_analysis_path.read_text(encoding="utf-8"))
    assert baseline_analysis["criterion_analyses"][0]["observed_status"] == "fail"
    strategist_prompt = (store.iteration_dir(1) / "strategist-prompt.md").read_text(encoding="utf-8")
    assert "AI DIAGNOSIS OF THE LATEST CRITERIA OUTPUTS" in strategist_prompt
    assert "The completion artifact is still missing" in strategist_prompt
    assert (store.iteration_dir(1) / "post-execution-analysis.json").exists()
    assert store.evaluation_analysis_path.exists()


def test_pause_interrupts_active_opencode_and_stop_exits(tmp_path: Path) -> None:
    fake = tmp_path / "slow_opencode.py"
    fake.write_text(
        "import sys, time\n"
        "_ = sys.stdin.read()\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    store = ProjectStore(tmp_path)
    store.initialize(model="fake/model")
    config = store.read_config()
    config.opencode_command = [sys.executable, str(fake)]
    config.poll_interval_seconds = 0.02
    config.iteration_delay_seconds = 0.01
    store.write_config(config)
    store.write_goal("Never reached in this test")
    store.write_criteria(
        CriteriaDocument(
            criteria=[
                CriterionDefinition(
                    id="missing",
                    description="missing.txt exists",
                    kind=CriterionKind.FILE_EXISTS,
                    path="missing.txt",
                )
            ]
        )
    )
    store.update_control(desired_state="running")

    async def scenario():
        task = asyncio.create_task(GoalAgentLoop(store).run_forever())
        for _ in range(100):
            if store.load_state().agents["strategist"].phase.value == "working":
                break
            await asyncio.sleep(0.02)
        store.update_control(desired_state="paused")
        for _ in range(100):
            if store.load_state().phase == RunPhase.PAUSED:
                break
            await asyncio.sleep(0.02)
        assert store.load_state().phase == RunPhase.PAUSED
        store.update_control(desired_state="stopped")
        return await asyncio.wait_for(task, timeout=5)

    state = asyncio.run(scenario())
    assert state.phase == RunPhase.STOPPED


def test_serial_executor_retries_synthetic_format_repair_report(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    store.initialize()
    loop = GoalAgentLoop(store)

    report = ExecutionReport(
        summary="fix curriculum_progression (format repair)",
        actions=["Create scripts/curriculum_controller.py"],
        files_changed=["scripts/curriculum_controller.py"],
    )

    assert loop._report_requires_implementation_retry(report)


def test_detects_executor_that_stops_before_tool_call() -> None:
    error = RuntimeError(
        "OpenCode did not return valid ExecutionReport JSON after 2 attempts: "
        "OpenCode completed without a final assistant text response. It emitted only event records "
        '(step_finish=1). {"tokens":{"output":1}}'
    )

    assert GoalAgentLoop._is_executor_tool_capability_error(error)


def test_serial_mode_rejects_duplicate_removal_of_external_benchmark() -> None:
    criterion = CriterionDefinition(
        id="tau_bench_score",
        description="The final model scores at least 70% on tau-bench.",
        kind=CriterionKind.COMMAND,
        command="python scripts/run_taubench.py",
    )
    diagnosis = SerialCriterionDiagnosis(
        summary="A comparison report already has a score.",
        root_cause="The benchmark is supposedly duplicated.",
        criteria_revision=CriteriaRevisionSuggestion(
            criterion_id="tau_bench_score",
            rationale="Remove the standalone official benchmark.",
            revision_reason="duplicate",
            remove_target=True,
        ),
    )

    error = GoalAgentLoop._criteria_revision_admission_error(criterion, diagnosis)

    assert error is not None
    assert "external benchmark" in error


def test_serial_mode_pauses_for_review_required_criteria_revision(tmp_path: Path) -> None:
    fake = tmp_path / "serial_noop_opencode.py"
    fake.write_text(
        r'''
import json
import sys

args = sys.argv[1:]
if args and args[0] == "models":
    print("fake/model")
    raise SystemExit(0)

_ = sys.stdin.read()
if "diagnostic criterion-repair strategist" in _:
    payload = {
        "classification": "implementation_defect",
        "summary": "The missing artifact still needs to be created.",
        "root_cause": "No file has been created.",
        "recommended_project_change": "Create missing.txt.",
        "executor_plan": ["Create missing.txt."],
        "criteria_revision": {
            "criterion_id": "missing-file",
            "rationale": "Example review-required suggestion.",
            "revision_reason": "not_meaningful",
            "proposed_criteria": [{
                "id": "missing-file-ready",
                "description": "missing.txt exists and is the validated completion artifact",
                "kind": "file_exists",
                "path": "missing.txt",
            }],
            "safeguards": ["User approval is required."],
            "approval_required": True,
        },
    }
else:
    payload = {
        "summary": "Inspected the failure but made no change.",
        "actions": ["Inspected the existing files"],
        "files_changed": [],
        "commands_run": [],
        "evidence": [],
        "blockers": [],
    }
text = "<GOAL_AGENT_JSON>\n" + json.dumps(payload) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type": "text", "sessionID": "fake-session", "part": {"type": "text", "text": text}}), flush=True)
''',
        encoding="utf-8",
    )

    store = ProjectStore(tmp_path)
    store.initialize(model="fake/model")
    config = store.read_config()
    config.opencode_command = [sys.executable, str(fake)]
    config.poll_interval_seconds = 0.01
    config.iteration_delay_seconds = 0.01
    config.opencode_timeout_seconds = 5
    config.criterion_serial_mode = True
    config.no_progress_rethink_after = 2
    store.write_config(config)
    store.write_goal("Create the missing completion artifact")
    store.write_criteria(
        CriteriaDocument(
            criteria=[
                CriterionDefinition(
                    id="missing-file",
                    description="missing.txt exists",
                    kind=CriterionKind.FILE_EXISTS,
                    path="missing.txt",
                )
            ]
        )
    )
    store.update_control(desired_state="running")

    async def scenario():
        task = asyncio.create_task(GoalAgentLoop(store).run_forever())
        for _ in range(500):
            state = store.load_state()
            if state.phase == RunPhase.PAUSED and "criteria review" in state.message:
                break
            await asyncio.sleep(0.02)
        state = store.load_state()
        assert state.phase == RunPhase.PAUSED
        # A proposed success-criteria change needs user review before the
        # executor is allowed to alter project artifacts around that new proxy.
        assert state.iteration == 1
        assert store.read_control().desired_state.value == "paused"
        assert state.consecutive_no_progress == 0
        assert state.serial_consecutive_no_progress == 0
        assert state.criteria_revision_suggestions[0].criterion_id == "missing-file"
        assert any(item.agent == "strategist" for item in state.agent_activity)
        assert not any(item.agent == "executor" for item in state.agent_activity)
        assert "Example review-required suggestion" in store.status_markdown_path.read_text(encoding="utf-8")
        store.update_control(desired_state="stopped")
        return await asyncio.wait_for(task, timeout=5)

    final_state = asyncio.run(scenario())
    assert final_state.phase == RunPhase.STOPPED
    events = store.read_events(limit=100)
    assert any(event.get("type") == "criteria_revision_review_required" for event in events)
    assert any(event.get("type") == "criteria_revision_suggested" for event in events)


def test_loop_auto_pauses_after_context_recovery_is_exhausted(tmp_path: Path) -> None:
    fake = tmp_path / "overflow_opencode.py"
    fake.write_text(
        r'''
import json
import sys
args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)
_ = sys.stdin.read()
print(json.dumps({
    "type": "error",
    "error": {
        "name": "ContextOverflowError",
        "data": {"message": "request (70793 tokens) exceeds the available context size (65536 tokens)"}
    }
}))
raise SystemExit(1)
''',
        encoding="utf-8",
    )
    store = ProjectStore(tmp_path)
    store.initialize(model="fake/model")
    config = store.read_config()
    config.opencode_command = [sys.executable, str(fake)]
    config.poll_interval_seconds = 0.01
    config.iteration_delay_seconds = 0.01
    config.opencode_timeout_seconds = 5
    store.write_config(config)
    store.write_goal("Create missing.txt")
    store.write_criteria(
        CriteriaDocument(
            criteria=[
                CriterionDefinition(
                    id="missing",
                    description="missing.txt exists",
                    kind=CriterionKind.FILE_EXISTS,
                    path="missing.txt",
                )
            ]
        )
    )
    store.update_control(desired_state="running")

    async def scenario():
        task = asyncio.create_task(GoalAgentLoop(store).run_forever())
        for _ in range(1200):
            state = store.load_state()
            if (
                state.phase == RunPhase.PAUSED
                and "context recovery" in state.message.lower()
            ):
                break
            await asyncio.sleep(0.02)
        state = store.load_state()
        assert state.phase == RunPhase.PAUSED
        assert store.read_control().desired_state.value == "paused"
        assert "context recovery" in state.message.lower()
        store.update_control(desired_state="stopped")
        return await asyncio.wait_for(task, timeout=5)

    final_state = asyncio.run(scenario())
    assert final_state.phase == RunPhase.STOPPED
    events = store.read_events(limit=100)
    assert any(event.get("type") == "context_recovery_exhausted" for event in events)
