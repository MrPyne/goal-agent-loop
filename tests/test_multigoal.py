import asyncio
import json
import sys
from pathlib import Path

from goal_agent.models import CriteriaDocument, CriterionDefinition, CriterionKind, RunPhase
from goal_agent.storage import ProjectStore
from goal_agent.supervisor import GoalSupervisor


def test_multiple_goals_have_isolated_state(tmp_path: Path) -> None:
    project = ProjectStore(tmp_path)
    project.initialize(model="fake/model")
    second = project.create_goal("second", title="Second goal", goal="Do another thing")

    project.write_goal("First goal")
    second.write_goal("Second goal")
    project.update_control(desired_state="running")

    assert project.list_goal_ids() == ["default", "second"]
    assert project.read_goal() == "First goal"
    assert second.read_goal() == "Second goal"
    assert second.read_control().desired_state.value == "paused"
    assert project.loop_lock_path != second.loop_lock_path


def test_supervisor_runs_two_goals_concurrently(tmp_path: Path) -> None:
    fake = tmp_path / "fake_opencode.py"
    fake.write_text(
        r'''
import json
import pathlib
import sys
import time

args = sys.argv[1:]
if args and args[0] == "models":
    print("fake/model")
    raise SystemExit(0)

prompt = sys.stdin.read()
target = "alpha.txt" if "alpha.txt" in prompt else "beta.txt"
time.sleep(0.05)
if "You are the strategist" in prompt:
    payload = {
        "hypothesis": f"Creating {target} will satisfy the criterion.",
        "rationale": "The required file is missing.",
        "expected_impact": "The file criterion will pass.",
        "target_criteria": ["done"],
        "plan": [f"Create {target}"],
        "avoid_repeating": []
    }
elif "You are the executor" in prompt:
    pathlib.Path(target).write_text("complete\n", encoding="utf-8")
    payload = {
        "summary": f"Created {target}",
        "actions": [f"Created {target}"],
        "files_changed": [target],
        "commands_run": [],
        "evidence": [f"{target} exists"],
        "blockers": []
    }
elif "You are the diagnostic evaluator" in prompt:
    passing = pathlib.Path(target).exists()
    payload = {
        "summary": f"{target} passes" if passing else f"{target} is missing",
        "progress_assessment": "The required artifact exists." if passing else "The artifact must be created.",
        "criterion_analyses": [{
            "criterion_id": "done",
            "observed_status": "pass" if passing else "fail",
            "interpretation": f"{target} exists" if passing else f"{target} is missing",
            "likely_causes": [] if passing else ["The file has not been created"],
            "useful_evidence": [target],
            "recommended_actions": [] if passing else [f"Create {target}"],
            "confidence": 1.0
        }],
        "cross_criterion_findings": [],
        "recommended_next_focus": [] if passing else [f"Create {target}"]
    }
else:
    payload = {"passed": True, "confidence": 1.0, "summary": "pass", "evidence": [], "missing": []}
print(json.dumps({
    "type": "text",
    "sessionID": "fake-session",
    "part": {"type": "text", "text": "<GOAL_AGENT_JSON>\n" + json.dumps(payload) + "\n</GOAL_AGENT_JSON>"}
}), flush=True)
''',
        encoding="utf-8",
    )

    project = ProjectStore(tmp_path)
    project.initialize(model="fake/model")
    second = project.create_goal("second", title="Second", goal="Create beta.txt")
    config = project.read_config()
    config.opencode_command = [sys.executable, str(fake)]
    config.poll_interval_seconds = 0.01
    config.iteration_delay_seconds = 0.01
    config.max_concurrent_goals = 2
    project.write_config(config)

    project.write_goal("Create alpha.txt")
    project.write_criteria(
        CriteriaDocument(criteria=[CriterionDefinition(id="done", description="alpha exists", kind=CriterionKind.FILE_EXISTS, path="alpha.txt")])
    )
    second.write_criteria(
        CriteriaDocument(criteria=[CriterionDefinition(id="done", description="beta exists", kind=CriterionKind.FILE_EXISTS, path="beta.txt")])
    )

    async def scenario() -> None:
        supervisor = GoalSupervisor(project)
        await asyncio.gather(supervisor.start("default"), supervisor.start("second"))
        for _ in range(1500):
            if (
                project.load_state().phase == RunPhase.ACHIEVED
                and second.load_state().phase == RunPhase.ACHIEVED
            ):
                break
            await asyncio.sleep(0.01)
        await supervisor.shutdown()

    asyncio.run(scenario())

    assert project.load_state().phase == RunPhase.ACHIEVED
    assert second.load_state().phase == RunPhase.ACHIEVED
    assert (tmp_path / "alpha.txt").exists()
    assert (tmp_path / "beta.txt").exists()
