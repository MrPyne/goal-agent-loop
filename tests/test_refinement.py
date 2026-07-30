from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from goal_agent.models import (
    CriterionDefinition,
    CriterionKind,
    RefinementMessage,
    SetupProposal,
)
from goal_agent.proposal_quality import assess_setup_proposal
from goal_agent.storage import ProjectStore
from goal_agent.web import create_app


def _poll(client: TestClient, job_id: str) -> dict:
    job: dict = {}
    for _ in range(600):
        job = client.get(f"/api/proposal-jobs/{job_id}").json()
        if job["status"] in {"completed", "failed", "cancelled"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"proposal job did not finish: {job}")


def test_proposal_quality_blocks_vague_ai_criterion() -> None:
    proposal = SetupProposal(
        refined_goal="Create a useful project dashboard",
        ready_to_finalize=True,
        criteria=[
            CriterionDefinition(
                id="quality",
                description="The dashboard is good and user friendly",
                kind=CriterionKind.AI_JUDGE,
                judge_prompt="Review the dashboard and decide whether it is good.",
            )
        ],
    )
    checked = assess_setup_proposal(proposal)
    assert checked.ready_to_finalize is False
    assert any(item.severity == "blocking" for item in checked.criteria_quality_issues)
    assert any("PASS" in item.suggested_fix for item in checked.criteria_quality_issues)


def test_proposal_quality_accepts_concrete_command_criterion() -> None:
    proposal = SetupProposal(
        refined_goal="Create output.txt with verified generated content",
        ready_to_finalize=True,
        readiness_reason="All decisions and checks are concrete.",
        criteria=[
            CriterionDefinition(
                id="verify-output",
                description="The project verification script confirms output.txt has the required content.",
                kind=CriterionKind.COMMAND,
                command="python verify_output.py",
                expected_exit_code=0,
            )
        ],
    )
    checked = assess_setup_proposal(proposal)
    assert checked.ready_to_finalize is True
    assert not [item for item in checked.criteria_quality_issues if item.severity == "blocking"]


def test_refinement_session_persists(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    store.initialize(model="fake/model")
    session = store.read_refinement_session()
    session.status = "refining"
    session.messages.append(RefinementMessage(role="user", content="Keep the API compatible."))
    store.write_refinement_session(session)

    loaded = ProjectStore(tmp_path).read_refinement_session()
    assert loaded.status == "refining"
    assert loaded.messages[0].content == "Keep the API compatible."


def test_iterative_refinement_can_finalize_goal_and_criteria(tmp_path: Path) -> None:
    fake = tmp_path / "refinement_opencode.py"
    fake.write_text(
        r'''
import json
import sys

args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)
prompt = sys.stdin.read()
answered = "The exact required count is 10" in prompt
if not answered:
    proposal = {
        "refined_goal": "Create a video publishing plan",
        "assistant_message": "I need the target publishing count before the stopping rule can be exact.",
        "clarifying_questions": ["How many videos must the finished plan include?"],
        "assumptions": ["The plan will be stored in channel-plan.md."],
        "criteria": [{
            "id": "plan-file",
            "description": "The channel plan file exists in the project.",
            "kind": "file_exists",
            "path": "channel-plan.md",
            "required": True
        }],
        "criteria_quality_issues": [],
        "ready_to_finalize": False,
        "readiness_reason": "The required video count is unresolved."
    }
else:
    proposal = {
        "refined_goal": "Create a channel-plan.md publishing plan for exactly 10 videos, with a title, audience, premise, and production checklist for every video.",
        "assistant_message": "The count is now resolved and the draft has exact automated checks.",
        "clarifying_questions": [],
        "assumptions": ["Markdown is an acceptable output format."],
        "criteria": [{
            "id": "verify-plan",
            "description": "The verification script confirms channel-plan.md contains exactly 10 complete video entries.",
            "kind": "command",
            "command": "python verify_channel_plan.py",
            "expected_exit_code": 0,
            "required": True
        }],
        "criteria_quality_issues": [],
        "ready_to_finalize": True,
        "readiness_reason": "All material decisions are resolved and the required outcome has an exact automated verification command."
    }
text = "<GOAL_AGENT_JSON>\n" + json.dumps(proposal) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type": "text", "part": {"type": "text", "text": text}}))
''',
        encoding="utf-8",
    )
    store = ProjectStore(tmp_path)
    store.initialize(model="fake/model")
    config = store.read_config()
    config.opencode_command = [sys.executable, str(fake)]
    config.poll_interval_seconds = 0.01
    config.opencode_timeout_seconds = 10
    store.write_config(config)
    store.write_goal("Create a video publishing plan")

    with TestClient(create_app(tmp_path)) as client:
        first = client.post(
            "/api/goals/default/proposal-jobs",
            json={"mode": "goal", "feedback": "Help make this exact."},
        )
        first_job = _poll(client, first.json()["id"])
        assert first_job["status"] == "completed", first_job
        assert first_job["result"]["ready_to_finalize"] is False
        session = client.get("/api/goals/default/refinement-session").json()
        assert session["status"] == "refining"
        assert len(session["messages"]) == 2

        second = client.post(
            "/api/goals/default/proposal-jobs",
            json={"mode": "goal", "feedback": "The exact required count is 10."},
        )
        second_job = _poll(client, second.json()["id"])
        assert second_job["status"] == "completed", second_job
        assert second_job["result"]["ready_to_finalize"] is True
        session = client.get("/api/goals/default/refinement-session").json()
        assert session["status"] == "ready"
        assert len(session["messages"]) == 4

        finalized = client.post(
            "/api/goals/default/refinement-session/finalize", json={"force": False}
        )
        assert finalized.status_code == 200, finalized.text
        assert finalized.json()["session"]["status"] == "finalized"
        assert "exactly 10 videos" in store.read_goal()
        assert store.read_criteria().criteria[0].command == "python verify_channel_plan.py"


def test_finalize_rejects_unresolved_questions(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    store.initialize(model="fake/model")
    session = store.read_refinement_session()
    session.status = "refining"
    session.current_proposal = SetupProposal(
        refined_goal="Build a report",
        clarifying_questions=["Which data source should be authoritative?"],
        criteria=[
            CriterionDefinition(
                id="report",
                description="The expected report artifact exists at report.md.",
                kind=CriterionKind.FILE_EXISTS,
                path="report.md",
            )
        ],
        ready_to_finalize=False,
    )
    store.write_refinement_session(session)

    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/goals/default/refinement-session/finalize", json={"force": False}
        )
        assert response.status_code == 409
        assert "not ready" in response.json()["detail"].lower()


def test_refinement_context_is_bounded_without_deleting_history(tmp_path: Path) -> None:
    from goal_agent.models import CriteriaDocument
    from goal_agent.refinement_context import build_refinement_context

    store = ProjectStore(tmp_path)
    store.initialize(model="fake/model")
    session = store.read_refinement_session()
    for index in range(30):
        role = "user" if index % 2 == 0 else "assistant"
        session.messages.append(
            RefinementMessage(role=role, content=f"message-{index} " + ("x" * 5000))
        )

    context = build_refinement_context(
        session=session,
        saved_goal="Create a bounded refinement flow",
        saved_criteria=CriteriaDocument(),
        mode="goal",
    )

    assert len(session.messages) == 30
    assert session.compacted_message_count == 22
    assert session.conversation_summary
    assert len(context.transcript) < 48_500
    assert context.estimated_input_tokens < 13_000
    assert "message-29" in context.transcript


def test_refinement_retries_after_context_overflow(tmp_path: Path) -> None:
    fake = tmp_path / "overflow_then_compact_opencode.py"
    fake.write_text(
        r'''
import json
import sys

args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)
prompt = sys.stdin.read()
if "LOW-CONTEXT RECOVERY MODE" not in prompt:
    print(json.dumps({
        "type": "error",
        "error": {
            "name": "ContextOverflowError",
            "data": {"message": "request (67191 tokens) exceeds the available context size (65536 tokens)"}
        }
    }))
    raise SystemExit(1)
proposal = {
    "refined_goal": "Create a verified context-safe goal definition.",
    "assistant_message": "I recovered using the compact context path.",
    "clarifying_questions": [],
    "assumptions": [],
    "criteria": [{
        "id": "verify-context",
        "description": "The verification command confirms the context-safe implementation.",
        "kind": "command",
        "command": "python verify_context.py",
        "expected_exit_code": 0,
        "required": True
    }],
    "criteria_quality_issues": [],
    "ready_to_finalize": True,
    "readiness_reason": "The result has a deterministic verification command."
}
text = "<GOAL_AGENT_JSON>\n" + json.dumps(proposal) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type": "text", "part": {"type": "text", "text": text}}))
''',
        encoding="utf-8",
    )
    store = ProjectStore(tmp_path)
    store.initialize(model="fake/model")
    config = store.read_config()
    config.opencode_command = [sys.executable, str(fake)]
    config.poll_interval_seconds = 0.01
    config.opencode_timeout_seconds = 10
    store.write_config(config)
    store.write_goal("Create a context-safe goal definition")

    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/goals/default/proposal-jobs",
            json={"mode": "goal", "feedback": "Make it concrete."},
        )
        job = _poll(client, response.json()["id"])

    assert job["status"] == "completed", job
    assert job["result"]["context_info"]["mode"] == "compact_retry"
    assert job["result"]["context_info"]["overflow_retries"] == 1
    session = store.read_refinement_session()
    assert session.context_overflow_retries == 1
    assert session.last_context_mode == "compact_retry"
    assert session.current_proposal is not None
    assert session.current_proposal.ready_to_finalize is True
