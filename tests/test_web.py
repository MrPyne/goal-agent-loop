from pathlib import Path

from fastapi.testclient import TestClient

from goal_agent.storage import ProjectStore
from goal_agent.web import create_app


def test_web_api_creates_and_updates_goal(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    store.initialize(model="fake/model")

    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/goals",
            json={"id": "web-goal", "title": "Web goal", "goal": "Create output.txt"},
        )
        assert response.status_code == 201
        assert response.json()["metadata"]["id"] == "web-goal"

        criteria = [
            {
                "id": "output",
                "description": "output.txt exists",
                "kind": "file_exists",
                "required": True,
                "override": "auto",
                "path": "output.txt",
                "expected_exit_code": 0,
                "regex": False,
                "case_sensitive": True,
                "evidence_paths": [],
                "confidence_threshold": 0.75,
            }
        ]
        assert client.put("/api/goals/web-goal/criteria", json={"criteria": criteria}).status_code == 200
        detail = client.get("/api/goals/web-goal").json()
        assert detail["criteria"]["criteria"][0]["id"] == "output"
        assert detail["paths"]["goal"].endswith("goals/web-goal/control/goal.md")


def test_legacy_layout_is_migrated(tmp_path: Path) -> None:
    legacy = tmp_path / ".goal-agent"
    (legacy / "control").mkdir(parents=True)
    (legacy / "status").mkdir()
    (legacy / "runs").mkdir()
    (legacy / "control" / "config.yaml").write_text(
        f"project_dir: {tmp_path}\nopencode_command:\n- opencode\n", encoding="utf-8"
    )
    (legacy / "control" / "goal.md").write_text("# Goal\n\nLegacy goal\n", encoding="utf-8")
    (legacy / "control" / "criteria.yaml").write_text("revision: 1\ncriteria: []\n", encoding="utf-8")
    (legacy / "control" / "control.yaml").write_text("desired_state: paused\nrevision: 1\n", encoding="utf-8")
    (legacy / "control" / "steering.md").write_text("# Steering Notes\n", encoding="utf-8")

    store = ProjectStore(tmp_path)
    store.require_initialized()

    assert store.read_goal() == "Legacy goal"
    assert store.config_path == legacy / "config.yaml"
    assert store.goal_path == legacy / "goals" / "default" / "control" / "goal.md"


def test_proposal_job_returns_result_and_progress(tmp_path: Path) -> None:
    import json
    import sys
    import time

    fake = tmp_path / "proposal_opencode.py"
    fake.write_text(
        r'''
import json
import sys

args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)
prompt = sys.stdin.read()
proposal = {
    "refined_goal": "Create a tested video channel project",
    "goal_rationale": "The outcome is specific and verifiable.",
    "clarifying_questions": [],
    "criteria": [
        {
            "id": "human-review",
            "description": "The user approves the channel plan",
            "kind": "manual",
            "required": True
        }
    ]
}
text = "<GOAL_AGENT_JSON>\n" + json.dumps(proposal) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type": "tool_use", "part": {"type": "tool", "tool": "read", "state": {"status": "completed"}, "output": "x" * 50000}}))
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
    store.write_goal("Build a video channel")

    with TestClient(create_app(tmp_path)) as client:
        started = client.post(
            "/api/goals/default/proposal-jobs",
            json={"mode": "goal", "feedback": "Make it measurable", "conversation": ""},
        )
        assert started.status_code == 202
        job_id = started.json()["id"]

        job = started.json()
        for _ in range(600):
            job = client.get(f"/api/proposal-jobs/{job_id}").json()
            if job["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)

        assert job["status"] == "completed", job
        assert job["stage"] == "completed"
        assert job["result"]["refined_goal"] == "Create a tested video channel project"
        assert job["result"]["criteria"][0]["id"] == "human-review"


def test_proposal_job_surfaces_invalid_opencode_output(tmp_path: Path) -> None:
    import sys
    import time

    fake = tmp_path / "invalid_proposal_opencode.py"
    fake.write_text(
        r'''
import json
import sys

args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)
_ = sys.stdin.read()
print(json.dumps({"type": "text", "part": {"type": "text", "text": "not structured json"}}))
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

    with TestClient(create_app(tmp_path)) as client:
        started = client.post(
            "/api/goals/default/proposal-jobs",
            json={"mode": "goal", "feedback": "", "conversation": ""},
        )
        job_id = started.json()["id"]
        job = started.json()
        for _ in range(600):
            job = client.get(f"/api/proposal-jobs/{job_id}").json()
            if job["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)

        assert job["status"] == "failed"
        assert "did not return valid SetupProposal JSON" in job["error"]


def test_gui_defaults_new_criteria_to_ai_review() -> None:
    asset = Path(__file__).parents[1] / "src" / "goal_agent" / "web_assets" / "app.js"
    text = asset.read_text(encoding="utf-8")
    assert 'kind: "ai_judge"' in text
    assert "Convert to AI review" in text
    assert "AI evaluation analysis" in text


def test_criteria_editor_is_contained_and_responsive() -> None:
    assets = Path(__file__).parents[1] / "src" / "goal_agent" / "web_assets"
    script = (assets / "app.js").read_text(encoding="utf-8")
    styles = (assets / "style.css").read_text(encoding="utf-8")

    assert 'class="definition-layout"' in script
    assert 'class="criterion-definition-grid"' in script
    assert 'class="criterion-options-grid"' in script
    assert 'class="criterion-editor-section criterion-evaluation-section"' in script
    assert ".criterion-editor { width: 100%; max-width: 100%; min-width: 0; overflow: hidden;" in styles
    assert "label > input, label > textarea, label > select { width: 100%; min-width: 0; }" in styles
    assert ".form-grid.two-column, .goal-definition-grid, .criterion-definition-grid, .criterion-options-grid, .criterion-specific { grid-template-columns: 1fr; }" in styles


def test_live_polling_preserves_focused_steering_input() -> None:
    assets = Path(__file__).parents[1] / "src" / "goal_agent" / "web_assets"
    script = (assets / "app.js").read_text(encoding="utf-8")
    html = (assets / "index.html").read_text(encoding="utf-8")

    assert "function captureSteeringDraft()" in script
    assert "function isUserEditingMainContent()" in script
    assert "state.pendingDetailRender = true" in script
    assert "state.steeringDrafts[state.selectedId] = event.target.value" in script
    assert '${esc(steeringDraft())}</textarea>' in script
    assert 'Date.now() < state.interactionHoldUntil' in script
    assert 'const detail = await api(`/api/goals/${encodeURIComponent(state.selectedId)}`);\n      applyPolledDetail(detail);' in script
    assert "/app.js?v=0.7.1" in html
