from pathlib import Path

from goal_agent.models import CriteriaDocument, CriterionDefinition, CriterionKind
from goal_agent.storage import ProjectStore


def test_initialize_and_persist(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    store.initialize(model="llama.cpp/test")

    assert store.exists()
    assert store.read_config().model == "llama.cpp/test"
    assert store.read_control().desired_state.value == "paused"

    store.write_goal("Make the tests pass")
    store.write_criteria(
        CriteriaDocument(
            criteria=[
                CriterionDefinition(
                    id="tests",
                    description="Tests pass",
                    kind=CriterionKind.COMMAND,
                    command="python -m pytest -q",
                )
            ]
        )
    )
    state = store.load_state()
    store.save_state(state)

    assert store.read_goal() == "Make the tests pass"
    assert store.status_markdown_path.exists()
    assert "Tests pass" in store.status_markdown_path.read_text(encoding="utf-8")


def test_control_update_is_persistent(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    store.initialize()
    original = store.read_control()
    updated = store.update_control(desired_state="running", note="test")

    assert updated.revision == original.revision + 1
    assert store.read_control().desired_state.value == "running"


def test_status_survives_temporarily_invalid_criteria(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    store.initialize()
    store.criteria_path.write_text("criteria: [", encoding="utf-8")
    state = store.load_state()
    store.save_state(state)
    status = store.status_markdown_path.read_text(encoding="utf-8")
    assert "temporarily invalid" in status
