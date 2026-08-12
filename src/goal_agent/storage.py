from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from filelock import FileLock

from .models import (
    AppConfig,
    ControlState,
    CriteriaDocument,
    EventRecord,
    GoalMetadata,
    GoalSummary,
    RefinementSession,
    RunPhase,
    RunState,
    utc_now,
)


AGENT_DIR_NAME = ".goal-agent"
DEFAULT_GOAL_ID = "default"
_GOAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ATOMIC_REPLACE_ATTEMPTS = 8
_ATOMIC_REPLACE_RETRY_SECONDS = 0.05


class ProjectStore:
    """Persistent file interface for one goal inside a project workspace.

    Project-wide configuration lives in ``.goal-agent/config.yaml``. Every goal has
    an independent control/status/runs tree under ``.goal-agent/goals/<goal-id>``.
    Creating another ProjectStore with the same project directory and a different
    goal ID is enough to operate the second goal concurrently.
    """

    def __init__(self, project_dir: Path | str, goal_id: str = DEFAULT_GOAL_ID):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.goal_id = self.validate_goal_id(goal_id)
        self.root = self.project_dir / AGENT_DIR_NAME
        self.goals_dir = self.root / "goals"
        self.goal_root = self.goals_dir / self.goal_id
        self.control_dir = self.goal_root / "control"
        self.status_dir = self.goal_root / "status"
        self.runs_dir = self.goal_root / "runs"
        self.lock_path = self.goal_root / ".state.lock"
        self.loop_lock_path = self.goal_root / ".loop.lock"

        self.config_path = self.root / "config.yaml"
        self.metadata_path = self.goal_root / "goal.yaml"
        self.goal_path = self.control_dir / "goal.md"
        self.criteria_path = self.control_dir / "criteria.yaml"
        self.control_path = self.control_dir / "control.yaml"
        self.steering_path = self.control_dir / "steering.md"
        self.refinement_path = self.control_dir / "refinement.json"

        self.state_path = self.status_dir / "state.json"
        self.status_markdown_path = self.status_dir / "STATUS.md"
        self.agents_path = self.status_dir / "agents.json"
        self.criteria_status_path = self.status_dir / "criteria.json"
        self.evaluation_analysis_path = self.status_dir / "evaluation-analysis.json"
        self.hypotheses_path = self.status_dir / "hypotheses.json"
        self.events_path = self.status_dir / "events.jsonl"

    @staticmethod
    def validate_goal_id(goal_id: str) -> str:
        candidate = goal_id.strip()
        if not _GOAL_ID_RE.fullmatch(candidate):
            raise ValueError(
                "goal_id must start with a letter or number and contain only letters, "
                "numbers, dots, underscores, or hyphens"
            )
        return candidate

    @classmethod
    def discover(
        cls, path: Path | str | None = None, goal_id: str = DEFAULT_GOAL_ID
    ) -> "ProjectStore":
        current = Path(path or Path.cwd()).expanduser().resolve()
        for candidate in [current, *current.parents]:
            if (candidate / AGENT_DIR_NAME).exists():
                return cls(candidate, goal_id=goal_id)
        return cls(current, goal_id=goal_id)

    def for_goal(self, goal_id: str) -> "ProjectStore":
        return ProjectStore(self.project_dir, goal_id=goal_id)

    def project_exists(self) -> bool:
        self._migrate_legacy_layout_if_needed()
        return self.config_path.exists()

    def exists(self) -> bool:
        self._migrate_legacy_layout_if_needed()
        return self.config_path.exists() and self.metadata_path.exists()

    def initialize(self, model: str | None = None, force: bool = False) -> None:
        if self.root.exists() and not force:
            raise FileExistsError(f"{self.root} already exists; use --force to replace it")
        if self.root.exists() and force:
            shutil.rmtree(self.root)

        self.root.mkdir(parents=True, exist_ok=True)
        self.goals_dir.mkdir(parents=True, exist_ok=True)
        config = AppConfig(project_dir=str(self.project_dir), model=model)
        self.write_yaml(self.config_path, config.model_dump(mode="json"))
        self.create_goal(
            goal_id=self.goal_id,
            title="Default Goal" if self.goal_id == DEFAULT_GOAL_ID else self.goal_id,
        )

    def create_goal(
        self,
        goal_id: str,
        *,
        title: str | None = None,
        goal: str | None = None,
        description: str = "",
    ) -> "ProjectStore":
        self.require_project_initialized()
        goal_store = self.for_goal(goal_id)
        if goal_store.goal_root.exists():
            raise FileExistsError(f"Goal '{goal_id}' already exists")

        goal_store.control_dir.mkdir(parents=True, exist_ok=False)
        goal_store.status_dir.mkdir(parents=True, exist_ok=True)
        goal_store.runs_dir.mkdir(parents=True, exist_ok=True)
        metadata = GoalMetadata(
            id=goal_store.goal_id,
            title=(title or goal_store.goal_id).strip(),
            description=description.strip(),
        )
        control = ControlState()
        criteria = CriteriaDocument()
        state = RunState(run_id=str(uuid.uuid4()), phase=RunPhase.IDLE)

        self.write_yaml(goal_store.metadata_path, metadata.model_dump(mode="json"))
        goal_store.atomic_write_text(
            goal_store.goal_path,
            "# Goal\n\n"
            + ((goal or "Describe the outcome this goal should achieve.").strip())
            + "\n",
        )
        self.write_yaml(goal_store.criteria_path, criteria.model_dump(mode="json"))
        self.write_yaml(goal_store.control_path, control.model_dump(mode="json"))
        goal_store.atomic_write_text(
            goal_store.steering_path,
            "# Steering Notes\n\n"
            "Add guidance here while this goal is running. The latest contents are read "
            "before every agent step.\n",
        )
        goal_store.save_state(state)
        goal_store.append_event(
            EventRecord(
                type="initialized",
                message=f"Goal '{goal_store.goal_id}' initialized",
                data={"goal_id": goal_store.goal_id},
            )
        )
        return goal_store

    def delete_goal(self, goal_id: str, *, force: bool = False) -> None:
        goal_store = self.for_goal(goal_id)
        goal_store.require_initialized()
        control = goal_store.read_control()
        if not force and control.desired_state.value == "running":
            raise RuntimeError("Stop the goal before deleting it")
        shutil.rmtree(goal_store.goal_root)

    def list_goal_ids(self, include_archived: bool = False) -> list[str]:
        self.require_project_initialized()
        values: list[str] = []
        if not self.goals_dir.exists():
            return values
        for path in sorted(self.goals_dir.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_dir() or not (path / "goal.yaml").exists():
                continue
            try:
                metadata = GoalMetadata.model_validate(self.read_yaml(path / "goal.yaml"))
            except Exception:
                continue
            if include_archived or not metadata.archived:
                values.append(metadata.id)
        return values

    def list_goal_summaries(
        self, *, active_goal_ids: set[str] | None = None, include_archived: bool = False
    ) -> list[GoalSummary]:
        active_goal_ids = active_goal_ids or set()
        summaries: list[GoalSummary] = []
        for goal_id in self.list_goal_ids(include_archived=include_archived):
            store = self.for_goal(goal_id)
            try:
                metadata = store.read_metadata()
                state = store.load_state()
                control = store.read_control()
                criteria = store.read_criteria()
                config = store.read_config()
            except Exception:
                continue
            required = [criterion for criterion in criteria.criteria if criterion.required]
            required_passed = sum(
                1
                for criterion in required
                if state.criteria_results.get(criterion.id)
                and state.criteria_results[criterion.id].passed
            )
            summaries.append(
                GoalSummary(
                    metadata=metadata,
                    phase=state.phase,
                    desired_state=control.desired_state,
                    iteration=state.iteration,
                    message=state.message,
                    model=control.model_override or config.model,
                    required_passed=required_passed,
                    required_total=len(required),
                    active=goal_id in active_goal_ids,
                    updated_at=state.updated_at,
                )
            )
        summaries.sort(
            key=lambda item: (item.phase.value == "achieved", item.updated_at), reverse=True
        )
        return summaries

    def require_project_initialized(self) -> None:
        self._migrate_legacy_layout_if_needed()
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"No {AGENT_DIR_NAME} workspace found at {self.project_dir}. "
                "Run 'goal-agent init'."
            )

    def require_initialized(self) -> None:
        self.require_project_initialized()
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Goal '{self.goal_id}' does not exist in {self.project_dir}. "
                "Create it in the GUI or with 'goal-agent goal-create'."
            )

    def read_config(self) -> AppConfig:
        self.require_project_initialized()
        return AppConfig.model_validate(self.read_yaml(self.config_path))

    def write_config(self, config: AppConfig) -> None:
        self.require_project_initialized()
        self.write_yaml(self.config_path, config.model_dump(mode="json"))

    def read_metadata(self) -> GoalMetadata:
        self.require_initialized()
        return GoalMetadata.model_validate(self.read_yaml(self.metadata_path))

    def write_metadata(self, metadata: GoalMetadata) -> None:
        metadata.updated_at = utc_now()
        self.write_yaml(self.metadata_path, metadata.model_dump(mode="json"))

    def read_goal(self) -> str:
        self.require_initialized()
        text = self.goal_path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if lines and lines[0].strip().lower() == "# goal":
            lines = lines[1:]
        return "\n".join(lines).strip()

    def write_goal(self, goal: str) -> None:
        self.atomic_write_text(self.goal_path, f"# Goal\n\n{goal.strip()}\n")
        if self.metadata_path.exists():
            metadata = self.read_metadata()
            self.write_metadata(metadata)

    def read_criteria(self) -> CriteriaDocument:
        self.require_initialized()
        return CriteriaDocument.model_validate(self.read_yaml(self.criteria_path))

    def write_criteria(self, criteria: CriteriaDocument) -> None:
        self.write_yaml(self.criteria_path, criteria.model_dump(mode="json"))

    def read_refinement_session(self) -> RefinementSession:
        self.require_initialized()
        if not self.refinement_path.exists():
            return RefinementSession()
        return RefinementSession.model_validate_json(
            self.refinement_path.read_text(encoding="utf-8")
        )

    def write_refinement_session(self, session: RefinementSession) -> None:
        self.require_initialized()
        session.updated_at = utc_now()
        with FileLock(str(self.lock_path)):
            self.atomic_write_text(
                self.refinement_path, session.model_dump_json(indent=2) + "\n"
            )

    def reset_refinement_session(self) -> RefinementSession:
        session = RefinementSession()
        self.write_refinement_session(session)
        return session

    def read_control(self) -> ControlState:
        self.require_initialized()
        return ControlState.model_validate(self.read_yaml(self.control_path))

    def update_control(self, **updates: Any) -> ControlState:
        self.require_initialized()
        with FileLock(str(self.lock_path)):
            control = ControlState.model_validate(self.read_yaml(self.control_path))
            data = control.model_dump()
            data.update(updates)
            data["revision"] = control.revision + 1
            data["updated_at"] = utc_now()
            updated = ControlState.model_validate(data)
            self.write_yaml(self.control_path, updated.model_dump(mode="json"))
            return updated

    def read_steering(self) -> str:
        self.require_initialized()
        return self.steering_path.read_text(encoding="utf-8").strip()

    def replace_steering(self, text: str) -> None:
        self.atomic_write_text(self.steering_path, text.rstrip() + "\n")

    def append_steering(self, message: str) -> None:
        self.require_initialized()
        timestamp = utc_now().isoformat()
        with self.steering_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n## {timestamp}\n\n{message.strip()}\n")

    def load_state(self) -> RunState:
        self.require_initialized()
        if not self.state_path.exists():
            return RunState(run_id=str(uuid.uuid4()))
        return RunState.model_validate_json(self.state_path.read_text(encoding="utf-8"))

    def save_state(self, state: RunState) -> None:
        state.updated_at = utc_now()
        payload = state.model_dump_json(indent=2)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            self.atomic_write_text(self.state_path, payload + "\n")
            self.atomic_write_text(
                self.agents_path,
                json.dumps(
                    {key: value.model_dump(mode="json") for key, value in state.agents.items()},
                    indent=2,
                )
                + "\n",
            )
            self.atomic_write_text(
                self.criteria_status_path,
                json.dumps(
                    {
                        key: value.model_dump(mode="json")
                        for key, value in state.criteria_results.items()
                    },
                    indent=2,
                )
                + "\n",
            )
            self.atomic_write_text(
                self.evaluation_analysis_path,
                json.dumps(
                    (
                        state.evaluation_analysis.model_dump(mode="json")
                        if state.evaluation_analysis
                        else None
                    ),
                    indent=2,
                )
                + "\n",
            )
            self.atomic_write_text(
                self.hypotheses_path,
                json.dumps(
                    [item.model_dump(mode="json") for item in state.hypotheses],
                    indent=2,
                )
                + "\n",
            )
            self.atomic_write_text(self.status_markdown_path, self.render_status_markdown(state))

    def append_event(self, event: EventRecord) -> None:
        self.status_dir.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def read_events(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.events_path.exists():
            return []
        lines = self.events_path.read_text(encoding="utf-8", errors="replace").splitlines()
        events: list[dict[str, Any]] = []
        for line in lines[-max(1, limit) :]:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        return events

    def iteration_dir(self, iteration: int) -> Path:
        path = self.runs_dir / f"iteration-{iteration:05d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_run_artifact(self, iteration: int, filename: str, content: str) -> Path:
        path = self.iteration_dir(iteration) / filename
        self.atomic_write_text(path, content)
        return path

    def render_status_markdown(self, state: RunState) -> str:
        goal = self.read_goal() if self.goal_path.exists() else ""
        metadata = self.read_metadata() if self.metadata_path.exists() else None
        criteria_error: str | None = None
        try:
            criteria = self.read_criteria().criteria if self.criteria_path.exists() else []
        except Exception as exc:
            criteria = []
            criteria_error = str(exc)
        lines = [
            f"# Goal Agent Status — {metadata.title if metadata else self.goal_id}",
            "",
            f"- **Goal ID:** `{self.goal_id}`",
            f"- **Phase:** `{state.phase.value}`",
            f"- **Iteration:** {state.iteration}",
            f"- **Message:** {state.message}",
            f"- **Updated:** {state.updated_at.isoformat()}",
            f"- **Active hypothesis:** `{state.active_hypothesis_id or 'none'}`",
            "",
            "## Goal",
            "",
            goal or "_No goal defined._",
            "",
            "## Agents",
            "",
            "| Agent | State | Current work | Detail |",
            "|---|---|---|---|",
        ]
        for name, agent in state.agents.items():
            detail = agent.detail.replace("|", "\\|").replace("\n", " ")[:300]
            task = agent.task.replace("|", "\\|").replace("\n", " ")[:200]
            lines.append(f"| {name} | `{agent.phase.value}` | {task} | {detail} |")

        lines.extend(["", "## Criteria", ""])
        if criteria_error:
            lines.extend([f"**criteria.yaml is temporarily invalid:** `{criteria_error}`", ""])
        lines.extend(
            [
                "| ID | Required | Result | Description | Evidence |",
                "|---|---:|---|---|---|",
            ]
        )
        for criterion in criteria:
            result = state.criteria_results.get(criterion.id)
            status = result.status if result else "unchecked"
            summary = result.summary if result else "Not checked"
            evidence = "; ".join(result.evidence[:2]) if result else ""
            description = criterion.description.replace("|", "\\|")
            result_detail = summary + (" — " + evidence if evidence else "")
            result_detail = result_detail.replace("|", "\\|")[:500]
            lines.append(
                f"| `{criterion.id}` | {'yes' if criterion.required else 'no'} | "
                f"**{status}** | {description} | {result_detail} |"
            )

        lines.extend(["", "## AI Evaluation Analysis", ""])
        if state.evaluation_analysis is None:
            lines.append("_No diagnostic analysis has been produced yet._")
        else:
            analysis = state.evaluation_analysis
            lines.extend(
                [
                    f"- **Evaluation:** {analysis.label}",
                    f"- **Source:** `{analysis.source}`",
                    f"- **Summary:** {analysis.summary}",
                    f"- **Progress:** {analysis.progress_assessment or 'Not specified'}",
                    "",
                ]
            )
            if analysis.recommended_next_focus:
                lines.append("**Recommended next focus:**")
                lines.extend(f"- {item}" for item in analysis.recommended_next_focus)
                lines.append("")
            for item in analysis.criterion_analyses:
                lines.extend(
                    [
                        f"### `{item.criterion_id}` — {item.observed_status}",
                        "",
                        item.interpretation,
                        "",
                    ]
                )
                if item.likely_causes:
                    lines.append("Likely causes:")
                    lines.extend(f"- {cause}" for cause in item.likely_causes)
                    lines.append("")
                if item.recommended_actions:
                    lines.append("Recommended actions:")
                    lines.extend(f"- {action}" for action in item.recommended_actions)
                    lines.append("")

        lines.extend(["", "## Criteria Revision Suggestions", ""])
        if not state.criteria_revision_suggestions:
            lines.append("_No review-required criteria revisions have been suggested._")
        else:
            lines.append("These are advisory only; review and apply them through criteria refinement.")
            lines.append("")
            for suggestion in state.criteria_revision_suggestions[-5:][::-1]:
                lines.extend(
                    [
                        f"### `{suggestion.criterion_id}` â€” approval required",
                        "",
                        suggestion.rationale,
                        "",
                    ]
                )
                if suggestion.safeguards:
                    lines.append("Safeguards:")
                    lines.extend(f"- {item}" for item in suggestion.safeguards)
                    lines.append("")
                if suggestion.proposed_criteria:
                    lines.append("Proposed replacement criteria:")
                    for item in suggestion.proposed_criteria:
                        lines.append(f"- `{item.id}`: {item.description}")
                    lines.append("")

        lines.extend(["", "## Recent Hypotheses", ""])
        if not state.hypotheses:
            lines.append("_No hypotheses have been proposed yet._")
        else:
            for hypothesis in state.hypotheses[-8:][::-1]:
                lines.extend(
                    [
                        f"### {hypothesis.id} — {hypothesis.status}",
                        "",
                        hypothesis.statement,
                        "",
                        f"**Outcome:** {hypothesis.outcome or 'Pending'}",
                        "",
                    ]
                )
        lines.extend(
            [
                "## Live Controls",
                "",
                "Edit these files while the process is running:",
                "",
                "- `../control/goal.md` — change this goal",
                "- `../control/criteria.yaml` — add, remove, or refine success criteria",
                "- `../control/steering.md` — add guidance for the next agent step",
                "- `../control/control.yaml` — set `desired_state` to `running`, `paused`, or `stopped`",
                f"- `{self.config_path}` — project-wide model and runtime settings",
                "",
            ]
        )
        return "\n".join(lines)

    def _migrate_legacy_layout_if_needed(self) -> None:
        legacy_control = self.root / "control"
        legacy_config = legacy_control / "config.yaml"
        if self.config_path.exists() or not legacy_config.exists():
            return

        migration_lock = FileLock(str(self.root / ".migration.lock"))
        with migration_lock:
            if self.config_path.exists() or not legacy_config.exists():
                return
            default_root = self.goals_dir / DEFAULT_GOAL_ID
            default_control = default_root / "control"
            default_status = default_root / "status"
            default_runs = default_root / "runs"
            self.goals_dir.mkdir(parents=True, exist_ok=True)
            default_control.mkdir(parents=True, exist_ok=True)

            shutil.move(str(legacy_config), str(self.config_path))
            for name in ("goal.md", "criteria.yaml", "control.yaml", "steering.md"):
                source = legacy_control / name
                target = default_control / name
                if source.exists() and not target.exists():
                    shutil.move(str(source), str(target))

            legacy_status = self.root / "status"
            if legacy_status.exists() and not default_status.exists():
                shutil.move(str(legacy_status), str(default_status))
            else:
                default_status.mkdir(parents=True, exist_ok=True)
            legacy_runs = self.root / "runs"
            if legacy_runs.exists() and not default_runs.exists():
                shutil.move(str(legacy_runs), str(default_runs))
            else:
                default_runs.mkdir(parents=True, exist_ok=True)

            metadata_path = default_root / "goal.yaml"
            if not metadata_path.exists():
                metadata = GoalMetadata(id=DEFAULT_GOAL_ID, title="Default Goal")
                self.write_yaml(metadata_path, metadata.model_dump(mode="json"))
            try:
                legacy_control.rmdir()
            except OSError:
                pass

    @staticmethod
    def read_yaml(path: Path) -> dict[str, Any]:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data or {}

    @staticmethod
    def write_yaml(path: Path, data: dict[str, Any]) -> None:
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        ProjectStore.atomic_write_text(path, text)

    @staticmethod
    def atomic_write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            # Windows can transiently reject a rename when an antivirus scan,
            # browser file watcher, or another short-lived reader still holds
            # the destination. The state lock serializes Goal Agent writers,
            # but cannot control those external handles.
            for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
                try:
                    os.replace(temp_name, path)
                    break
                except PermissionError:
                    if attempt == _ATOMIC_REPLACE_ATTEMPTS - 1:
                        raise
                    time.sleep(_ATOMIC_REPLACE_RETRY_SECONDS * (attempt + 1))
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
