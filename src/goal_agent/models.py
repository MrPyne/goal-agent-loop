from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DesiredState(str, Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class RunPhase(str, Enum):
    IDLE = "idle"
    SETUP = "setup"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ACHIEVED = "achieved"
    ERROR = "error"


class AgentPhase(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    BLOCKED = "blocked"
    COMPLETE = "complete"
    ERROR = "error"


class CriterionKind(str, Enum):
    COMMAND = "command"
    FILE_EXISTS = "file_exists"
    FILE_CONTAINS = "file_contains"
    AI_JUDGE = "ai_judge"
    MANUAL = "manual"


class OverrideMode(str, Enum):
    AUTO = "auto"
    PASS = "pass"
    FAIL = "fail"


class AppConfig(BaseModel):
    project_dir: str
    opencode_command: list[str] = Field(default_factory=lambda: ["opencode"])
    model: str | None = None
    attach_url: str | None = None
    attach_username: str | None = None
    attach_password_env: str | None = None
    strategist_agent: str = "plan"
    executor_agent: str = "build"
    evaluator_agent: str = "plan"
    auto_approve: bool = True
    poll_interval_seconds: float = 0.5
    iteration_delay_seconds: float = 2.0
    opencode_timeout_seconds: int = 1800
    criterion_timeout_seconds: int = 300
    model_context_tokens: int | None = Field(default=None, ge=4096, le=2_000_000)
    max_iterations: int | None = None
    max_recent_hypotheses: int = 12
    no_progress_rethink_after: int = 3
    criterion_serial_mode: bool = Field(
        default=False,
        description=(
            "When True, fix criteria one at a time in declaration order rather than proposing "
            "free-form hypotheses. Each failing required criterion is targeted individually; "
            "once all pass, a final combined check is run. If that check reveals regressions "
            "the serial loop restarts."
        ),
    )
    status_refresh_seconds: float = 0.75
    max_concurrent_goals: int = Field(default=2, ge=1, le=32)
    gui_auto_resume_running_goals: bool = True
    gui_host: str = "127.0.0.1"
    gui_port: int = Field(default=8765, ge=1, le=65535)

    @property
    def project_path(self) -> Path:
        return Path(self.project_dir).expanduser().resolve()


class GoalMetadata(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    title: str
    description: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    archived: bool = False


class ControlState(BaseModel):
    desired_state: DesiredState = DesiredState.PAUSED
    revision: int = 1
    model_override: str | None = None
    note: str = ""
    updated_at: datetime = Field(default_factory=utc_now)


class CriterionDefinition(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    description: str
    kind: CriterionKind
    required: bool = True
    override: OverrideMode = OverrideMode.AUTO

    command: str | None = None
    expected_exit_code: int = 0
    timeout_seconds: int | None = None
    stdout_contains: str | None = None
    stderr_contains: str | None = None
    stdout_regex: str | None = None
    stderr_regex: str | None = None
    output_case_sensitive: bool = True
    output_judge_prompt: str | None = None
    output_confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    path: str | None = None
    contains: str | None = None
    regex: bool = False
    case_sensitive: bool = True

    judge_prompt: str | None = None
    evidence_paths: list[str] = Field(default_factory=list)
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_kind_fields(self) -> "CriterionDefinition":
        if self.kind == CriterionKind.COMMAND and not self.command:
            raise ValueError("command criteria require 'command'")
        if self.kind in {CriterionKind.FILE_EXISTS, CriterionKind.FILE_CONTAINS} and not self.path:
            raise ValueError(f"{self.kind.value} criteria require 'path'")
        if self.kind == CriterionKind.FILE_CONTAINS and self.contains is None:
            raise ValueError("file_contains criteria require 'contains'")
        if self.kind == CriterionKind.AI_JUDGE and not self.judge_prompt:
            raise ValueError("ai_judge criteria require 'judge_prompt'")
        return self


class CriteriaDocument(BaseModel):
    revision: int = 1
    criteria: list[CriterionDefinition] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self) -> "CriteriaDocument":
        ids = [item.id for item in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion IDs must be unique")
        return self


class CriterionResult(BaseModel):
    criterion_id: str
    passed: bool = False
    status: Literal["pass", "fail", "error", "unchecked"] = "unchecked"
    summary: str = "Not checked"
    evidence: list[str] = Field(default_factory=list)
    checked_at: datetime | None = None
    duration_seconds: float | None = None
    error: str | None = None
    confidence: float | None = None
    evaluation_method: Literal[
        "command",
        "file_exists",
        "file_contains",
        "ai_judge",
        "human_override",
        "human_required",
    ] | None = None


class CriterionAnalysis(BaseModel):
    criterion_id: str
    observed_status: Literal["pass", "fail", "error", "unchecked"]
    interpretation: str
    likely_causes: list[str] = Field(default_factory=list)
    useful_evidence: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class EvaluationAnalysis(BaseModel):
    iteration: int = 0
    label: str = ""
    summary: str
    progress_assessment: str = ""
    material_progress: bool = False
    progress_evidence: list[str] = Field(default_factory=list)
    criterion_analyses: list[CriterionAnalysis] = Field(default_factory=list)
    cross_criterion_findings: list[str] = Field(default_factory=list)
    recommended_next_focus: list[str] = Field(default_factory=list)
    analyzed_at: datetime = Field(default_factory=utc_now)
    source: Literal["ai", "fallback"] = "ai"


class AgentStatus(BaseModel):
    name: str
    phase: AgentPhase = AgentPhase.IDLE
    task: str = "Idle"
    detail: str = ""
    started_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class Hypothesis(BaseModel):
    id: str
    iteration: int
    statement: str
    rationale: str
    expected_impact: str
    target_criteria: list[str] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    status: Literal[
        "proposed", "active", "supported", "refuted", "inconclusive", "abandoned"
    ] = "proposed"
    evidence: list[str] = Field(default_factory=list)
    outcome: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RunState(BaseModel):
    run_id: str
    phase: RunPhase = RunPhase.IDLE
    iteration: int = 0
    started_at: datetime | None = None
    ended_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    message: str = "Initialized"
    active_hypothesis_id: str | None = None
    criteria_results: dict[str, CriterionResult] = Field(default_factory=dict)
    evaluation_analysis: EvaluationAnalysis | None = None
    agents: dict[str, AgentStatus] = Field(
        default_factory=lambda: {
            "strategist": AgentStatus(name="strategist"),
            "executor": AgentStatus(name="executor"),
            "evaluator": AgentStatus(name="evaluator"),
        }
    )
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    consecutive_no_progress: int = 0
    last_error: str | None = None
    # Serial criterion mode: tracks the criterion currently being targeted.
    # None  = find next failing required criterion.
    # "__final_check__" = all passed individually; run full combined check.
    serial_target_criterion: str | None = None


class GoalSummary(BaseModel):
    metadata: GoalMetadata
    phase: RunPhase
    desired_state: DesiredState
    iteration: int
    message: str
    model: str | None = None
    required_passed: int = 0
    required_total: int = 0
    active: bool = False
    updated_at: datetime


class CriteriaQualityIssue(BaseModel):
    criterion_id: str | None = None
    issue: str
    suggested_fix: str = ""
    severity: Literal["warning", "blocking"] = "blocking"


class SetupProposal(BaseModel):
    refined_goal: str
    goal_rationale: str = ""
    assistant_message: str = ""
    clarifying_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    criteria: list[CriterionDefinition] = Field(default_factory=list)
    criteria_quality_issues: list[CriteriaQualityIssue] = Field(default_factory=list)
    ready_to_finalize: bool = False
    readiness_reason: str = ""


class RefinementMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    created_at: datetime = Field(default_factory=utc_now)


class RefinementSession(BaseModel):
    revision: int = 1
    status: Literal["not_started", "refining", "ready", "finalized"] = "not_started"
    messages: list[RefinementMessage] = Field(default_factory=list)
    current_proposal: SetupProposal | None = None
    # The full message list remains available to the UI.  These fields hold a
    # bounded representation used for model prompts so long refinement chats do
    # not consume the provider's entire context window.
    conversation_summary: str = ""
    compacted_message_count: int = Field(default=0, ge=0)
    last_prompt_chars: int = Field(default=0, ge=0)
    last_estimated_input_tokens: int = Field(default=0, ge=0)
    last_context_mode: Literal["bounded", "compact_retry"] = "bounded"
    context_overflow_retries: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)
    finalized_at: datetime | None = None


class StrategyDecision(BaseModel):
    hypothesis: str
    rationale: str
    expected_impact: str
    target_criteria: list[str] = Field(default_factory=list)
    plan: list[str] = Field(default_factory=list)
    avoid_repeating: list[str] = Field(default_factory=list)


class ExecutionReport(BaseModel):
    summary: str
    actions: list[str] = Field(default_factory=list)
    files_changed: list[str] = Field(default_factory=list)
    commands_run: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)


class JudgeDecision(BaseModel):
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str
    evidence: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)


class EventRecord(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    type: str
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
