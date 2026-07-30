import asyncio
import sys
from pathlib import Path

from goal_agent.criteria import CriteriaEvaluator, all_required_pass
from goal_agent.models import (
    AppConfig,
    CriterionDefinition,
    CriterionKind,
    CriterionResult,
    JudgeDecision,
)


class DummyRunner:
    pass


def test_file_criteria(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\n\n## Usage\n", encoding="utf-8")
    config = AppConfig(project_dir=str(tmp_path), poll_interval_seconds=0.01)
    evaluator = CriteriaEvaluator(config, DummyRunner())  # type: ignore[arg-type]

    exists = CriterionDefinition(
        id="readme",
        description="README exists",
        kind=CriterionKind.FILE_EXISTS,
        path="README.md",
    )
    contains = CriterionDefinition(
        id="usage",
        description="Usage documented",
        kind=CriterionKind.FILE_CONTAINS,
        path="README.md",
        contains="## Usage",
    )

    exists_result = asyncio.run(
        evaluator.evaluate_one(exists, goal="g", steering="", model=None)
    )
    contains_result = asyncio.run(
        evaluator.evaluate_one(contains, goal="g", steering="", model=None)
    )
    assert exists_result.passed
    assert contains_result.passed


def test_command_criterion(tmp_path: Path) -> None:
    config = AppConfig(project_dir=str(tmp_path), poll_interval_seconds=0.01)
    evaluator = CriteriaEvaluator(config, DummyRunner())  # type: ignore[arg-type]
    criterion = CriterionDefinition(
        id="command",
        description="Command exits zero",
        kind=CriterionKind.COMMAND,
        command=f'"{sys.executable}" -c "print(123)"',
    )
    result = asyncio.run(
        evaluator.evaluate_one(criterion, goal="g", steering="", model=None)
    )
    assert result.passed
    assert any("123" in evidence for evidence in result.evidence)


def test_all_required_pass_requires_at_least_one_required() -> None:
    optional = CriterionDefinition(
        id="optional",
        description="Optional",
        kind=CriterionKind.MANUAL,
        required=False,
    )
    assert not all_required_pass([optional], {})

    required = CriterionDefinition(
        id="required",
        description="Required",
        kind=CriterionKind.MANUAL,
    )
    results = {
        "required": CriterionResult(
            criterion_id="required", passed=True, status="pass", summary="ok"
        )
    }
    assert all_required_pass([required], results)


class JudgeRunner:
    async def run_structured(self, *args, **kwargs):
        return (
            JudgeDecision(
                passed=True,
                confidence=0.93,
                summary="The documented behavior is present and supported by project evidence.",
                evidence=["README.md documents the completed behavior"],
                missing=[],
            ),
            None,
        )


def test_ai_judge_can_pass_qualitative_criterion_each_evaluation(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("The interface is clear and complete.\n", encoding="utf-8")
    config = AppConfig(project_dir=str(tmp_path), poll_interval_seconds=0.01)
    evaluator = CriteriaEvaluator(config, JudgeRunner())  # type: ignore[arg-type]
    criterion = CriterionDefinition(
        id="quality",
        description="The interface is clear and complete",
        kind=CriterionKind.AI_JUDGE,
        judge_prompt="Inspect the interface artifacts and decide whether they are clear and complete.",
        evidence_paths=["README.md"],
        confidence_threshold=0.8,
    )

    result = asyncio.run(
        evaluator.evaluate_one(criterion, goal="Ship a usable interface", steering="", model=None)
    )

    assert result.passed
    assert result.status == "pass"
    assert result.evaluation_method == "ai_judge"
    assert result.confidence == 0.93


def test_manual_criterion_is_explicitly_human_only(tmp_path: Path) -> None:
    config = AppConfig(project_dir=str(tmp_path), poll_interval_seconds=0.01)
    evaluator = CriteriaEvaluator(config, DummyRunner())  # type: ignore[arg-type]
    criterion = CriterionDefinition(
        id="approval",
        description="A person approves the final appearance",
        kind=CriterionKind.MANUAL,
    )

    result = asyncio.run(
        evaluator.evaluate_one(criterion, goal="g", steering="", model=None)
    )

    assert not result.passed
    assert result.evaluation_method == "human_required"
    assert "Human approval is required" in result.summary
