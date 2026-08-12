import asyncio
import hashlib
import json
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


def test_command_criterion_supports_stdout_and_regex_checks(tmp_path: Path) -> None:
    config = AppConfig(project_dir=str(tmp_path), poll_interval_seconds=0.01)
    evaluator = CriteriaEvaluator(config, DummyRunner())  # type: ignore[arg-type]
    criterion = CriterionDefinition(
        id="command-output",
        description="Command output includes required markers",
        kind=CriterionKind.COMMAND,
        command=f'"{sys.executable}" -c "print(\'STATUS: PASS\')"',
        stdout_contains="STATUS:",
        stdout_regex=r"PASS$",
    )

    result = asyncio.run(
        evaluator.evaluate_one(criterion, goal="g", steering="", model=None)
    )

    assert result.passed
    assert any("stdout matched" in evidence for evidence in result.evidence)


def test_named_external_benchmark_requires_official_provenance(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "livebench_results.json").write_text(
        json.dumps({"score_valid": True, "metrics": {"overall_score": 0.99}}),
        encoding="utf-8",
    )
    config = AppConfig(project_dir=str(tmp_path), poll_interval_seconds=0.01)
    evaluator = CriteriaEvaluator(config, DummyRunner())  # type: ignore[arg-type]
    criterion = CriterionDefinition(
        id="livebench_score",
        description="The model scores at least 70% on LiveBench.",
        kind=CriterionKind.COMMAND,
        command=f'"{sys.executable}" -c "print(123)"',
    )

    result = asyncio.run(evaluator.evaluate_one(criterion, goal="g", steering="", model=None))

    assert not result.passed
    assert "official-run provenance" in result.summary


def test_named_external_benchmark_accepts_hashed_official_artifact(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    scripts = tmp_path / "scripts"
    logs.mkdir()
    scripts.mkdir()
    raw = logs / "livebench_raw.jsonl"
    raw.write_text('{"sample":"official result"}\n', encoding="utf-8")
    (scripts / "run_livebench.py").write_text("# official runner adapter\n", encoding="utf-8")
    (logs / "livebench_results.json").write_text(
        json.dumps(
            {
                "score_valid": True,
                "metrics": {"overall_score": 0.99},
                "provenance": {
                    "official_source": "https://github.com/livebench/livebench",
                    "dataset_revision": "2025-02-01",
                    "raw_results_path": "logs/livebench_raw.jsonl",
                    "raw_results_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    config = AppConfig(project_dir=str(tmp_path), poll_interval_seconds=0.01)
    evaluator = CriteriaEvaluator(config, DummyRunner())  # type: ignore[arg-type]
    criterion = CriterionDefinition(
        id="livebench_score",
        description="The model scores at least 70% on LiveBench.",
        kind=CriterionKind.COMMAND,
        command=f'"{sys.executable}" scripts/run_livebench.py',
    )

    result = asyncio.run(evaluator.evaluate_one(criterion, goal="g", steering="", model=None))

    assert result.passed


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


class CapturingJudgeRunner:
    def __init__(self) -> None:
        self.prompt = ""
        self.kwargs = {}

    async def run_structured(self, prompt, *args, **kwargs):
        self.prompt = prompt
        self.kwargs = kwargs
        return (
            JudgeDecision(
                passed=True,
                confidence=0.9,
                summary="Evidence satisfies the rubric.",
                evidence=["report says PASS"],
                missing=[],
            ),
            None,
        )


class CommandJudgeRunner:
    def __init__(self) -> None:
        self.prompt = ""

    async def run_structured(self, prompt, *args, **kwargs):
        self.prompt = prompt
        return (
            JudgeDecision(
                passed=True,
                confidence=0.91,
                summary="Output rubric satisfied by concrete output evidence.",
                evidence=["stdout contains STATUS: PASS"],
                missing=[],
            ),
            None,
        )


def test_ai_judge_receives_bounded_evidence_without_tools(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text("PASS only if this exact evidence is present.\n" + "x" * 50000, encoding="utf-8")
    runner = CapturingJudgeRunner()
    config = AppConfig(project_dir=str(tmp_path), poll_interval_seconds=0.01)
    evaluator = CriteriaEvaluator(config, runner)  # type: ignore[arg-type]
    criterion = CriterionDefinition(
        id="report-quality",
        description="The report meets the rubric",
        kind=CriterionKind.AI_JUDGE,
        judge_prompt="PASS only if the report contains the required evidence. FAIL if it does not.",
        evidence_paths=["report.md"],
        confidence_threshold=0.8,
    )

    result = asyncio.run(
        evaluator.evaluate_one(criterion, goal="Produce the report", steering="", model=None)
    )

    assert result.passed
    assert "report.md" in runner.prompt
    assert "PASS only if this exact evidence is present" in runner.prompt
    assert "middle omitted by Goal Agent evidence budget" in runner.prompt
    assert runner.kwargs["profile"] == "judge"
    assert len(runner.prompt) < 60000


def test_command_can_use_ai_judge_on_output(tmp_path: Path) -> None:
    runner = CommandJudgeRunner()
    config = AppConfig(project_dir=str(tmp_path), poll_interval_seconds=0.01)
    evaluator = CriteriaEvaluator(config, runner)  # type: ignore[arg-type]
    criterion = CriterionDefinition(
        id="judge-output",
        description="The smoke check output confirms success criteria",
        kind=CriterionKind.COMMAND,
        command=f'"{sys.executable}" -c "print(\'STATUS: PASS\')"',
        output_judge_prompt=(
            "PASS only if stdout confirms STATUS: PASS and no failure marker appears. "
            "FAIL if the output is ambiguous or indicates failure."
        ),
        output_confidence_threshold=0.8,
    )

    result = asyncio.run(
        evaluator.evaluate_one(criterion, goal="Run smoke check", steering="", model=None)
    )

    assert result.passed
    assert result.confidence == 0.91
    assert "STATUS: PASS" in runner.prompt
