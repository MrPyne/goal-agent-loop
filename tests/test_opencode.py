import asyncio
import sys
from pathlib import Path

from goal_agent.models import AppConfig, StrategyDecision
from goal_agent.opencode import OpenCodeRunner, extract_json_payload


def test_extract_marker_json() -> None:
    text = '''noise
<GOAL_AGENT_JSON>
{"hypothesis":"x","rationale":"y","expected_impact":"z","target_criteria":[],"plan":[],"avoid_repeating":[]}
</GOAL_AGENT_JSON>
'''
    payload = extract_json_payload(text)
    decision = StrategyDecision.model_validate(payload)
    assert decision.hypothesis == "x"


def test_extract_fenced_json() -> None:
    assert extract_json_payload('```json\n{"passed": true}\n```') == {"passed": True}


def _config(tmp_path: Path, script: Path, *, auto_approve: bool = True) -> AppConfig:
    return AppConfig(
        project_dir=str(tmp_path),
        opencode_command=[sys.executable, str(script)],
        auto_approve=auto_approve,
        poll_interval_seconds=0.01,
    )


def test_detects_current_dangerous_permission_flag(tmp_path: Path) -> None:
    fake = tmp_path / "current_opencode.py"
    fake.write_text(
        "import sys\n"
        "assert sys.argv[1:] == ['run', '--help']\n"
        "print('  --dangerously-skip-permissions  auto-approve permissions')\n",
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))

    flag = asyncio.run(runner.detect_auto_approve_flag())
    command = runner.build_command(
        model="fake/model",
        agent="plan",
        title="test",
        auto_approve_flag=flag,
    )

    assert flag == "--dangerously-skip-permissions"
    assert "--dangerously-skip-permissions" in command
    assert "--auto" not in command


def test_detects_legacy_auto_permission_flag(tmp_path: Path) -> None:
    fake = tmp_path / "legacy_opencode.py"
    fake.write_text(
        "import sys\n"
        "assert sys.argv[1:] == ['run', '--help']\n"
        "print('  --auto  auto-approve permissions')\n",
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))

    assert asyncio.run(runner.detect_auto_approve_flag()) == "--auto"


def test_run_uses_detected_permission_flag(tmp_path: Path) -> None:
    fake = tmp_path / "strict_opencode.py"
    fake.write_text(
        r'''
import json
import sys

args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)
if "--auto" in args or "--dangerously-skip-permissions" not in args:
    print("unsupported permission flag", file=sys.stderr)
    raise SystemExit(1)
_ = sys.stdin.read()
print(json.dumps({"type": "text", "part": {"type": "text", "text": "ok"}}))
''',
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))

    result = asyncio.run(runner.run("hello", model="fake/model", agent="plan"))

    assert result.exit_code == 0
    assert result.text == "ok"


def test_auto_approve_can_be_disabled_without_probe(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.py"
    runner = OpenCodeRunner(_config(tmp_path, missing, auto_approve=False))

    assert asyncio.run(runner.detect_auto_approve_flag()) is None


def test_run_handles_json_event_larger_than_asyncio_default_line_limit(tmp_path: Path) -> None:
    fake = tmp_path / "large_event_opencode.py"
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
    "type": "tool_use",
    "part": {
        "type": "tool",
        "tool": "read",
        "state": {"status": "completed"},
        "output": "x" * 200000,
    },
}))
print(json.dumps({"type": "text", "part": {"type": "text", "text": "ok"}}))
''',
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))

    result = asyncio.run(runner.run("hello", model="fake/model", agent="plan"))

    assert result.exit_code == 0
    assert result.text == "ok"
    assert len(result.events) == 2


def test_run_surfaces_context_overflow_with_token_counts(tmp_path: Path) -> None:
    import pytest

    from goal_agent.opencode import OpenCodeContextOverflowError

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
        "data": {
            "message": "request (67191 tokens) exceeds the available context size (65536 tokens), try increasing it",
            "responseBody": "{\"error\":{\"type\":\"exceed_context_size_error\",\"n_prompt_tokens\":67191,\"n_ctx\":65536}}"
        }
    }
}))
raise SystemExit(1)
''',
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))

    with pytest.raises(OpenCodeContextOverflowError) as caught:
        asyncio.run(runner.run("hello", model="fake/model", agent="plan"))

    assert caught.value.requested_tokens == 67191
    assert caught.value.context_size == 65536
    assert "67,191" in str(caught.value)


def test_run_structured_recovers_overflow_in_fresh_compact_session(tmp_path: Path) -> None:
    import json

    fake = tmp_path / "recovering_opencode.py"
    fake.write_text(
        r'''
import json
import os
import pathlib
import sys

args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)

root = pathlib.Path.cwd()
counter = root / "calls.txt"
count = int(counter.read_text() or "0") if counter.exists() else 0
count += 1
counter.write_text(str(count))
prompt = sys.stdin.read()
(root / f"prompt-{count}.txt").write_text(prompt, encoding="utf-8")
(root / f"env-{count}.txt").write_text(os.environ.get("OPENCODE_CONFIG_CONTENT", ""), encoding="utf-8")

if count == 1:
    print(json.dumps({
        "type": "error",
        "error": {
            "name": "ContextOverflowError",
            "data": {"message": "request (70793 tokens) exceeds the available context size (65536 tokens)"}
        }
    }))
    raise SystemExit(1)

payload = {
    "hypothesis": "Use the focused failing evidence.",
    "rationale": "The compact retry has enough context.",
    "expected_impact": "The target criterion should improve.",
    "target_criteria": ["c1"],
    "plan": ["Inspect one relevant file"],
    "avoid_repeating": []
}
text = "<GOAL_AGENT_JSON>\n" + json.dumps(payload) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type": "text", "part": {"type": "text", "text": text}}))
''',
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))
    statuses: list[tuple[str, str]] = []

    decision, _ = asyncio.run(
        runner.run_structured(
            "You are the strategist. Inspect the project as needed.",
            StrategyDecision,
            model="fake/model",
            agent="plan",
            status_callback=lambda kind, detail: statuses.append((kind, detail)),
        )
    )

    assert decision.target_criteria == ["c1"]
    assert (tmp_path / "calls.txt").read_text() == "2"
    retry_prompt = (tmp_path / "prompt-2.txt").read_text(encoding="utf-8")
    assert "CONTEXT-OVERFLOW RECOVERY MODE" in retry_prompt
    assert "Do not delegate to subagents" in retry_prompt
    inline = json.loads((tmp_path / "env-2.txt").read_text(encoding="utf-8"))
    assert inline["compaction"]["auto"] is True
    assert inline["compaction"]["prune"] is True
    assert inline["compaction"]["reserved"] >= 12000
    assert any(kind == "context_recovery" for kind, _ in statuses)


def test_run_structured_raises_after_context_recovery_exhausted(tmp_path: Path) -> None:
    import pytest

    from goal_agent.opencode import OpenCodeContextOverflowError

    fake = tmp_path / "always_overflow_opencode.py"
    fake.write_text(
        r'''
import json
import pathlib
import sys
args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)
count_file = pathlib.Path("count.txt")
count = int(count_file.read_text() or "0") if count_file.exists() else 0
count_file.write_text(str(count + 1))
_ = sys.stdin.read()
print(json.dumps({"type":"error","error":{"name":"ContextOverflowError","data":{"message":"request (70793 tokens) exceeds the available context size (65536 tokens)"}}}))
raise SystemExit(1)
''',
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))

    with pytest.raises(OpenCodeContextOverflowError):
        asyncio.run(
            runner.run_structured(
                "You are the strategist.",
                StrategyDecision,
                model="fake/model",
                agent="plan",
                context_overflow_retries=2,
            )
        )

    assert (tmp_path / "count.txt").read_text() == "3"


def test_analysis_profile_disables_tools_without_max_step_prefill(tmp_path: Path) -> None:
    import json

    fake = tmp_path / "profile_opencode.py"
    fake.write_text(
        r'''
import json
import os
import pathlib
import sys
args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)
pathlib.Path("captured-env.json").write_text(os.environ["OPENCODE_CONFIG_CONTENT"], encoding="utf-8")
_ = sys.stdin.read()
payload = {
    "hypothesis": "Use supplied evidence only",
    "rationale": "No project tools are needed",
    "expected_impact": "Avoid context growth",
    "target_criteria": ["c1"],
    "plan": ["Act on the evidence"],
    "avoid_repeating": []
}
text = "<GOAL_AGENT_JSON>\n" + json.dumps(payload) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type":"text","part":{"type":"text","text":text}}))
''',
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))

    asyncio.run(
        runner.run_structured(
            "Choose a hypothesis from supplied evidence.",
            StrategyDecision,
            model="fake/model",
            agent="plan",
            profile="analysis",
        )
    )

    inline = json.loads((tmp_path / "captured-env.json").read_text(encoding="utf-8"))
    assert inline["compaction"]["auto"] is True
    assert inline["compaction"]["prune"] is True
    assert inline["compaction"]["reserved"] >= 24000
    agent = inline["agent"]["plan"]
    assert "steps" not in agent
    assert "maxSteps" not in agent
    assert agent["permission"] == {"*": "deny"}
    assert agent["mode"] == "primary"
    assert "response-only strategist" in agent["prompt"]


def test_executor_tool_budget_starts_fresh_stricter_retry(tmp_path: Path) -> None:
    import json

    from goal_agent.models import ExecutionReport

    fake = tmp_path / "budgeted_executor.py"
    fake.write_text(
        r'''
import json
import os
import pathlib
import sys
args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)
count_file = pathlib.Path("calls.txt")
count = int(count_file.read_text() or "0") if count_file.exists() else 0
count += 1
count_file.write_text(str(count))
pathlib.Path(f"env-{count}.json").write_text(os.environ["OPENCODE_CONFIG_CONTENT"], encoding="utf-8")
_ = sys.stdin.read()
if count == 1:
    print(json.dumps({
        "type": "tool_use",
        "part": {
            "type": "tool",
            "tool": "read",
            "state": {"status": "completed"},
            "output": "x" * 130000
        }
    }), flush=True)
    raise SystemExit(0)
payload = {
    "summary": "Completed a focused retry",
    "actions": ["Used exact evidence"],
    "files_changed": [],
    "commands_run": [],
    "evidence": ["retry succeeded"],
    "blockers": []
}
text = "<GOAL_AGENT_JSON>\n" + json.dumps(payload) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type":"text","part":{"type":"text","text":text}}))
''',
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))
    statuses: list[tuple[str, str]] = []

    report, _ = asyncio.run(
        runner.run_structured(
            "Make one bounded change.",
            ExecutionReport,
            model="fake/model",
            agent="build",
            attempts=1,
            profile="executor",
            status_callback=lambda kind, detail: statuses.append((kind, detail)),
        )
    )

    assert report.summary == "Completed a focused retry"
    assert (tmp_path / "calls.txt").read_text() == "2"
    first = json.loads((tmp_path / "env-1.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "env-2.json").read_text(encoding="utf-8"))
    assert first["agent"]["build"]["steps"] == 6
    assert second["agent"]["build"]["steps"] == 3
    assert second["agent"]["build"]["permission"].get("task", "deny") == "deny"
    assert any(kind == "context_recovery" for kind, _ in statuses)


def test_extract_json_payload_from_opencode_ndjson_text_event() -> None:
    payload = {
        "hypothesis": "Use bounded evidence",
        "rationale": "The final assistant text is separate from tool events",
        "expected_impact": "Parsing succeeds",
        "target_criteria": ["c1"],
        "plan": ["Return JSON"],
        "avoid_repeating": [],
    }
    assistant_text = (
        "<GOAL_AGENT_JSON>\n"
        + __import__("json").dumps(payload)
        + "\n</GOAL_AGENT_JSON>"
    )
    stream = "\n".join(
        [
            '{"type":"step_start","sessionID":"ses_1","part":{"type":"step-start"}}',
            '{"type":"tool_use","sessionID":"ses_1","part":{"type":"tool","tool":"read","state":{"status":"completed","output":"README"}}}',
            __import__("json").dumps(
                {
                    "type": "text",
                    "sessionID": "ses_1",
                    "part": {"type": "text", "text": assistant_text},
                }
            ),
        ]
    )

    assert extract_json_payload(stream) == payload


def test_extract_json_payload_reports_event_stream_without_assistant_text() -> None:
    import pytest

    stream = "\n".join(
        [
            '{"type":"step_start","sessionID":"ses_1","part":{"type":"step-start"}}',
            '{"type":"tool_use","sessionID":"ses_1","part":{"type":"tool","tool":"read","state":{"status":"completed","output":"README"}}}',
            '{"type":"step_finish","sessionID":"ses_1","part":{"type":"step-finish","reason":"stop"}}',
        ]
    )

    with pytest.raises(ValueError) as caught:
        extract_json_payload(stream)

    message = str(caught.value)
    assert "without a final assistant text response" in message
    assert "tool_use=1" in message
    assert "Extra data" not in message


def test_refinement_profile_is_tool_free_without_max_step_prefill(tmp_path: Path) -> None:
    import json

    from goal_agent.models import SetupProposal

    fake = tmp_path / "refinement_profile_opencode.py"
    fake.write_text(
        r'''
import json
import os
import pathlib
import sys
args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)
pathlib.Path("refinement-env.json").write_text(os.environ["OPENCODE_CONFIG_CONTENT"], encoding="utf-8")
_ = sys.stdin.read()
payload = {
    "refined_goal": "Create a verified project plan.",
    "assistant_message": "The proposal is concrete.",
    "clarifying_questions": [],
    "assumptions": [],
    "criteria": [{
        "id": "verify-plan",
        "description": "The verification command confirms the plan.",
        "kind": "command",
        "command": "python verify_plan.py",
        "expected_exit_code": 0,
        "required": True
    }],
    "criteria_quality_issues": [],
    "ready_to_finalize": True,
    "readiness_reason": "A deterministic command verifies the result."
}
text = "<GOAL_AGENT_JSON>\n" + json.dumps(payload) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type":"text","part":{"type":"text","text":text}}))
''',
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))

    proposal, _ = asyncio.run(
        runner.run_structured(
            "Use the supplied snapshot and return a proposal.",
            SetupProposal,
            model="fake/model",
            agent="plan",
            profile="refinement",
        )
    )

    assert proposal.ready_to_finalize is True
    inline = json.loads((tmp_path / "refinement-env.json").read_text(encoding="utf-8"))
    agent = inline["agent"]["plan"]
    assert "steps" not in agent
    assert "maxSteps" not in agent
    assert agent["permission"] == {"*": "deny"}
    assert agent["mode"] == "primary"
    assert "response-only goal and success-criteria refiner" in agent["prompt"]


def test_structured_retry_recovers_when_first_run_has_only_tool_events(tmp_path: Path) -> None:
    import json

    from goal_agent.models import SetupProposal

    fake = tmp_path / "no_final_then_valid_opencode.py"
    fake.write_text(
        r'''
import json
import pathlib
import sys
args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)
count_path = pathlib.Path("calls.txt")
count = int(count_path.read_text() or "0") if count_path.exists() else 0
count += 1
count_path.write_text(str(count))
_ = sys.stdin.read()
if count == 1:
    print(json.dumps({"type":"step_start","sessionID":"ses_1","part":{"type":"step-start"}}))
    print(json.dumps({"type":"tool_use","sessionID":"ses_1","part":{"type":"tool","tool":"read","state":{"status":"completed","output":"README"}}}))
    print(json.dumps({"type":"step_finish","sessionID":"ses_1","part":{"type":"step-finish","reason":"stop"}}))
    raise SystemExit(0)
payload = {
    "refined_goal": "Create a verified project plan.",
    "assistant_message": "Recovered with an answer-only retry.",
    "clarifying_questions": [],
    "assumptions": [],
    "criteria": [{
        "id": "verify-plan",
        "description": "The verification command confirms the plan.",
        "kind": "command",
        "command": "python verify_plan.py",
        "expected_exit_code": 0,
        "required": True
    }],
    "criteria_quality_issues": [],
    "ready_to_finalize": True,
    "readiness_reason": "A deterministic command verifies the result."
}
text = "<GOAL_AGENT_JSON>\n" + json.dumps(payload) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type":"text","part":{"type":"text","text":text}}))
''',
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))
    statuses: list[tuple[str, str]] = []

    proposal, _ = asyncio.run(
        runner.run_structured(
            "Return a proposal.",
            SetupProposal,
            model="fake/model",
            agent="plan",
            profile="default",
            status_callback=lambda kind, detail: statuses.append((kind, detail)),
        )
    )

    assert proposal.ready_to_finalize is True
    assert (tmp_path / "calls.txt").read_text() == "2"
    assert any(
        kind == "retry"
        and (
            "without a final assistant text response" in detail
            or "prose instead of json" in detail.lower()
        )
        for kind, detail in statuses
    )


def test_refinement_avoids_one_token_empty_response_caused_by_steps_one(tmp_path: Path) -> None:
    """Regression for local OpenAI-compatible models that EOS on max-step prefill."""

    import json

    from goal_agent.models import SetupProposal

    fake = tmp_path / "empty_on_max_steps_opencode.py"
    fake.write_text(
        r'''
import json
import os
import sys

args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)

config = json.loads(os.environ["OPENCODE_CONFIG_CONTENT"])
agent_name = args[args.index("--agent") + 1]
agent = config["agent"][agent_name]
_ = sys.stdin.read()

# Reproduce the user's observed OpenCode stream: max-step mode causes the
# OpenAI-compatible local model to emit only EOS (one output token), so the CLI
# produces start/finish records and no text event.
if agent.get("steps") == 1 or agent.get("maxSteps") == 1:
    print(json.dumps({"type":"step_start","sessionID":"ses_empty","part":{"type":"step-start"}}))
    print(json.dumps({"type":"step_finish","sessionID":"ses_empty","part":{"type":"step-finish","reason":"stop","tokens":{"total":9341,"input":578,"output":1,"reasoning":0,"cache":{"write":0,"read":8762}}}}))
    raise SystemExit(0)

assert agent["permission"] == {"*": "deny"}
assert "response-only goal and success-criteria refiner" in agent["prompt"]
payload = {
    "refined_goal": "Create a verified goal definition.",
    "assistant_message": "The response-only call completed normally.",
    "clarifying_questions": [],
    "assumptions": [],
    "criteria": [{
        "id": "verify-goal",
        "description": "The verification command confirms the goal definition.",
        "kind": "command",
        "command": "python verify_goal.py",
        "expected_exit_code": 0,
        "required": True
    }],
    "criteria_quality_issues": [],
    "ready_to_finalize": True,
    "readiness_reason": "A deterministic command verifies the result."
}
text = "<GOAL_AGENT_JSON>\n" + json.dumps(payload) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type":"text","part":{"type":"text","text":text}}))
''',
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))

    proposal, result = asyncio.run(
        runner.run_structured(
            "Return the setup proposal using supplied evidence.",
            SetupProposal,
            model="fake/model",
            agent="plan",
            profile="refinement",
        )
    )

    assert result.text
    assert proposal.ready_to_finalize is True


def test_adaptive_reserved_tokens_for_smaller_context_window(tmp_path: Path, monkeypatch) -> None:
    import json

    from goal_agent.models import SetupProposal

    fake = tmp_path / "adaptive_reserved_opencode.py"
    fake.write_text(
        r'''
import json
import os
import pathlib
import sys
args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)
pathlib.Path("adaptive-env.json").write_text(os.environ["OPENCODE_CONFIG_CONTENT"], encoding="utf-8")
_ = sys.stdin.read()
payload = {
    "refined_goal": "Keep context budgets bounded.",
    "assistant_message": "Adaptive reserve applied.",
    "clarifying_questions": [],
    "assumptions": [],
    "criteria": [{
        "id": "verify-bounds",
        "description": "A bounded command confirms safety.",
        "kind": "command",
        "command": "python -c \"print('ok')\"",
        "expected_exit_code": 0,
        "required": True
    }],
    "criteria_quality_issues": [],
    "ready_to_finalize": True,
    "readiness_reason": "Bounded criteria are present."
}
text = "<GOAL_AGENT_JSON>\n" + json.dumps(payload) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type":"text","part":{"type":"text","text":text}}))
''',
        encoding="utf-8",
    )

    inline = {
        "model": "llama.cpp/qwen3-coder:a3b",
        "provider": {
            "llama.cpp": {
                "models": {
                    "qwen3-coder:a3b": {
                        "limit": {"context": 32768, "output": 4096}
                    }
                }
            }
        },
    }
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps(inline))
    runner = OpenCodeRunner(_config(tmp_path, fake))

    proposal, _ = asyncio.run(
        runner.run_structured(
            "Return the setup proposal.",
            SetupProposal,
            model="llama.cpp/qwen3-coder:a3b",
            agent="plan",
            profile="refinement",
        )
    )

    assert proposal.ready_to_finalize is True
    applied = json.loads((tmp_path / "adaptive-env.json").read_text(encoding="utf-8"))
    assert applied["compaction"]["reserved"] < 24000
    assert applied["compaction"]["reserved"] >= 4096


def test_response_only_stall_retries_and_recovers(tmp_path: Path, monkeypatch) -> None:
    import json

    from goal_agent.models import SetupProposal

    fake = tmp_path / "stall_then_valid_opencode.py"
    fake.write_text(
        r'''
import json
import pathlib
import sys
import time

args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)

count_path = pathlib.Path("calls.txt")
count = int(count_path.read_text() or "0") if count_path.exists() else 0
count += 1
count_path.write_text(str(count))
_ = sys.stdin.read()

if count == 1:
    while True:
        print(json.dumps({"type": "step_start", "part": {"type": "step-start"}}), flush=True)
        time.sleep(0.05)

payload = {
    "refined_goal": "Recovered after stall",
    "assistant_message": "Returned after retry.",
    "clarifying_questions": [],
    "assumptions": [],
    "criteria": [{
        "id": "verify-goal",
        "description": "Verification command exists.",
        "kind": "command",
        "command": "python -m pytest",
        "expected_exit_code": 0,
        "required": True
    }],
    "criteria_quality_issues": [],
    "ready_to_finalize": True,
    "readiness_reason": "Recovered"
}
text = "<GOAL_AGENT_JSON>\n" + json.dumps(payload) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type": "text", "part": {"type": "text", "text": text}}), flush=True)
''',
        encoding="utf-8",
    )
    monkeypatch.setenv("GOAL_AGENT_RESPONSE_ONLY_STALL_SECONDS", "1")
    runner = OpenCodeRunner(_config(tmp_path, fake))
    statuses: list[tuple[str, str]] = []

    proposal, _ = asyncio.run(
        runner.run_structured(
            "Return the setup proposal.",
            SetupProposal,
            model="fake/model",
            agent="plan",
            profile="refinement",
            attempts=2,
            status_callback=lambda kind, detail: statuses.append((kind, detail)),
        )
    )

    assert proposal.ready_to_finalize is True
    assert (tmp_path / "calls.txt").read_text() == "2"
    assert any(kind == "retry" and "no assistant text" in detail.lower() for kind, detail in statuses)


def test_refinement_prose_response_is_repaired_to_structured_json(tmp_path: Path) -> None:
    import json

    from goal_agent.models import SetupProposal

    fake = tmp_path / "prose_then_repair_opencode.py"
    fake.write_text(
        r'''
import json
import pathlib
import sys

args = sys.argv[1:]
if args == ["run", "--help"]:
    print("--dangerously-skip-permissions")
    raise SystemExit(0)

count_path = pathlib.Path("calls.txt")
count = int(count_path.read_text() or "0") if count_path.exists() else 0
count += 1
count_path.write_text(str(count))
prompt = sys.stdin.read()

if "FORMAT-REPAIR MODE" not in prompt:
    prose = """
# Goal & Criteria Review
The current criteria validate only internal suites and miss external benchmark coverage.
I recommend splitting into internal validation and external benchmark tracks.
"""
    print(json.dumps({"type": "text", "part": {"type": "text", "text": prose}}))
    raise SystemExit(0)

payload = {
    "refined_goal": "Validate internal evals and define external benchmark track.",
    "assistant_message": "Converted prior prose into structured proposal.",
    "clarifying_questions": ["Which external benchmark should be prioritized first?"],
    "assumptions": ["Internal eval suites remain authoritative until external infra is added."],
    "criteria": [{
        "id": "verify-internal-gates",
        "description": "Promotion gate command passes with expected output.",
        "kind": "command",
        "command": "python -m pytest",
        "expected_exit_code": 0,
        "required": True
    }],
    "criteria_quality_issues": [],
    "ready_to_finalize": False,
    "readiness_reason": "External benchmark scope requires one decision."
}
text = "<GOAL_AGENT_JSON>\n" + json.dumps(payload) + "\n</GOAL_AGENT_JSON>"
print(json.dumps({"type": "text", "part": {"type": "text", "text": text}}))
''',
        encoding="utf-8",
    )
    runner = OpenCodeRunner(_config(tmp_path, fake))

    proposal, _ = asyncio.run(
        runner.run_structured(
            "Return the setup proposal.",
            SetupProposal,
            model="fake/model",
            agent="plan",
            profile="refinement",
            attempts=1,
        )
    )

    assert proposal.ready_to_finalize is False
    assert proposal.clarifying_questions
    assert (tmp_path / "calls.txt").read_text() == "2"
