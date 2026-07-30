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
