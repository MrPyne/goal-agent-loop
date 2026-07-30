from pathlib import Path

import pytest

from goal_agent.command_resolver import (
    CommandResolutionError,
    prepare_command,
    resolve_executable,
)


def test_resolve_explicit_executable(tmp_path: Path) -> None:
    executable = tmp_path / "opencode.exe"
    executable.write_text("fake", encoding="utf-8")

    resolution = resolve_executable(str(executable), windows=True)

    assert resolution.found
    assert resolution.path == executable.resolve()
    assert resolution.kind == "executable"


def test_windows_cmd_shim_is_wrapped_with_cmd_exe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shim = tmp_path / "opencode.cmd"
    shim.write_text("@echo off\r\n", encoding="utf-8")
    command_processor = tmp_path / "cmd.exe"
    command_processor.write_text("fake", encoding="utf-8")
    monkeypatch.setenv("COMSPEC", str(command_processor))

    invocation = prepare_command(
        [str(shim), "run", "--dir", r"C:\Projects\My Project", "--auto"],
        windows=True,
    )

    assert invocation[:4] == [str(command_processor), "/d", "/s", "/c"]
    assert str(shim.resolve()) in invocation[4]
    assert 'C:\\Projects\\My Project' in invocation[4]
    assert "--auto" in invocation[4]


def test_windows_powershell_shim_is_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shim = tmp_path / "opencode.ps1"
    shim.write_text("exit 0\n", encoding="utf-8")
    powershell = tmp_path / "powershell.exe"
    powershell.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(
        "goal_agent.command_resolver.shutil.which",
        lambda name: str(powershell) if "powershell" in name else None,
    )

    invocation = prepare_command([str(shim), "models"], windows=True)

    assert invocation[0] == str(powershell)
    assert "-File" in invocation
    assert str(shim.resolve()) in invocation
    assert invocation[-1] == "models"


def test_missing_command_has_windows_diagnostics() -> None:
    with pytest.raises(CommandResolutionError, match="Get-Command opencode"):
        prepare_command(["definitely-not-installed-opencode"], windows=True)
