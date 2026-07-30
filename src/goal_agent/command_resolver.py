from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


WINDOWS_SCRIPT_SUFFIXES = {".cmd", ".bat"}
POWERSHELL_SCRIPT_SUFFIXES = {".ps1"}
WINDOWS_EXECUTABLE_SUFFIXES = (".exe", ".com", ".cmd", ".bat", ".ps1")


@dataclass(slots=True, frozen=True)
class ExecutableResolution:
    configured: str
    path: Path | None
    source: str

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def kind(self) -> str:
        if self.path is None:
            return "missing"
        suffix = self.path.suffix.lower()
        if suffix in WINDOWS_SCRIPT_SUFFIXES:
            return "cmd-script"
        if suffix in POWERSHELL_SCRIPT_SUFFIXES:
            return "powershell-script"
        return "executable"


class CommandResolutionError(FileNotFoundError):
    pass


def resolve_executable(executable: str, *, windows: bool | None = None) -> ExecutableResolution:
    """Resolve a configured command, including Windows npm/bun shims.

    ``asyncio.create_subprocess_exec`` cannot reliably launch ``.cmd``/``.bat``
    files directly on Windows. Resolution is kept separate from invocation so
    validation can report the actual file that will be used.
    """

    configured = os.path.expandvars(os.path.expanduser(executable.strip()))
    if not configured:
        return ExecutableResolution(executable, None, "empty")

    is_windows = os.name == "nt" if windows is None else windows
    explicit = _looks_like_path(configured)
    if explicit:
        path = Path(configured)
        if path.is_file():
            return ExecutableResolution(executable, path.resolve(), "configured-path")
        if is_windows and not path.suffix:
            for suffix in WINDOWS_EXECUTABLE_SUFFIXES:
                candidate = Path(f"{configured}{suffix}")
                if candidate.is_file():
                    return ExecutableResolution(executable, candidate.resolve(), "configured-path")
        return ExecutableResolution(executable, None, "configured-path-missing")

    found = shutil.which(configured)
    if found:
        return ExecutableResolution(executable, Path(found).resolve(), "PATH")

    if is_windows:
        for directory in _windows_search_directories():
            for name in _windows_candidate_names(configured):
                candidate = directory / name
                if candidate.is_file():
                    return ExecutableResolution(executable, candidate.resolve(), "Windows user path")

    return ExecutableResolution(executable, None, "not-found")


def prepare_command(command: Sequence[str], *, windows: bool | None = None) -> list[str]:
    """Return a subprocess-safe command for the current platform.

    On Windows, npm commonly exposes OpenCode as ``opencode.cmd``. CreateProcess
    does not execute command scripts itself, so they must be passed through
    ``cmd.exe``. PowerShell scripts receive a similarly explicit launcher.
    """

    if not command:
        raise CommandResolutionError("The configured OpenCode command is empty.")

    is_windows = os.name == "nt" if windows is None else windows
    resolution = resolve_executable(command[0], windows=is_windows)
    if not resolution.path:
        raise CommandResolutionError(
            _missing_command_message(command[0], windows=is_windows)
        )

    resolved = str(resolution.path)
    remainder = [str(part) for part in command[1:]]
    suffix = resolution.path.suffix.lower()

    if is_windows and suffix in WINDOWS_SCRIPT_SUFFIXES:
        command_processor = _resolve_command_processor()
        # Supplying one fully quoted command string after /c avoids cmd.exe
        # reinterpreting later arguments as options to itself.
        command_line = subprocess.list2cmdline([resolved, *remainder])
        return [command_processor, "/d", "/s", "/c", command_line]

    if is_windows and suffix in POWERSHELL_SCRIPT_SUFFIXES:
        powershell = _resolve_powershell()
        return [
            powershell,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            resolved,
            *remainder,
        ]

    return [resolved, *remainder]


def render_command(command: Sequence[str], *, windows: bool | None = None) -> str:
    is_windows = os.name == "nt" if windows is None else windows
    if is_windows:
        return subprocess.list2cmdline([str(part) for part in command])
    import shlex

    return " ".join(shlex.quote(str(part)) for part in command)


def _looks_like_path(value: str) -> bool:
    path = Path(value)
    return path.is_absolute() or any(separator in value for separator in ("/", "\\"))


def _windows_candidate_names(command: str) -> Iterable[str]:
    path = Path(command)
    if path.suffix:
        yield command
        return
    for suffix in WINDOWS_EXECUTABLE_SUFFIXES:
        yield f"{command}{suffix}"


def _windows_search_directories() -> list[Path]:
    directories: list[Path] = []

    def add(value: str | os.PathLike[str] | None) -> None:
        if not value:
            return
        expanded = os.path.expandvars(os.path.expanduser(str(value))).strip().strip('"')
        if not expanded:
            return
        candidate = Path(expanded)
        if candidate not in directories:
            directories.append(candidate)

    # Current process PATH first.
    for item in os.environ.get("PATH", "").split(os.pathsep):
        add(item)

    # A GUI process may have been started before npm/bun updated PATH. Read the
    # current user and machine PATH directly from the registry as a fallback.
    try:
        import winreg  # type: ignore[attr-defined]

        registry_locations = (
            (winreg.HKEY_CURRENT_USER, r"Environment"),
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ),
        )
        for hive, key_name in registry_locations:
            try:
                with winreg.OpenKey(hive, key_name) as key:
                    value, _ = winreg.QueryValueEx(key, "Path")
                for item in str(value).split(";"):
                    add(item)
            except OSError:
                continue
    except (ImportError, AttributeError):
        pass

    appdata = os.environ.get("APPDATA")
    localappdata = os.environ.get("LOCALAPPDATA")
    userprofile = os.environ.get("USERPROFILE") or str(Path.home())

    add(Path(appdata) / "npm" if appdata else None)
    add(Path(userprofile) / ".bun" / "bin")
    add(Path(userprofile) / ".opencode" / "bin")
    add(Path(localappdata) / "Microsoft" / "WinGet" / "Links" if localappdata else None)
    add(Path(localappdata) / "Programs" / "opencode" if localappdata else None)
    return directories


def _resolve_command_processor() -> str:
    configured = os.environ.get("COMSPEC")
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("cmd.exe") or shutil.which("cmd")
    if found:
        return found
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    candidate = Path(system_root) / "System32" / "cmd.exe"
    if candidate.is_file():
        return str(candidate)
    raise CommandResolutionError("Windows command processor cmd.exe could not be found.")


def _resolve_powershell() -> str:
    for name in ("pwsh.exe", "powershell.exe", "pwsh", "powershell"):
        found = shutil.which(name)
        if found:
            return found
    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    candidate = (
        Path(system_root)
        / "System32"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    if candidate.is_file():
        return str(candidate)
    raise CommandResolutionError("PowerShell could not be found to run the configured .ps1 command.")


def _missing_command_message(executable: str, *, windows: bool) -> str:
    message = f"OpenCode executable could not be resolved: {executable}"
    if windows:
        message += (
            ". The app checked PATH, the current Windows user/machine PATH, "
            "%APPDATA%\\npm, %USERPROFILE%\\.bun\\bin, "
            "%USERPROFILE%\\.opencode\\bin, and WinGet links. "
            "In PowerShell, run `Get-Command opencode | Format-List CommandType,Source,Path` "
            "to see which launcher your shell is using."
        )
    return message
