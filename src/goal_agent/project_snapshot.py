from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

_MAX_FILES = 8
_MAX_FILE_CHARS = 10_000
_MAX_TOTAL_CHARS = 42_000
_MAX_ROOT_ENTRIES = 80

_EXCLUDED_DIRS = {
    ".git",
    ".goal-agent",
    ".venv",
    "venv",
    "node_modules",
    "dist",
    "build",
    "target",
    "coverage",
    "vendor",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tmp",
    "tmp",
    "logs",
}

_TEXT_SUFFIXES = {
    ".txt",
    ".md",
    ".rst",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".tsv",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".css",
    ".scss",
    ".java",
    ".kt",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".sh",
    ".ps1",
    ".bat",
    ".cmd",
    ".xml",
    ".sql",
}

# Files that usually explain the purpose, current state, build, and verification
# contract of a project.  Goal Agent reads them itself so the OpenCode refinement
# agent can remain tool-free and cannot spend its entire step budget exploring.
_PREFERRED_NAMES = (
    "PROJECT_CHARTER.md",
    "README.md",
    "README.rst",
    "README.txt",
    "MODEL_CARD.md",
    "CURRICULUM.md",
    "AGENTS.md",
    "CONTEXT.md",
    "pyproject.toml",
    "package.json",
    "requirements.txt",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "Dockerfile",
)


def _verification_research(root: Path, root_files: dict[str, Path]) -> str:
    """Infer likely verification commands from project metadata and layout."""

    hints: list[str] = []

    pyproject = root_files.get("pyproject.toml")
    if pyproject is not None:
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8", errors="replace"))
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        if isinstance(data, dict):
            tool = data.get("tool")
            if isinstance(tool, dict) and "pytest" in tool:
                hints.append("python: pytest detected in pyproject.toml -> try `pytest -q`")
            project = data.get("project")
            scripts = project.get("scripts") if isinstance(project, dict) else None
            if isinstance(scripts, dict) and scripts:
                names = ", ".join(sorted(str(key) for key in scripts.keys())[:8])
                hints.append(
                    "python: project scripts detected in pyproject.toml "
                    f"({names}) -> consider command criteria against those entry points"
                )

    package_json = root_files.get("package.json")
    if package_json is not None:
        try:
            pkg = json.loads(package_json.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            pkg = {}
        scripts = pkg.get("scripts") if isinstance(pkg, dict) else None
        if isinstance(scripts, dict) and scripts:
            script_names = set(str(name) for name in scripts.keys())
            for candidate in ("test", "lint", "build", "check"):
                if candidate in script_names:
                    hints.append(
                        f"node: package script '{candidate}' detected -> try `npm run {candidate}`"
                    )

    if "cargo.toml" in root_files:
        hints.append("rust: Cargo.toml detected -> try `cargo test` and/or `cargo clippy -- -D warnings`")
    if "go.mod" in root_files:
        hints.append("go: go.mod detected -> try `go test ./...`")

    makefile = root_files.get("makefile")
    if makefile is not None:
        try:
            text = makefile.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        targets = sorted(set(re.findall(r"^(test|check|lint|ci)\s*:", text, flags=re.MULTILINE)))
        for target in targets[:6]:
            hints.append(f"build: make target '{target}' detected -> try `make {target}`")

    if (root / "tests").is_dir() and not any("pytest" in hint for hint in hints):
        hints.append("tests/: directory detected -> likely Python tests, try `pytest -q`")
    if (root / ".github" / "workflows").is_dir():
        hints.append("ci: .github/workflows detected -> mirror CI commands in required criteria")

    if not hints:
        return "VERIFICATION RESEARCH\nNo explicit verification surface detected; infer checks from project docs and root listing."
    return "VERIFICATION RESEARCH\n" + "\n".join(f"- {hint}" for hint in hints[:12])


def _clip(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 3 // 5
    tail = max_chars - head
    return (
        text[:head].rstrip()
        + "\n\n… [middle omitted from project snapshot] …\n\n"
        + text[-tail:].lstrip()
    )


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_SUFFIXES or path.name.lower() in {
        "dockerfile",
        "makefile",
        "readme",
        "license",
        "agents.md",
        "context.md",
    }


def _safe_root_listing(project_path: Path) -> str:
    entries: list[str] = []
    try:
        children = sorted(project_path.iterdir(), key=lambda item: item.name.lower())
    except OSError as exc:
        return f"Root listing unavailable: {exc}"

    for child in children:
        if child.name in _EXCLUDED_DIRS:
            continue
        suffix = "/" if child.is_dir() else ""
        entries.append(child.name + suffix)
        if len(entries) >= _MAX_ROOT_ENTRIES:
            entries.append("… [additional root entries omitted]")
            break
    return "\n".join(entries) if entries else "(no visible project files)"


def collect_project_snapshot(project_path: Path) -> str:
    """Return a bounded, read-only project overview for goal refinement.

    The snapshot deliberately favors project contracts and documentation rather
    than source-code breadth.  It is stable, inspectable, and small enough to be
    embedded in a single tool-free OpenCode request.
    """

    root = project_path.resolve()
    selected: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            return
        if resolved in seen or not resolved.is_file() or not _is_text_file(resolved):
            return
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            return
        if any(part in _EXCLUDED_DIRS for part in relative.parts):
            return
        seen.add(resolved)
        selected.append(resolved)

    # Case-insensitive preferred-name lookup at the project root.
    try:
        root_files = {
            item.name.lower(): item
            for item in root.iterdir()
            if item.is_file()
        }
    except OSError:
        root_files = {}
    for name in _PREFERRED_NAMES:
        candidate = root_files.get(name.lower())
        if candidate is not None:
            add(candidate)
        if len(selected) >= _MAX_FILES:
            break

    # Do not fall back to arbitrary source files. Refinement only needs a
    # high-level project contract, and including helper scripts can leak test
    # fixtures, generated prompts, or unrelated implementation details into the
    # goal-definition conversation. The root listing still exposes the shape of
    # projects that do not have preferred documents.

    chunks = [
        "PROJECT ROOT LISTING\n" + _safe_root_listing(root),
        _verification_research(root, root_files),
    ]
    used = 0
    for path in selected:
        if used >= _MAX_TOTAL_CHARS:
            break
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            size = path.stat().st_size
        except OSError as exc:
            relative = path.relative_to(root).as_posix()
            chunks.append(f"--- {relative} ---\n[Could not read: {exc}]")
            continue
        available = min(_MAX_FILE_CHARS, _MAX_TOTAL_CHARS - used)
        clipped = _clip(content, available)
        used += len(clipped)
        relative = path.relative_to(root).as_posix()
        chunks.append(f"--- {relative} ({size} bytes) ---\n{clipped}")

    if not selected:
        chunks.append("No preferred readable project documents were found.")

    return "\n\n".join(chunks)
