from pathlib import Path

from goal_agent.project_snapshot import collect_project_snapshot


def test_project_snapshot_prefers_contract_files_and_excludes_noisy_trees(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Project purpose and usage", encoding="utf-8")
    (tmp_path / "PROJECT_CHARTER.md").write_text("Exact project goal", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pydantic", encoding="utf-8")
    noisy = tmp_path / ".venv"
    noisy.mkdir()
    (noisy / "secret.txt").write_text("must not be included", encoding="utf-8")

    snapshot = collect_project_snapshot(tmp_path)

    assert "README.md" in snapshot
    assert "Project purpose and usage" in snapshot
    assert "PROJECT_CHARTER.md" in snapshot
    assert "Exact project goal" in snapshot
    assert "must not be included" not in snapshot
    assert ".venv/" not in snapshot


def test_project_snapshot_includes_verification_research_hints(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Project purpose and usage", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "demo"
version = "0.1.0"

[tool.pytest.ini_options]
addopts = "-q"
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()

    snapshot = collect_project_snapshot(tmp_path)

    assert "VERIFICATION RESEARCH" in snapshot
    assert "pytest" in snapshot.lower()
