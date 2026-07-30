from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

from goal_agent.web import create_app


def test_gui_can_create_switch_and_remove_projects(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    project_a = tmp_path / "project-a"
    project_b = tmp_path / "project-b"

    with TestClient(create_app(registry_path=registry)) as client:
        empty = client.get("/api/projects")
        assert empty.status_code == 200
        assert empty.json()["projects"] == []

        created_a = client.post(
            "/api/projects/create",
            json={"path": str(project_a), "title": "Project A", "model": "fake/a"},
        )
        assert created_a.status_code == 201
        id_a = created_a.json()["id"]
        assert (project_a / ".goal-agent" / "config.yaml").exists()

        created_b = client.post(
            "/api/projects/create",
            json={"path": str(project_b), "title": "Project B", "model": "fake/b"},
        )
        assert created_b.status_code == 201
        id_b = created_b.json()["id"]
        assert id_a != id_b

        projects = client.get("/api/projects").json()
        assert len(projects["projects"]) == 2
        assert projects["active_project_id"] == id_b

        assert client.post(f"/api/projects/{id_a}/activate").status_code == 200
        goal = client.post(
            f"/api/goals?project_id={id_a}",
            json={"id": "gui-goal", "title": "GUI goal", "goal": "Do the work"},
        )
        assert goal.status_code == 201
        assert (project_a / ".goal-agent" / "goals" / "gui-goal").exists()
        assert not (project_b / ".goal-agent" / "goals" / "gui-goal").exists()

        assert client.delete(f"/api/projects/{id_b}").status_code == 204
        assert project_b.exists()  # unregistering never deletes the workspace
        remaining = client.get("/api/projects").json()["projects"]
        assert [item["id"] for item in remaining] == [id_a]


def test_gui_exposes_validation_and_complete_config(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    project = tmp_path / "project"

    with TestClient(create_app(registry_path=registry)) as client:
        created = client.post(
            "/api/projects/create",
            json={"path": str(project), "title": "Project", "model": "fake/model"},
        ).json()
        project_id = created["id"]

        patch = client.patch(
            f"/api/project/config?project_id={project_id}",
            json={
                "opencode_command": [sys.executable],
                "attach_url": "http://127.0.0.1:4096",
                "attach_username": "local-user",
                "attach_password_env": "OPENCODE_PASSWORD",
                "max_iterations": 25,
                "max_recent_hypotheses": 20,
                "poll_interval_seconds": 0.2,
                "status_refresh_seconds": 0.3,
                "gui_host": "127.0.0.1",
                "gui_port": 8999,
            },
        )
        assert patch.status_code == 200
        config = patch.json()
        assert config["max_iterations"] == 25
        assert config["attach_username"] == "local-user"
        assert config["gui_port"] == 8999

        criteria = [
            {
                "id": "manual-check",
                "description": "A human verifies the result",
                "kind": "manual",
                "required": True,
                "override": "auto",
            }
        ]
        assert client.put(
            f"/api/goals/default/criteria?project_id={project_id}",
            json={"criteria": criteria},
        ).status_code == 200

        validation = client.post(
            f"/api/project/validate?project_id={project_id}"
        )
        assert validation.status_code == 200
        assert validation.json()["valid"] is True
        goal_validation = validation.json()["goals"]["default"]
        assert goal_validation["warnings"]
        assert "cannot pass autonomously" in goal_validation["warnings"][0]

        files = client.get(f"/api/project/files?project_id={project_id}").json()
        assert files["config"].endswith(".goal-agent/config.yaml")
        assert "default" in files["goals"]
        assert files["goals"]["default"]["runs"].endswith("goals/default/runs")
        assert files["goals"]["default"]["evaluation_analysis"].endswith(
            "goals/default/status/evaluation-analysis.json"
        )


def test_gui_can_browse_create_and_discover_project_folders(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    root = tmp_path / "workspaces"
    root.mkdir()
    regular = root / "regular"
    regular.mkdir()
    project = root / "existing-project"

    with TestClient(create_app(registry_path=registry)) as client:
        locations = client.get("/api/system/locations")
        assert locations.status_code == 200
        assert "default_projects_dir" in locations.json()

        nested = root / "new-parent" / "new-project"
        info = client.get("/api/system/path-info", params={"path": str(nested)})
        assert info.status_code == 200
        assert info.json()["exists"] is False
        assert info.json()["writable"] is True
        assert info.json()["nearest_existing_parent"] == str(root)

        folder = client.post(
            "/api/system/folders/create", json={"path": str(root / "created-in-gui")}
        )
        assert folder.status_code == 201
        assert (root / "created-in-gui").is_dir()

        created = client.post(
            "/api/projects/create",
            json={"path": str(project), "title": "Existing project", "model": "fake/model"},
        )
        assert created.status_code == 201

        listing = client.get("/api/system/folders", params={"path": str(root)}).json()
        item = next(value for value in listing["directories"] if value["name"] == project.name)
        assert item["initialized_project"] is True

        discovered = client.post(
            "/api/system/projects/discover",
            json={"roots": [str(root)], "max_depth": 3, "max_results": 10},
        )
        assert discovered.status_code == 200
        match = next(value for value in discovered.json()["projects"] if value["path"] == str(project))
        assert match["registered"] is True


def test_gui_native_folder_picker_endpoint_can_be_mocked(tmp_path: Path, monkeypatch) -> None:
    from goal_agent import web

    selected = tmp_path / "selected"
    selected.mkdir()
    monkeypatch.setattr(
        web,
        "_native_pick_directory",
        lambda initial_path, title, must_exist: str(selected),
    )

    with TestClient(create_app(registry_path=tmp_path / "registry.yaml")) as client:
        response = client.post(
            "/api/system/folders/pick",
            json={"initial_path": str(tmp_path), "title": "Choose", "must_exist": True},
        )
        assert response.status_code == 200
        assert response.json() == {"selected": str(selected), "cancelled": False}


def test_friendly_project_ui_assets_are_served(tmp_path: Path) -> None:
    with TestClient(create_app(registry_path=tmp_path / "registry.yaml")) as client:
        html = client.get("/")
        assert html.status_code == 200
        assert "Choose project folder" in html.text
        assert "folder-picker-modal" in html.text
        assert "/friendly.js" in html.text

        script = client.get("/friendly.js")
        assert script.status_code == 200
        assert "chooseNativeFolder" in script.text
        assert "discoverProjects" in script.text
