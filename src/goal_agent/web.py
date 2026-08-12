from __future__ import annotations

import asyncio
import ipaddress
import os
import shutil
import string
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .command_resolver import resolve_executable
from .models import (
    AppConfig,
    apply_criteria_revision as build_criteria_revision,
    CriteriaDocument,
    CriterionDefinition,
    EventRecord,
    GoalMetadata,
    OverrideMode,
    RefinementMessage,
    SetupProposal,
    utc_now,
)
from .opencode import OpenCodeContextOverflowError, OpenCodeError, OpenCodeRunner
from .project_registry import ProjectEntry, ProjectRegistry
from .proposal_quality import assess_setup_proposal
from .project_snapshot import collect_project_snapshot
from .proposal_jobs import ProposalJobManager, StatusCallback
from .refinement_context import build_refinement_context
from .prompts import criteria_refinement_prompt, setup_prompt
from .storage import ProjectStore
from .supervisor import ConcurrencyLimitReached, GoalSupervisor


ASSET_DIR = Path(__file__).with_name("web_assets")


class FolderPickRequest(BaseModel):
    initial_path: str = ""
    title: str = "Choose a folder"
    must_exist: bool = True


class FolderCreateRequest(BaseModel):
    path: str = Field(min_length=1)


class RevealPathRequest(BaseModel):
    path: str = Field(min_length=1)


class ProjectDiscoveryRequest(BaseModel):
    roots: list[str] = Field(default_factory=list)
    max_depth: int = Field(default=4, ge=1, le=8)
    max_results: int = Field(default=50, ge=1, le=200)


class ProjectCreateRequest(BaseModel):
    path: str = Field(min_length=1)
    title: str = ""
    model: str | None = None
    force: bool = False


class ProjectOpenRequest(BaseModel):
    path: str = Field(min_length=1)
    title: str = ""


class ProjectMetadataPatch(BaseModel):
    title: str = Field(min_length=1)


class GoalCreateRequest(BaseModel):
    id: str
    title: str
    goal: str = ""
    description: str = ""


class GoalMetadataPatch(BaseModel):
    title: str
    description: str = ""
    archived: bool = False


class GoalTextRequest(BaseModel):
    goal: str


class CriteriaUpdateRequest(BaseModel):
    criteria: list[CriterionDefinition]


class SteeringRequest(BaseModel):
    message: str = Field(min_length=1)


class ActionRequest(BaseModel):
    action: Literal["start", "resume", "pause", "stop"]


class BulkActionRequest(BaseModel):
    action: Literal["start", "pause", "stop"]


class GoalModelRequest(BaseModel):
    model: str | None = None


class OverrideRequest(BaseModel):
    value: OverrideMode


class ConfigPatch(BaseModel):
    model: str | None = None
    opencode_command: list[str] | None = None
    attach_url: str | None = None
    attach_username: str | None = None
    attach_password_env: str | None = None
    strategist_agent: str | None = None
    executor_agent: str | None = None
    evaluator_agent: str | None = None
    auto_approve: bool | None = None
    poll_interval_seconds: float | None = Field(default=None, ge=0.05)
    iteration_delay_seconds: float | None = Field(default=None, ge=0)
    opencode_timeout_seconds: int | None = Field(default=None, ge=1)
    criterion_timeout_seconds: int | None = Field(default=None, ge=1)
    max_iterations: int | None = Field(default=None, ge=1)
    max_recent_hypotheses: int | None = Field(default=None, ge=1)
    no_progress_rethink_after: int | None = Field(default=None, ge=1)
    status_refresh_seconds: float | None = Field(default=None, ge=0.05)
    max_concurrent_goals: int | None = Field(default=None, ge=1, le=32)
    gui_auto_resume_running_goals: bool | None = None
    gui_host: str | None = None
    gui_port: int | None = Field(default=None, ge=1, le=65535)


class RefineGoalRequest(BaseModel):
    feedback: str = ""
    conversation: str = ""


class RefineCriteriaRequest(BaseModel):
    feedback: str = Field(min_length=1)


class ProposalJobRequest(BaseModel):
    mode: Literal["goal", "criteria"]
    feedback: str = ""
    conversation: str = ""


class FinalizeRefinementRequest(BaseModel):
    force: bool = False


@dataclass
class ProjectRuntime:
    entry: ProjectEntry
    store: ProjectStore
    supervisor: GoalSupervisor
    auto_resumed: bool = False


class ControlCenter:
    def __init__(
        self,
        initial_project_dir: Path | str | None = None,
        *,
        registry_path: Path | str | None = None,
    ):
        self.registry = ProjectRegistry(registry_path)
        self.runtimes: dict[str, ProjectRuntime] = {}
        if initial_project_dir is not None:
            candidate = ProjectStore.discover(initial_project_dir)
            if candidate.project_exists():
                self.registry.add(candidate.project_dir, activate=True)

    def list_projects(self) -> list[dict[str, Any]]:
        document = self.registry.read()
        values: list[dict[str, Any]] = []
        for entry in sorted(document.projects, key=lambda item: item.last_opened_at, reverse=True):
            path = entry.project_path
            store = ProjectStore(path)
            exists = path.exists()
            initialized = exists and store.project_exists()
            goal_count = 0
            active_count = 0
            if initialized:
                try:
                    goal_count = len(store.list_goal_ids(include_archived=True))
                except Exception:
                    goal_count = 0
                runtime = self.runtimes.get(entry.id)
                active_count = len(runtime.supervisor.active_goal_ids()) if runtime else 0
            values.append(
                {
                    **entry.model_dump(mode="json"),
                    "exists": exists,
                    "initialized": initialized,
                    "goal_count": goal_count,
                    "active_goal_count": active_count,
                    "active": document.active_project_id == entry.id,
                }
            )
        return values

    def resolve_entry(self, project_id: str | None = None) -> ProjectEntry:
        try:
            if project_id:
                return self.registry.get(project_id)
            entry = self.registry.active()
            if entry is None:
                raise HTTPException(status_code=404, detail="No project is selected")
            return entry
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    async def runtime(self, project_id: str | None = None) -> ProjectRuntime:
        entry = self.resolve_entry(project_id)
        existing = self.runtimes.get(entry.id)
        if existing is None:
            store = ProjectStore(entry.project_path)
            try:
                store.require_project_initialized()
            except FileNotFoundError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            existing = ProjectRuntime(entry=entry, store=store, supervisor=GoalSupervisor(store))
            self.runtimes[entry.id] = existing
        if not existing.auto_resumed:
            existing.auto_resumed = True
            await existing.supervisor.auto_resume()
        return existing

    async def shutdown(self) -> None:
        for runtime in list(self.runtimes.values()):
            await runtime.supervisor.shutdown()

    async def unregister(self, project_id: str) -> None:
        runtime = self.runtimes.pop(project_id, None)
        if runtime:
            await runtime.supervisor.shutdown()
        try:
            self.registry.remove(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


def _normalize_user_path(value: str | Path, *, fallback: Path | None = None) -> Path:
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        return (fallback or Path.home()).expanduser().resolve()
    expanded = os.path.expandvars(raw)
    return Path(expanded).expanduser().resolve()


def _is_local_request(request: Request) -> bool:
    host = request.client.host if request.client else ""
    if host in {"localhost", "testclient"}:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _require_local_request(request: Request) -> None:
    if not _is_local_request(request):
        raise HTTPException(
            status_code=403,
            detail="Local filesystem browsing is only available from this computer.",
        )


def _default_projects_dir() -> Path:
    home = Path.home()
    candidates = [
        home / "projects",
        home / "Projects",
        home / "source",
        home / "Documents" / "Projects",
    ]
    return next((candidate for candidate in candidates if candidate.is_dir()), home / "Projects")


def _quick_locations(center: ControlCenter) -> list[dict[str, str]]:
    home = Path.home()
    candidates: list[tuple[str, Path, str]] = [
        ("Home", home, "home"),
        ("Projects", _default_projects_dir(), "projects"),
        ("Desktop", home / "Desktop", "desktop"),
        ("Documents", home / "Documents", "documents"),
        ("Downloads", home / "Downloads", "downloads"),
        ("Current folder", Path.cwd(), "current"),
    ]
    for entry in center.registry.list():
        candidates.append((entry.title, entry.project_path, "project"))
        candidates.append((f"{entry.title} parent", entry.project_path.parent, "recent"))
    if os.name == "nt":
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:/")
            if drive.exists():
                candidates.append((f"{letter}: drive", drive, "drive"))

    values: list[dict[str, str]] = []
    seen: set[str] = set()
    for label, path, kind in candidates:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            continue
        key = str(resolved).casefold() if os.name == "nt" else str(resolved)
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        values.append({"label": label, "path": str(resolved), "kind": kind})
    return values


def _nearest_existing_directory(path: Path) -> Path:
    current = path if path.is_dir() else path.parent
    while not current.exists() and current.parent != current:
        current = current.parent
    return current if current.is_dir() else Path.home().resolve()


def _path_info(path: Path) -> dict[str, Any]:
    exists = path.exists()
    is_dir = path.is_dir() if exists else False
    initialized = is_dir and (path / ".goal-agent" / "config.yaml").exists()
    discovered_root: str | None = None
    if is_dir:
        try:
            discovered = ProjectStore.discover(path)
            if discovered.project_exists():
                discovered_root = str(discovered.project_dir)
        except Exception:
            discovered_root = None
    parent = path.parent
    nearest_existing_parent = _nearest_existing_directory(path)
    writable_parent = os.access(nearest_existing_parent, os.W_OK)
    has_contents = False
    if is_dir:
        try:
            has_contents = next(path.iterdir(), None) is not None
        except OSError:
            has_contents = False
    return {
        "path": str(path),
        "name": path.name or str(path),
        "exists": exists,
        "is_dir": is_dir,
        "initialized_project": initialized,
        "discovered_project_root": discovered_root,
        "writable": os.access(path, os.W_OK) if is_dir else writable_parent,
        "parent_exists": parent.exists(),
        "nearest_existing_parent": str(nearest_existing_parent),
        "has_contents": has_contents,
    }


def _native_pick_directory(initial_path: Path, title: str, must_exist: bool) -> str | None:
    initial_directory = _nearest_existing_directory(initial_path)
    if os.name == "nt":
        powershell = shutil.which("powershell.exe") or shutil.which("pwsh.exe") or shutil.which("pwsh")
        if powershell:
            escaped_initial = str(initial_directory).replace("'", "''")
            escaped_title = title.replace("'", "''")
            script = f"""
Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '{escaped_title}'
$dialog.SelectedPath = '{escaped_initial}'
$dialog.ShowNewFolderButton = $true
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
$owner.ShowInTaskbar = $false
$owner.Opacity = 0
$owner.Show()
try {{
  if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {{
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Write-Output $dialog.SelectedPath
  }}
}} finally {{
  $owner.Close()
  $owner.Dispose()
  $dialog.Dispose()
}}
"""
            completed = subprocess.run(
                [powershell, "-NoProfile", "-Sta", "-Command", script],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
                check=False,
            )
            if completed.returncode == 0:
                selected = completed.stdout.strip().splitlines()
                if selected:
                    return str(Path(selected[-1]).expanduser().resolve())
                return None

    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError("The native folder picker is unavailable. Use the built-in folder browser.") from exc

    root = tk.Tk()
    try:
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass
        root.update_idletasks()
        selected = filedialog.askdirectory(
            parent=root,
            initialdir=str(initial_directory),
            title=title,
            mustexist=must_exist,
        )
        return str(Path(selected).expanduser().resolve()) if selected else None
    finally:
        root.destroy()


def _reveal_path(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return
    subprocess.Popen(["xdg-open", str(path)])


def _discover_projects(roots: list[Path], max_depth: int, max_results: int) -> list[dict[str, Any]]:
    ignored = {".git", ".venv", "venv", "node_modules", "__pycache__", ".cache", "AppData"}
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        root_parts = len(root.parts)
        for current, dirnames, _filenames in os.walk(root):
            current_path = Path(current)
            depth = len(current_path.parts) - root_parts
            dirnames[:] = [
                name for name in dirnames
                if name not in ignored and not (name.startswith(".") and name != ".goal-agent")
            ]
            if ".goal-agent" in dirnames:
                candidate = current_path.resolve()
                key = str(candidate).casefold() if os.name == "nt" else str(candidate)
                if key not in seen and (candidate / ".goal-agent" / "config.yaml").exists():
                    seen.add(key)
                    found.append({"path": str(candidate), "title": candidate.name})
                    if len(found) >= max_results:
                        return found
                dirnames.remove(".goal-agent")
            if depth >= max_depth:
                dirnames.clear()
    return found


def create_app(
    project_dir: Path | str | None = None,
    *,
    registry_path: Path | str | None = None,
) -> FastAPI:
    center = ControlCenter(project_dir, registry_path=registry_path)
    proposal_jobs = ProposalJobManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        active = center.registry.active()
        if active is not None:
            try:
                await center.runtime(active.id)
            except HTTPException:
                pass
        yield
        await proposal_jobs.shutdown()
        await center.shutdown()

    app = FastAPI(
        title="Goal Agent Control Center",
        version="0.6.7",
        lifespan=lifespan,
    )
    app.state.control_center = center
    app.state.proposal_jobs = proposal_jobs

    @app.get("/api/projects")
    async def list_projects() -> dict[str, Any]:
        active = center.registry.active()
        return {
            "projects": center.list_projects(),
            "active_project_id": active.id if active else None,
            "registry_path": str(center.registry.path),
        }

    @app.get("/api/system/locations")
    async def system_locations(request: Request) -> dict[str, Any]:
        _require_local_request(request)
        return {
            "home": str(Path.home().resolve()),
            "default_projects_dir": str(_default_projects_dir().resolve()),
            "locations": _quick_locations(center),
            "native_picker_available": True,
        }

    @app.get("/api/system/path-info")
    async def path_info(request: Request, path: str = Query(default="")) -> dict[str, Any]:
        _require_local_request(request)
        return _path_info(_normalize_user_path(path, fallback=_default_projects_dir()))

    @app.get("/api/system/folders")
    async def browse_folders(
        request: Request,
        path: str = Query(default=""),
        search: str = Query(default=""),
        include_hidden: bool = Query(default=False),
    ) -> dict[str, Any]:
        _require_local_request(request)
        current = _normalize_user_path(path, fallback=_default_projects_dir())
        if not current.exists():
            current = _nearest_existing_directory(current)
        if not current.is_dir():
            raise HTTPException(status_code=422, detail=f"Not a folder: {current}")
        query = search.casefold().strip()
        directories: list[dict[str, Any]] = []
        try:
            children = sorted(
                (child for child in current.iterdir() if child.is_dir()),
                key=lambda child: child.name.casefold(),
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=f"Cannot read {current}: {exc}") from exc
        for child in children:
            if not include_hidden and child.name.startswith("."):
                continue
            if query and query not in child.name.casefold():
                continue
            directories.append(
                {
                    "name": child.name,
                    "path": str(child.resolve()),
                    "initialized_project": (child / ".goal-agent" / "config.yaml").exists(),
                }
            )
        return {
            **_path_info(current),
            "parent": str(current.parent.resolve()) if current.parent != current else None,
            "directories": directories,
            "locations": _quick_locations(center),
        }

    @app.post("/api/system/folders/pick")
    async def pick_folder(request: Request, payload: FolderPickRequest) -> dict[str, Any]:
        _require_local_request(request)
        initial = _normalize_user_path(payload.initial_path, fallback=_default_projects_dir())
        try:
            selected = await asyncio.to_thread(
                _native_pick_directory, initial, payload.title, payload.must_exist
            )
        except Exception as exc:
            raise HTTPException(status_code=501, detail=str(exc)) from exc
        return {"selected": selected, "cancelled": selected is None}

    @app.post("/api/system/folders/create", status_code=201)
    async def create_folder(request: Request, payload: FolderCreateRequest) -> dict[str, Any]:
        _require_local_request(request)
        target = _normalize_user_path(payload.path)
        try:
            target.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=f"Folder already exists: {target}") from exc
        except OSError as exc:
            raise HTTPException(status_code=422, detail=f"Could not create {target}: {exc}") from exc
        return _path_info(target)

    @app.post("/api/system/reveal")
    async def reveal_path(request: Request, payload: RevealPathRequest) -> dict[str, str]:
        _require_local_request(request)
        target = _normalize_user_path(payload.path)
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"Path does not exist: {target}")
        try:
            await asyncio.to_thread(_reveal_path, target)
        except Exception as exc:
            raise HTTPException(status_code=501, detail=f"Could not open folder: {exc}") from exc
        return {"path": str(target)}

    @app.post("/api/system/projects/discover")
    async def discover_projects(
        request: Request, payload: ProjectDiscoveryRequest
    ) -> dict[str, Any]:
        _require_local_request(request)
        roots = [
            _normalize_user_path(value)
            for value in payload.roots
            if str(value).strip()
        ]
        if not roots:
            roots = [Path(item["path"]) for item in _quick_locations(center) if item["kind"] in {"projects", "documents", "current"}]
        discovered = await asyncio.to_thread(
            _discover_projects, roots, payload.max_depth, payload.max_results
        )
        registered_paths = {
            str(entry.project_path).casefold() if os.name == "nt" else str(entry.project_path)
            for entry in center.registry.list()
        }
        for item in discovered:
            key = item["path"].casefold() if os.name == "nt" else item["path"]
            item["registered"] = key in registered_paths
        return {"projects": discovered, "roots": [str(root) for root in roots]}

    @app.post("/api/projects/create", status_code=201)
    async def create_project(request: ProjectCreateRequest) -> dict[str, Any]:
        path = _normalize_user_path(request.path)
        path.mkdir(parents=True, exist_ok=True)
        existing_id = center.registry.project_id_for_path(path)
        existing_runtime = center.runtimes.get(existing_id)
        if request.force and existing_runtime is not None:
            if existing_runtime.supervisor.active_goal_ids():
                raise HTTPException(
                    status_code=409,
                    detail="Stop all active goals before replacing this project's workspace",
                )
            await existing_runtime.supervisor.shutdown()
            center.runtimes.pop(existing_id, None)
        store = ProjectStore(path)
        try:
            store.initialize(model=request.model, force=request.force)
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        entry = center.registry.add(path, title=request.title or path.name, activate=True)
        runtime = await center.runtime(entry.id)
        return await _project_detail(runtime)

    @app.post("/api/projects/open", status_code=201)
    async def open_project(request: ProjectOpenRequest) -> dict[str, Any]:
        path = _normalize_user_path(request.path)
        store = ProjectStore.discover(path)
        try:
            store.require_project_initialized()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        entry = center.registry.add(
            store.project_dir, title=request.title or store.project_dir.name, activate=True
        )
        runtime = await center.runtime(entry.id)
        return await _project_detail(runtime)

    @app.post("/api/projects/{project_id}/activate")
    async def activate_project(project_id: str) -> dict[str, Any]:
        try:
            center.registry.activate(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return await _project_detail(await center.runtime(project_id))

    @app.patch("/api/projects/{project_id}")
    async def patch_project(project_id: str, request: ProjectMetadataPatch) -> dict[str, Any]:
        entry = center.resolve_entry(project_id)
        updated = center.registry.add(entry.project_path, title=request.title, activate=False)
        runtime = center.runtimes.get(project_id)
        if runtime:
            runtime.entry = updated
        return updated.model_dump(mode="json")

    @app.delete("/api/projects/{project_id}", status_code=204)
    async def unregister_project(project_id: str) -> None:
        await center.unregister(project_id)

    @app.get("/api/project")
    async def get_project(project_id: str | None = Query(default=None)) -> dict[str, Any]:
        return await _project_detail(await center.runtime(project_id))

    @app.patch("/api/project/config")
    async def patch_config(
        request: ConfigPatch, project_id: str | None = Query(default=None)
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        config = runtime.store.read_config()
        updates = request.model_dump(exclude_unset=True)
        for key, value in updates.items():
            setattr(config, key, value)
        config.project_dir = str(runtime.store.project_dir)
        validated = AppConfig.model_validate(config.model_dump())
        runtime.store.write_config(validated)
        return validated.model_dump(mode="json")

    @app.post("/api/project/validate")
    async def validate_project(project_id: str | None = Query(default=None)) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        return _validate_project(runtime.store)

    @app.get("/api/project/files")
    async def project_files(project_id: str | None = Query(default=None)) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = runtime.store
        goals: dict[str, Any] = {}
        for goal_id in store.list_goal_ids(include_archived=True):
            goal = store.for_goal(goal_id)
            goals[goal_id] = _goal_paths(goal)
        return {
            "registry": _display_path(center.registry.path),
            "project": _display_path(store.project_dir),
            "config": _display_path(store.config_path),
            "agent_root": _display_path(store.root),
            "goals": goals,
        }

    @app.get("/api/models")
    async def get_models(project_id: str | None = Query(default=None)) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        try:
            models = await OpenCodeRunner(runtime.store.read_config()).list_models()
        except OpenCodeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {"models": models}

    @app.get("/api/goals")
    async def list_goals(
        include_archived: bool = False, project_id: str | None = Query(default=None)
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        summaries = runtime.store.list_goal_summaries(
            active_goal_ids=runtime.supervisor.active_goal_ids(), include_archived=include_archived
        )
        return {"goals": [item.model_dump(mode="json") for item in summaries]}

    @app.post("/api/goals", status_code=201)
    async def create_goal(
        request: GoalCreateRequest, project_id: str | None = Query(default=None)
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        try:
            store = runtime.store.create_goal(
                request.id,
                title=request.title,
                goal=request.goal or None,
                description=request.description,
            )
        except (ValueError, FileExistsError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return await _goal_detail(store, runtime.supervisor)

    @app.get("/api/goals/{goal_id}")
    async def get_goal(
        goal_id: str, project_id: str | None = Query(default=None)
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        return await _goal_detail(store, runtime.supervisor)

    @app.patch("/api/goals/{goal_id}/metadata")
    async def patch_goal_metadata(
        goal_id: str,
        request: GoalMetadataPatch,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        current = store.read_metadata()
        updated = GoalMetadata(
            id=current.id,
            title=request.title.strip(),
            description=request.description.strip(),
            archived=request.archived,
            created_at=current.created_at,
            updated_at=current.updated_at,
        )
        store.write_metadata(updated)
        store.append_event(EventRecord(type="metadata_modified", message=updated.title))
        return updated.model_dump(mode="json")

    @app.put("/api/goals/{goal_id}/goal")
    async def update_goal(
        goal_id: str,
        request: GoalTextRequest,
        project_id: str | None = Query(default=None),
    ) -> dict[str, str]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        if not request.goal.strip():
            raise HTTPException(status_code=422, detail="Goal cannot be empty")
        store.write_goal(request.goal)
        store.append_event(EventRecord(type="goal_modified", message=request.goal.strip()))
        return {"goal": store.read_goal()}

    @app.put("/api/goals/{goal_id}/criteria")
    async def update_criteria(
        goal_id: str,
        request: CriteriaUpdateRequest,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        current = store.read_criteria()
        document = CriteriaDocument(revision=current.revision + 1, criteria=request.criteria)
        store.write_criteria(document)
        store.append_event(
            EventRecord(
                type="criteria_modified",
                message=f"Criteria revision {document.revision} saved",
                data={"count": len(document.criteria)},
            )
        )
        return document.model_dump(mode="json")

    @app.post("/api/goals/{goal_id}/criteria-revisions/{criterion_id}/apply")
    async def apply_criteria_revision(
        goal_id: str,
        criterion_id: str,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Apply one strategist proposal, then restart that goal from a clean serial state."""

        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        state = store.load_state()
        suggestion = next(
            (
                item
                for item in state.criteria_revision_suggestions
                if item.criterion_id == criterion_id
            ),
            None,
        )
        if suggestion is None:
            raise HTTPException(status_code=404, detail="Criteria revision suggestion was not found")
        current = store.read_criteria()
        try:
            document, already_current = build_criteria_revision(current, suggestion)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        if already_current:
            state.criteria_revision_suggestions = [
                item for item in state.criteria_revision_suggestions if item.criterion_id != criterion_id
            ]
            state.message = f"Dismissed already-applied criteria revision for {criterion_id}"
            store.save_state(state)
            store.append_event(
                EventRecord(
                    type="criteria_revision_already_current",
                    message=f"Dismissed already-applied criteria revision for {criterion_id}",
                    data={"criterion_id": criterion_id, "revision": current.revision},
                )
            )
            return await _goal_detail(store, runtime.supervisor)

        store.write_criteria(document)

        state.criteria_revision_suggestions = [
            item for item in state.criteria_revision_suggestions if item.criterion_id != criterion_id
        ]
        state.serial_target_criterion = None
        state.serial_consecutive_no_progress = 0
        state.consecutive_no_progress = 0
        state.serial_strict_recovery = False
        state.last_error = None
        state.message = f"Applied criteria revision for {criterion_id}; restarting"
        store.save_state(state)
        store.append_event(
            EventRecord(
                type="criteria_revision_applied",
                message=f"Applied approved criteria revision for {criterion_id}",
                data={"criterion_id": criterion_id, "revision": document.revision},
            )
        )
        try:
            await runtime.supervisor.restart(
                goal_id, note=f"Applied criteria revision for {criterion_id}; restarted from GUI"
            )
        except ConcurrencyLimitReached as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return await _goal_detail(store, runtime.supervisor)

    @app.post("/api/goals/{goal_id}/criteria-revisions/{criterion_id}/dismiss")
    async def dismiss_criteria_revision(
        goal_id: str,
        criterion_id: str,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Dismiss an advisory revision without changing criteria or restarting work."""

        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        state = store.load_state()
        if not any(
            item.criterion_id == criterion_id
            for item in state.criteria_revision_suggestions
        ):
            raise HTTPException(status_code=404, detail="Criteria revision suggestion was not found")
        state.criteria_revision_suggestions = [
            item
            for item in state.criteria_revision_suggestions
            if item.criterion_id != criterion_id
        ]
        state.serial_target_criterion = None
        state.serial_consecutive_no_progress = 0
        state.consecutive_no_progress = 0
        state.serial_strict_recovery = False
        state.message = f"Ignored criteria revision suggestion for {criterion_id}; ready to resume"
        store.save_state(state)
        store.append_event(
            EventRecord(
                type="criteria_revision_dismissed",
                message=f"Ignored criteria revision suggestion for {criterion_id}",
                data={"criterion_id": criterion_id},
            )
        )
        return await _goal_detail(store, runtime.supervisor)

    @app.post("/api/goals/{goal_id}/setup-complete")
    async def setup_complete(
        goal_id: str, project_id: str | None = Query(default=None)
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        store.update_control(desired_state="paused", note="Setup complete from GUI")
        state = store.load_state()
        state.message = "Setup complete; paused"
        store.save_state(state)
        store.append_event(EventRecord(type="setup_complete", message="GUI setup completed"))
        return await _goal_detail(store, runtime.supervisor)

    @app.post("/api/goals/{goal_id}/steering")
    async def add_steering(
        goal_id: str,
        request: SteeringRequest,
        project_id: str | None = Query(default=None),
    ) -> dict[str, str]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        store.append_steering(request.message)
        store.append_event(EventRecord(type="user_steering", message=request.message))
        return {"steering": store.read_steering()}

    @app.get("/api/goals/{goal_id}/runs")
    async def list_runs(
        goal_id: str,
        project_id: str | None = Query(default=None),
        limit: int = Query(default=5, ge=1, le=50),
    ) -> dict[str, Any]:
        """Return a summary of recent iteration run artifacts."""
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        runs_dir = store.runs_dir
        if not runs_dir.exists():
            return {"runs": []}
        iterations = sorted(
            [d for d in runs_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
            reverse=True,
        )[:limit]
        runs = []
        for d in reversed(iterations):
            files = sorted(f.name for f in d.iterdir() if f.is_file())
            runs.append({"iteration": d.name, "files": files})
        return {"runs": runs}

    @app.get("/api/goals/{goal_id}/runs/{iteration}/{filename:path}")
    async def get_run_artifact(
        goal_id: str,
        iteration: str,
        filename: str,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        """Return the text content of a specific run artifact."""
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        # Validate iteration name to prevent path traversal
        if not iteration.startswith("iteration-") or "/" in iteration or ".." in iteration:
            raise HTTPException(status_code=400, detail="Invalid iteration name")
        if ".." in filename or filename.startswith("/"):
            raise HTTPException(status_code=400, detail="Invalid filename")
        path = store.runs_dir / iteration / filename
        if not path.exists() or not path.is_file():
            raise HTTPException(status_code=404, detail="Artifact not found")
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"iteration": iteration, "filename": filename, "content": content}

    @app.post("/api/goals/{goal_id}/action")
    async def goal_action(
        goal_id: str,
        request: ActionRequest,
        project_id: str | None = Query(default=None),
    ) -> dict[str, str]:
        runtime = await center.runtime(project_id)
        _goal_store(runtime.store, goal_id)
        try:
            if request.action == "start":
                await runtime.supervisor.start(goal_id)
            elif request.action == "resume":
                await runtime.supervisor.resume(goal_id)
            elif request.action == "pause":
                await runtime.supervisor.pause(goal_id)
            else:
                await runtime.supervisor.stop(goal_id)
        except ConcurrencyLimitReached as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "goal_id": goal_id,
            "action": request.action,
            "task": runtime.supervisor.task_status(goal_id),
        }

    @app.post("/api/goals/actions")
    async def bulk_action(
        request: BulkActionRequest, project_id: str | None = Query(default=None)
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        if request.action == "start":
            return {"results": await runtime.supervisor.start_all()}
        if request.action == "pause":
            await runtime.supervisor.pause_all()
        else:
            await runtime.supervisor.stop_all()
        return {"action": request.action}

    @app.put("/api/goals/{goal_id}/model")
    async def set_goal_model(
        goal_id: str,
        request: GoalModelRequest,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        control = store.update_control(
            model_override=request.model,
            note=f"Goal model changed to {request.model or 'project default'}",
        )
        return control.model_dump(mode="json")

    @app.put("/api/goals/{goal_id}/criteria/{criterion_id}/override")
    async def set_criterion_override(
        goal_id: str,
        criterion_id: str,
        request: OverrideRequest,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        document = store.read_criteria()
        criterion = next((item for item in document.criteria if item.id == criterion_id), None)
        if criterion is None:
            raise HTTPException(status_code=404, detail=f"Unknown criterion: {criterion_id}")
        criterion.override = request.value
        document.revision += 1
        store.write_criteria(document)
        store.append_event(
            EventRecord(type="criterion_override", message=f"{criterion_id} -> {request.value.value}")
        )
        return document.model_dump(mode="json")


    @app.get("/api/goals/{goal_id}/refinement-session")
    async def get_refinement_session(
        goal_id: str,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        return store.read_refinement_session().model_dump(mode="json")

    @app.post("/api/goals/{goal_id}/refinement-session/reset")
    async def reset_refinement_session(
        goal_id: str,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        session = store.reset_refinement_session()
        store.append_event(
            EventRecord(type="refinement_reset", message="Goal refinement conversation reset")
        )
        return session.model_dump(mode="json")

    @app.post("/api/goals/{goal_id}/refinement-session/finalize")
    async def finalize_refinement_session(
        goal_id: str,
        request: FinalizeRefinementRequest,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        session = store.read_refinement_session()
        if session.current_proposal is None:
            raise HTTPException(status_code=409, detail="No refinement proposal is available to finalize")
        proposal = assess_setup_proposal(
            session.current_proposal,
            project_path=store.read_config().project_path,
        )
        blockers = [item for item in proposal.criteria_quality_issues if item.severity == "blocking"]
        if not request.force and (
            not proposal.ready_to_finalize or proposal.clarifying_questions or blockers
        ):
            details = [proposal.readiness_reason]
            details.extend(item.issue for item in blockers[:5])
            raise HTTPException(
                status_code=409,
                detail="The proposal is not ready to finalize: " + " ".join(item for item in details if item),
            )

        current = store.read_criteria()
        store.write_goal(proposal.refined_goal)
        store.write_criteria(
            CriteriaDocument(revision=current.revision + 1, criteria=proposal.criteria)
        )
        store.update_control(
            desired_state="paused",
            note="Goal and success criteria finalized through AI refinement",
        )
        state = store.load_state()
        state.message = "Goal and criteria finalized; paused"
        store.save_state(state)
        session.current_proposal = proposal
        session.status = "finalized"
        session.revision += 1
        session.finalized_at = utc_now()
        session.messages.append(
            RefinementMessage(
                role="system",
                content="The user finalized and saved this goal and its success criteria.",
            )
        )
        store.write_refinement_session(session)
        store.append_event(
            EventRecord(
                type="refinement_finalized",
                message="Goal and success criteria finalized",
                data={"criteria_count": len(proposal.criteria)},
            )
        )
        return {
            "session": session.model_dump(mode="json"),
            "goal": store.read_goal(),
            "criteria": store.read_criteria().model_dump(mode="json"),
        }

    @app.post("/api/goals/{goal_id}/proposal-jobs", status_code=202)
    async def create_proposal_job(
        goal_id: str,
        request: ProposalJobRequest,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        config = store.read_config()
        control = store.read_control()
        session = store.read_refinement_session()
        feedback = request.feedback.strip()
        if feedback:
            session.messages.append(RefinementMessage(role="user", content=feedback))
        elif not session.messages:
            session.messages.append(
                RefinementMessage(
                    role="user",
                    content=(
                        "Help me refine this goal and create concrete success criteria. "
                        "Ask any material clarifying questions before finalizing."
                    ),
                )
            )
        if session.started_at is None:
            session.started_at = utc_now()
        session.status = "refining"
        session.revision += 1
        store.write_refinement_session(session)

        async def worker(status_callback: StatusCallback) -> dict[str, Any]:
            runner = OpenCodeRunner(config)
            current_session = store.read_refinement_session()
            saved_goal = store.read_goal()
            saved_criteria = store.read_criteria()
            project_snapshot = collect_project_snapshot(config.project_path)
            prompt_context = build_refinement_context(
                session=current_session,
                saved_goal=saved_goal,
                saved_criteria=saved_criteria,
                mode=request.mode,
                legacy_context=request.conversation,
                aggressive=False,
            )
            store.write_refinement_session(current_session)
            status_callback(
                "context_budget",
                f"Using about {prompt_context.estimated_input_tokens:,} input tokens; "
                f"{prompt_context.compacted_message_count} older messages compacted",
            )

            try:
                proposal, _ = await runner.run_structured(
                    setup_prompt(
                        saved_goal,
                        prompt_context.transcript,
                        project_snapshot=project_snapshot,
                    ),
                    SetupProposal,
                    model=control.model_override or config.model,
                    agent=config.strategist_agent,
                    title=f"Refine goal and criteria: {goal_id}",
                    status_callback=status_callback,
                    attempts=4,
                    profile="refinement",
                )
            except OpenCodeContextOverflowError as first_overflow:
                # OpenCode's own tool/file history can exhaust the provider context
                # even when Goal Agent's stdin prompt is bounded. Retry once with a
                # smaller transcript and explicit no-broad-inspection instructions.
                current_session = store.read_refinement_session()
                current_session.context_overflow_retries += 1
                compact_context = build_refinement_context(
                    session=current_session,
                    saved_goal=saved_goal,
                    saved_criteria=saved_criteria,
                    mode=request.mode,
                    legacy_context="",
                    aggressive=True,
                )
                store.write_refinement_session(current_session)
                status_callback(
                    "context_retry",
                    f"The model context filled ({first_overflow}). Retrying with about "
                    f"{compact_context.estimated_input_tokens:,} input tokens and restricted project inspection.",
                )
                try:
                    proposal, _ = await runner.run_structured(
                        setup_prompt(
                            saved_goal,
                            compact_context.transcript,
                            low_context=True,
                            project_snapshot=project_snapshot,
                        ),
                        SetupProposal,
                        model=control.model_override or config.model,
                        agent=config.strategist_agent,
                        title=f"Refine goal and criteria (compact): {goal_id}",
                        status_callback=status_callback,
                        attempts=3,
                        profile="refinement",
                    )
                except OpenCodeContextOverflowError as second_overflow:
                    requested = second_overflow.requested_tokens
                    context_size = second_overflow.context_size
                    numbers = (
                        f" The retry requested {requested:,} tokens against a {context_size:,}-token window."
                        if requested is not None and context_size is not None
                        else ""
                    )
                    raise OpenCodeError(
                        "Goal refinement exceeded the model context window even after Goal Agent "
                        "compacted the saved conversation and restricted project inspection."
                        + numbers
                        + " For a local llama.cpp server, increase its context setting (for example "
                        "from -c 65536 to -c 131072 if memory permits), or use a model/server with "
                        "a larger context window. The refinement conversation was preserved."
                    ) from second_overflow

            proposal = assess_setup_proposal(proposal, project_path=config.project_path)
            latest = store.read_refinement_session()
            assistant_text = proposal.assistant_message.strip() or proposal.readiness_reason
            if proposal.clarifying_questions:
                assistant_text += "\n\nQuestions:\n" + "\n".join(
                    f"- {question}" for question in proposal.clarifying_questions
                )
            latest.messages.append(
                RefinementMessage(role="assistant", content=assistant_text.strip())
            )
            latest.current_proposal = proposal
            latest.status = "ready" if proposal.ready_to_finalize else "refining"
            latest.revision += 1
            store.write_refinement_session(latest)
            payload = proposal.model_dump(mode="json")
            payload["refinement_session"] = latest.model_dump(mode="json")
            payload["context_info"] = {
                "mode": latest.last_context_mode,
                "estimated_input_tokens": latest.last_estimated_input_tokens,
                "prompt_chars": latest.last_prompt_chars,
                "compacted_message_count": latest.compacted_message_count,
                "overflow_retries": latest.context_overflow_retries,
            }
            return payload

        job = proposal_jobs.start(
            project_id=runtime.entry.id,
            goal_id=goal_id,
            mode=request.mode,
            worker=worker,
        )
        return job.to_dict()

    @app.get("/api/proposal-jobs/{job_id}")
    async def get_proposal_job(
        job_id: str,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        entry = center.resolve_entry(project_id)
        try:
            job = proposal_jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if job.project_id != entry.id:
            raise HTTPException(status_code=404, detail="Proposal job not found in this project")
        return job.to_dict()

    @app.delete("/api/proposal-jobs/{job_id}")
    async def cancel_proposal_job(
        job_id: str,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        entry = center.resolve_entry(project_id)
        try:
            job = proposal_jobs.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if job.project_id != entry.id:
            raise HTTPException(status_code=404, detail="Proposal job not found in this project")
        return (await proposal_jobs.cancel(job_id)).to_dict()

    @app.post("/api/goals/{goal_id}/refine-goal")
    async def refine_goal(
        goal_id: str,
        request: RefineGoalRequest,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        config = store.read_config()
        control = store.read_control()
        context = request.conversation.strip()
        if request.feedback.strip():
            context += f"\nUser feedback: {request.feedback.strip()}"
        try:
            proposal, _ = await OpenCodeRunner(config).run_structured(
                setup_prompt(
                    store.read_goal(),
                    context,
                    project_snapshot=collect_project_snapshot(config.project_path),
                ),
                SetupProposal,
                model=control.model_override or config.model,
                agent=config.strategist_agent,
                title=f"Refine goal: {goal_id}",
                attempts=4,
                profile="refinement",
            )
        except OpenCodeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return proposal.model_dump(mode="json")

    @app.post("/api/goals/{goal_id}/refine-criteria")
    async def refine_criteria(
        goal_id: str,
        request: RefineCriteriaRequest,
        project_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        runtime = await center.runtime(project_id)
        store = _goal_store(runtime.store, goal_id)
        config = store.read_config()
        control = store.read_control()
        try:
            document, _ = await OpenCodeRunner(config).run_structured(
                criteria_refinement_prompt(
                    store.read_goal(), store.read_criteria(), request.feedback
                ),
                CriteriaDocument,
                model=control.model_override or config.model,
                agent=config.strategist_agent,
                title=f"Refine criteria: {goal_id}",
                profile="analysis",
            )
        except OpenCodeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return document.model_dump(mode="json")

    @app.delete("/api/goals/{goal_id}", status_code=204)
    async def delete_goal(
        goal_id: str,
        project_id: str | None = Query(default=None),
        force: bool = Query(default=False),
    ) -> None:
        runtime = await center.runtime(project_id)
        if runtime.supervisor.task_status(goal_id) == "active":
            raise HTTPException(
                status_code=409,
                detail="Stop this goal and wait for its loop task to exit before deleting it",
            )
        try:
            runtime.store.delete_goal(goal_id, force=force)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(ASSET_DIR / "index.html")

    @app.get("/{asset_path:path}")
    async def assets(asset_path: str) -> FileResponse:
        candidate = (ASSET_DIR / asset_path).resolve()
        if not str(candidate).startswith(str(ASSET_DIR.resolve())) or not candidate.is_file():
            return FileResponse(ASSET_DIR / "index.html")
        return FileResponse(candidate)

    return app


async def _project_detail(runtime: ProjectRuntime) -> dict[str, Any]:
    config = runtime.store.read_config()
    return {
        "id": runtime.entry.id,
        "title": runtime.entry.title,
        "project_dir": _display_path(runtime.store.project_dir),
        "agent_root": _display_path(runtime.store.root),
        "config": config.model_dump(mode="json"),
        "active_goal_ids": sorted(runtime.supervisor.active_goal_ids()),
        "paths": {
            "project": _display_path(runtime.store.project_dir),
            "agent_root": _display_path(runtime.store.root),
            "config": _display_path(runtime.store.config_path),
        },
    }


def _validate_project(project_store: ProjectStore) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    config = project_store.read_config()
    executable = config.opencode_command[0] if config.opencode_command else ""
    resolution = resolve_executable(executable)
    executable_ok = resolution.found
    if executable_ok:
        executable_detail = (
            f"{resolution.path} ({resolution.kind}; resolved from {resolution.source})"
        )
    else:
        executable_detail = f"Not found: {executable or '(empty command)'}"
        if os.name == "nt":
            executable_detail += (
                ". Run Get-Command opencode | Format-List CommandType,Source,Path in PowerShell."
            )
    checks.append(
        {
            "name": "OpenCode executable",
            "passed": executable_ok,
            "detail": executable_detail,
        }
    )
    goal_results: dict[str, Any] = {}
    for goal_id in project_store.list_goal_ids(include_archived=True):
        store = project_store.for_goal(goal_id)
        errors: list[str] = []
        warnings: list[str] = []
        try:
            goal = store.read_goal()
            if not goal:
                errors.append("Goal is empty")
        except Exception as exc:
            errors.append(f"Goal cannot be read: {exc}")
        try:
            criteria = store.read_criteria()
            if not criteria.criteria:
                errors.append("No criteria are defined")
            elif not any(item.required for item in criteria.criteria):
                errors.append("No required criteria are defined")
            human_only = [
                item.id
                for item in criteria.criteria
                if item.required
                and item.kind.value == "manual"
                and item.override.value == "auto"
            ]
            if human_only:
                warnings.append(
                    "Required human-only criteria cannot pass autonomously: "
                    + ", ".join(human_only)
                    + ". Change them to AI evidence review or provide a human override."
                )
        except Exception as exc:
            errors.append(f"Criteria cannot be read: {exc}")
        try:
            store.read_control()
        except Exception as exc:
            errors.append(f"Control state cannot be read: {exc}")
        goal_results[goal_id] = {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        }
    checks.append(
        {
            "name": "Goal definitions",
            "passed": bool(goal_results) and all(item["valid"] for item in goal_results.values()),
            "detail": f"{sum(1 for item in goal_results.values() if item['valid'])}/{len(goal_results)} valid",
        }
    )
    return {
        "valid": all(item["passed"] for item in checks),
        "checks": checks,
        "goals": goal_results,
        "model": config.model or "OpenCode default",
        "project_dir": str(project_store.project_dir),
    }


def _goal_store(project_store: ProjectStore, goal_id: str) -> ProjectStore:
    try:
        store = project_store.for_goal(goal_id)
        store.require_initialized()
        return store
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _goal_paths(store: ProjectStore) -> dict[str, str]:
    return {
        "metadata": _display_path(store.metadata_path),
        "goal": _display_path(store.goal_path),
        "criteria": _display_path(store.criteria_path),
        "steering": _display_path(store.steering_path),
        "refinement": _display_path(store.refinement_path),
        "control": _display_path(store.control_path),
        "status": _display_path(store.status_markdown_path),
        "agents": _display_path(store.agents_path),
        "criteria_status": _display_path(store.criteria_status_path),
        "evaluation_analysis": _display_path(store.evaluation_analysis_path),
        "hypotheses": _display_path(store.hypotheses_path),
        "events": _display_path(store.events_path),
        "runs": _display_path(store.runs_dir),
    }


def _display_path(path: Path) -> str:
    """Serialize filesystem paths in a platform-stable slash format for clients."""

    return path.as_posix()


async def _goal_detail(store: ProjectStore, supervisor: GoalSupervisor) -> dict[str, Any]:
    metadata = store.read_metadata()
    config = store.read_config()
    control = store.read_control()
    return {
        "metadata": metadata.model_dump(mode="json"),
        "goal": store.read_goal(),
        "criteria": store.read_criteria().model_dump(mode="json"),
        "control": control.model_dump(mode="json"),
        "state": store.load_state().model_dump(mode="json"),
        "steering": store.read_steering(),
        "refinement": store.read_refinement_session().model_dump(mode="json"),
        "events": store.read_events(limit=150),
        "effective_model": control.model_override or config.model,
        "task_status": supervisor.task_status(store.goal_id),
        "paths": _goal_paths(store),
    }
