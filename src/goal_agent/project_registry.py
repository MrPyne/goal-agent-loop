from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from filelock import FileLock
from pydantic import BaseModel, Field

from .models import utc_now
from .storage import ProjectStore


class ProjectEntry(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    title: str
    path: str
    created_at: datetime = Field(default_factory=utc_now)
    last_opened_at: datetime = Field(default_factory=utc_now)

    @property
    def project_path(self) -> Path:
        return Path(self.path).expanduser().resolve()


class ProjectRegistryDocument(BaseModel):
    version: int = 1
    active_project_id: str | None = None
    projects: list[ProjectEntry] = Field(default_factory=list)


class ProjectRegistry:
    """Persistent list of projects known to the local GUI.

    The registry only stores project paths and display metadata. Project state remains
    inside each project's ``.goal-agent`` directory.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or self.default_path()).expanduser().resolve()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @staticmethod
    def default_path() -> Path:
        override = os.environ.get("GOAL_AGENT_REGISTRY_PATH")
        if override:
            return Path(override)
        if os.name == "nt":
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
            return base / "goal-agent" / "projects.yaml"
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return base / "goal-agent" / "projects.yaml"

    def read(self) -> ProjectRegistryDocument:
        if not self.path.exists():
            return ProjectRegistryDocument()
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            return ProjectRegistryDocument.model_validate(data)
        except Exception:
            # A damaged registry must not make project workspaces unusable.
            return ProjectRegistryDocument()

    def write(self, document: ProjectRegistryDocument) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(document.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
        ProjectStore.atomic_write_text(self.path, text)

    def list(self) -> list[ProjectEntry]:
        document = self.read()
        return sorted(document.projects, key=lambda item: item.last_opened_at, reverse=True)

    def get(self, project_id: str) -> ProjectEntry:
        document = self.read()
        entry = next((item for item in document.projects if item.id == project_id), None)
        if entry is None:
            raise KeyError(f"Unknown project: {project_id}")
        return entry

    def add(self, path: Path | str, *, title: str | None = None, activate: bool = True) -> ProjectEntry:
        project_path = Path(path).expanduser().resolve()
        project_id = self.project_id_for_path(project_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            document = self.read()
            now = utc_now()
            existing = next((item for item in document.projects if item.id == project_id), None)
            if existing:
                existing.path = str(project_path)
                existing.title = (title or existing.title or project_path.name).strip()
                existing.last_opened_at = now
                entry = existing
            else:
                entry = ProjectEntry(
                    id=project_id,
                    title=(title or project_path.name or str(project_path)).strip(),
                    path=str(project_path),
                    created_at=now,
                    last_opened_at=now,
                )
                document.projects.append(entry)
            if activate:
                document.active_project_id = entry.id
            self.write(document)
            return entry

    def remove(self, project_id: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            document = self.read()
            before = len(document.projects)
            document.projects = [item for item in document.projects if item.id != project_id]
            if len(document.projects) == before:
                raise KeyError(f"Unknown project: {project_id}")
            if document.active_project_id == project_id:
                document.active_project_id = document.projects[0].id if document.projects else None
            self.write(document)

    def activate(self, project_id: str) -> ProjectEntry:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.lock_path)):
            document = self.read()
            entry = next((item for item in document.projects if item.id == project_id), None)
            if entry is None:
                raise KeyError(f"Unknown project: {project_id}")
            entry.last_opened_at = utc_now()
            document.active_project_id = project_id
            self.write(document)
            return entry

    def active(self) -> ProjectEntry | None:
        document = self.read()
        if document.active_project_id:
            entry = next(
                (item for item in document.projects if item.id == document.active_project_id), None
            )
            if entry:
                return entry
        return document.projects[0] if document.projects else None

    @staticmethod
    def project_id_for_path(path: Path | str) -> str:
        project_path = Path(path).expanduser().resolve()
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", project_path.name).strip("-.") or "project"
        canonical = str(project_path).casefold() if os.name == "nt" else str(project_path)
        suffix = uuid.uuid5(uuid.NAMESPACE_URL, canonical).hex[:8]
        return f"{slug}-{suffix}"
