from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from uuid import uuid4

StatusCallback = Callable[[str, str], None]
ProposalWorker = Callable[[StatusCallback], Awaitable[dict[str, Any]]]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ProposalJob:
    id: str
    project_id: str
    goal_id: str
    mode: str
    status: str = "queued"
    stage: str = "queued"
    detail: str = "Waiting to start"
    created_at: str = field(default_factory=_utc_iso)
    updated_at: str = field(default_factory=_utc_iso)
    result: dict[str, Any] | None = None
    error: str | None = None
    task: asyncio.Task[None] | None = field(default=None, repr=False)

    def update(self, stage: str, detail: str) -> None:
        self.stage = stage
        self.detail = detail
        self.updated_at = _utc_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "goal_id": self.goal_id,
            "mode": self.mode,
            "status": self.status,
            "stage": self.stage,
            "detail": self.detail,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result": self.result,
            "error": self.error,
        }


class ProposalJobManager:
    """Runs GUI proposal generation without holding an HTTP request open."""

    def __init__(self) -> None:
        self.jobs: dict[str, ProposalJob] = {}

    def start(
        self,
        *,
        project_id: str,
        goal_id: str,
        mode: str,
        worker: ProposalWorker,
    ) -> ProposalJob:
        job = ProposalJob(
            id=uuid4().hex,
            project_id=project_id,
            goal_id=goal_id,
            mode=mode,
        )
        self.jobs[job.id] = job

        async def execute() -> None:
            job.status = "running"
            job.update("starting", "Starting OpenCode")

            def status_callback(stage: str, detail: str) -> None:
                if job.status == "running":
                    job.update(stage, detail)

            try:
                job.result = await worker(status_callback)
                job.status = "completed"
                job.update("completed", "Proposal is ready")
            except asyncio.CancelledError:
                job.status = "cancelled"
                job.update("cancelled", "Proposal generation was cancelled")
                raise
            except Exception as exc:  # surfaced through the polling endpoint
                job.status = "failed"
                job.error = str(exc)
                job.update("failed", "Proposal generation failed")

        job.task = asyncio.create_task(execute(), name=f"proposal-{goal_id}-{job.id[:8]}")
        return job

    def get(self, job_id: str) -> ProposalJob:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"Unknown proposal job: {job_id}") from exc

    async def cancel(self, job_id: str) -> ProposalJob:
        job = self.get(job_id)
        if job.task and not job.task.done():
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
        elif job.status in {"queued", "running"}:
            job.status = "cancelled"
            job.update("cancelled", "Proposal generation was cancelled")
        return job

    async def shutdown(self) -> None:
        tasks = [job.task for job in self.jobs.values() if job.task and not job.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
