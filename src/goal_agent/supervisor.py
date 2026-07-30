from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from .loop import GoalAgentLoop, LoopAlreadyRunning
from .models import EventRecord
from .storage import ProjectStore


class ConcurrencyLimitReached(RuntimeError):
    pass


@dataclass
class TaskInfo:
    goal_id: str
    task: asyncio.Task[Any]


class GoalSupervisor:
    """Owns multiple per-goal loop tasks inside one GUI/server process."""

    def __init__(self, project_store: ProjectStore):
        self.project_store = project_store
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        self._lock = asyncio.Lock()

    def active_goal_ids(self) -> set[str]:
        self._discard_finished()
        return {goal_id for goal_id, task in self._tasks.items() if not task.done()}

    def task_status(self, goal_id: str) -> str:
        task = self._tasks.get(goal_id)
        if task is None:
            return "not-started"
        if task.cancelled():
            return "cancelled"
        if task.done():
            return "finished"
        return "active"

    async def start(self, goal_id: str, *, note: str = "Started from GUI") -> None:
        async with self._lock:
            self._discard_finished()
            store = self.project_store.for_goal(goal_id)
            store.require_initialized()
            task = self._tasks.get(goal_id)
            running_count = self._running_desired_count(exclude=goal_id)
            max_concurrent = store.read_config().max_concurrent_goals
            if task is not None and not task.done():
                if store.read_control().desired_state.value == "running":
                    return
                if running_count >= max_concurrent:
                    raise ConcurrencyLimitReached(
                        f"The project allows {max_concurrent} concurrently running goals. "
                        "Pause or stop another goal, or raise max_concurrent_goals in settings."
                    )
                store.update_control(desired_state="running", note=note)
                return

            if running_count >= max_concurrent:
                raise ConcurrencyLimitReached(
                    f"The project allows {max_concurrent} concurrently running goals. "
                    "Pause or stop another goal, or raise max_concurrent_goals in settings."
                )

            store.update_control(desired_state="running", note=note)
            task = asyncio.create_task(self._run_goal(store), name=f"goal-agent:{goal_id}")
            self._tasks[goal_id] = task

    async def pause(self, goal_id: str, *, note: str = "Paused from GUI") -> None:
        store = self.project_store.for_goal(goal_id)
        store.require_initialized()
        store.update_control(desired_state="paused", note=note)

    async def resume(self, goal_id: str) -> None:
        await self.start(goal_id, note="Resumed from GUI")

    async def stop(self, goal_id: str, *, note: str = "Stopped from GUI") -> None:
        store = self.project_store.for_goal(goal_id)
        store.require_initialized()
        store.update_control(desired_state="stopped", note=note)

    async def start_all(self) -> dict[str, str]:
        results: dict[str, str] = {}
        for goal_id in self.project_store.list_goal_ids():
            try:
                await self.start(goal_id, note="Started by Start All")
                results[goal_id] = "started"
            except ConcurrencyLimitReached as exc:
                results[goal_id] = str(exc)
                break
            except Exception as exc:
                results[goal_id] = str(exc)
        return results

    async def pause_all(self) -> None:
        for goal_id in self.project_store.list_goal_ids():
            await self.pause(goal_id, note="Paused by Pause All")

    async def stop_all(self) -> None:
        for goal_id in self.project_store.list_goal_ids():
            await self.stop(goal_id, note="Stopped by Stop All")

    async def auto_resume(self) -> None:
        config = self.project_store.read_config()
        if not config.gui_auto_resume_running_goals:
            return
        for goal_id in self.project_store.list_goal_ids():
            store = self.project_store.for_goal(goal_id)
            if store.read_control().desired_state.value != "running":
                continue
            try:
                await self.start(goal_id, note="Automatically resumed by GUI")
            except ConcurrencyLimitReached:
                break

    async def shutdown(self) -> None:
        tasks = [task for task in self._tasks.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._discard_finished()

    async def _run_goal(self, store: ProjectStore) -> None:
        try:
            await GoalAgentLoop(store).run_forever()
        except LoopAlreadyRunning as exc:
            store.append_event(
                EventRecord(type="supervisor_error", message=str(exc), data={"goal_id": store.goal_id})
            )
            raise
        except asyncio.CancelledError:
            store.append_event(
                EventRecord(
                    type="supervisor_shutdown",
                    message="GUI process stopped; running intent was preserved for restart",
                    data={"goal_id": store.goal_id},
                )
            )
            raise
        except Exception as exc:
            store.append_event(
                EventRecord(type="supervisor_error", message=str(exc), data={"goal_id": store.goal_id})
            )
            raise

    def _running_desired_count(self, *, exclude: str | None = None) -> int:
        count = 0
        for goal_id in self.project_store.list_goal_ids():
            if goal_id == exclude:
                continue
            store = self.project_store.for_goal(goal_id)
            try:
                if store.read_control().desired_state.value == "running":
                    count += 1
            except Exception:
                continue
        return count

    def _discard_finished(self) -> None:
        finished = [goal_id for goal_id, task in self._tasks.items() if task.done()]
        for goal_id in finished:
            task = self._tasks.pop(goal_id)
            try:
                task.exception()
            except (asyncio.CancelledError, Exception):
                pass
