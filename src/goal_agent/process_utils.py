from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from typing import Any


def process_group_kwargs() -> dict[str, Any]:
    """Start a subprocess in a group that can be terminated as one tree."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


async def terminate_process_tree(
    process: asyncio.subprocess.Process,
    *,
    grace_seconds: float = 5.0,
) -> None:
    if process.returncode is not None:
        return

    if os.name == "nt":
        # taskkill /T includes commands OpenCode launched beneath itself.
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
        try:
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
        except asyncio.TimeoutError:
            process.kill()
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_seconds)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()
