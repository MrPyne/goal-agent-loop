from __future__ import annotations

import asyncio
import os
import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Prompt

from .command_resolver import resolve_executable
from .loop import GoalAgentLoop, LoopAlreadyRunning
from .models import AppConfig, EventRecord, OverrideMode
from .opencode import OpenCodeError, OpenCodeRunner
from .project_registry import ProjectRegistry
from .setup_wizard import SetupWizard
from .storage import DEFAULT_GOAL_ID, ProjectStore
from .supervisor import GoalSupervisor

app = typer.Typer(
    name="goal-agent",
    help="Run one or more persistent OpenCode-backed goal loops until explicit criteria pass.",
    no_args_is_help=True,
)
console = Console()

ProjectOption = Annotated[
    Path,
    typer.Option(
        "--project",
        "-p",
        help="Target project. Defaults to the current directory or nearest .goal-agent parent.",
    ),
]
GoalOption = Annotated[
    str,
    typer.Option("--goal-id", "-g", help="Goal ID inside the project workspace."),
]


def _store(project: Path | None, goal_id: str = DEFAULT_GOAL_ID) -> ProjectStore:
    return ProjectStore.discover(project, goal_id=goal_id)


@app.command()
def init(
    project: Annotated[Path, typer.Argument(help="Project the autonomous loops will work in")] = Path.cwd(),
    model: Annotated[str | None, typer.Option("--model", "-m", help="OpenCode provider/model ID")] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace the existing workspace")] = False,
) -> None:
    """Create the project workspace and its default goal."""
    store = ProjectStore(project)
    try:
        store.initialize(model=model, force=force)
    except FileExistsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Initialized[/green] {store.root}")
    console.print("Next: run [bold]goal-agent gui[/bold] or [bold]goal-agent setup[/bold].")


@app.command("goals")
def goals(project: ProjectOption = Path.cwd()) -> None:
    """List every goal and its current state."""
    store = _store(project)
    store.require_project_initialized()
    summaries = store.list_goal_summaries(include_archived=True)
    if not summaries:
        console.print("No goals exist.")
        return
    for item in summaries:
        archived = " [archived]" if item.metadata.archived else ""
        console.print(
            f"[bold]{item.metadata.id}[/bold] — {item.metadata.title}{archived} | "
            f"{item.phase.value} | {item.required_passed}/{item.required_total} required | "
            f"iteration {item.iteration}"
        )


@app.command("goal-create")
def goal_create(
    goal_id: Annotated[str, typer.Argument(help="Stable ID for the new goal")],
    project: ProjectOption = Path.cwd(),
    title: Annotated[str | None, typer.Option("--title", help="Display title")] = None,
    goal: Annotated[str | None, typer.Option("--goal", help="Initial goal text")] = None,
) -> None:
    """Add another independent goal to an initialized project."""
    store = _store(project)
    try:
        created = store.create_goal(goal_id, title=title, goal=goal)
    except (ValueError, FileExistsError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Created[/green] {created.goal_id} at {created.goal_root}")


@app.command("goal-delete")
def goal_delete(
    goal_id: Annotated[str, typer.Argument()],
    project: ProjectOption = Path.cwd(),
    force: Annotated[bool, typer.Option("--force", help="Delete even if marked running")] = False,
) -> None:
    """Permanently delete one goal and its history."""
    store = _store(project)
    try:
        store.delete_goal(goal_id, force=force)
    except (FileNotFoundError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[red]Deleted[/red] {goal_id}")


@app.command()
def setup(
    project: ProjectOption = Path.cwd(),
    goal_id: GoalOption = DEFAULT_GOAL_ID,
    goal: Annotated[str | None, typer.Option("--goal", help="Initial rough goal")] = None,
    model: Annotated[str | None, typer.Option("--model", "-m", help="OpenCode provider/model ID")] = None,
) -> None:
    """Collaboratively refine one goal and its success criteria with OpenCode."""
    store = _store(project, goal_id)
    store.require_initialized()
    try:
        asyncio.run(SetupWizard(store, console).run(rough_goal=goal, model=model))
    except (OpenCodeError, KeyboardInterrupt) as exc:
        console.print(f"[red]Setup stopped: {exc}[/red]")
        raise typer.Exit(1)


@app.command("run")
def run_loop(
    project: ProjectOption = Path.cwd(), goal_id: GoalOption = DEFAULT_GOAL_ID
) -> None:
    """Start or restart one persistent goal loop."""
    store = _store(project, goal_id)
    store.require_initialized()
    store.update_control(desired_state="running", note="Started from goal-agent run")
    loop = GoalAgentLoop(store)
    console.print(f"[green]Running goal[/green] {goal_id} in {store.project_dir}")
    console.print(f"Live status: {store.status_markdown_path}")
    try:
        state = asyncio.run(loop.run_forever())
    except LoopAlreadyRunning as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(2)
    except KeyboardInterrupt:
        store.update_control(desired_state="paused", note="Paused by Ctrl+C")
        console.print("\n[yellow]Paused.[/yellow] Run the command again to restart.")
        raise typer.Exit(130)
    console.print(f"Loop exited with phase [bold]{state.phase.value}[/bold]: {state.message}")


@app.command("run-all")
def run_all(project: ProjectOption = Path.cwd()) -> None:
    """Run multiple goals in one process, up to max_concurrent_goals."""
    store = _store(project)
    store.require_project_initialized()

    async def scenario() -> None:
        supervisor = GoalSupervisor(store)
        results = await supervisor.start_all()
        for goal_id, result in results.items():
            console.print(f"{goal_id}: {result}")
        try:
            while supervisor.active_goal_ids():
                await asyncio.sleep(1)
        finally:
            await supervisor.shutdown()

    try:
        asyncio.run(scenario())
    except KeyboardInterrupt:
        console.print("\n[yellow]Supervisor stopped. Running intent is preserved.[/yellow]")


@app.command()
def pause(
    project: ProjectOption = Path.cwd(), goal_id: GoalOption = DEFAULT_GOAL_ID
) -> None:
    """Pause one loop, including an in-progress OpenCode subprocess."""
    store = _store(project, goal_id)
    control = store.update_control(desired_state="paused", note="Paused by user")
    console.print(f"[yellow]Pause requested[/yellow] for {goal_id} (revision {control.revision})")


@app.command()
def resume(
    project: ProjectOption = Path.cwd(), goal_id: GoalOption = DEFAULT_GOAL_ID
) -> None:
    """Request running state for one goal."""
    store = _store(project, goal_id)
    control = store.update_control(desired_state="running", note="Resumed by user")
    console.print(f"[green]Running requested[/green] for {goal_id} (revision {control.revision})")


@app.command()
def stop(
    project: ProjectOption = Path.cwd(), goal_id: GoalOption = DEFAULT_GOAL_ID
) -> None:
    """Stop one loop safely; its history remains restartable."""
    store = _store(project, goal_id)
    control = store.update_control(desired_state="stopped", note="Stopped by user")
    console.print(f"[red]Stop requested[/red] for {goal_id} (revision {control.revision})")


@app.command()
def status(
    project: ProjectOption = Path.cwd(),
    goal_id: GoalOption = DEFAULT_GOAL_ID,
    watch: Annotated[bool, typer.Option("--watch", "-w", help="Refresh until Ctrl+C")] = False,
) -> None:
    """Show one goal's agents, criteria, and hypothesis status."""
    store = _store(project, goal_id)
    store.require_initialized()
    try:
        while True:
            if watch:
                console.clear()
            if store.status_markdown_path.exists():
                console.print(store.status_markdown_path.read_text(encoding="utf-8"))
            else:
                console.print("No status has been written yet.")
            if not watch:
                return
            time.sleep(1)
    except KeyboardInterrupt:
        return


@app.command()
def steer(
    message: Annotated[str, typer.Argument(help="Guidance read before the next agent step")],
    project: ProjectOption = Path.cwd(),
    goal_id: GoalOption = DEFAULT_GOAL_ID,
) -> None:
    """Append live steering to one goal without stopping it."""
    store = _store(project, goal_id)
    store.append_steering(message)
    store.append_event(EventRecord(type="user_steering", message=message))
    console.print(f"[green]Steering added[/green] to {store.steering_path}")


@app.command("set-goal")
def set_goal(
    goal: Annotated[str, typer.Argument(help="Replacement goal")],
    project: ProjectOption = Path.cwd(),
    goal_id: GoalOption = DEFAULT_GOAL_ID,
) -> None:
    """Replace one goal while preserving its state and history."""
    store = _store(project, goal_id)
    store.write_goal(goal)
    store.append_event(EventRecord(type="goal_modified", message=goal))
    console.print(f"[green]Goal updated[/green] in {store.goal_path}")


@app.command("models")
def models(project: ProjectOption = Path.cwd()) -> None:
    """List model IDs visible to the configured OpenCode installation."""
    store = _store(project)
    store.require_project_initialized()
    try:
        values = asyncio.run(OpenCodeRunner(store.read_config()).list_models())
    except OpenCodeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    for value in values:
        console.print(value)


@app.command("select-model")
def select_model(
    project: ProjectOption = Path.cwd(),
    goal_id: GoalOption = DEFAULT_GOAL_ID,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    goal_override: Annotated[
        bool,
        typer.Option("--goal-override", help="Change only this goal instead of the project default"),
    ] = False,
) -> None:
    """Choose the project default model or a per-goal override."""
    store = _store(project, goal_id)
    store.require_initialized()
    config = store.read_config()
    if model is None:
        values = asyncio.run(OpenCodeRunner(config).list_models())
        if not values:
            console.print("[red]OpenCode returned no models.[/red]")
            raise typer.Exit(1)
        for index, value in enumerate(values, start=1):
            console.print(f"{index:>3}. {value}")
        selection = int(Prompt.ask("Select model number"))
        if selection < 1 or selection > len(values):
            raise typer.BadParameter("Selection is outside the model list")
        model = values[selection - 1]
    if goal_override:
        store.update_control(model_override=model, note=f"Goal model changed to {model}")
    else:
        config.model = model
        store.write_config(config)
    console.print(f"[green]Model selected:[/green] {model}")


@app.command("criterion-override")
def criterion_override(
    criterion_id: Annotated[str, typer.Argument()],
    value: Annotated[OverrideMode, typer.Argument(help="auto, pass, or fail")],
    project: ProjectOption = Path.cwd(),
    goal_id: GoalOption = DEFAULT_GOAL_ID,
) -> None:
    """Apply or clear a human override for a success criterion."""
    store = _store(project, goal_id)
    document = store.read_criteria()
    found = False
    for criterion in document.criteria:
        if criterion.id == criterion_id:
            criterion.override = value
            found = True
            break
    if not found:
        console.print(f"[red]Unknown criterion: {criterion_id}[/red]")
        raise typer.Exit(1)
    document.revision += 1
    store.write_criteria(document)
    store.append_event(EventRecord(type="criterion_override", message=f"{criterion_id} -> {value.value}"))
    console.print(f"[green]{criterion_id} override set to {value.value}[/green]")


@app.command("files")
def files(
    project: ProjectOption = Path.cwd(), goal_id: GoalOption = DEFAULT_GOAL_ID
) -> None:
    """Print editable and generated files for one goal."""
    store = _store(project, goal_id)
    store.require_initialized()
    console.print("[bold]Project configuration[/bold]")
    console.print(store.config_path)
    console.print("\n[bold]Editable goal files[/bold]")
    for path in [store.goal_path, store.criteria_path, store.steering_path, store.control_path]:
        console.print(path)
    console.print("\n[bold]Generated status files[/bold]")
    for path in [
        store.status_markdown_path,
        store.agents_path,
        store.criteria_status_path,
        store.evaluation_analysis_path,
        store.hypotheses_path,
        store.events_path,
    ]:
        console.print(path)


@app.command()
def validate(
    project: ProjectOption = Path.cwd(), goal_id: GoalOption = DEFAULT_GOAL_ID
) -> None:
    """Validate one goal and confirm the OpenCode executable is available."""
    store = _store(project, goal_id)
    store.require_initialized()
    config = store.read_config()
    goal = store.read_goal()
    criteria = store.read_criteria()
    store.read_control()
    executable = config.opencode_command[0] if config.opencode_command else ""
    resolution = resolve_executable(executable)
    if not resolution.found:
        console.print(f"[red]OpenCode executable not found: {executable or '(empty command)'}[/red]")
        if os.name == "nt":
            console.print(
                "Run [bold]Get-Command opencode | Format-List CommandType,Source,Path[/bold] "
                "to inspect the PowerShell launcher."
            )
        raise typer.Exit(1)
    if not goal:
        console.print("[red]Goal is empty.[/red]")
        raise typer.Exit(1)
    if not criteria.criteria:
        console.print("[red]No criteria are defined.[/red]")
        raise typer.Exit(1)
    human_only = [
        item.id
        for item in criteria.criteria
        if item.required and item.kind.value == "manual" and item.override.value == "auto"
    ]
    if human_only:
        console.print(
            "[yellow]Warning:[/yellow] required human-only criteria cannot pass autonomously: "
            + ", ".join(human_only)
        )
        console.print(
            "Change them to ai_judge when project evidence is sufficient, or provide a human override."
        )
    console.print(
        f"[green]Valid[/green]: goal={goal_id}; {len(criteria.criteria)} criteria; "
        f"model={store.read_control().model_override or config.model or 'OpenCode default'}"
    )
    console.print(
        f"OpenCode: {resolution.path} "
        f"([dim]{resolution.kind}, resolved from {resolution.source}[/dim])"
    )


@app.command()
def gui(
    project: ProjectOption = Path.cwd(),
    host: Annotated[str | None, typer.Option("--host", help="Bind address")] = None,
    port: Annotated[int | None, typer.Option("--port", help="TCP port")] = None,
    open_browser: Annotated[
        bool, typer.Option("--open-browser/--no-open-browser", help="Open the dashboard in a browser")
    ] = True,
) -> None:
    """Launch the local multi-goal web dashboard and supervisor."""
    import uvicorn

    from .web import create_app

    candidate = ProjectStore.discover(project)
    initial_project = candidate.project_dir if candidate.project_exists() else None
    launch_project = initial_project
    if initial_project is not None:
        config = candidate.read_config()
    else:
        active_entry = ProjectRegistry().active()
        active_store = ProjectStore(active_entry.project_path) if active_entry else None
        if active_store is not None and active_store.project_exists():
            launch_project = active_store.project_dir
            config = active_store.read_config()
        else:
            config = AppConfig(project_dir=str(Path(project).expanduser().resolve()))
    selected_host = host or config.gui_host
    selected_port = port or config.gui_port
    display_host = "127.0.0.1" if selected_host in {"0.0.0.0", "::"} else selected_host
    url = f"http://{display_host}:{selected_port}"
    console.print(f"[green]Goal Agent GUI[/green] {url}")
    if launch_project is not None:
        console.print(f"Initial project: {launch_project}")
    else:
        console.print("No initialized project was found. Create or open one from the dashboard.")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(launch_project), host=selected_host, port=selected_port, log_level="info")


def main() -> None:
    app()
