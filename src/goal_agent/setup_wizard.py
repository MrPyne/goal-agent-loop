from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .models import CriteriaDocument, SetupProposal
from .opencode import OpenCodeRunner
from .proposal_quality import assess_setup_proposal
from .prompts import criteria_refinement_prompt, setup_prompt
from .project_snapshot import collect_project_snapshot
from .storage import ProjectStore


class SetupWizard:
    def __init__(self, store: ProjectStore, console: Console | None = None):
        self.store = store
        self.console = console or Console()

    async def run(self, rough_goal: str | None = None, model: str | None = None) -> None:
        config = self.store.read_config()
        if model:
            config.model = model
            self.store.write_config(config)
        runner = OpenCodeRunner(config)
        project_snapshot = collect_project_snapshot(config.project_path)
        rough_goal = rough_goal or Prompt.ask("Describe the outcome you want")
        answers: list[str] = []

        while True:
            self.console.print("\n[bold]AI goal refinement[/bold]")
            proposal, _ = await runner.run_structured(
                setup_prompt(
                    rough_goal,
                    "\n".join(answers),
                    project_snapshot=project_snapshot,
                ),
                SetupProposal,
                model=config.model,
                agent=config.strategist_agent,
                title="Goal loop setup: refine goal",
                attempts=4,
                profile="refinement",
            )
            proposal = assess_setup_proposal(proposal, project_path=config.project_path)
            if proposal.assistant_message:
                self.console.print(Panel(proposal.assistant_message, title="AI collaborator"))
            if proposal.clarifying_questions:
                self.console.print("The AI needs a few clarifications:")
                for question in proposal.clarifying_questions:
                    answer = Prompt.ask(question)
                    answers.append(f"Q: {question}\nA: {answer}")
                continue

            blockers = [
                item for item in proposal.criteria_quality_issues if item.severity == "blocking"
            ]
            if blockers or not proposal.ready_to_finalize:
                self.console.print("[yellow]The draft is not ready to finalize.[/yellow]")
                if proposal.readiness_reason:
                    self.console.print(proposal.readiness_reason)
                for item in blockers:
                    prefix = f"[{item.criterion_id}] " if item.criterion_id else ""
                    self.console.print(f"- {prefix}{item.issue}")
                feedback = Prompt.ask(
                    "Reply with corrections, decisions, or ask for a final concrete-criteria review"
                )
                answers.append(f"User feedback: {feedback}")
                continue

            self.console.print(Panel(proposal.refined_goal, title="Proposed goal"))
            if proposal.goal_rationale:
                self.console.print(f"[dim]{proposal.goal_rationale}[/dim]")
            action = Prompt.ask(
                "Goal action",
                choices=["accept", "edit", "retry"],
                default="accept",
            )
            if action == "retry":
                feedback = Prompt.ask("What should change?")
                answers.append(f"User feedback: {feedback}")
                continue
            goal = proposal.refined_goal
            if action == "edit":
                goal = _edit_text(
                    self.store.goal_root / "goal-edit.tmp.md",
                    proposal.refined_goal,
                    "Edit the goal, save, and close the editor.",
                )
            break

        criteria = CriteriaDocument(revision=1, criteria=proposal.criteria)
        while True:
            self._show_criteria(criteria)
            has_required = any(item.required for item in criteria.criteria)
            human_only = [
                item.id
                for item in criteria.criteria
                if item.required and item.kind.value == "manual" and item.override.value == "auto"
            ]
            if human_only:
                self.console.print(
                    "[yellow]Human-only required criteria cannot pass autonomously:[/yellow] "
                    + ", ".join(human_only)
                )
                self.console.print(
                    "Use ai_judge when the evaluator can decide from project evidence, or keep manual only for intentional personal approval."
                )
            if not criteria.criteria or not has_required:
                self.console.print(
                    "[red]At least one required criterion is needed before the loop can stop safely.[/red]"
                )
                feedback = Prompt.ask("Describe the missing proof of success")
            elif Confirm.ask("Do these criteria prove the goal is achieved?", default=True):
                break
            else:
                feedback = Prompt.ask("Describe how the criteria should change")
            criteria, _ = await runner.run_structured(
                criteria_refinement_prompt(goal, criteria, feedback),
                CriteriaDocument,
                model=config.model,
                agent=config.strategist_agent,
                title="Goal loop setup: refine criteria",
                profile="analysis",
            )
            criteria.revision += 1

        self.store.write_goal(goal)
        self.store.write_criteria(criteria)
        self.store.update_control(
            desired_state="paused",
            note="Setup complete. Run 'goal-agent run' when ready.",
        )
        state = self.store.load_state()
        state.message = "Setup complete; paused"
        self.store.save_state(state)
        self.console.print("\n[green]Goal and criteria saved.[/green]")
        self.console.print(f"Goal: {self.store.goal_path}")
        self.console.print(f"Criteria: {self.store.criteria_path}")

    def _show_criteria(self, criteria: CriteriaDocument) -> None:
        table = Table(title="Proposed success criteria")
        table.add_column("ID")
        table.add_column("Kind")
        table.add_column("Required")
        table.add_column("Description")
        table.add_column("Check")
        for criterion in criteria.criteria:
            check = criterion.command or criterion.path or criterion.judge_prompt or "manual override"
            table.add_row(
                criterion.id,
                criterion.kind.value,
                "yes" if criterion.required else "no",
                criterion.description,
                check or "",
            )
        self.console.print(table)


def _edit_text(path: Path, initial: str, instruction: str) -> str:
    import click

    edited = click.edit(initial + "\n", extension=".md")
    if edited is None or not edited.strip():
        raise RuntimeError(instruction)
    return edited.strip()
