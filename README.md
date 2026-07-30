# Goal Agent Loop

A persistent Python orchestration layer for OpenCode that works toward explicit goals until their success criteria pass. Version 0.6.4 makes live dashboard polling interaction-safe so status updates no longer replace focused controls or erase an unfinished steering note.

Each goal follows the same evidence-driven cycle:

1. Refine the goal with the user.
2. Define measurable success criteria.
3. Evaluate every criterion against the current project state.
4. Analyze all passing, failing, and errored evidence with the evaluator AI.
5. Form one falsifiable root-cause hypothesis.
6. Let OpenCode execute the plan.
7. Re-evaluate and diagnose the actual result.
8. Repeat until every required criterion passes.

The project is intended for local models through OpenCode, including OpenCode connected to `llama-server`, Ollama, LM Studio, or another OpenAI-compatible endpoint.

## Main features

- Local web control center with no Node.js frontend.
- Create, discover, register, remove, and switch projects entirely from the GUI.
- Native system folder picker plus an in-app folder browser with quick locations and new-folder creation.
- Automatic project-folder suggestions, live path validation, and workspace detection.
- Persistent per-user project registry; project data remains inside each project.
- Multiple independent goals in each project.
- Configurable concurrent goal execution.
- Persistent multi-turn AI goal and criteria refinement in both the GUI and terminal.
- Saved refinement conversations per goal, with explicit assumptions, unresolved questions, readiness status, and one-click finalization.
- Context-safe refinement prompts that preserve the full UI transcript while compacting older model context and retaining recent corrections plus the current draft.
- Automatic recovery from OpenCode context overflow for every agent role using a fresh process, smaller task brief, restricted inspection, pruned tool history, and reserved context headroom.
- Automatic pause after recovery is genuinely exhausted, preventing an endless loop of identical context errors.
- A deterministic criteria-quality guard that blocks finalization of vague or non-verifiable stopping rules.
- Cancellable background proposal jobs with live OpenCode progress, parsing status, retries, and persistent error messages.
- Strategist, executor, and evaluator roles for every goal.
- Deterministic command and file checks plus qualitative AI evidence reviews that can pass criteria autonomously.
- Structured AI diagnosis of every passing, failing, and errored criterion result on every evaluation.
- Human-only approval gates remain available as an explicit exception and are clearly warned about in the GUI and validation.
- Live goal, criteria, steering, model, and control changes without resetting history.
- Interaction-safe live polling: focused inputs remain mounted, per-goal steering drafts survive refreshes and tab changes, and queued status updates apply after editing ends.
- Pause, resume, stop, and restart with persistent state.
- Hypothesis and event history for each goal.
- Automatic stop when every required criterion passes.
- Automatic migration of version 0.1 single-goal workspaces into a `default` goal.

Within one goal, the strategist, executor, and evaluator run sequentially. Different goals may run concurrently.

## Requirements

- Python 3.11 or newer.
- OpenCode installed and available as `opencode`.
- At least one model configured in OpenCode.
- A Git repository or disposable working copy is strongly recommended.



## Interaction-safe live updates

The dashboard continues polling while goals run, but it no longer rebuilds the goal view while you are typing in an input, textarea, or select control. An unfinished **Steer the loop** note is kept in browser memory per goal, survives tab changes and ordinary status refreshes, and is cleared only after the steering note is successfully submitted. Status data continues to be fetched in the background and the newest view is applied after editing ends.

## Success criteria editor

The browser editor presents success criteria as contained, responsive cards. Each criterion separates its definition, checking method, and evaluation details so long IDs, descriptions, judge prompts, and evidence paths remain readable without forcing horizontal scrolling. The criteria panel uses the full available content width and collapses to one-column fields on narrow windows.

## Installation

### Windows

From the extracted project directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\bootstrap.ps1
```

Manual installation:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

### Linux or macOS

```bash
./scripts/bootstrap.sh
```

Verify the installation:

```powershell
goal-agent --help
opencode --version
```

## Configure OpenCode for local llama.cpp

OpenCode owns the provider configuration. Goal Agent invokes OpenCode instead of calling the model server directly.

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server local",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1"
      },
      "models": {
        "qwen3-coder:a3b": {
          "name": "Qwen Coder local",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  },
  "model": "llama.cpp/qwen3-coder:a3b"
}
```

A copy is included at `examples/opencode.llama-cpp.json`.

Confirm that OpenCode can see the model:

```powershell
opencode models
```

### Context-window protection

Goal Agent starts each OpenCode role as a new non-continued session. For child runs it also supplies an inline OpenCode
compaction override equivalent to:

```json
{
  "compaction": {
    "auto": true,
    "prune": true,
    "reserved": 12000
  }
}
```

This override is passed through `OPENCODE_CONFIG_CONTENT`; it does not edit the project's `opencode.json`. Existing
inline JSON settings are preserved. Pruning removes older tool results and the reserved buffer leaves room for tool
schemas, a final model response, and compaction.

If OpenCode still reports a context overflow, Goal Agent starts a fresh recovery process rather than continuing the
failed session. The recovery prompt preserves the task while limiting broad searches, subagents, large files, noisy
commands, dependency trees, generated output, media, and other common sources of context growth. It retries twice,
with a smaller file and prompt budget on the second attempt.

If all fresh-session retries overflow, the goal is auto-paused once. Increase the `llama-server` context, reduce very
large evidence or steering text, narrow the goal, or remove unusually large project instructions before resuming. The
loop no longer repeats the same failing strategist/executor call indefinitely.

## Recommended first run: GUI

Launch the control center from any directory:

```powershell
goal-agent gui
```

The default address is:

```text
http://127.0.0.1:8765
```

Use **Projects** in the top bar to either:

- enter a project name and let Goal Agent suggest a folder automatically;
- choose a folder through the native system picker or the built-in folder browser;
- search recent projects or scan common locations for existing `.goal-agent` workspaces;
- create folders directly inside the browser;
- open/register an existing initialized project;
- switch between known projects without restarting the GUI;
- open a project's directory in the system file manager;
- remove a project from the dashboard without deleting its files.

Manual path entry remains available as a fallback, but it is no longer required for normal setup. The GUI validates a chosen path immediately and explains whether it will create a new directory, initialize an existing source folder, or open an existing Goal Agent workspace.

You can still open a particular project immediately:

```powershell
goal-agent gui -p C:\Users\mattp\projects\my-project
```

The dashboard provides:

- a saved **Goal definition conversation** where you can answer the AI's questions and keep revising the same draft;
- a readiness indicator that stays in **Still refining** while questions or blocking criteria-quality issues remain;
- concrete criteria previews showing the exact command, file evidence, or AI-review pass/fail rubric;
- **Finalize and save**, which writes the accepted goal and complete criteria directly to the project only after readiness checks pass;
- guided project creation, folder browsing, discovery, registration, switching, and validation;
- goal cards with phase and criteria progress, including archived goals;
- automatic goal IDs generated from goal titles, with manual IDs available under Advanced;
- live strategist, executor, and evaluator activity;
- start, pause, resume, and stop controls;
- project-wide Start All, Pause All, and Stop All controls;
- project-default and per-goal model selection;
- every project configuration field supported by the CLI/file configuration;
- structured success-criterion editing and human overrides;
- AI-assisted goal and criteria refinement;
- a **Save & finish setup** action equivalent to the terminal setup completion;
- live steering messages;
- hypothesis and event history;
- project and goal file paths;
- safe goal deletion and project removal from the dashboard.

Use another host or port when needed:

```powershell
goal-agent gui --host 0.0.0.0 --port 9000
```

Binding to `0.0.0.0` exposes the control center to the local network. The GUI has no authentication, so use `127.0.0.1` unless network access is deliberately required.

### GUI and CLI feature parity

| CLI capability | GUI location |
|---|---|
| `init` | Projects → Create project |
| `goals`, `goal-create`, `goal-delete` | Goal sidebar and goal management |
| `setup` | Goal & criteria → AI refinement → Save & finish setup |
| `run`, `pause`, `resume`, `stop` | Goal header controls |
| `run-all` | Top-bar Start all |
| `status --watch` | Live Overview tab |
| `steer` | Overview → Steer the loop |
| `set-goal` | Goal & criteria editor |
| `models`, `select-model` | Model selector and Runtime settings |
| `criterion-override` | Criterion override selectors |
| `files` | Files tab |
| `validate` | Top-bar Validate button |

The CLI and GUI operate on the same files. Changes made in either interface are visible to the other.

## Iterative goal and criteria finalization

Open **Goal & criteria → Refine with AI**. The conversation is stored at:

```text
.goal-agent/goals/<goal-id>/control/refinement.json
```

Each AI turn returns a complete current draft rather than a partial answer. The draft includes the refined goal, assumptions, remaining clarifying questions, all proposed criteria, criteria-quality findings, and a readiness decision. You can close the modal or restart the GUI and continue where you left off.

A proposal cannot be finalized while it has unanswered material questions or blocking quality findings. In particular, AI-reviewed criteria must define repeatable evidence and explicit `PASS only if ...` and `FAIL if ...` rules. Vague outcomes such as “the interface is good” or “the program works correctly” must be converted into observable behavior, thresholds, required artifacts, automated tests, or a strict evidence checklist.

**Apply draft to editor** is available for manual review without finalizing. **Finalize and save** writes the goal and criteria immediately, records the conversation as finalized, and leaves the loop paused until you start it.

## Project registry

The GUI stores only its list of known project paths in a per-user registry:

- Windows: `%APPDATA%\goal-agent\projects.yaml`
- Linux/macOS: `$XDG_CONFIG_HOME/goal-agent/projects.yaml` or `~/.config/goal-agent/projects.yaml`

Goals, state, criteria, history, and configuration remain under each project's `.goal-agent` directory. Removing a project from the dashboard never deletes that directory.

## Multiple goals

The initial workspace contains a goal named `default`. Add goals from the GUI or terminal:

```powershell
goal-agent goal-create repair-auth `
  -p C:\Users\mattp\projects\my-project `
  --title "Repair authentication" `
  --goal "Make login and token refresh reliable and prove it with tests"
```

List goals:

```powershell
goal-agent goals -p C:\Users\mattp\projects\my-project
```

Use `--goal-id` or `-g` with goal-specific commands:

```powershell
goal-agent setup -p C:\path\to\project -g repair-auth
goal-agent validate -p C:\path\to\project -g repair-auth
goal-agent run -p C:\path\to\project -g repair-auth
goal-agent pause -p C:\path\to\project -g repair-auth
goal-agent status -w -p C:\path\to\project -g repair-auth
```

Run several goals in one terminal process:

```powershell
goal-agent run-all -p C:\path\to\project
```

The GUI is generally easier for concurrent operation.

### Concurrency warning

Concurrent goals share the target project directory. Two executor agents can therefore edit the same file at the same time.

Use concurrent goals when their work is independent. For overlapping code changes, use separate Git worktrees or separate project directories. The GUI displays this warning in the goal sidebar.

The default concurrency limit is two goals. Change it in GUI settings or `.goal-agent/config.yaml`:

```yaml
max_concurrent_goals: 2
```

## AI-assisted setup

### GUI

Open a goal, select **Goal & criteria**, then use:

- **AI refine goal** to produce a refined outcome, clarifying questions, and proposed criteria;
- **AI improve criteria** to strengthen the existing stopping conditions.

AI proposals are applied to the editor first. Review and save them before the loop uses them.

### Terminal

```powershell
goal-agent setup -p C:\path\to\project -g default
```

The terminal wizard asks for the rough goal, collects material clarifications, proposes a final goal, and iteratively refines the criteria with the user.

## File layout

Project-wide configuration:

```text
.goal-agent/
  config.yaml
```

Per-goal state:

```text
.goal-agent/
  goals/
    default/
      goal.yaml
      control/
        goal.md
        criteria.yaml
        steering.md
        control.yaml
      status/
        STATUS.md
        state.json
        agents.json
        criteria.json
        evaluation-analysis.json
        hypotheses.json
        events.jsonl
      runs/
        iteration-00001/
        iteration-00002/
```

Each goal has its own loop lock, state, controls, events, and run artifacts. This isolation allows one goal to pause or complete without stopping the others.

The GUI and file interface use the same files. Editing `goal.md`, `criteria.yaml`, `steering.md`, or `control.yaml` externally remains supported while the process is running.

Print paths for one goal:

```powershell
goal-agent files -p C:\path\to\project -g default
```

## Goal controls

Start or restart a goal:

```powershell
goal-agent run -p C:\path\to\project -g default
```

Pause it, including termination of the active OpenCode subprocess:

```powershell
goal-agent pause -p C:\path\to\project -g default
```

Resume it:

```powershell
goal-agent resume -p C:\path\to\project -g default
```

Stop it and allow the loop process to exit:

```powershell
goal-agent stop -p C:\path\to\project -g default
```

State and hypothesis history remain available after stopping. Work already performed by an interrupted OpenCode process is not rolled back automatically.

## Live steering

Append guidance without stopping the goal:

```powershell
goal-agent steer `
  "Do not replace the parser. Fix the retry boundary and add regression tests." `
  -p C:\path\to\project `
  -g repair-auth
```

The loop rereads goal, criteria, steering, controls, and configuration at phase boundaries.

## Model selection

List models:

```powershell
goal-agent models -p C:\path\to\project
```

Set the project default:

```powershell
goal-agent select-model `
  -p C:\path\to\project `
  --model "llama.cpp/qwen3-coder:a3b"
```

Set a per-goal override:

```powershell
goal-agent select-model `
  -p C:\path\to\project `
  -g repair-auth `
  --model "ollama/qwen3-coder" `
  --goal-override
```

The effective model is resolved before every agent phase, so changes do not reset the goal's history.

## Criteria types

Criteria are the stopping authority. The executor's own statement that it succeeded is not sufficient.

### Command

```yaml
- id: tests
  description: All tests pass
  kind: command
  required: true
  override: auto
  command: python -m pytest -q
  expected_exit_code: 0
  timeout_seconds: 300
```

### File existence

```yaml
- id: migration-created
  description: The database migration exists
  kind: file_exists
  required: true
  override: auto
  path: migrations/0042_retry_state.sql
```

### File content

```yaml
- id: docs-retry
  description: Retry behaviour is documented
  kind: file_contains
  required: true
  override: auto
  path: README.md
  contains: Retry behaviour
  regex: false
  case_sensitive: true
```

### AI judgment

```yaml
- id: understandable-errors
  description: Import errors are understandable to a non-technical operator
  kind: ai_judge
  required: true
  override: auto
  judge_prompt: >-
    Pass only when the output identifies the row, explains the problem,
    and provides an actionable correction.
  evidence_paths:
    - src/importer
    - tests
  confidence_threshold: 0.85
```

Prefer deterministic criteria whenever possible. Use AI evidence review for outcomes that are qualitative but can still be judged from concrete project artifacts, behavior, tests, screenshots, logs, or documentation. The evaluator runs this check every time criteria are evaluated and can pass it when the evidence and confidence threshold are sufficient.

### Human approval only

```yaml
- id: physical-device-check
  description: The workflow succeeds on the physical test device
  kind: manual
  required: true
  override: auto
```

This type cannot pass through the autonomous loop. Use it only when a person must personally verify something that cannot be established from project evidence. New criteria created in the GUI default to `ai_judge`, and manual criteria include a **Convert to AI review** action.

After personally verifying it:

```powershell
goal-agent criterion-override physical-device-check pass `
  -p C:\path\to\project `
  -g default
```

Clear the override with `auto`.

## Iteration flow

For each running goal:

1. Read the latest project and goal controls.
2. Evaluate every criterion using its configured method:
   - automated command/test;
   - file existence or content;
   - AI evidence review;
   - explicit human override for a human-only gate.
3. Ask the evaluator AI to analyze every pass, fail, and error result. The raw criterion result remains authoritative; the AI cannot rewrite a deterministic result.
4. Stop immediately if all required criteria pass.
5. Give the raw results and structured diagnosis to the strategist.
6. Ask the strategist for one falsifiable root-cause hypothesis.
7. Persist the hypothesis and plan.
8. Ask the executor to perform the work.
9. Re-read user changes and evaluate every criterion again.
10. Analyze the new outputs against the earlier results, including concrete partial progress within criteria that still fail.
11. Mark the hypothesis supported, refuted, or inconclusive.
12. Repeat using the accumulated evidence.

The latest diagnosis is written to `status/evaluation-analysis.json`, included in `STATUS.md`, displayed in the GUI, and saved with each iteration as `baseline-analysis.json` and `post-execution-analysis.json`. After repeated no-progress iterations, the strategist is instructed to challenge assumptions and select a materially different direction.

## AI diagnosis of criterion output

Every baseline and post-execution evaluation produces two layers of evidence:

1. **Criterion results** decide whether each criterion passes. Commands, files, and prior AI-judge decisions are the stopping authority.
2. **Evaluation analysis** explains what those results mean, diagnoses likely causes, identifies passing behavior that must be preserved, finds cross-criterion patterns, and recommends the next focus.

The diagnostic evaluator is read-only. Its `observed_status` values are normalized back to the actual criterion results before they are saved, so it cannot turn a failed deterministic test into a pass. When OpenCode cannot return a valid diagnosis, the loop records the error and creates a result-based fallback analysis instead of losing the raw checks.

When post-execution evidence shows concrete partial improvement but a criterion has not crossed its final pass threshold, the analysis can mark `material_progress: true`. That evidence is included in the hypothesis outcome so iterative work is not treated as a complete failure merely because the final condition remains unmet.

## Project configuration

`.goal-agent/config.yaml` is shared by all goals:

```yaml
project_dir: C:\path\to\project
opencode_command:
  - opencode
model: llama.cpp/qwen3-coder:a3b
attach_url: null
attach_username: null
attach_password_env: null
strategist_agent: plan
executor_agent: build
evaluator_agent: plan
auto_approve: true
poll_interval_seconds: 0.5
iteration_delay_seconds: 2.0
opencode_timeout_seconds: 1800
criterion_timeout_seconds: 300
max_iterations: null
max_recent_hypotheses: 12
no_progress_rethink_after: 3
status_refresh_seconds: 0.75
max_concurrent_goals: 2
gui_auto_resume_running_goals: true
gui_host: 127.0.0.1
gui_port: 8765
```

`auto_approve: true` adds OpenCode's automatic approval flag. Explicit denials in OpenCode configuration still apply.

`max_iterations` pauses each goal after the configured number of total iterations instead of falsely declaring success.

### Persistent OpenCode server

To reduce repeated OpenCode startup and MCP initialization cost:

```powershell
opencode serve --port 4096
```

Then configure:

```yaml
attach_url: http://127.0.0.1:4096
```

All goals can attach to the same OpenCode server. Confirm that the selected local model backend can handle the desired parallel request count.

## Version 0.1 migration

Opening an old workspace automatically moves the previous single-goal files into:

```text
.goal-agent/goals/default/
```

The old project configuration becomes:

```text
.goal-agent/config.yaml
```

No goal state or run history is intentionally discarded.

## Safety

The executor can edit files and run commands in the target project.

Recommended precautions:

- Use a dedicated Git branch or worktree for every independently running goal.
- Keep secrets and unrelated personal data outside the target directory.
- Prefer deterministic, difficult-to-game success criteria.
- Review event logs and iteration artifacts.
- Start with small goals before unattended operation.
- Do not expose the unauthenticated GUI to an untrusted network.
- Use a container, VM, or restricted operating-system account for stronger isolation.

## Development

Run all tests:

```powershell
python -m pytest
```

The current suite includes storage, automated and AI-judged criteria, grounded evaluation analysis, large OpenCode event streaming, context-overflow detection and compact retry, bounded refinement history, structured proposal jobs, invalid-output reporting, full-loop, active-pause, migration, web API, and two-goal concurrent execution tests.

Run directly from source:

```powershell
$env:PYTHONPATH = "src"
python -m goal_agent --help
```

## Windows OpenCode command resolution

OpenCode installed with npm is commonly exposed as `opencode.cmd`, while PowerShell may display it simply as `opencode`. Goal Agent resolves the actual launcher before starting a subprocess and automatically runs `.cmd`/`.bat` shims through `cmd.exe` and `.ps1` shims through PowerShell.

Resolution checks the current process PATH, the latest user and machine PATH values from the Windows registry, and common npm, Bun, OpenCode, and WinGet locations. This also handles a dashboard process whose PATH became stale after OpenCode was installed.

Inspect what PowerShell is using with:

```powershell
Get-Command opencode | Format-List CommandType,Source,Path,Definition
where.exe opencode
```

A temporary workaround for older Goal Agent releases is to set **Runtime settings → OpenCode connection → OpenCode command** to:

```text
cmd.exe /d /s /c opencode
```

Version 0.4.1 and newer should normally keep the simpler default value `opencode`.


## Proposal generation diagnostics

The GUI starts **AI refine goal** and **AI improve criteria** as background jobs. The dialog polls the job and shows the latest OpenCode stage, including startup, tool activity, response receipt, schema validation, and format retries. You can close the dialog or cancel the active proposal without freezing the rest of the dashboard.

OpenCode's `--format json` output is newline-delimited JSON. Tool-result events may be much larger than Python's default 64 KiB stream line limit, so Goal Agent reads the stream in chunks and supports individual records up to 32 MiB. Reader failures terminate the OpenCode process and surface an error instead of waiting for the normal OpenCode timeout.

### Context-window protection

Long refinement conversations do not send the entire raw transcript back to the model on every turn. Goal Agent keeps every message in `refinement.json` for the GUI, while the model prompt contains:

- a deterministic summary of older user decisions and AI questions;
- the eight most recent messages;
- a bounded copy of the current draft proposal;
- bounded saved criteria when criteria-only refinement is requested.

The normal Goal Agent input budget is intentionally kept well below the model context size so OpenCode has room for its own system instructions, file reads, and tool results. The refinement dialog reports the estimated Goal Agent input size and how many older messages were compacted.

If OpenCode still emits `ContextOverflowError`, Goal Agent recognizes the structured error, records the token counts when available, and automatically retries once with a smaller prompt plus instructions to avoid broad searches, large files, dependency trees, generated output, logs, and subagents. The full saved conversation is not deleted.

If the compact retry also exceeds the context window, increase the local model server context if memory permits. For llama.cpp, that normally means restarting `llama-server` with a larger `-c` value, for example:

```powershell
-c 131072
```

A larger context consumes more KV-cache memory, so use the highest value that remains stable on the selected hardware. Reducing the target project's unrelated files or using a dedicated worktree can also reduce OpenCode inspection overhead.
