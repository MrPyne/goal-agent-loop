const state = {
  projects: [],
  projectId: null,
  project: null,
  goals: [],
  includeArchived: false,
  selectedId: null,
  detail: null,
  activeTab: "overview",
  editorDirty: false,
  models: [],
  aiMode: "goal",
  aiProposal: null,
  aiConversation: "",
  aiSession: null,
  aiJobId: null,
  aiLastSubmittedFeedback: "",
  aiRetryFeedback: "",
  polling: false,
  pendingDetailRender: false,
  agentChat: { goalId: null, agentName: null, autoRefresh: false, refreshTimer: null, runs: [] },
  steeringDrafts: {},
  lastDetailSignature: null,
  interactionHoldUntil: 0,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = (value = "") => String(value).replace(/[&<>'"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[c]));
const pct = (part, total) => total ? Math.round(part * 100 / total) : 0;
const fmtTime = value => value ? new Date(value).toLocaleString() : "—";
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function latestUserMessage(messages = []) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role === "user") {
      const content = String(message.content || "").trim();
      if (content) return content;
    }
  }
  return "";
}


function captureSteeringDraft() {
  if (!state.selectedId) return;
  const input = $("#steering-message");
  if (input) state.steeringDrafts[state.selectedId] = input.value;
}

function steeringDraft(goalId = state.selectedId) {
  return goalId ? (state.steeringDrafts[goalId] || "") : "";
}

function isEditableElement(element) {
  return !!(element && element.matches?.('input, textarea, select, [contenteditable="true"]'));
}

function isSelectionInsideMainContent() {
  const selection = window.getSelection?.();
  if (!selection || selection.rangeCount < 1 || selection.isCollapsed) return false;
  const range = selection.getRangeAt(0);
  const mainContent = $("#main-content");
  if (!mainContent) return false;
  const anchorNode = selection.anchorNode;
  const focusNode = selection.focusNode;
  return !!(
    (anchorNode && mainContent.contains(anchorNode))
    || (focusNode && mainContent.contains(focusNode))
    || (range.commonAncestorContainer && mainContent.contains(range.commonAncestorContainer))
  );
}

function isUserEditingMainContent() {
  const active = document.activeElement;
  const focusedEditor = !!(active && active.closest?.("#main-content") && isEditableElement(active));
  return focusedEditor || isSelectionInsideMainContent() || Date.now() < state.interactionHoldUntil;
}

function detailSignature(detail) {
  if (!detail) return "";
  return JSON.stringify({
    metadata: detail.metadata,
    goal: detail.goal,
    criteria: detail.criteria,
    control: detail.control,
    state: detail.state,
    events: detail.events,
    paths: detail.paths,
  });
}

function applyPolledDetail(detail) {
  const signature = detailSignature(detail);
  const changed = signature !== state.lastDetailSignature;
  state.detail = detail;
  if (!changed) {
    updateHeaderStatusOnly();
    return;
  }
  if (isUserEditingMainContent()) {
    captureSteeringDraft();
    state.pendingDetailRender = true;
    updateHeaderStatusOnly();
    return;
  }
  state.pendingDetailRender = false;
  state.lastDetailSignature = signature;
  renderDetail();
}

async function api(path, options = {}, scoped = true) {
  let target = path;
  if (scoped && state.projectId && !path.startsWith("/api/projects")) {
    const joiner = path.includes("?") ? "&" : "?";
    target = `${path}${joiner}project_id=${encodeURIComponent(state.projectId)}`;
  }
  const config = { ...options, headers: { "Content-Type": "application/json", ...(options.headers || {}) } };
  const response = await fetch(target, config);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { const body = await response.json(); message = body.detail || message; } catch (_) {}
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

function toast(message, kind = "success") {
  const node = document.createElement("div");
  node.className = `toast ${kind}`;
  node.textContent = message;
  $("#toast-stack").appendChild(node);
  setTimeout(() => node.remove(), 4200);
}

function openModal(id) { $(`#${id}`).classList.remove("hidden"); }
function closeModal(id) { $(`#${id}`).classList.add("hidden"); }

async function boot() {
  bindGlobalEvents();
  await refreshProjects();
  if (state.projectId) await loadProject();
  else { renderNoProject(); openProjectManager(); }
  setInterval(poll, 1400);
}

async function refreshProjects() {
  const data = await api("/api/projects", {}, false);
  state.projects = data.projects || [];
  if (!state.projectId || !state.projects.some(p => p.id === state.projectId)) {
    state.projectId = data.active_project_id || state.projects[0]?.id || null;
  }
  renderProjectSelector();
}

function renderProjectSelector() {
  const selector = $("#project-selector");
  selector.innerHTML = state.projects.length
    ? state.projects.map(p => `<option value="${esc(p.id)}" ${p.id === state.projectId ? "selected" : ""}>${esc(p.title)}</option>`).join("")
    : `<option value="">No projects</option>`;
  selector.disabled = !state.projects.length;
}

async function loadProject() {
  state.selectedId = null; state.detail = null; state.goals = []; state.models = [];
  await api(`/api/projects/${encodeURIComponent(state.projectId)}/activate`, { method: "POST" }, false);
  await refreshProject();
  await refreshGoals();
  if (state.goals.length) await selectGoal(state.goals[0].metadata.id);
  else renderNoGoals();
}

async function refreshProject() {
  if (!state.projectId) return;
  state.project = await api("/api/project");
  $("#project-path").textContent = state.project.project_dir;
  renderConcurrency();
}

async function refreshGoals() {
  if (!state.projectId) return;
  const data = await api(`/api/goals?include_archived=${state.includeArchived}`);
  state.goals = data.goals;
  renderGoalList();
  renderConcurrency();
}

function renderNoProject() {
  state.project = null; state.goals = []; state.selectedId = null; state.detail = null;
  $("#project-path").textContent = "No project selected";
  $("#goal-list").innerHTML = `<div class="muted small" style="padding:18px 8px">Create or open a project.</div>`;
  $("#main-content").innerHTML = `<section class="empty-state welcoming-empty"><div class="empty-icon">▣</div><h2>Choose your first project</h2><p>A project is the folder the agents will work in. You can create a new workspace or open an existing one without typing its path.</p><div class="empty-choice-grid"><button class="empty-choice" data-open-projects data-project-mode-shortcut="create"><span>＋</span><strong>Create a project</strong><small>Choose a name and folder</small></button><button class="empty-choice" data-open-projects data-project-mode-shortcut="open"><span>⌂</span><strong>Open existing</strong><small>Browse this computer</small></button></div></section>`;
  $("#concurrency").textContent = "0 running";
}

function renderNoGoals() {
  $("#main-content").innerHTML = `<section class="empty-state"><div class="empty-icon">◎</div><h2>No active goals</h2><p>Create a goal for this project to begin.</p><button class="button primary" data-open-new-goal>Create a goal</button></section>`;
}

async function poll() {
  if (state.polling) return;
  state.polling = true;
  try {
    await refreshProjects();
    if (!state.projectId) { renderNoProject(); return; }
    await refreshProject();
    await refreshGoals();
    if (state.selectedId && state.activeTab !== "definition") {
      const detail = await api(`/api/goals/${encodeURIComponent(state.selectedId)}`);
      applyPolledDetail(detail);
    } else if (state.selectedId) {
      updateHeaderStatusOnly();
    }
  } catch (error) {
    console.error(error);
  } finally {
    state.polling = false;
  }
}

function renderConcurrency() {
  if (!state.project) return;
  const running = state.goals.filter(g => g.desired_state === "running").length;
  const max = state.project.config.max_concurrent_goals;
  $("#concurrency").textContent = `${running}/${max} running`;
}

function renderGoalList() {
  const filter = ($("#goal-filter").value || "").toLowerCase();
  const goals = state.goals.filter(item =>
    `${item.metadata.title} ${item.metadata.id} ${item.metadata.description}`.toLowerCase().includes(filter)
  );
  $("#goal-list").innerHTML = goals.length ? goals.map(item => {
    const progress = pct(item.required_passed, item.required_total);
    return `<article class="goal-card ${item.metadata.id === state.selectedId ? "selected" : ""}" data-select-goal="${esc(item.metadata.id)}">
      <div class="goal-card-top">
        <span class="goal-title">${esc(item.metadata.title)}${item.metadata.archived ? ` <span class="badge">archived</span>` : ""}</span>
        <span class="state-dot ${esc(item.phase)}" title="${esc(item.phase)}"></span>
      </div>
      <div class="goal-id">${esc(item.metadata.id)}</div>
      <div class="goal-card-footer">
        <div class="mini-progress"><span style="width:${progress}%"></span></div>
        <span class="progress-label">${item.required_passed}/${item.required_total}</span>
      </div>
    </article>`;
  }).join("") : `<div class="muted small" style="padding:18px 8px">No matching goals.</div>`;
}

async function selectGoal(goalId) {
  captureSteeringDraft();
  state.selectedId = goalId;
  state.editorDirty = false;
  state.pendingDetailRender = false;
  state.detail = await api(`/api/goals/${encodeURIComponent(goalId)}`);
  state.lastDetailSignature = detailSignature(state.detail);
  renderGoalList();
  renderDetail();
}

function updateHeaderStatusOnly() {
  const summary = state.goals.find(g => g.metadata.id === state.selectedId);
  if (!summary) return;
  const badge = $("#detail-phase");
  if (badge) {
    badge.className = `badge ${summary.phase}`;
    badge.textContent = summary.phase;
  }
}

function renderDetail() {
  if (!state.detail) return;
  captureSteeringDraft();
  const d = state.detail;
  const phase = d.state.phase;
  const running = d.control.desired_state === "running";
  const modelOptions = buildModelOptions(d);
  $("#main-content").innerHTML = `<div class="detail">
    <div class="detail-heading">
      <div>
        <span class="eyebrow">${esc(d.metadata.id)}</span>
        <div class="detail-title-row"><h2>${esc(d.metadata.title)}</h2><span id="detail-phase" class="badge ${esc(phase)}">${esc(phase)}</span></div>
        <p class="detail-subtitle">Iteration ${d.state.iteration} · ${esc(d.state.message)}</p>
      </div>
      <div class="goal-actions">
        <div class="model-control">
          <select id="goal-model" title="Model for this goal">${modelOptions}</select>
          <button class="icon-button" data-load-models title="Reload OpenCode models">↻</button>
        </div>
        ${running ? `<button class="button ghost" data-action="pause">Pause</button>` : `<button class="button ghost" data-action="resume">Resume</button>`}
        <button class="button ghost danger-text" data-action="stop">Stop</button>
        ${!running ? `<button class="button primary" data-action="start">Start loop</button>` : ""}
      </div>
    </div>
    <nav class="tabbar">
      ${tabButton("overview", "Overview")}
      ${tabButton("definition", "Goal & criteria")}
      ${tabButton("history", "History")}
      ${tabButton("files", "Files")}
    </nav>
    <div id="tab-content">${renderActiveTab()}</div>
  </div>`;
}

function buildModelOptions(d) {
  const current = d.control.model_override || "";
  const defaultLabel = state.project?.config?.model || "OpenCode default";
  const values = [...new Set([current, ...(state.models || [])].filter(Boolean))];
  return `<option value="">Project default · ${esc(defaultLabel)}</option>` + values.map(model =>
    `<option value="${esc(model)}" ${model === current ? "selected" : ""}>${esc(model)}</option>`
  ).join("");
}

function tabButton(id, label) {
  return `<button class="tab-button ${state.activeTab === id ? "active" : ""}" data-tab="${id}">${label}</button>`;
}

function renderActiveTab() {
  if (state.activeTab === "definition") return renderDefinition();
  if (state.activeTab === "history") return renderHistory();
  if (state.activeTab === "files") return renderFiles();
  return renderOverview();
}

function renderOverview() {
  const d = state.detail;
  const definitions = d.criteria.criteria;
  const results = d.state.criteria_results || {};
  const required = definitions.filter(c => c.required);
  const passed = required.filter(c => results[c.id]?.passed).length;
  const checked = Object.values(results).filter(result => result && result.status && result.status !== "unchecked").length;
  const totalCriteria = definitions.length;
  const agents = Object.entries(d.state.agents || {});
  const liveAgent = selectLiveAgent(d.state.agents || {});
  const evaluatorAgent = d.state.agents?.evaluator || null;
  const activeHypothesis = [...(d.state.hypotheses || [])].reverse().find(h => h.id === d.state.active_hypothesis_id) || [...(d.state.hypotheses || [])].reverse()[0];
  const evaluationAnalysis = d.state.evaluation_analysis || null;
  return `<div class="grid two">
    <div class="grid">
      <section class="card live-status-card ${d.control.desired_state === "running" ? "live" : ""}">
        <div class="card-heading"><h3>Loop status</h3><span class="badge ${esc(d.state.phase)}">${esc(d.state.phase)}</span></div>
        <div class="live-status-message">${esc(d.state.message || "No current status")}</div>
        <div class="live-status-grid">
          <div><span class="eyebrow">Desired state</span><div>${esc(d.control.desired_state)}</div></div>
          <div><span class="eyebrow">Iteration</span><div>${esc(String(d.state.iteration))}</div></div>
          <div><span class="eyebrow">Last update</span><div>${esc(fmtTime(d.state.updated_at))}</div></div>
          <div><span class="eyebrow">Last error</span><div>${esc(d.state.last_error || "None")}</div></div>
          <div><span class="eyebrow">Criteria checked</span><div>${esc(`${checked}/${totalCriteria}`)}</div></div>
          <div><span class="eyebrow">Required passing</span><div>${esc(`${passed}/${required.length}`)}</div></div>
        </div>
        ${evaluatorAgent ? `<div class="live-status-progress"><strong>Current evaluation step</strong><p>${esc(evaluatorAgent.task || "Evaluating criteria")}</p><small>${esc(evaluatorAgent.detail || "Checking the next success criterion…")}</small></div>` : ""}
        ${liveAgent ? `<div class="live-status-agent"><span class="eyebrow">Active agent</span><strong>${esc(liveAgent.name)}</strong><span class="badge ${esc(liveAgent.phase)}">${esc(liveAgent.phase)}</span><div>${esc(liveAgent.task || "Idle")}</div><p>${esc(liveAgent.detail || "No detailed progress yet.")}</p></div>` : ""}
        ${d.state.phase === "running" ? `<p class="live-status-note">The loop is active. If the task looks unchanged, the current step may be evaluating criteria, waiting on OpenCode, or preparing the next hypothesis.</p>` : ""}
      </section>
      <section class="card">
        <div class="card-heading"><h3>Goal</h3><button class="button ghost" data-tab="definition">Edit definition</button></div>
        <div class="goal-copy">${esc(d.goal)}</div>
      </section>
      <section class="grid three">
        ${agents.map(([name, agent]) => `<article class="card agent-card">
          <div class="agent-top"><span class="agent-name">${esc(name)}</span><span class="badge ${esc(agent.phase)}">${esc(agent.phase)}</span></div>
          <div class="agent-task">${esc(agent.task)}</div>
          <div class="agent-detail">${esc(agent.detail || "No current detail")}</div>
          <button class="button ghost agent-chat-btn" data-agent="${esc(name)}" style="margin-top:8px;font-size:0.78em">View chat \u2192</button>
        </article>`).join("")}
      </section>
      <section class="card">
        <div class="card-heading"><h3>Success criteria</h3><span class="muted small">${passed} of ${required.length} required passing</span></div>
        <div class="progress-row"><div class="big-progress"><span style="width:${pct(passed, required.length)}%"></span></div><strong>${pct(passed, required.length)}%</strong></div>
        <div class="criteria-list" style="margin-top:15px">${definitions.length ? definitions.map(c => criterionStatus(c, results[c.id])).join("") : `<div class="muted small">No criteria defined yet.</div>`}</div>
      </section>
    </div>
    <div class="grid">
      <section class="card">
        <div class="card-heading"><h3>Current hypothesis</h3>${activeHypothesis ? `<span class="badge ${esc(activeHypothesis.status)}">${esc(activeHypothesis.status)}</span>` : ""}</div>
        ${activeHypothesis ? `<div class="hypothesis-title">${esc(activeHypothesis.statement)}</div>
          <p class="hypothesis-outcome">${esc(activeHypothesis.rationale)}</p>
          <p class="hypothesis-outcome"><strong>Expected:</strong> ${esc(activeHypothesis.expected_impact)}</p>
          <p class="hypothesis-outcome"><strong>Outcome:</strong> ${esc(activeHypothesis.outcome || "Pending")}</p>` : `<div class="muted small">The strategist has not proposed a hypothesis yet.</div>`}
      </section>
      ${renderEvaluationAnalysis(evaluationAnalysis)}
      <section class="card">
        <div class="card-heading"><h3>Steer the loop</h3><span class="muted small">Read before the next step</span></div>
        <textarea id="steering-message" rows="5" style="width:100%" placeholder="Add a constraint, correction, observation, or suggested direction…">${esc(steeringDraft())}</textarea>
        <button class="button primary full-width" style="margin-top:9px" data-add-steering>Add steering note</button>
      </section>
      <section class="card">
        <div class="card-heading"><h3>Recent activity</h3><button class="button ghost" data-tab="history">View all</button></div>
        ${renderEventList((d.events || []).slice(-6).reverse())}
      </section>
    </div>
  </div>`;
}

function selectLiveAgent(agents) {
  const values = Object.values(agents);
  return values
    .slice()
    .sort((left, right) => {
      const phaseRank = phase => phase === "working" ? 0 : phase === "waiting" ? 1 : phase === "blocked" ? 2 : phase === "error" ? 3 : 4;
      const leftRank = phaseRank(left.phase);
      const rightRank = phaseRank(right.phase);
      if (leftRank !== rightRank) return leftRank - rightRank;
      const leftTime = left.updated_at ? new Date(left.updated_at).getTime() : 0;
      const rightTime = right.updated_at ? new Date(right.updated_at).getTime() : 0;
      return rightTime - leftTime;
    })[0] || null;
}

function renderEvaluationAnalysis(analysis) {
  if (!analysis) return `<section class="card"><div class="card-heading"><h3>AI evaluation analysis</h3><span class="badge">every loop</span></div><div class="muted small">After criteria are checked, the evaluator AI diagnoses the pass/fail evidence here before the strategist chooses the next hypothesis.</div></section>`;
  const items = analysis.criterion_analyses || [];
  const focus = analysis.recommended_next_focus || [];
  return `<section class="card evaluation-analysis-card">
    <div class="card-heading"><h3>AI evaluation analysis</h3><span class="badge ${analysis.source === "ai" ? "pass" : "error"}">${esc(analysis.source || "ai")}</span></div>
    <div class="analysis-label">${esc(analysis.label || "Latest evaluation")} · iteration ${esc(analysis.iteration ?? "-")}</div>
    <div class="analysis-summary">${esc(analysis.summary || "Analysis complete")}</div>
    ${analysis.material_progress ? `<div class="analysis-progress-badge">Concrete partial progress detected</div>` : ""}
    ${analysis.progress_assessment ? `<p class="hypothesis-outcome"><strong>Progress:</strong> ${esc(analysis.progress_assessment)}</p>` : ""}
    ${analysis.progress_evidence?.length ? `<p class="hypothesis-outcome"><strong>Progress evidence:</strong> ${esc(analysis.progress_evidence.join("; "))}</p>` : ""}
    ${items.length ? `<details class="analysis-details"><summary>Criterion-by-criterion diagnosis (${items.length})</summary><div class="analysis-list">${items.map(item => `<article class="analysis-item"><div><span class="criterion-id">${esc(item.criterion_id)}</span> <span class="badge ${esc(item.observed_status)}">${esc(item.observed_status)}</span></div><p>${esc(item.interpretation)}</p>${item.likely_causes?.length ? `<small><strong>Likely causes:</strong> ${esc(item.likely_causes.join("; "))}</small>` : ""}${item.recommended_actions?.length ? `<small><strong>Next:</strong> ${esc(item.recommended_actions.join("; "))}</small>` : ""}</article>`).join("")}</div></details>` : ""}
    ${focus.length ? `<div class="analysis-focus"><strong>Recommended next focus</strong>${focus.map(item => `<div>• ${esc(item)}</div>`).join("")}</div>` : ""}
  </section>`;
}

function criterionStatus(c, result) {
  const status = result?.status || "unchecked";
  const summary = result?.summary || "Not checked";
  const isTimeout = status === "timeout";
  const timeoutNote = isTimeout ? ` <span class="timeout-note" title="This criterion command took longer than its configured timeout. The evaluator aborted it to avoid blocking the loop. Consider increasing timeout_seconds in the criterion definition.">⏱ timed out</span>` : "";
  return `<article class="criterion-status">
    <div class="criterion-header">
      <div><span class="criterion-id">${esc(c.id)}</span>${c.required ? ` <span class="badge">required</span>` : ""}</div>
      <div class="criterion-tools"><span class="badge ${esc(status)}">${esc(status)}</span>
        <select data-criterion-override="${esc(c.id)}" title="Human override">
          <option value="auto" ${c.override === "auto" ? "selected" : ""}>Auto</option>
          <option value="pass" ${c.override === "pass" ? "selected" : ""}>Force pass</option>
          <option value="fail" ${c.override === "fail" ? "selected" : ""}>Force fail</option>
        </select>
      </div>
    </div>
    <div class="criterion-description">${esc(c.description)}</div>
    <div class="criterion-summary">${esc(summary)}${timeoutNote}${result?.evaluation_method ? ` <span class="evaluation-method">${esc(result.evaluation_method.replaceAll("_", " "))}</span>` : ""}${result?.evidence?.length ? ` · ${esc(result.evidence.slice(0,2).join("; "))}` : ""}</div>
  </article>`;
}

function renderDefinition() {
  const d = state.detail;
  return `<div class="definition-actions">
    <div><button class="button ghost" data-ai="goal">AI refine goal</button> <button class="button ghost" data-ai="criteria">AI improve criteria</button></div>
    <div><button class="button ghost" data-add-criterion>Add criterion</button> <button class="button ghost" data-setup-complete>Save & finish setup</button> <button class="button primary" data-save-definition>Save changes</button></div>
  </div>
  <div class="definition-layout">
    <section class="card goal-definition-card">
      <div class="card-heading"><h3>Goal definition</h3><span class="muted small">Describe the outcome the loop should achieve</span></div>
      <div class="goal-definition-grid">
        <label>Title<input id="edit-title" value="${esc(d.metadata.title)}"></label>
        <label>Description<textarea id="edit-description" rows="3">${esc(d.metadata.description || "")}</textarea></label>
        <label class="full">Goal<textarea id="edit-goal" rows="8">${esc(d.goal)}</textarea></label>
      </div>
    </section>
    <section class="card criteria-card">
      <div class="card-heading criteria-card-heading">
        <div><h3>Success criteria</h3><p class="section-help">Each required criterion must pass before the loop can finish.</p></div>
        <span class="quiet-pill">${d.criteria.criteria.length} criterion${d.criteria.criteria.length === 1 ? "" : "a"}</span>
      </div>
      <div id="criteria-editor" class="criteria-editor">${d.criteria.criteria.map((c, i) => criterionEditor(c, i)).join("") || `<div class="empty-criteria"><strong>No criteria yet</strong><span>Add at least one concrete, required success criterion.</span></div>`}</div>
    </section>
  </div>`;
}

function criterionEditor(c, index) {
  const kind = c.kind;
  const kinds = [["command","Automated command / test"],["file_exists","File exists"],["file_contains","File contains text"],["ai_judge","AI evidence review"],["manual","Human approval only"]];
  const kindLabel = kinds.find(([value]) => value === kind)?.[1] || kind;
  let specific = "";
  let sectionNote = "Configure the evidence used to decide whether this criterion passes.";
  if (kind === "command") {
    sectionNote = "The command is run on every evaluation; its exit code is authoritative.";
    specific = `
      <label class="full">Command<span class="field-help">Run relative to the project folder.</span><input data-field="command" value="${esc(c.command || "")}"></label>
      <label>Expected exit code<input type="number" data-field="expected_exit_code" value="${c.expected_exit_code ?? 0}"></label>
      <label>Timeout seconds<span class="field-help">Leave blank to use the project default.</span><input type="number" data-field="timeout_seconds" value="${c.timeout_seconds ?? ""}" placeholder="Project default"></label>`;
  }
  if (kind === "file_exists") {
    sectionNote = "Passes when the path exists inside the project.";
    specific = `<label class="full">File path<span class="field-help">Use a path relative to the project root.</span><input data-field="path" value="${esc(c.path || "")}" placeholder="relative/to/project"></label>`;
  }
  if (kind === "file_contains") {
    sectionNote = "Passes when the selected file contains the required text or pattern.";
    specific = `
      <label>File path<input data-field="path" value="${esc(c.path || "")}"></label>
      <label>Required text or pattern<input data-field="contains" value="${esc(c.contains || "")}"></label>
      <div class="criterion-toggle-group full">
        <label class="checkbox-row"><input type="checkbox" data-field="regex" ${c.regex ? "checked" : ""}>Regular expression</label>
        <label class="checkbox-row"><input type="checkbox" data-field="case_sensitive" ${c.case_sensitive !== false ? "checked" : ""}>Case sensitive</label>
      </div>`;
  }
  if (kind === "ai_judge") {
    sectionNote = "The evaluator AI must use these exact rules and concrete project evidence.";
    specific = `
      <label class="full judge-prompt-field">Judge prompt<span class="field-help">Include explicit PASS only if and FAIL if rules.</span><textarea data-field="judge_prompt" rows="6">${esc(c.judge_prompt || "")}</textarea></label>
      <label class="evidence-paths-field">Evidence paths<span class="field-help">One project-relative path per line.</span><textarea data-field="evidence_paths" rows="4" placeholder="one path per line">${esc((c.evidence_paths || []).join("\n"))}</textarea></label>
      <label class="confidence-field">Confidence threshold<span class="field-help">Minimum confidence required to pass.</span><input type="number" min="0" max="1" step="0.05" data-field="confidence_threshold" value="${c.confidence_threshold ?? .75}"></label>`;
  }
  if (kind === "manual") {
    sectionNote = "Manual criteria require a person to explicitly mark pass or fail.";
    specific = `<div class="manual-criterion-warning full"><div><strong>Human approval only</strong><p>This type cannot be passed by the autonomous loop. Use it only when a person must personally approve the result.</p></div><button class="button primary compact" type="button" data-convert-ai="${index}">Convert to AI review</button></div>`;
  }
  return `<article class="criterion-editor" data-criterion-index="${index}">
    <header class="criterion-editor-header">
      <div class="criterion-title-block">
        <span class="criterion-number">${index + 1}</span>
        <div class="criterion-title-copy"><strong>Criterion ${index + 1}</strong><span>${esc(kindLabel)}${c.required ? " · Required" : " · Optional"}</span></div>
      </div>
      <button class="button ghost danger-text compact" type="button" data-remove-criterion="${index}">Remove</button>
    </header>

    <section class="criterion-editor-section">
      <div class="criterion-section-heading"><strong>Definition</strong><span>Name the outcome clearly and concretely.</span></div>
      <div class="criterion-definition-grid">
        <label>ID<span class="field-help">Stable machine-readable name.</span><input data-field="id" value="${esc(c.id)}"></label>
        <label>Description<span class="field-help">The observable result that must be true.</span><input data-field="description" value="${esc(c.description)}"></label>
      </div>
    </section>

    <section class="criterion-editor-section criterion-options-section">
      <div class="criterion-section-heading"><strong>Checking method</strong><span>Choose how the loop decides pass or fail.</span></div>
      <div class="criterion-options-grid">
        <label>How it is checked<select data-field="kind">${kinds.map(([k,label]) => `<option value="${k}" ${k === kind ? "selected" : ""}>${label}</option>`).join("")}</select></label>
        <label>Override<select data-field="override"><option value="auto" ${c.override === "auto" ? "selected" : ""}>Automatic</option><option value="pass" ${c.override === "pass" ? "selected" : ""}>Force pass</option><option value="fail" ${c.override === "fail" ? "selected" : ""}>Force fail</option></select></label>
        <label class="required-toggle"><span><strong>Required</strong><small>The goal cannot finish until this passes.</small></span><input type="checkbox" data-field="required" ${c.required ? "checked" : ""}></label>
      </div>
    </section>

    <section class="criterion-editor-section criterion-evaluation-section">
      <div class="criterion-section-heading"><strong>Evaluation details</strong><span>${esc(sectionNote)}</span></div>
      <div class="criterion-specific">${specific}</div>
    </section>
  </article>`;
}

function collectCriteria() {
  return $$("[data-criterion-index]").map(card => {
    const get = name => card.querySelector(`[data-field="${name}"]`);
    const val = name => get(name)?.value ?? null;
    const checked = name => !!get(name)?.checked;
    const kind = val("kind");
    const item = {
      id: val("id").trim(), description: val("description").trim(), kind,
      required: checked("required"), override: val("override") || "auto",
      expected_exit_code: 0, regex: false, case_sensitive: true,
      evidence_paths: [], confidence_threshold: .75,
    };
    if (kind === "command") {
      item.command = val("command").trim();
      item.expected_exit_code = Number(val("expected_exit_code") || 0);
      item.timeout_seconds = val("timeout_seconds") ? Number(val("timeout_seconds")) : null;
    }
    if (kind === "file_exists") item.path = val("path").trim();
    if (kind === "file_contains") {
      item.path = val("path").trim(); item.contains = val("contains");
      item.regex = checked("regex"); item.case_sensitive = checked("case_sensitive");
    }
    if (kind === "ai_judge") {
      item.judge_prompt = val("judge_prompt").trim();
      item.evidence_paths = val("evidence_paths").split(/\r?\n|,/).map(x => x.trim()).filter(Boolean);
      item.confidence_threshold = Number(val("confidence_threshold") || .75);
    }
    return item;
  });
}

function renderHistory() {
  const hypotheses = [...(state.detail.state.hypotheses || [])].reverse();
  return `<div class="grid two">
    <section class="card"><div class="card-heading"><h3>Hypotheses</h3><span class="muted small">${hypotheses.length} recorded</span></div>
      ${hypotheses.length ? hypotheses.map(h => `<article class="hypothesis-card">
        <div class="hypothesis-meta"><span>${esc(h.id)}</span><span class="badge ${esc(h.status)}">${esc(h.status)}</span><span>iteration ${h.iteration}</span></div>
        <div class="hypothesis-title">${esc(h.statement)}</div>
        <div class="hypothesis-outcome"><strong>Rationale:</strong> ${esc(h.rationale)}</div>
        <div class="hypothesis-outcome"><strong>Plan:</strong> ${esc((h.plan || []).join(" → "))}</div>
        <div class="hypothesis-outcome"><strong>Outcome:</strong> ${esc(h.outcome || "Pending")}</div>
      </article>`).join("") : `<div class="muted small">No hypotheses yet.</div>`}
    </section>
    <section class="card"><div class="card-heading"><h3>Event log</h3><span class="muted small">latest 150</span></div>${renderEventList([...(state.detail.events || [])].reverse())}</section>
  </div>`;
}

function renderEventList(events) {
  return `<div class="event-list">${events.length ? events.map(event => `<div class="event">
    <span class="event-time">${esc(fmtTime(event.timestamp))}</span>
    <span class="event-type">${esc(event.type)}</span>
    <span class="event-message">${esc(event.message)}</span>
  </div>`).join("") : `<div class="muted small">No events recorded.</div>`}</div>`;
}

function renderFiles() {
  const d = state.detail;
  const projectPaths = { project: state.project.project_dir, agent_root: state.project.agent_root, config: state.project.paths?.config || `${state.project.agent_root}\\config.yaml` };
  return `<div class="grid two">
    <section class="card"><h3>Project files</h3><p class="muted small">These are the same files shown by the CLI <code>files</code> command. External edits are reread by the loop.</p>
      <div class="code-paths">${Object.entries(projectPaths).map(([name, path]) => `<div><span class="eyebrow">${esc(name)}</span><code>${esc(path)}</code></div>`).join("")}</div>
    </section>
    <section class="card"><h3>Goal files</h3>
      <div class="code-paths">${Object.entries(d.paths).map(([name, path]) => `<div><span class="eyebrow">${esc(name)}</span><code>${esc(path)}</code></div>`).join("")}</div>
    </section>
    <section class="card"><h3>Goal management</h3>
      <p class="muted small">Archiving hides a goal from the normal list while retaining its complete history. Deleting permanently removes this goal's control, status, and run files.</p>
      <label class="checkbox-row"><input id="archive-goal" type="checkbox" ${d.metadata.archived ? "checked" : ""}>Archive this goal</label>
      <button class="button ghost" style="margin-top:12px" data-save-archive>Save archive setting</button>
      <hr style="border:0;border-top:1px solid var(--border);margin:22px 0">
      <button class="button danger" data-delete-goal>Delete goal permanently</button>
    </section>
  </div>`;
}

function bindGlobalEvents() {
  $("#new-goal-button").addEventListener("click", () => state.projectId ? openModal("new-goal-modal") : openProjectManager());
  $("#empty-new-goal").addEventListener("click", () => state.projectId ? openModal("new-goal-modal") : openProjectManager());
  $("#settings-button").addEventListener("click", () => state.projectId ? openSettings() : openProjectManager());
  $("#projects-button").addEventListener("click", openProjectManager);
  $("#validate-button").addEventListener("click", validateProject);
  $("#project-selector").addEventListener("change", async event => { state.projectId = event.target.value || null; if (state.projectId) await loadProject(); else renderNoProject(); });
  $("#goal-filter").addEventListener("input", renderGoalList);
  $("#show-archived").addEventListener("change", async event => { state.includeArchived = event.target.checked; await refreshGoals(); if (state.selectedId && !state.goals.some(g => g.metadata.id === state.selectedId)) { state.selectedId = null; state.detail = null; state.goals.length ? await selectGoal(state.goals[0].metadata.id) : renderNoGoals(); } });
  $$(".modal-close").forEach(button => button.addEventListener("click", () => closeModal(button.dataset.close)));
  $("#new-goal-form").addEventListener("submit", createGoal);
  $("#create-project-form").addEventListener("submit", createProject);
  $("#open-project-form").addEventListener("submit", openProject);
  $("#settings-form").addEventListener("submit", saveSettings);
  $("#ai-generate").addEventListener("click", generateAIProposal);
  $("#ai-retry-last").addEventListener("click", retryLastAIMessage);
  $("#ai-cancel").addEventListener("click", cancelAIProposal);
  $("#ai-reset").addEventListener("click", resetAIRefinement);
  $("#ai-apply").addEventListener("click", applyAIProposal);
  $("#ai-finalize").addEventListener("click", finalizeAIProposal);

  document.addEventListener("click", async event => {
    const close = event.target.closest("[data-close]");
    if (close) { closeModal(close.dataset.close); return; }
    if (event.target.closest("[data-open-projects]")) {
      const shortcut = event.target.closest("[data-project-mode-shortcut]")?.dataset.projectModeShortcut;
      openProjectManager();
      if (shortcut && typeof switchProjectMode === "function") switchProjectMode(shortcut);
      return;
    }
    if (event.target.closest("[data-open-new-goal]")) return openModal("new-goal-modal");
    const activateProjectButton = event.target.closest("[data-project-activate]");
    if (activateProjectButton) return activateProject(activateProjectButton.dataset.projectActivate);
    const removeProjectButton = event.target.closest("[data-project-remove]");
    if (removeProjectButton) return removeProject(removeProjectButton.dataset.projectRemove);
    const select = event.target.closest("[data-select-goal]");
    if (select) return selectGoal(select.dataset.selectGoal);
    const tab = event.target.closest("[data-tab]");
    if (tab) { state.activeTab = tab.dataset.tab; renderDetail(); return; }
    const action = event.target.closest("[data-action]");
    if (action) return runAction(action.dataset.action);
    const bulk = event.target.closest("[data-bulk]");
    if (bulk) return runBulkAction(bulk.dataset.bulk);
    if (event.target.closest("[data-load-models]")) return loadModels();
    if (event.target.closest("[data-add-steering]")) return addSteering();
    if (event.target.closest("[data-save-definition]")) return saveDefinition();
    if (event.target.closest("[data-setup-complete]")) return completeSetup();
    if (event.target.closest("[data-add-criterion]")) return addCriterion();
    const convertAI = event.target.closest("[data-convert-ai]");
    if (convertAI) return convertCriterionToAI(Number(convertAI.dataset.convertAi));
    const remove = event.target.closest("[data-remove-criterion]");
    if (remove) return removeCriterion(Number(remove.dataset.removeCriterion));
    const ai = event.target.closest("[data-ai]");
    if (ai) return openAI(ai.dataset.ai);
    if (event.target.closest("[data-save-archive]")) return saveArchive();
    if (event.target.closest("[data-delete-goal]")) return deleteGoal();
  });

  document.addEventListener("change", async event => {
    if (event.target.id === "goal-model") return setGoalModel(event.target.value);
    if (event.target.matches("[data-criterion-override]")) return setOverride(event.target.dataset.criterionOverride, event.target.value);
    if (event.target.matches('[data-field="kind"]')) {
      captureDefinitionText();
      state.detail.criteria.criteria = collectCriteria();
      state.editorDirty = true;
      renderDetail();
    }
  });

  document.addEventListener("input", event => {
    if (event.target.id === "steering-message" && state.selectedId) {
      state.steeringDrafts[state.selectedId] = event.target.value;
      state.interactionHoldUntil = Date.now() + 1000;
    }
    if (state.activeTab === "definition" && event.target.closest("#tab-content")) state.editorDirty = true;
  });

  document.addEventListener("focusout", event => {
    // Do not redraw immediately: focusout fires before a button click. Hold the
    // current DOM briefly so the intended click can complete against the same node.
    if (event.target.closest?.("#main-content") && isEditableElement(event.target)) {
      state.interactionHoldUntil = Date.now() + 750;
    }
    captureSteeringDraft();
  });

  document.addEventListener("selectionchange", () => {
    if (isSelectionInsideMainContent()) {
      // Keep the selected text stable while polling continues in the background.
      state.interactionHoldUntil = Date.now() + 2500;
    }
  });

  document.addEventListener("mousedown", event => {
    if (event.target.closest?.("#main-content")) {
      state.interactionHoldUntil = Date.now() + 1500;
    }
  });

  document.addEventListener("copy", event => {
    if (event.target.closest?.("#main-content") || isSelectionInsideMainContent()) {
      state.interactionHoldUntil = Date.now() + 2000;
    }
  });
}

function openProjectManager() {
  renderProjectManager();
  openModal("projects-modal");
  if (typeof loadFriendlySystemInfo === "function") loadFriendlySystemInfo();
  if (typeof validateProjectPath === "function") { validateProjectPath("create"); validateProjectPath("open"); }
}

function renderProjectManager() {
  const list = $("#project-manager-list");
  const filter = ($("#project-filter")?.value || "").trim().toLowerCase();
  const projects = state.projects.filter(project => `${project.title} ${project.path}`.toLowerCase().includes(filter));
  if (!projects.length) {
    list.innerHTML = `<div class="project-list-empty"><span>${state.projects.length ? "⌕" : "▣"}</span><strong>${state.projects.length ? "No matching projects" : "No recent projects yet"}</strong><p>${state.projects.length ? "Try a different search." : "Create a project or use Find projects to locate existing workspaces."}</p></div>`;
    return;
  }
  list.innerHTML = projects.map(project => `<article class="managed-project ${project.id === state.projectId ? "selected" : ""}">
    <div class="project-avatar">${esc((project.title || "P").slice(0, 1).toUpperCase())}</div>
    <div class="managed-project-copy">
      <strong>${esc(project.title)}</strong>
      <code title="${esc(project.path)}">${esc(project.path)}</code>
      <span class="muted small">${project.initialized ? `${project.goal_count} goal${project.goal_count === 1 ? "" : "s"} · ${project.active_goal_count} active` : "Workspace unavailable"}</span>
    </div>
    <div class="managed-project-actions">
      <button class="icon-button compact-icon" type="button" data-reveal-project="${esc(project.path)}" title="Open folder">↗</button>
      <button class="button ${project.id === state.projectId ? "primary" : "ghost"} compact" data-project-activate="${esc(project.id)}" ${project.initialized ? "" : "disabled"}>${project.id === state.projectId ? "Current" : "Open"}</button>
      <button class="icon-button compact-icon danger-text" data-project-remove="${esc(project.id)}" title="Remove from dashboard">×</button>
    </div>
  </article>`).join("");
}

async function createProject(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const payload = {
    path: String(form.get("path") || "").trim(),
    title: String(form.get("title") || "").trim(),
    model: String(form.get("model") || "").trim() || null,
    force: !!form.get("force"),
  };
  try {
    const project = await api("/api/projects/create", { method: "POST", body: JSON.stringify(payload) }, false);
    state.projectId = project.id;
    event.target.reset();
    await refreshProjects();
    await loadProject();
    closeModal("projects-modal");
    toast("Project created");
  } catch (error) { toast(error.message, "error"); }
}

async function openProject(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const payload = { path: String(form.get("path") || "").trim(), title: String(form.get("title") || "").trim() };
  try {
    const project = await api("/api/projects/open", { method: "POST", body: JSON.stringify(payload) }, false);
    state.projectId = project.id;
    event.target.reset();
    await refreshProjects();
    await loadProject();
    closeModal("projects-modal");
    toast("Project opened");
  } catch (error) { toast(error.message, "error"); }
}

async function activateProject(projectId) {
  try {
    state.projectId = projectId;
    await api(`/api/projects/${encodeURIComponent(projectId)}/activate`, { method: "POST" }, false);
    await refreshProjects();
    await loadProject();
    closeModal("projects-modal");
    toast("Project switched");
  } catch (error) { toast(error.message, "error"); }
}

async function removeProject(projectId) {
  const project = state.projects.find(item => item.id === projectId);
  if (!confirm(`Remove '${project?.title || projectId}' from this dashboard? Its files will not be deleted.`)) return;
  try {
    await api(`/api/projects/${encodeURIComponent(projectId)}`, { method: "DELETE" }, false);
    if (state.projectId === projectId) state.projectId = null;
    await refreshProjects();
    renderProjectManager();
    if (state.projectId) await loadProject(); else renderNoProject();
    toast("Project removed from dashboard");
  } catch (error) { toast(error.message, "error"); }
}

async function validateProject() {
  if (!state.projectId) return openProjectManager();
  openModal("validation-modal");
  $("#validation-result").innerHTML = `<div class="working"><span class="spinner"></span> Validating project…</div>`;
  try {
    const result = await api("/api/project/validate", { method: "POST" });
    const goalRows = Object.entries(result.goals || {}).map(([id, value]) => {
      const warnings = value.warnings || [];
      const stateClass = value.valid ? (warnings.length ? "warning" : "pass") : "fail";
      const icon = value.valid ? (warnings.length ? "!" : "✓") : "×";
      return `<div class="validation-row ${stateClass}"><span>${icon}</span><div><strong>${esc(id)}</strong>${value.errors.length ? `<ul>${value.errors.map(error => `<li>${esc(error)}</li>`).join("")}</ul>` : `<p>Goal and criteria are valid.</p>`}${warnings.length ? `<ul class="validation-warnings">${warnings.map(warning => `<li>${esc(warning)}</li>`).join("")}</ul>` : ""}</div></div>`;
    }).join("");
    $("#validation-result").innerHTML = `<div class="validation-summary ${result.valid ? "pass" : "fail"}">${result.valid ? "Project is ready" : "Project needs attention"}</div>
      ${result.checks.map(check => `<div class="validation-row ${check.passed ? "pass" : "fail"}"><span>${check.passed ? "✓" : "×"}</span><div><strong>${esc(check.name)}</strong><p>${esc(check.detail)}</p></div></div>`).join("")}
      <h3>Goals</h3>${goalRows || `<p class="muted small">No goals found.</p>`}`;
  } catch (error) { $("#validation-result").innerHTML = `<div class="validation-summary fail">${esc(error.message)}</div>`; }
}

async function createGoal(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const payload = Object.fromEntries(form.entries());
  payload.title = String(payload.title || "").trim();
  payload.id = String(payload.id || "").trim() || payload.title.toLowerCase().replace(/[^a-z0-9._-]+/g, "-").replace(/^[._-]+|[._-]+$/g, "") || `goal-${Date.now()}`;
  try {
    const detail = await api("/api/goals", { method: "POST", body: JSON.stringify(payload) });
    closeModal("new-goal-modal"); event.target.reset();
    const goalIdInput = $("#new-goal-id"); if (goalIdInput) delete goalIdInput.dataset.manual;
    await refreshGoals();
    state.selectedId = detail.metadata.id; state.detail = detail; state.activeTab = "definition";
    renderDetail(); toast("Goal created");
  } catch (error) { toast(error.message, "error"); }
}

async function runAction(action) {
  try {
    await api(`/api/goals/${encodeURIComponent(state.selectedId)}/action`, { method: "POST", body: JSON.stringify({ action }) });
    toast(`${action} requested`); await poll();
  } catch (error) { toast(error.message, "error"); }
}

async function runBulkAction(action) {
  try {
    const result = await api("/api/goals/actions", { method: "POST", body: JSON.stringify({ action }) });
    const failures = Object.entries(result.results || {}).filter(([, value]) => value !== "started");
    toast(failures.length ? failures.map(([id,v]) => `${id}: ${v}`).join("; ") : `${action} all requested`, failures.length ? "error" : "success");
    await poll();
  } catch (error) { toast(error.message, "error"); }
}

async function loadModels() {
  try {
    toast("Loading models…");
    const result = await api("/api/models"); state.models = result.models; renderDetail();
    toast(`${state.models.length} models loaded`);
  } catch (error) { toast(error.message, "error"); }
}

async function setGoalModel(model) {
  try {
    await api(`/api/goals/${encodeURIComponent(state.selectedId)}/model`, { method: "PUT", body: JSON.stringify({ model: model || null }) });
    state.detail = await api(`/api/goals/${encodeURIComponent(state.selectedId)}`); renderDetail(); toast("Goal model updated");
  } catch (error) { toast(error.message, "error"); }
}

async function addSteering() {
  const input = $("#steering-message");
  const message = (input?.value ?? steeringDraft()).trim();
  if (!message) return;
  try {
    await api(`/api/goals/${encodeURIComponent(state.selectedId)}/steering`, { method: "POST", body: JSON.stringify({ message }) });
    state.steeringDrafts[state.selectedId] = "";
    if (input) input.value = "";
    toast("Steering note added");
  } catch (error) { toast(error.message, "error"); }
}

async function setOverride(criterionId, value) {
  try {
    await api(`/api/goals/${encodeURIComponent(state.selectedId)}/criteria/${encodeURIComponent(criterionId)}/override`, { method: "PUT", body: JSON.stringify({ value }) });
    state.detail = await api(`/api/goals/${encodeURIComponent(state.selectedId)}`); renderDetail(); toast("Override updated");
  } catch (error) { toast(error.message, "error"); }
}

function captureDefinitionText() {
  const title = $("#edit-title");
  if (!title) return;
  state.detail.metadata.title = title.value;
  state.detail.metadata.description = $("#edit-description").value;
  state.detail.goal = $("#edit-goal").value;
}

function convertCriterionToAI(index) {
  captureDefinitionText();
  const current = collectCriteria();
  const item = current[index];
  if (!item) return;
  item.kind = "ai_judge";
  item.judge_prompt = `Inspect the project and determine whether this criterion is satisfied using concrete evidence: ${item.description}`;
  item.evidence_paths = [];
  item.confidence_threshold = .75;
  item.override = "auto";
  state.detail.criteria.criteria = current;
  state.editorDirty = true;
  renderDetail();
  toast("Criterion will now be checked by AI each loop");
}

function addCriterion() {
  captureDefinitionText();
  const current = collectCriteria();
  current.push({ id: `criterion-${current.length + 1}`, description: "Describe measurable proof of success", kind: "ai_judge", judge_prompt: "Inspect the project and decide whether this outcome is satisfied now using concrete evidence. Fail when evidence is missing or ambiguous.", required: true, override: "auto", expected_exit_code: 0, regex: false, case_sensitive: true, evidence_paths: [], confidence_threshold: .75 });
  state.detail.criteria.criteria = current; state.editorDirty = true; renderDetail();
}

function removeCriterion(index) {
  captureDefinitionText();
  const current = collectCriteria(); current.splice(index, 1);
  state.detail.criteria.criteria = current; state.editorDirty = true; renderDetail();
}

async function saveDefinition() {
  try {
    const criteria = collectCriteria();
    if (!criteria.some(c => c.required)) throw new Error("At least one criterion must be required");
    const title = $("#edit-title").value.trim();
    const description = $("#edit-description").value;
    const goal = $("#edit-goal").value.trim();
    if (!title || !goal) throw new Error("Title and goal are required");
    await api(`/api/goals/${encodeURIComponent(state.selectedId)}/metadata`, { method: "PATCH", body: JSON.stringify({ title, description, archived: state.detail.metadata.archived }) });
    await api(`/api/goals/${encodeURIComponent(state.selectedId)}/goal`, { method: "PUT", body: JSON.stringify({ goal }) });
    await api(`/api/goals/${encodeURIComponent(state.selectedId)}/criteria`, { method: "PUT", body: JSON.stringify({ criteria }) });
    state.editorDirty = false;
    state.detail = await api(`/api/goals/${encodeURIComponent(state.selectedId)}`);
    await refreshGoals(); renderDetail(); toast("Goal and criteria saved"); return true;
  } catch (error) { toast(error.message, "error"); return false; }
}

async function completeSetup() {
  if (!await saveDefinition()) return;
  try {
    state.detail = await api(`/api/goals/${encodeURIComponent(state.selectedId)}/setup-complete`, { method: "POST" });
    renderDetail(); await refreshGoals(); toast("Setup marked complete; goal is paused and ready to start");
  } catch (error) { toast(error.message, "error"); }
}

function openSettings() {
  const c = state.project.config;
  $("#settings-form").innerHTML = `
    <section class="settings-section full"><div class="settings-section-heading"><span class="settings-icon">✦</span><div><h3>Everyday settings</h3><p>These are the settings most projects need.</p></div></div><div class="form-grid two-column settings-grid">
      <label>Default model<span>Used unless a goal selects its own model</span><input name="model" value="${esc(c.model || "")}" placeholder="Use the OpenCode default"></label>
      <label>Goals allowed to run together<input name="max_concurrent_goals" type="number" min="1" max="32" value="${c.max_concurrent_goals}"></label>
      <label class="checkbox-row full"><input name="auto_approve" type="checkbox" ${c.auto_approve ? "checked" : ""}>Allow OpenCode tool actions automatically</label>
      <label class="checkbox-row full"><input name="gui_auto_resume_running_goals" type="checkbox" ${c.gui_auto_resume_running_goals ? "checked" : ""}>Resume goals that were running when the GUI restarts</label>
    </div></section>

    <details class="settings-details full"><summary><span>OpenCode connection</span><small>Command, server attachment, and agent names</small></summary><div class="form-grid two-column settings-details-body">
      <label>OpenCode command<input name="opencode_command" value="${esc((c.opencode_command || []).join(" "))}"></label>
      <label>Attach URL<input name="attach_url" value="${esc(c.attach_url || "")}" placeholder="http://127.0.0.1:4096"></label>
      <label>Attach username<input name="attach_username" value="${esc(c.attach_username || "")}"></label>
      <label>Password environment variable<input name="attach_password_env" value="${esc(c.attach_password_env || "")}" placeholder="OPENCODE_PASSWORD"></label>
      <label>Strategist agent<input name="strategist_agent" value="${esc(c.strategist_agent)}"></label>
      <label>Executor agent<input name="executor_agent" value="${esc(c.executor_agent)}"></label>
      <label>Evaluator agent<input name="evaluator_agent" value="${esc(c.evaluator_agent)}"></label>
    </div></details>

    <details class="settings-details full"><summary><span>Loop tuning</span><small>Timing, limits, and hypothesis retention</small></summary><div class="form-grid two-column settings-details-body">
      <label>Iteration delay seconds<input name="iteration_delay_seconds" type="number" min="0" step=".1" value="${c.iteration_delay_seconds}"></label>
      <label>Loop poll interval seconds<input name="poll_interval_seconds" type="number" min=".05" step=".05" value="${c.poll_interval_seconds}"></label>
      <label>Status refresh seconds<input name="status_refresh_seconds" type="number" min=".05" step=".05" value="${c.status_refresh_seconds}"></label>
      <label>OpenCode timeout seconds<input name="opencode_timeout_seconds" type="number" min="1" value="${c.opencode_timeout_seconds}"></label>
      <label>Criterion timeout seconds<input name="criterion_timeout_seconds" type="number" min="1" value="${c.criterion_timeout_seconds}"></label>
      <label>Maximum iterations<span>Leave blank for unlimited</span><input name="max_iterations" type="number" min="1" value="${c.max_iterations ?? ""}" placeholder="Unlimited"></label>
      <label>Recent hypotheses retained<input name="max_recent_hypotheses" type="number" min="1" value="${c.max_recent_hypotheses}"></label>
      <label>Rethink after no-progress iterations<input name="no_progress_rethink_after" type="number" min="1" value="${c.no_progress_rethink_after}"></label>
    </div></details>

    <details class="settings-details full"><summary><span>GUI server</span><small>Applied the next time the GUI starts</small></summary><div class="form-grid two-column settings-details-body">
      <label>GUI host<input name="gui_host" value="${esc(c.gui_host)}"></label>
      <label>GUI port<input name="gui_port" type="number" min="1" max="65535" value="${c.gui_port}"></label>
    </div></details>
    <div class="form-actions full sticky-form-actions"><button type="button" class="button ghost modal-close" data-close="settings-modal">Cancel</button><button type="submit" class="button primary">Save settings</button></div>`;
  openModal("settings-modal");
}

async function saveSettings(event) {
  event.preventDefault();
  const f = new FormData(event.target);
  const optionalNumber = name => String(f.get(name) || "").trim() === "" ? null : Number(f.get(name));
  const payload = {
    model: f.get("model") || null,
    opencode_command: String(f.get("opencode_command") || "opencode").match(/(?:[^\s"]+|"[^"]*")+/g)?.map(x => x.replace(/^"|"$/g, "")) || ["opencode"],
    attach_url: f.get("attach_url") || null,
    attach_username: f.get("attach_username") || null,
    attach_password_env: f.get("attach_password_env") || null,
    max_concurrent_goals: Number(f.get("max_concurrent_goals")),
    iteration_delay_seconds: Number(f.get("iteration_delay_seconds")),
    poll_interval_seconds: Number(f.get("poll_interval_seconds")),
    status_refresh_seconds: Number(f.get("status_refresh_seconds")),
    opencode_timeout_seconds: Number(f.get("opencode_timeout_seconds")),
    criterion_timeout_seconds: Number(f.get("criterion_timeout_seconds")),
    max_iterations: optionalNumber("max_iterations"),
    max_recent_hypotheses: Number(f.get("max_recent_hypotheses")),
    strategist_agent: f.get("strategist_agent"), executor_agent: f.get("executor_agent"), evaluator_agent: f.get("evaluator_agent"),
    no_progress_rethink_after: Number(f.get("no_progress_rethink_after")),
    gui_host: f.get("gui_host"), gui_port: Number(f.get("gui_port")),
    auto_approve: !!f.get("auto_approve"), gui_auto_resume_running_goals: !!f.get("gui_auto_resume_running_goals"),
  };
  try {
    await api("/api/project/config", { method: "PATCH", body: JSON.stringify(payload) });
    closeModal("settings-modal"); await refreshProject(); toast("Settings saved");
  } catch (error) { toast(error.message, "error"); }
}

async function openAI(mode) {
  state.aiMode = mode; state.aiProposal = null; state.aiConversation = ""; state.aiJobId = null; state.aiSession = null;
  state.aiLastSubmittedFeedback = "";
  state.aiRetryFeedback = "";
  $("#ai-modal-title").textContent = mode === "goal" ? "Finalize goal and success criteria" : "Make success criteria concrete";
  $("#ai-help").textContent = mode === "goal"
    ? "Keep replying until the questions are resolved and the AI marks the draft ready. The conversation is saved with this goal."
    : "Ask the AI to replace vague criteria with exact evidence, thresholds, commands, and strict AI-review rubrics.";
  $("#ai-feedback").value = "";
  $("#ai-result").className = "ai-result empty-result";
  $("#ai-result").textContent = "Loading the saved refinement conversation…";
  $("#ai-working").classList.add("hidden");
  $("#ai-cancel").classList.add("hidden");
  $("#ai-apply").classList.add("hidden");
  $("#ai-finalize").classList.add("hidden");
  updateAIRetryButton();
  openModal("ai-modal");
  try {
    state.aiSession = await api(`/api/goals/${encodeURIComponent(state.selectedId)}/refinement-session`);
    const priorUserMessage = latestUserMessage(state.aiSession?.messages || []);
    if (priorUserMessage) state.aiLastSubmittedFeedback = priorUserMessage;
    state.aiProposal = state.aiSession.current_proposal || null;
    renderAIRefinement();
  } catch (error) {
    $("#ai-result").className = "ai-result proposal-error";
    $("#ai-result").innerHTML = `<span class="eyebrow">Could not load refinement</span><p>${esc(error.message)}</p>`;
  }
}

function updateAIRetryButton() {
  const button = $("#ai-retry-last");
  const fallbackRetry = latestUserMessage(state.aiSession?.messages || []);
  const retryText = (state.aiRetryFeedback || "").trim() || (state.aiLastSubmittedFeedback || "").trim() || fallbackRetry;
  const hasRetry = !!retryText;
  button.disabled = !!state.aiJobId || !hasRetry;
  if (!hasRetry) {
    button.textContent = "Retry last failed message";
    button.title = "Becomes available after a refinement turn fails";
    return;
  }
  const preview = retryText.length > 72
    ? `${retryText.slice(0, 69)}...`
    : retryText;
  button.textContent = `Retry failed message: \"${preview}\"`;
  button.title = "Resend the last message that failed to complete";
}

function renderAIConversation() {
  const messages = state.aiSession?.messages || [];
  const node = $("#ai-conversation");
  if (!messages.length) {
    node.innerHTML = `<p class="muted small">No messages yet. Start refinement and the AI will examine the saved goal and criteria.</p>`;
  } else {
    node.innerHTML = messages.map(message => `<div class="ai-message ${esc(message.role)}"><span class="ai-message-meta">${message.role === "user" ? "You" : message.role === "assistant" ? "AI collaborator" : "Status"}</span>${esc(message.content)}</div>`).join("");
    node.scrollTop = node.scrollHeight;
  }
  $("#ai-generate").textContent = messages.length ? "Send reply and continue" : "Start refinement";
}

function renderAIRefinement() {
  renderAIConversation();
  updateAIRetryButton();
  const proposal = state.aiProposal;
  const status = state.aiSession?.status || "not_started";
  const readiness = $("#ai-readiness");
  readiness.className = `badge ${status === "ready" || status === "finalized" ? "pass" : status === "refining" ? "warning" : ""}`;
  readiness.textContent = status === "finalized" ? "Finalized" : status === "ready" ? "Ready to finalize" : status === "refining" ? "Still refining" : "Not reviewed";
  if (!proposal) {
    $("#ai-result").className = "ai-result empty-result";
    $("#ai-result").textContent = "Start the conversation to create a goal and criteria draft.";
    $("#ai-apply").classList.add("hidden");
    $("#ai-finalize").classList.add("hidden");
    return;
  }

  const issues = proposal.criteria_quality_issues || [];
  const blockers = issues.filter(item => item.severity === "blocking");
  const ready = !!proposal.ready_to_finalize && !(proposal.clarifying_questions || []).length && !blockers.length;
  const qualityHtml = issues.length ? `<div class="proposal-section"><h3>Criteria quality review</h3><ul class="proposal-quality-list">${issues.map(item => `<li class="proposal-quality-item ${esc(item.severity || "blocking")}"><strong>${item.criterion_id ? `<code>${esc(item.criterion_id)}</code>: ` : ""}${esc(item.issue)}</strong>${item.suggested_fix ? `<div class="muted small">Fix: ${esc(item.suggested_fix)}</div>` : ""}</li>`).join("")}</ul></div>` : `<div class="proposal-section"><h3>Criteria quality review</h3><p class="muted small">No quality issues were reported.</p></div>`;
  const contextStats = state.aiSession || {};
  const contextHtml = contextStats.compacted_message_count || contextStats.context_overflow_retries
    ? `<div class="proposal-section context-note"><h3>Context management</h3><p class="muted small">${contextStats.compacted_message_count ? `${esc(contextStats.compacted_message_count)} older messages were compacted while the full conversation remains visible here. ` : ""}${contextStats.context_overflow_retries ? `Goal Agent recovered from ${esc(contextStats.context_overflow_retries)} context overflow${contextStats.context_overflow_retries === 1 ? "" : "s"} using a smaller retry prompt. ` : ""}${contextStats.last_estimated_input_tokens ? `Latest Goal Agent input estimate: about ${esc(contextStats.last_estimated_input_tokens.toLocaleString())} tokens.` : ""}</p></div>`
    : "";
  const questions = proposal.clarifying_questions || [];
  const assumptions = proposal.assumptions || [];
  $("#ai-result").className = "ai-result";
  $("#ai-result").innerHTML = `
    <div class="readiness-card ${ready ? "ready" : "refining"}"><strong>${ready ? "Ready for finalization" : "More refinement is needed"}</strong><p class="muted small">${esc(proposal.readiness_reason || "No readiness explanation was supplied.")}</p></div>
    ${proposal.assistant_message ? `<div class="proposal-section"><span class="eyebrow">AI summary</span><p>${esc(proposal.assistant_message)}</p></div>` : ""}
    <div class="proposal-section"><span class="eyebrow">Proposed goal</span><div class="proposal-goal">${esc(proposal.refined_goal)}</div>${proposal.goal_rationale ? `<p class="muted small">${esc(proposal.goal_rationale)}</p>` : ""}</div>
    ${questions.length ? `<div class="proposal-section"><h3>Questions to answer</h3><ol class="proposal-questions">${questions.map(q => `<li>${esc(q)}</li>`).join("")}</ol><p class="muted small">Answer these in the reply box. The next AI turn will revise the full draft.</p></div>` : ""}
    ${assumptions.length ? `<div class="proposal-section"><h3>Current assumptions</h3><ul class="proposal-assumptions">${assumptions.map(a => `<li>${esc(a)}</li>`).join("")}</ul></div>` : ""}
    ${contextHtml}
    <div class="proposal-section"><h3>Concrete success criteria</h3>${proposalCriteria(proposal.criteria || [])}</div>
    ${qualityHtml}`;
  $("#ai-apply").classList.remove("hidden");
  $("#ai-finalize").classList.remove("hidden");
  $("#ai-finalize").disabled = !ready || status === "finalized";
  $("#ai-finalize").title = ready ? "Save this goal and criteria as the finalized definition" : "Resolve the remaining questions and blocking quality issues first";
  $("#ai-finalize").textContent = status === "finalized" ? "Finalized" : "Finalize and save";
}

function updateAIJobProgress(job) {
  const stageNames = {
    queued: "Waiting to start…",
    starting: "Starting OpenCode…",
    attempt: "AI is reviewing the bounded conversation…",
    context_budget: "Preparing a context-safe refinement prompt…",
    context_retry: "Context filled — retrying with a compact prompt…",
    started: "OpenCode is running…",
    text: "Receiving the revised draft…",
    tool: "AI is inspecting the project…",
    parsing: "Checking the structured proposal…",
    retry: "Repairing the proposal format…",
    complete: "Checking criteria quality…",
    completed: "AI turn complete",
    failed: "AI turn failed",
    cancelled: "AI turn cancelled",
  };
  $("#ai-working-stage").textContent = stageNames[job.stage] || "OpenCode is refining the goal and criteria…";
  $("#ai-working-detail").textContent = job.detail || `Status: ${job.status}`;
}

async function waitForAIProposal(jobId, projectId) {
  while (true) {
    await sleep(700);
    const job = await api(`/api/proposal-jobs/${encodeURIComponent(jobId)}?project_id=${encodeURIComponent(projectId)}`, {}, false);
    updateAIJobProgress(job);
    if (job.status === "completed") return job.result;
    if (job.status === "failed") throw new Error(job.error || "OpenCode could not complete this refinement turn.");
    if (job.status === "cancelled") throw new Error("The refinement turn was cancelled.");
  }
}

async function generateAIProposal(forcedFeedback = null) {
  const inputValue = typeof forcedFeedback === "string" ? forcedFeedback : $("#ai-feedback").value;
  const feedback = inputValue.trim();
  const projectId = state.projectId;
  const goalId = state.selectedId;
  const mode = state.aiMode;
  if ((state.aiSession?.messages || []).length && !feedback) {
    toast("Write a reply, answer a question, or ask for a final readiness review", "error");
    $("#ai-feedback").focus();
    return;
  }
  if (feedback) {
    state.aiLastSubmittedFeedback = feedback;
  }
  $("#ai-working").classList.remove("hidden");
  $("#ai-cancel").classList.remove("hidden");
  $("#ai-generate").disabled = true;
  $("#ai-retry-last").disabled = true;
  $("#ai-reset").disabled = true;
  $("#ai-working-stage").textContent = "Starting refinement turn…";
  $("#ai-working-detail").textContent = "The AI will return a complete revised draft, not just a partial answer.";
  try {
    const body = { mode, feedback, conversation: "" };
    const job = await api(`/api/goals/${encodeURIComponent(goalId)}/proposal-jobs?project_id=${encodeURIComponent(projectId)}`, { method: "POST", body: JSON.stringify(body) }, false);
    state.aiJobId = job.id;
    updateAIJobProgress(job);
    const result = await waitForAIProposal(job.id, projectId);
    state.aiProposal = result;
    state.aiSession = result.refinement_session || await api(`/api/goals/${encodeURIComponent(goalId)}/refinement-session`);
    if (state.aiSession.current_proposal) state.aiProposal = state.aiSession.current_proposal;
    $("#ai-feedback").value = "";
    state.aiRetryFeedback = "";
    renderAIRefinement();
  } catch (error) {
    state.aiRetryFeedback = feedback || state.aiLastSubmittedFeedback || "";
    $("#ai-result").className = "ai-result proposal-error";
    $("#ai-result").innerHTML = `<span class="eyebrow">Refinement error</span><p>${esc(error.message)}</p><p class="muted small">Your saved conversation remains available. Retry this turn after reviewing the error.</p>`;
    toast(error.message, "error");
    try {
      state.aiSession = await api(`/api/goals/${encodeURIComponent(goalId)}/refinement-session`);
      renderAIConversation();
    } catch (_) {}
    updateAIRetryButton();
  } finally {
    state.aiJobId = null;
    $("#ai-working").classList.add("hidden");
    $("#ai-cancel").classList.add("hidden");
    $("#ai-generate").disabled = false;
    $("#ai-retry-last").disabled = false;
    $("#ai-reset").disabled = false;
    updateAIRetryButton();
  }
}

async function retryLastAIMessage() {
  const retryFeedback = (state.aiRetryFeedback || "").trim()
    || (state.aiLastSubmittedFeedback || "").trim()
    || latestUserMessage(state.aiSession?.messages || []);
  if (!retryFeedback) {
    toast("No failed message is available to retry", "error");
    return;
  }
  state.aiRetryFeedback = retryFeedback;
  $("#ai-feedback").value = retryFeedback;
  toast("Retrying your last message…");
  await generateAIProposal(retryFeedback);
}

async function cancelAIProposal() {
  if (!state.aiJobId || !state.projectId) return;
  $("#ai-cancel").disabled = true;
  try {
    await api(`/api/proposal-jobs/${encodeURIComponent(state.aiJobId)}?project_id=${encodeURIComponent(state.projectId)}`, { method: "DELETE" }, false);
    $("#ai-working-stage").textContent = "Cancelling AI turn…";
    $("#ai-working-detail").textContent = "Stopping the OpenCode process.";
  } catch (error) {
    toast(error.message, "error");
  } finally {
    $("#ai-cancel").disabled = false;
  }
}

async function resetAIRefinement() {
  if (state.aiJobId) return toast("Cancel the active AI turn first", "error");
  if (!confirm("Start a new refinement conversation? The current conversation and draft will be replaced, but saved goal files will not change.")) return;
  try {
    state.aiSession = await api(`/api/goals/${encodeURIComponent(state.selectedId)}/refinement-session/reset`, { method: "POST", body: "{}" });
    state.aiProposal = null;
    state.aiLastSubmittedFeedback = "";
    state.aiRetryFeedback = "";
    $("#ai-feedback").value = "";
    renderAIRefinement();
    toast("Refinement conversation reset");
  } catch (error) { toast(error.message, "error"); }
}

function proposalCriteria(criteria) {
  if (!criteria.length) return `<p class="muted small">No criteria proposed.</p>`;
  const labels = { command: "automated command / test", file_exists: "file exists", file_contains: "file content", ai_judge: "AI evidence review", manual: "human approval only" };
  const humanOnly = criteria.filter(c => c.kind === "manual" && c.required);
  return `${humanOnly.length ? `<div class="proposal-warning"><strong>Human-only stopping gate</strong><p>${humanOnly.length} required criterion${humanOnly.length === 1 ? "" : "a"} cannot pass autonomously. Keep this only when personal approval is intentional.</p></div>` : ""}<div class="criteria-list">${criteria.map(c => {
    const proof = c.kind === "command" ? `<strong>Run:</strong> <code>${esc(c.command || "")}</code><br><strong>Pass:</strong> exit code ${esc(c.expected_exit_code ?? 0)}`
      : c.kind === "file_exists" ? `<strong>Pass:</strong> <code>${esc(c.path || "")}</code> exists`
      : c.kind === "file_contains" ? `<strong>Inspect:</strong> <code>${esc(c.path || "")}</code><br><strong>Required content:</strong> ${esc(c.contains || "")}`
      : c.kind === "ai_judge" ? `<strong>Evidence paths:</strong> ${esc((c.evidence_paths || []).join(", ") || "project workspace")}<br><strong>Decision rubric:</strong> ${esc(c.judge_prompt || "")}`
      : `<strong>Pass:</strong> explicit human override`;
    return `<div class="criterion-status"><div class="criterion-header"><span class="criterion-id">${esc(c.id)}</span><span class="badge ${c.kind === "manual" ? "error" : ""}">${esc(labels[c.kind] || c.kind)}</span></div><div class="criterion-description">${esc(c.description)}</div><div class="criterion-proof">${proof}</div></div>`;
  }).join("")}</div>`;
}

function applyAIProposal() {
  if (!state.aiProposal) return;
  state.detail.goal = state.aiProposal.refined_goal;
  if (state.aiProposal.criteria?.length) state.detail.criteria.criteria = state.aiProposal.criteria;
  state.activeTab = "definition"; state.editorDirty = true; closeModal("ai-modal"); renderDetail(); toast("Draft applied locally; review it and save, or return to refinement to finalize it");
}

async function finalizeAIProposal() {
  if (!state.aiProposal || !state.selectedId) return;
  $("#ai-finalize").disabled = true;
  try {
    const result = await api(`/api/goals/${encodeURIComponent(state.selectedId)}/refinement-session/finalize`, { method: "POST", body: JSON.stringify({ force: false }) });
    state.aiSession = result.session;
    state.detail.goal = result.goal;
    state.detail.criteria = result.criteria;
    state.detail.refinement = result.session;
    state.editorDirty = false;
    renderAIRefinement();
    renderDetail();
    toast("Goal and concrete success criteria finalized and saved");
  } catch (error) {
    toast(error.message, "error");
    $("#ai-finalize").disabled = false;
  }
}

async function saveArchive() {
  const archived = $("#archive-goal").checked;
  try {
    await api(`/api/goals/${encodeURIComponent(state.selectedId)}/metadata`, { method: "PATCH", body: JSON.stringify({ title: state.detail.metadata.title, description: state.detail.metadata.description, archived }) });
    state.detail.metadata.archived = archived; await refreshGoals();
    if (archived) {
      state.selectedId = null; state.detail = null;
      if (state.goals.length) await selectGoal(state.goals[0].metadata.id);
      else $("#main-content").innerHTML = `<section class="empty-state"><div class="empty-icon">◎</div><h2>No active goals</h2><p>Create a goal or unarchive one through the file interface.</p></section>`;
    }
    toast("Archive setting saved");
  } catch (error) { toast(error.message, "error"); }
}

async function deleteGoal() {
  if (!confirm(`Permanently delete goal '${state.selectedId}' and all of its run history?`)) return;
  try {
    await api(`/api/goals/${encodeURIComponent(state.selectedId)}`, { method: "DELETE" });
    state.selectedId = null; state.detail = null; await refreshGoals();
    if (state.goals.length) await selectGoal(state.goals[0].metadata.id);
    else $("#main-content").innerHTML = `<section class="empty-state"><div class="empty-icon">◎</div><h2>No goals</h2><p>Create a goal to begin.</p><button class="button primary" onclick="document.querySelector('#new-goal-button').click()">Create a goal</button></section>`;
    toast("Goal deleted");
  } catch (error) { toast(error.message, "error"); }
}

boot().catch(error => toast(error.message, "error"));

// ─── Agent Chat Modal ─────────────────────────────────────────────────────────

const AGENT_FILE_HINTS = {
  strategist: ["strategist-prompt.md", "strategist-output.txt"],
  executor:   ["executor-prompt.md", "executor-output.txt", "serial-fix-", "executor-verify-output.txt", "executor-retry-output.txt"],
  evaluator:  ["baseline-analysis-prompt.md", "baseline-analysis-output.txt", "post-execution-analysis-output.txt"],
};

function agentFileScore(name, agentName) {
  const hints = AGENT_FILE_HINTS[agentName] || [];
  for (let i = 0; i < hints.length; i++) {
    if (name.startsWith(hints[i]) || name.includes(hints[i])) return hints.length - i;
  }
  return -1;
}

async function openAgentChat(agentName) {
  const d = state.detail;
  if (!d) return;
  const goalId = d.metadata?.id || state.selectedId;
  state.agentChat.goalId = goalId;
  state.agentChat.agentName = agentName;

  const modal = $("#agent-chat-modal");
  modal.classList.remove("hidden");
  $("#agent-chat-eyebrow").textContent = agentName.charAt(0).toUpperCase() + agentName.slice(1) + " agent";
  $("#agent-chat-title").textContent = "Chat history";
  const subtitle = d.state?.agents?.[agentName];
  $("#agent-chat-subtitle").textContent = subtitle ? `${subtitle.phase} — ${subtitle.task || "Idle"}` : "";

  await loadAgentChatRuns();
}

async function loadAgentChatRuns() {
  const { goalId, agentName } = state.agentChat;
  if (!goalId) return;
  const qs = state.projectId ? `?project_id=${encodeURIComponent(state.projectId)}` : "";
  let data;
  try {
    data = await api.get(`/api/goals/${encodeURIComponent(goalId)}/runs${qs}`, { limit: 10 });
  } catch { return; }
  state.agentChat.runs = data.runs || [];

  const iterSel = $("#agent-chat-iter-select");
  const prevIter = iterSel.value;
  iterSel.innerHTML = state.agentChat.runs.map(r =>
    `<option value="${esc(r.iteration)}">${esc(r.iteration)}</option>`).join("");
  if (prevIter && [...iterSel.options].some(o => o.value === prevIter)) iterSel.value = prevIter;

  await loadAgentChatFiles();
}

async function loadAgentChatFiles() {
  const { agentName, runs } = state.agentChat;
  const iterSel = $("#agent-chat-iter-select");
  const iteration = iterSel.value;
  if (!iteration) return;

  const run = runs.find(r => r.iteration === iteration);
  const files = run ? run.files : [];

  // Sort: agent-relevant files first, then everything else alphabetically
  const sorted = [...files].sort((a, b) => {
    const sa = agentFileScore(a, agentName), sb = agentFileScore(b, agentName);
    if (sa !== sb) return sb - sa;
    return a.localeCompare(b);
  });

  const fileSel = $("#agent-chat-file-select");
  const prevFile = fileSel.value;
  fileSel.innerHTML = sorted.map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join("");
  // Auto-select best match for this agent
  const best = sorted.find(f => agentFileScore(f, agentName) >= 0) || sorted[0];
  if (prevFile && sorted.includes(prevFile)) fileSel.value = prevFile;
  else if (best) fileSel.value = best;

  await loadAgentChatArtifact();
}

async function loadAgentChatArtifact() {
  const { goalId } = state.agentChat;
  const iteration = $("#agent-chat-iter-select")?.value;
  const filename = $("#agent-chat-file-select")?.value;
  if (!goalId || !iteration || !filename) return;

  const contentEl = $("#agent-chat-content");
  contentEl.innerHTML = `<p class="muted small" style="padding:16px">Loading…</p>`;

  const qs = state.projectId ? `?project_id=${encodeURIComponent(state.projectId)}` : "";
  let data;
  try {
    data = await api.get(`/api/goals/${encodeURIComponent(goalId)}/runs/${encodeURIComponent(iteration)}/${encodeURIComponent(filename)}${qs}`);
  } catch (err) {
    contentEl.innerHTML = `<p class="muted small" style="padding:16px;color:var(--red)">Error: ${esc(err.message)}</p>`;
    return;
  }

  const raw = data.content || "";
  const isMarkdown = filename.endsWith(".md");
  const isJson = filename.endsWith(".json") || raw.trimStart().startsWith("{") || raw.trimStart().startsWith("[");

  // Strip GOAL_AGENT_JSON wrappers for readability
  const display = raw.replace(/<GOAL_AGENT_JSON>/g, "").replace(/<\/GOAL_AGENT_JSON>/g, "").trim();

  contentEl.innerHTML = `
    <div class="agent-chat-meta">
      <span class="muted small">${esc(iteration)} / ${esc(filename)}</span>
      <span class="muted small">${display.length.toLocaleString()} chars</span>
    </div>
    <pre class="agent-chat-pre">${esc(display)}</pre>`;
}

function startAgentChatAutoRefresh() {
  stopAgentChatAutoRefresh();
  state.agentChat.refreshTimer = setInterval(async () => {
    if ($("#agent-chat-modal").classList.contains("hidden")) {
      stopAgentChatAutoRefresh();
      return;
    }
    await loadAgentChatRuns();
  }, 3000);
}

function stopAgentChatAutoRefresh() {
  if (state.agentChat.refreshTimer) {
    clearInterval(state.agentChat.refreshTimer);
    state.agentChat.refreshTimer = null;
  }
}

// Wire up agent chat modal events
document.addEventListener("click", async (evt) => {
  const btn = evt.target.closest(".agent-chat-btn");
  if (btn) { await openAgentChat(btn.dataset.agent); return; }

  const close = evt.target.closest("[data-close='agent-chat-modal']");
  if (close || evt.target.id === "agent-chat-modal") {
    $("#agent-chat-modal").classList.add("hidden");
    stopAgentChatAutoRefresh();
  }
}, true);

document.addEventListener("change", async (evt) => {
  if (evt.target.id === "agent-chat-iter-select") await loadAgentChatFiles();
  if (evt.target.id === "agent-chat-file-select") await loadAgentChatArtifact();
  if (evt.target.id === "agent-chat-auto-refresh") {
    state.agentChat.autoRefresh = evt.target.checked;
    evt.target.checked ? startAgentChatAutoRefresh() : stopAgentChatAutoRefresh();
  }
}, true);

document.addEventListener("click", async (evt) => {
  if (evt.target.id === "agent-chat-refresh") await loadAgentChatRuns();
}, true);
