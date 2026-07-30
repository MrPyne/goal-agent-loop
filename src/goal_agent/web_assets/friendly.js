const friendlyState = {
  system: null,
  picker: {
    targetId: null,
    purpose: "open",
    currentPath: "",
    parent: null,
    locations: [],
  },
  pathTimers: new Map(),
};

function friendlySlug(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^[._-]+|[._-]+$/g, "")
    .slice(0, 80) || "project";
}

function pathJoin(base, child) {
  const separator = String(base).includes("\\") ? "\\" : "/";
  return `${String(base).replace(/[\\/]+$/, "")}${separator}${String(child).replace(/^[\\/]+/, "")}`;
}

function setPathStatus(id, kind, message) {
  const node = document.getElementById(id);
  if (!node) return;
  node.className = `path-status ${kind}`;
  node.textContent = message;
}

async function friendlyJson(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}

async function loadFriendlySystemInfo() {
  try {
    friendlyState.system = await friendlyJson("/api/system/locations");
    renderCreateLocationChips();
    const title = document.getElementById("create-project-title");
    const path = document.getElementById("create-project-path");
    if (title && path && !path.value && title.value) {
      path.value = pathJoin(friendlyState.system.default_projects_dir, friendlySlug(title.value));
      path.dataset.auto = "true";
    }
  } catch (error) {
    console.warn("Friendly filesystem helpers unavailable", error);
  }
}

function renderCreateLocationChips() {
  const host = document.getElementById("create-location-chips");
  if (!host || !friendlyState.system) return;
  const useful = friendlyState.system.locations.filter(item => ["projects", "documents", "desktop", "current", "recent"].includes(item.kind)).slice(0, 5);
  host.innerHTML = useful.length
    ? `<span>Quick locations:</span>${useful.map(item => `<button type="button" class="location-chip" data-create-location="${esc(item.path)}">${esc(item.label)}</button>`).join("")}`
    : "";
}

function suggestCreateProjectPath() {
  const title = document.getElementById("create-project-title");
  const input = document.getElementById("create-project-path");
  if (!title || !input || !friendlyState.system) return;
  if (input.dataset.auto === "false" && input.value.trim()) return;
  input.value = pathJoin(friendlyState.system.default_projects_dir, friendlySlug(title.value));
  input.dataset.auto = "true";
  validateProjectPath("create");
}

function schedulePathValidation(purpose) {
  clearTimeout(friendlyState.pathTimers.get(purpose));
  friendlyState.pathTimers.set(purpose, setTimeout(() => validateProjectPath(purpose), 250));
}

async function validateProjectPath(purpose) {
  const input = document.getElementById(`${purpose}-project-path`);
  const submit = document.getElementById(`${purpose}-project-submit`);
  if (!input || !submit) return;
  const value = input.value.trim();
  const title = document.getElementById("create-project-title")?.value.trim();
  if (!value) {
    setPathStatus(`${purpose}-project-path-status`, "neutral", purpose === "create" ? "Choose where the project should live." : "Choose a folder containing a Goal Agent project.");
    submit.disabled = true;
    return;
  }
  try {
    const info = await friendlyJson(`/api/system/path-info?path=${encodeURIComponent(value)}`);
    input.value = info.path;
    if (purpose === "create") {
      const force = !!document.querySelector('#create-project-form input[name="force"]')?.checked;
      if (!title) {
        setPathStatus("create-project-path-status", "neutral", "Enter a project name first.");
        submit.disabled = true;
      } else if (info.initialized_project && !force) {
        setPathStatus("create-project-path-status", "warning", "A Goal Agent workspace already exists here. Open it instead, or use the advanced replace option.");
        submit.disabled = true;
      } else if (info.initialized_project && force) {
        setPathStatus("create-project-path-status", "warning", "This will replace the existing Goal Agent workspace. Project source files are not removed.");
        submit.disabled = false;
      } else if (info.exists && !info.is_dir) {
        setPathStatus("create-project-path-status", "error", "That path is a file, not a folder.");
        submit.disabled = true;
      } else if (!info.writable) {
        setPathStatus("create-project-path-status", "error", "Goal Agent cannot write to this location.");
        submit.disabled = true;
      } else if (info.exists && info.has_contents) {
        setPathStatus("create-project-path-status", "warning", "This folder already contains files. Goal Agent will add its .goal-agent workspace without deleting anything.");
        submit.disabled = false;
      } else if (info.exists) {
        setPathStatus("create-project-path-status", "success", "This existing folder is ready to initialize.");
        submit.disabled = false;
      } else {
        const parentNote = info.parent_exists ? "" : ` Parent folders will also be created under ${info.nearest_existing_parent}.`;
        setPathStatus("create-project-path-status", "success", `This folder will be created automatically.${parentNote}`);
        submit.disabled = false;
      }
    } else {
      if (info.discovered_project_root) {
        const suffix = info.discovered_project_root === info.path ? "" : ` Project root: ${info.discovered_project_root}`;
        setPathStatus("open-project-path-status", "success", `Goal Agent workspace found.${suffix}`);
        submit.disabled = false;
      } else if (!info.exists) {
        setPathStatus("open-project-path-status", "error", "That folder does not exist.");
        submit.disabled = true;
      } else {
        setPathStatus("open-project-path-status", "error", "No .goal-agent workspace was found in this folder or its parents.");
        submit.disabled = true;
      }
    }
  } catch (error) {
    setPathStatus(`${purpose}-project-path-status`, "error", error.message);
    submit.disabled = true;
  }
}

function switchProjectMode(mode) {
  document.querySelectorAll("[data-project-mode]").forEach(button => button.classList.toggle("active", button.dataset.projectMode === mode));
  document.querySelectorAll("[data-project-panel]").forEach(panel => panel.classList.toggle("hidden", panel.dataset.projectPanel !== mode));
  const focusTarget = mode === "create" ? document.getElementById("create-project-title") : document.querySelector('[data-browse-path="open-project-path"]');
  setTimeout(() => focusTarget?.focus(), 30);
}

async function chooseNativeFolder(targetId, purpose, initialPath = "") {
  const input = document.getElementById(targetId);
  const initial = initialPath || input?.value || friendlyState.system?.default_projects_dir || "";
  try {
    const result = await friendlyJson("/api/system/folders/pick", {
      method: "POST",
      body: JSON.stringify({
        initial_path: initial,
        title: purpose === "create" ? "Choose the project folder" : "Choose an existing Goal Agent project",
        must_exist: true,
      }),
    });
    if (!result.selected) return;
    if (input) {
      input.value = result.selected;
      input.dataset.auto = "false";
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
  } catch (error) {
    toast(`${error.message} Opening the built-in folder browser instead.`, "error");
    await openFolderBrowser(targetId, purpose, initial);
  }
}

async function openFolderBrowser(targetId, purpose, initialPath = "") {
  friendlyState.picker.targetId = targetId;
  friendlyState.picker.purpose = purpose;
  const input = document.getElementById(targetId);
  friendlyState.picker.currentPath = initialPath || input?.value || friendlyState.system?.default_projects_dir || "";
  document.getElementById("folder-picker-title").textContent = purpose === "create" ? "Choose the project folder" : "Choose an existing project";
  document.getElementById("select-folder-button").textContent = purpose === "create" ? "Use this folder" : "Select project folder";
  document.getElementById("folder-search").value = "";
  openModal("folder-picker-modal");
  await browseFolder(friendlyState.picker.currentPath);
}

async function browseFolder(path, search = "") {
  const list = document.getElementById("folder-list");
  list.innerHTML = `<div class="working"><span class="spinner"></span> Reading folders…</div>`;
  try {
    const data = await friendlyJson(`/api/system/folders?path=${encodeURIComponent(path || "")}&search=${encodeURIComponent(search)}`);
    friendlyState.picker.currentPath = data.path;
    friendlyState.picker.parent = data.parent;
    friendlyState.picker.locations = data.locations || [];
    document.getElementById("folder-current-path").value = data.path;
    document.getElementById("folder-up-button").disabled = !data.parent;
    document.getElementById("folder-picker-status").textContent = data.initialized_project
      ? "This folder contains a Goal Agent workspace."
      : friendlyState.picker.purpose === "open"
        ? "Navigate to a project containing a .goal-agent folder."
        : "Select this folder, or create a new folder inside it.";
    document.getElementById("folder-picker-status").className = `path-status ${data.initialized_project ? "success" : "neutral"}`;
    renderFolderQuickLocations();
    list.innerHTML = data.directories.length
      ? data.directories.map(item => `<button type="button" class="folder-row" data-folder-nav="${esc(item.path)}"><span class="folder-icon">▰</span><span class="folder-copy"><strong>${esc(item.name)}</strong>${item.initialized_project ? `<small>Goal Agent project</small>` : `<small>Folder</small>`}</span>${item.initialized_project ? `<span class="project-found-badge">Project</span>` : ""}<span class="folder-chevron">›</span></button>`).join("")
      : `<div class="folder-empty">No matching subfolders. You can select the current folder.</div>`;
  } catch (error) {
    list.innerHTML = `<div class="folder-empty error-copy">${esc(error.message)}</div>`;
  }
}

function renderFolderQuickLocations() {
  const host = document.getElementById("folder-quick-locations");
  host.innerHTML = friendlyState.picker.locations.map(item => `<button type="button" class="quick-location ${item.path === friendlyState.picker.currentPath ? "active" : ""}" data-folder-nav="${esc(item.path)}"><span>${item.kind === "drive" ? "◫" : item.kind === "project" ? "◆" : "▰"}</span><span>${esc(item.label)}</span></button>`).join("");
}

function selectCurrentFolder() {
  const input = document.getElementById(friendlyState.picker.targetId);
  if (!input) return;
  input.value = friendlyState.picker.currentPath;
  input.dataset.auto = "false";
  closeModal("folder-picker-modal");
  input.dispatchEvent(new Event("input", { bubbles: true }));
}

async function createFolderFromPicker() {
  const name = prompt("New folder name");
  if (!name?.trim()) return;
  const target = pathJoin(friendlyState.picker.currentPath, name.trim());
  try {
    const created = await friendlyJson("/api/system/folders/create", { method: "POST", body: JSON.stringify({ path: target }) });
    await browseFolder(created.path);
    toast("Folder created");
  } catch (error) { toast(error.message, "error"); }
}

async function discoverProjects() {
  const button = document.getElementById("discover-projects-button");
  const host = document.getElementById("discovered-projects");
  button.disabled = true;
  button.textContent = "Searching…";
  host.classList.remove("hidden");
  host.innerHTML = `<div class="working"><span class="spinner"></span> Searching common project folders…</div>`;
  try {
    const result = await friendlyJson("/api/system/projects/discover", { method: "POST", body: JSON.stringify({}) });
    const unregistered = result.projects.filter(item => !item.registered);
    host.innerHTML = `<div class="discovered-heading"><strong>Projects found</strong><button class="icon-button compact-icon" type="button" data-hide-discovered>×</button></div>${unregistered.length
      ? unregistered.map(item => `<article class="discovered-project"><div><strong>${esc(item.title)}</strong><code>${esc(item.path)}</code></div><button class="button ghost compact" type="button" data-open-discovered="${esc(item.path)}">Open</button></article>`).join("")
      : `<p class="muted small">No unregistered Goal Agent projects were found in the common locations. Use Choose project folder to browse anywhere.</p>`}`;
  } catch (error) {
    host.innerHTML = `<div class="error-copy">${esc(error.message)}</div>`;
  } finally {
    button.disabled = false;
    button.textContent = "Find projects";
  }
}

async function revealFriendlyPath(path) {
  try {
    await friendlyJson("/api/system/reveal", { method: "POST", body: JSON.stringify({ path }) });
  } catch (error) { toast(error.message, "error"); }
}

function initializeFriendlyEvents() {
  document.addEventListener("click", async event => {
    const mode = event.target.closest("[data-project-mode]");
    if (mode) return switchProjectMode(mode.dataset.projectMode);

    const browse = event.target.closest("[data-browse-path]");
    if (browse) return chooseNativeFolder(browse.dataset.browsePath, browse.dataset.pickerPurpose || "open");

    const browser = event.target.closest("[data-open-folder-browser]");
    if (browser) return openFolderBrowser(browser.dataset.openFolderBrowser, browser.dataset.pickerPurpose || "open");

    const location = event.target.closest("[data-create-location]");
    if (location) {
      const input = document.getElementById("create-project-path");
      const title = document.getElementById("create-project-title").value;
      input.value = pathJoin(location.dataset.createLocation, friendlySlug(title));
      input.dataset.auto = "false";
      return validateProjectPath("create");
    }

    const navigate = event.target.closest("[data-folder-nav]");
    if (navigate) return browseFolder(navigate.dataset.folderNav, document.getElementById("folder-search").value);

    const reveal = event.target.closest("[data-reveal-project]");
    if (reveal) return revealFriendlyPath(reveal.dataset.revealProject);

    const discovered = event.target.closest("[data-open-discovered]");
    if (discovered) {
      switchProjectMode("open");
      const input = document.getElementById("open-project-path");
      input.value = discovered.dataset.openDiscovered;
      await validateProjectPath("open");
      return;
    }

    if (event.target.closest("#discover-projects-button")) return discoverProjects();
    if (event.target.closest("[data-hide-discovered]")) return document.getElementById("discovered-projects").classList.add("hidden");
    if (event.target.closest("#folder-up-button") && friendlyState.picker.parent) return browseFolder(friendlyState.picker.parent);
    if (event.target.closest("#folder-go-button")) return browseFolder(document.getElementById("folder-current-path").value);
    if (event.target.closest("#native-folder-picker-button")) {
      const resultTarget = friendlyState.picker.targetId;
      await chooseNativeFolder(resultTarget, friendlyState.picker.purpose, friendlyState.picker.currentPath);
      const value = document.getElementById(resultTarget)?.value;
      if (value) await browseFolder(value);
      return;
    }
    if (event.target.closest("#new-folder-button")) return createFolderFromPicker();
    if (event.target.closest("#select-folder-button")) return selectCurrentFolder();
  });

  document.addEventListener("input", event => {
    if (event.target.id === "create-project-title") {
      suggestCreateProjectPath();
      return;
    }
    if (event.target.id === "create-project-path") {
      if (event.isTrusted) event.target.dataset.auto = "false";
      return schedulePathValidation("create");
    }
    if (event.target.id === "open-project-path") return schedulePathValidation("open");
    if (event.target.id === "project-filter") {
      renderProjectManager();
      return;
    }
    if (event.target.id === "folder-search") return browseFolder(friendlyState.picker.currentPath, event.target.value);
    if (event.target.id === "new-goal-title") {
      const id = document.getElementById("new-goal-id");
      if (!id.dataset.manual) id.value = friendlySlug(event.target.value);
    }
    if (event.target.id === "new-goal-id") event.target.dataset.manual = "true";
  });

  document.addEventListener("change", event => {
    if (event.target.matches('#create-project-form input[name="force"]')) validateProjectPath("create");
  });

  document.getElementById("folder-current-path")?.addEventListener("keydown", event => {
    if (event.key === "Enter") { event.preventDefault(); browseFolder(event.target.value); }
  });
  document.getElementById("folder-search")?.addEventListener("keydown", event => {
    if (event.key === "Escape") { event.target.value = ""; browseFolder(friendlyState.picker.currentPath); }
  });
}

initializeFriendlyEvents();
loadFriendlySystemInfo();
