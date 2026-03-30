const DEFAULTS = {
  bridgeBaseUrl: "http://127.0.0.1:8765",
  autoCapture: true,
};

let liveDrafts = [];

function setStatus(value) {
  document.getElementById("status").textContent = value;
}

function previewText(value, limit = 180) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) {
    return "";
  }
  return text.length > limit ? `${text.slice(0, limit - 1)}…` : text;
}

function formatFetchedAt(value) {
  if (!value) {
    return "No cached live drafts yet.";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Cached live drafts available.";
  }
  return `Cached ${date.toLocaleString()}`;
}

function renderDrafts(payload, fetchedAt) {
  const items = Array.isArray(payload?.items) ? payload.items : [];
  liveDrafts = items.filter((item) => item?.live && item?.recipient_profile_url);
  document.getElementById("draft-count").textContent = String(liveDrafts.length);
  document.getElementById("draft-meta").textContent = formatFetchedAt(fetchedAt);

  const list = document.getElementById("draft-list");
  list.textContent = "";
  if (!items.length) {
    const empty = document.createElement("p");
    empty.className = "meta";
    empty.textContent = "No live outreach drafts are cached yet.";
    list.appendChild(empty);
    return;
  }

  for (const draft of items) {
    const card = document.createElement("article");
    card.className = "draft-card";

    const title = document.createElement("h3");
    title.textContent = `${draft.priority_rank || "?"}. ${draft.recipient_name || "Unknown recipient"}`;
    card.appendChild(title);

    const role = document.createElement("p");
    role.className = "draft-role";
    role.textContent = `${draft.company_name || "Unknown company"} | ${draft.role_title || "Unknown role"}`;
    card.appendChild(role);

    const history = document.createElement("p");
    history.className = "draft-history";
    history.textContent = draft.message_history_count
      ? `Message history: ${draft.message_history_count} recent messages`
      : "Message history: none captured";
    card.appendChild(history);

    if (Array.isArray(draft.target_names) && draft.target_names.length) {
      const targets = document.createElement("p");
      targets.className = "draft-targets";
      targets.textContent = `Targets: ${draft.target_names.join(", ")}`;
      card.appendChild(targets);
    }

    const preview = document.createElement("p");
    preview.className = "draft-preview";
    preview.textContent = previewText(draft.message_body);
    card.appendChild(preview);

    const actions = document.createElement("div");
    actions.className = "actions";
    const openButton = document.createElement("button");
    openButton.type = "button";
    openButton.textContent = draft.live ? "Open Draft" : "Unavailable";
    openButton.disabled = !draft.live || !draft.recipient_profile_url;
    openButton.addEventListener("click", async () => {
      setStatus(`Opening draft for ${draft.recipient_name}...`);
      const response = await browser.runtime.sendMessage({
        type: "JOB_AGENT_OPEN_OUTREACH_DRAFT",
        payload: draft,
      });
      if (response?.ok === false) {
        setStatus(`Could not prefill draft (${response.error || "unknown error"}).`);
        return;
      }
      setStatus(`Opened prefilled draft for ${draft.recipient_name}.`);
    });
    actions.appendChild(openButton);
    card.appendChild(actions);

    list.appendChild(card);
  }
}

async function loadSettings() {
  const settings = await browser.storage.local.get(DEFAULTS);
  document.getElementById("bridge-url").value = settings.bridgeBaseUrl || DEFAULTS.bridgeBaseUrl;
  document.getElementById("auto-capture").checked = settings.autoCapture !== false;
}

async function saveSettings() {
  const bridgeBaseUrl = document.getElementById("bridge-url").value.trim() || DEFAULTS.bridgeBaseUrl;
  const autoCapture = document.getElementById("auto-capture").checked;
  await browser.storage.local.set({ bridgeBaseUrl, autoCapture });
  setStatus("Saved bridge settings.");
}

async function pingBridge() {
  setStatus("Checking bridge...");
  const response = await browser.runtime.sendMessage({ type: "JOB_AGENT_EXTENSION_PING" });
  if (response && response.ok) {
    setStatus(`Bridge reachable at ${response.bridgeBaseUrl}`);
    return;
  }
  const details = response?.error ? ` (${response.error})` : "";
  setStatus(`Workflow bridge not reachable yet${details}. Expected unless a run is actively in Firefox-extension capture.`);
}

async function loadCachedDrafts() {
  const response = await browser.runtime.sendMessage({ type: "JOB_AGENT_GET_LIVE_OUTREACH" });
  renderDrafts(response?.payload || { items: [] }, response?.fetchedAt || null);
}

async function refreshDrafts() {
  setStatus("Refreshing live drafts...");
  const response = await browser.runtime.sendMessage({ type: "JOB_AGENT_REFRESH_LIVE_OUTREACH" });
  renderDrafts(response?.payload || { items: [] }, response?.fetchedAt || null);
  setStatus("Live drafts refreshed.");
}

async function openAllDrafts() {
  if (!liveDrafts.length) {
    setStatus("No live drafts are ready to open.");
    return;
  }
  setStatus(`Opening ${liveDrafts.length} live drafts...`);
  const response = await browser.runtime.sendMessage({
    type: "JOB_AGENT_OPEN_ALL_OUTREACH_DRAFTS",
    payload: {
      drafts: liveDrafts,
    },
  });
  const failures = Array.isArray(response) ? response.filter((result) => result?.ok === false) : [];
  if (failures.length) {
    setStatus(`Opened drafts with ${failures.length} issue(s).`);
    return;
  }
  setStatus(`Opened ${liveDrafts.length} prefilled draft tabs.`);
}

document.getElementById("save-settings").addEventListener("click", () => {
  void saveSettings();
});

document.getElementById("ping-bridge").addEventListener("click", () => {
  void pingBridge();
});

document.getElementById("refresh-drafts").addEventListener("click", () => {
  void refreshDrafts();
});

document.getElementById("open-all-drafts").addEventListener("click", () => {
  void openAllDrafts();
});

void loadSettings();
void loadCachedDrafts();
