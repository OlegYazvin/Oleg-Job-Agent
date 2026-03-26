const DEFAULT_SETTINGS = {
  bridgeBaseUrl: "http://127.0.0.1:8765",
  autoCapture: true,
  liveOutreachCache: {
    generated_at: null,
    items: [],
  },
  liveOutreachFetchedAt: null,
};

const LIVE_OUTREACH_PATH = "/api/linkedin-extension/live-outreach";
const historyQueues = new Map();
const activeHistoryRuns = new Set();

async function getSettings() {
  const stored = await browser.storage.local.get(DEFAULT_SETTINGS);
  return {
    bridgeBaseUrl: (stored.bridgeBaseUrl || DEFAULT_SETTINGS.bridgeBaseUrl).replace(/\/+$/, ""),
    autoCapture: stored.autoCapture !== false,
  };
}

async function getCachedLiveOutreach() {
  const stored = await browser.storage.local.get({
    liveOutreachCache: DEFAULT_SETTINGS.liveOutreachCache,
    liveOutreachFetchedAt: DEFAULT_SETTINGS.liveOutreachFetchedAt,
  });
  return {
    payload: stored.liveOutreachCache || DEFAULT_SETTINGS.liveOutreachCache,
    fetchedAt: stored.liveOutreachFetchedAt || null,
  };
}

async function setCachedLiveOutreach(payload) {
  await browser.storage.local.set({
    liveOutreachCache: payload,
    liveOutreachFetchedAt: new Date().toISOString(),
  });
}

async function postToBridge(path, payload) {
  const settings = await getSettings();
  const response = await fetch(`${settings.bridgeBaseUrl}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Bridge request failed: ${response.status}`);
  }
}

async function fetchLiveOutreachFromBridge() {
  const settings = await getSettings();
  const response = await fetch(`${settings.bridgeBaseUrl}${LIVE_OUTREACH_PATH}`);
  if (!response.ok) {
    throw new Error(`Live outreach fetch failed: ${response.status}`);
  }
  const payload = await response.json();
  if (payload && typeof payload === "object") {
    await setCachedLiveOutreach(payload);
  }
  return payload;
}

function normalizeName(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function normalizeProfileUrl(url) {
  if (!url) {
    return "";
  }
  try {
    const parsed = new URL(url);
    return `${parsed.origin}${parsed.pathname}`;
  } catch (error) {
    return String(url).split("?")[0];
  }
}

function ensureQueue(sessionId) {
  if (!historyQueues.has(sessionId)) {
    historyQueues.set(sessionId, new Map());
  }
  return historyQueues.get(sessionId);
}

function queueHistoryTargets(sessionId, contacts) {
  const queue = ensureQueue(sessionId);
  for (const contact of contacts || []) {
    if (contact.connection_degree === "1st" && contact.name && contact.profile_url) {
      const key = normalizeProfileUrl(contact.profile_url) || normalizeName(contact.name);
      if (key) {
        queue.set(key, { name: contact.name, profile_url: normalizeProfileUrl(contact.profile_url) });
      }
    }
    for (const connectorName of contact.connected_first_order_names || []) {
      const normalizedName = normalizeName(connectorName);
      if (normalizedName && !queue.has(normalizedName)) {
        queue.set(normalizedName, { name: connectorName, profile_url: "" });
      }
    }
    for (const [connectorName, connectorProfileUrl] of Object.entries(contact.connected_first_order_profile_urls || {})) {
      const key = normalizeProfileUrl(connectorProfileUrl) || normalizeName(connectorName);
      if (key) {
        queue.set(key, { name: connectorName, profile_url: normalizeProfileUrl(connectorProfileUrl) });
      }
    }
  }
}

function takeHistoryBatch(sessionId, size = 6) {
  const queue = ensureQueue(sessionId);
  const batch = [];
  for (const [key, value] of queue.entries()) {
    batch.push(value);
    queue.delete(key);
    if (batch.length >= size) {
      break;
    }
  }
  return batch;
}

async function waitForTabReady(tabId, timeoutMs = 15000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    const tab = await browser.tabs.get(tabId);
    if (tab.status === "complete") {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
}

async function sendMessageWithRetry(tabId, message, attempts = 8) {
  let lastError = null;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      return await browser.tabs.sendMessage(tabId, message);
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 600));
    }
  }
  throw lastError || new Error("Unable to reach content script.");
}

async function runHistoryCapture(sessionId) {
  if (activeHistoryRuns.has(sessionId)) {
    return;
  }
  const queue = ensureQueue(sessionId);
  if (!queue.size) {
    return;
  }
  activeHistoryRuns.add(sessionId);

  let tabId = null;
  try {
    const tab = await browser.tabs.create({
      url: `https://www.linkedin.com/messaging/?job_agent_session=${encodeURIComponent(sessionId)}`,
      active: false,
    });
    tabId = tab.id;
    await waitForTabReady(tabId);

    while (ensureQueue(sessionId).size) {
      const batch = takeHistoryBatch(sessionId);
      if (!batch.length) {
        break;
      }
      const response = await sendMessageWithRetry(tabId, {
        type: "JOB_AGENT_SCRAPE_MESSAGE_HISTORIES",
        payload: {
          sessionId,
          items: batch,
        },
      });
      if (response && Array.isArray(response.histories) && response.histories.length) {
        await postToBridge("/api/linkedin-extension/message-histories", {
          session_id: sessionId,
          histories: response.histories,
        });
      }
      await new Promise((resolve) => setTimeout(resolve, 600));
    }
  } finally {
    activeHistoryRuns.delete(sessionId);
    if (tabId !== null) {
      try {
        await browser.tabs.remove(tabId);
      } catch (error) {
      }
    }
    if (ensureQueue(sessionId).size) {
      setTimeout(() => {
        void runHistoryCapture(sessionId);
      }, 1000);
    }
  }
}

async function openAndPrefillDraft(draft) {
  const recipientProfileUrl = normalizeProfileUrl(draft?.recipient_profile_url || draft?.recipientProfileUrl || "");
  const messageBody = String(draft?.message_body || draft?.messageBody || "").trim();
  if (!recipientProfileUrl || !messageBody) {
    throw new Error("Draft is missing a recipient profile URL or message body.");
  }

  const tab = await browser.tabs.create({
    url: recipientProfileUrl,
    active: true,
  });
  await waitForTabReady(tab.id);

  const response = await sendMessageWithRetry(tab.id, {
    type: "JOB_AGENT_OPEN_AND_PREFILL_MESSAGE",
    payload: {
      recipientName: draft?.recipient_name || draft?.recipientName || "",
      messageBody,
    },
  });
  return {
    tabId: tab.id,
    ok: response?.ok !== false,
    error: response?.error || null,
  };
}

async function openAllDrafts(drafts) {
  const results = [];
  for (const draft of drafts || []) {
    if (!draft?.live || !(draft?.recipient_profile_url || draft?.recipientProfileUrl)) {
      continue;
    }
    try {
      results.push(await openAndPrefillDraft(draft));
    } catch (error) {
      results.push({ ok: false, error: String(error) });
    }
    await new Promise((resolve) => setTimeout(resolve, 900));
  }
  return results;
}

async function refreshLiveOutreachCache() {
  try {
    return await fetchLiveOutreachFromBridge();
  } catch (error) {
    return null;
  }
}

setInterval(() => {
  void refreshLiveOutreachCache();
}, 5000);

browser.runtime.onMessage.addListener((message, sender) => {
  if (!message || typeof message !== "object") {
    return undefined;
  }
  if (message.type === "JOB_AGENT_SEARCH_RESULTS") {
    return (async () => {
      const settings = await getSettings();
      if (!settings.autoCapture) {
        return { accepted: false, skipped: true };
      }
      await postToBridge("/api/linkedin-extension/search-results", message.payload);
      queueHistoryTargets(message.payload.session_id, message.payload.contacts || []);
      await runHistoryCapture(message.payload.session_id);
      return { accepted: true };
    })();
  }
  if (message.type === "JOB_AGENT_EXTENSION_PING") {
    return (async () => {
      const settings = await getSettings();
      try {
        const response = await fetch(`${settings.bridgeBaseUrl}/health`);
        return { ok: response.ok, bridgeBaseUrl: settings.bridgeBaseUrl };
      } catch (error) {
        return { ok: false, bridgeBaseUrl: settings.bridgeBaseUrl, error: String(error) };
      }
    })();
  }
  if (message.type === "JOB_AGENT_GET_LIVE_OUTREACH") {
    return getCachedLiveOutreach();
  }
  if (message.type === "JOB_AGENT_REFRESH_LIVE_OUTREACH") {
    return (async () => {
      const payload = await refreshLiveOutreachCache();
      if (payload) {
        return {
          payload,
          fetchedAt: new Date().toISOString(),
        };
      }
      return getCachedLiveOutreach();
    })();
  }
  if (message.type === "JOB_AGENT_OPEN_OUTREACH_DRAFT") {
    return openAndPrefillDraft(message.payload || {});
  }
  if (message.type === "JOB_AGENT_OPEN_ALL_OUTREACH_DRAFTS") {
    return openAllDrafts(message.payload?.drafts || []);
  }
  return undefined;
});
