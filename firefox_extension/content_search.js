(function () {
  function captureContextFromUrl(rawUrl) {
    try {
      const parsed = new URL(rawUrl, window.location.origin);
      const params = parsed.searchParams;
      const sessionId = params.get("job_agent_session");
      const degree = params.get("job_agent_degree");
      if (!sessionId || !degree) {
        return null;
      }
      return { sessionId, degree, pageUrl: parsed.toString() };
    } catch (error) {
      return null;
    }
  }

  const directContext = captureContextFromUrl(window.location.href);
  const loginRedirectContext = (() => {
    const params = new URLSearchParams(window.location.search);
    const sessionRedirect = params.get("session_redirect");
    if (!sessionRedirect) {
      return null;
    }
    const candidate = sessionRedirect.startsWith("http")
      ? sessionRedirect
      : `${window.location.origin}${sessionRedirect}`;
    return captureContextFromUrl(candidate);
  })();
  const captureContext = directContext || loginRedirectContext;
  const sessionId = captureContext?.sessionId || null;
  const degree = captureContext?.degree || null;
  if (!sessionId || !degree) {
    return;
  }

  let sent = false;
  let inFlight = false;
  let sendAttempts = 0;
  const MAX_SEND_ATTEMPTS = 5;

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function normalizeProfileUrl(href) {
    if (!href) {
      return "";
    }
    try {
      const parsed = new URL(href, window.location.origin);
      return `${parsed.origin}${parsed.pathname}`;
    } catch (error) {
      return String(href).split("?")[0];
    }
  }

  function normalizeName(value) {
    return String(value || "")
      .toLowerCase()
      .replace(/[^a-z0-9]/g, "");
  }

  function isPlaceholderName(value) {
    return /^\d+\s+(other|others|more)$/i.test(String(value || "").trim());
  }

  function cleanPersonName(value) {
    let candidate = String(value || "")
      .replace(/\u00a0/g, " ")
      .replace(/\b(mutual|shared) connections?\b.*$/i, "")
      .replace(/\bfollow\b.*$/i, "")
      .trim()
      .replace(/^[\-•\s]+|[\-•\s]+$/g, "")
      .replace(/\s+/g, " ");
    if (!candidate || isPlaceholderName(candidate)) {
      return "";
    }
    const parts = candidate.split(" ");
    if (parts.length >= 3 && /^[a-z]{1,2}$/.test(parts[parts.length - 1])) {
      candidate = parts.slice(0, -1).join(" ").trim();
    }
    return isPlaceholderName(candidate) ? "" : candidate;
  }

  function dedupeNames(values) {
    const names = [];
    const seen = new Set();
    for (const value of values || []) {
      const cleaned = cleanPersonName(value);
      const normalized = normalizeName(cleaned);
      if (!cleaned || !normalized || seen.has(normalized)) {
        continue;
      }
      seen.add(normalized);
      names.push(cleaned);
    }
    return names;
  }

  function extractConnectionDegree(text) {
    const normalized = String(text || "").replace(/\s+/g, " ");
    if (/(?:^|\s|\u2022)1st(?:\s|$)/.test(normalized)) {
      return "1st";
    }
    if (/(?:^|\s|\u2022)2nd(?:\s|$)/.test(normalized)) {
      return "2nd";
    }
    return null;
  }

  function extractName(text) {
    const lines = String(text || "")
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    return lines[0] || "";
  }

  function extractMutualConnectionNames(text) {
    const relevantLines = String(text || "")
      .split("\n")
      .map((line) => line.trim())
      .filter((line) => /\b(mutual|shared) connections?\b/i.test(line));
    if (!relevantLines.length) {
      return [];
    }
    const cleaned = relevantLines
      .join(" ")
      .replace(/\bmutual connections?\b/gi, "")
      .replace(/\bshared connections?\b/gi, "")
      .replace(/\bis\b|\bare\b/gi, "");
    return cleaned
      .replace(/\sand\s/gi, ",")
      .split(",")
      .map((part) => cleanPersonName(part))
      .filter(Boolean);
  }

  function finalizeResult(result) {
    if (!result || !result.href || !result.text) {
      return null;
    }
    const lines = String(result.text)
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (!lines.length) {
      return null;
    }
    const connectionDegree = extractConnectionDegree(result.text);
    if (!connectionDegree) {
      return null;
    }
    const filteredLines = lines.filter((line) => line !== connectionDegree && line !== `• ${connectionDegree}`);
    const headline = filteredLines[1] || "";
    const companyText = headline || filteredLines[2] || "";
    const connectedFirstOrderProfileUrls = {};
    const connectedFirstOrderNames = dedupeNames(extractMutualConnectionNames(result.text));
    for (const mutual of result.mutuals || []) {
      const cleanedName = cleanPersonName(mutual.name);
      if (!cleanedName || !mutual.profile_url) {
        continue;
      }
      connectedFirstOrderProfileUrls[cleanedName] = mutual.profile_url;
      if (!connectedFirstOrderNames.includes(cleanedName)) {
        connectedFirstOrderNames.push(cleanedName);
      }
    }
    const cleanedName = cleanPersonName(lines[0]);
    if (!cleanedName) {
      return null;
    }
    return {
      name: cleanedName,
      profile_url: normalizeProfileUrl(result.href),
      raw_text: result.text,
      headline,
      company_text: companyText,
      connection_degree: connectionDegree,
      mutual_connection_names: [...connectedFirstOrderNames],
      connected_first_order_names: [...connectedFirstOrderNames],
      connected_first_order_profile_urls: connectedFirstOrderProfileUrls,
    };
  }

  function parseCurrentPageContacts() {
    const anchors = Array.from(document.querySelectorAll('a[href*="/in/"]')).map((anchor) => ({
      href: normalizeProfileUrl(anchor.href),
      text: (anchor.innerText || anchor.textContent || "").trim(),
    }));
    const results = [];
    let current = null;

    for (const anchor of anchors) {
      if (!anchor.href || !anchor.text) {
        continue;
      }
      if (extractConnectionDegree(anchor.text)) {
        if (current) {
          const finalized = finalizeResult(current);
          if (finalized) {
            results.push(finalized);
          }
        }
        current = { href: anchor.href, text: anchor.text, mutuals: [] };
        continue;
      }
      if (!current) {
        continue;
      }
      if (anchor.href === current.href) {
        continue;
      }
      if (normalizeName(anchor.text) === normalizeName(extractName(current.text))) {
        continue;
      }
      current.mutuals.push({
        name: cleanPersonName(anchor.text),
        profile_url: anchor.href,
      });
    }

    if (current) {
      const finalized = finalizeResult(current);
      if (finalized) {
        results.push(finalized);
      }
    }

    const deduped = new Map();
    for (const result of results) {
      if (result.connection_degree !== degree) {
        continue;
      }
      if (!deduped.has(result.profile_url)) {
        deduped.set(result.profile_url, result);
      }
    }
    return Array.from(deduped.values());
  }

  async function loadAllResults() {
    for (let attempt = 0; attempt < 4; attempt += 1) {
      window.scrollBy(0, window.innerHeight * 2);
      await sleep(900);
    }
  }

  function nextButton() {
    return (
      document.querySelector('button[aria-label="Next"]') ||
      document.querySelector('button.artdeco-pagination__button--next') ||
      document.querySelector('button[aria-label*="Next"]')
    );
  }

  function isDisabled(button) {
    if (!button) {
      return true;
    }
    return button.disabled || button.getAttribute("aria-disabled") === "true";
  }

  async function collectAcrossPages(maxPages = 3) {
    const contacts = new Map();
    for (let pageIndex = 0; pageIndex < maxPages; pageIndex += 1) {
      await loadAllResults();
      for (const contact of parseCurrentPageContacts()) {
        contacts.set(contact.profile_url, contact);
      }
      const button = nextButton();
      if (!button || isDisabled(button)) {
        break;
      }
      button.click();
      await sleep(2200);
    }
    return Array.from(contacts.values());
  }

  function retryDelayMs(attemptNumber) {
    return Math.min(8000, 1200 * attemptNumber);
  }

  async function sendCapture() {
    if (sent || inFlight || sendAttempts >= MAX_SEND_ATTEMPTS) {
      return;
    }
    inFlight = true;
    sendAttempts += 1;
    try {
      const loginRequired =
        window.location.pathname.includes("/uas/login") || window.location.pathname.includes("/login");
      const contacts = loginRequired ? [] : await collectAcrossPages(3);
      const response = await browser.runtime.sendMessage({
        type: "JOB_AGENT_SEARCH_RESULTS",
        payload: {
          session_id: sessionId,
          degree,
          page_url: captureContext?.pageUrl || window.location.href,
          login_required: loginRequired,
          contacts,
        },
      });
      if (response?.accepted || response?.skipped) {
        sent = true;
        return;
      }
    } catch (error) {
    } finally {
      inFlight = false;
    }
    if (!sent && sendAttempts < MAX_SEND_ATTEMPTS) {
      setTimeout(() => {
        void sendCapture();
      }, retryDelayMs(sendAttempts));
    }
  }

  window.addEventListener("load", () => {
    setTimeout(() => {
      void sendCapture();
    }, 1500);
  });

  setTimeout(() => {
    void sendCapture();
  }, 7000);
})();
