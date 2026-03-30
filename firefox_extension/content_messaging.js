(function () {
  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function visibleText(element) {
    return (element?.innerText || element?.textContent || "").trim();
  }

  function findMessageScroller() {
    const selectors = [
      ".msg-s-message-list-content",
      ".msg-s-message-list",
      ".msg-overlay-conversation-bubble-list",
      '[data-view-name="message-thread"]',
    ];
    for (const selector of selectors) {
      const node = document.querySelector(selector);
      if (node) {
        return node;
      }
    }
    return null;
  }

  async function loadOlderMessages() {
    const scroller = findMessageScroller();
    if (!scroller) {
      return;
    }
    for (let attempt = 0; attempt < 4; attempt += 1) {
      scroller.scrollTop = 0;
      await sleep(500);
    }
  }

  function readVisibleMessages(limit = 20) {
    const selectors = [
      ".msg-s-message-list__event",
      ".msg-s-message-group__messages li",
      ".msg-s-event-listitem",
    ];
    for (const selector of selectors) {
      const nodes = Array.from(document.querySelectorAll(selector));
      if (!nodes.length) {
        continue;
      }
      const texts = nodes
        .map((node) => visibleText(node))
        .filter(Boolean)
        .slice(-limit);
      if (texts.length) {
        return texts;
      }
    }
    return [];
  }

  function findSearchBox() {
    const selectors = [
      'input[placeholder*="Search messages"]',
      'input[aria-label*="Search messages"]',
      'input[placeholder*="Search"]',
    ];
    for (const selector of selectors) {
      const node = document.querySelector(selector);
      if (node) {
        return node;
      }
    }
    return null;
  }

  async function waitForSearchBox(timeoutMs = 12000) {
    const startedAt = Date.now();
    while (Date.now() - startedAt < timeoutMs) {
      const input = findSearchBox();
      if (input) {
        return input;
      }
      await sleep(400);
    }
    return null;
  }

  async function clearSearchBox(input) {
    input.focus();
    input.value = "";
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await sleep(300);
  }

  async function searchConversation(name) {
    for (let attempt = 0; attempt < 3; attempt += 1) {
      const searchBox = await waitForSearchBox();
      if (!searchBox) {
        await sleep(600);
        continue;
      }
      await clearSearchBox(searchBox);
      searchBox.focus();
      searchBox.value = name;
      searchBox.dispatchEvent(new Event("input", { bubbles: true }));
      searchBox.dispatchEvent(new Event("change", { bubbles: true }));
      await sleep(1400 + attempt * 300);

      const candidates = Array.from(document.querySelectorAll("li, div, a")).filter((node) =>
        visibleText(node).includes(name)
      );
      const conversation = candidates.find((node) => {
        const text = visibleText(node);
        return text.includes(name) && text.length < 500;
      });
      if (!conversation) {
        await sleep(500);
        continue;
      }
      conversation.click();
      await sleep(1600);
      return true;
    }
    return false;
  }

  async function scrapeMessageHistory(item) {
    const found = await searchConversation(item.name);
    if (!found) {
      return {
        name: item.name,
        profile_url: item.profile_url || "",
        messages: [],
      };
    }
    await loadOlderMessages();
    return {
      name: item.name,
      profile_url: item.profile_url || "",
      messages: readVisibleMessages(20),
    };
  }

  browser.runtime.onMessage.addListener((message) => {
    if (!message || message.type !== "JOB_AGENT_SCRAPE_MESSAGE_HISTORIES") {
      return undefined;
    }
    return (async () => {
      const histories = [];
      for (const item of message.payload.items || []) {
        histories.push(await scrapeMessageHistory(item));
      }
      return { histories };
    })();
  });
})();
