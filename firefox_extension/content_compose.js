(function () {
  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function visibleText(element) {
    return (element?.innerText || element?.textContent || "").trim();
  }

  function candidateMessageButtons() {
    return Array.from(document.querySelectorAll("button, a")).filter((node) => {
      const text = visibleText(node);
      return /^message$/i.test(text) || /^send message$/i.test(text);
    });
  }

  async function openComposerIfNeeded() {
    if (findComposer()) {
      return true;
    }
    const button = candidateMessageButtons()[0];
    if (!button) {
      return false;
    }
    button.click();
    for (let attempt = 0; attempt < 10; attempt += 1) {
      await sleep(400);
      if (findComposer()) {
        return true;
      }
    }
    return false;
  }

  function findComposer() {
    const selectors = [
      '.msg-form__contenteditable[contenteditable="true"]',
      'div[role="textbox"][contenteditable="true"]',
      'textarea[name="message"]',
      "textarea",
    ];
    for (const selector of selectors) {
      const node = document.querySelector(selector);
      if (node) {
        return node;
      }
    }
    return null;
  }

  function setComposerText(node, text) {
    node.focus();
    if (node.matches("textarea")) {
      node.value = text;
      node.dispatchEvent(new Event("input", { bubbles: true }));
      node.dispatchEvent(new Event("change", { bubbles: true }));
      return;
    }
    try {
      document.execCommand("selectAll", false, null);
      document.execCommand("insertText", false, text);
    } catch (error) {
      node.textContent = text;
    }
    node.dispatchEvent(new InputEvent("input", { bubbles: true, data: text, inputType: "insertText" }));
  }

  browser.runtime.onMessage.addListener((message) => {
    if (!message || message.type !== "JOB_AGENT_OPEN_AND_PREFILL_MESSAGE") {
      return undefined;
    }
    return (async () => {
      const payload = message.payload || {};
      const messageBody = String(payload.messageBody || "").trim();
      if (!messageBody) {
        return { ok: false, error: "Missing message body." };
      }
      const opened = await openComposerIfNeeded();
      if (!opened) {
        return { ok: false, error: "LinkedIn message composer was not available on this page." };
      }
      const composer = findComposer();
      if (!composer) {
        return { ok: false, error: "Composer was not found after opening the message dialog." };
      }
      setComposerText(composer, messageBody);
      await sleep(200);
      return { ok: true };
    })();
  });
})();
