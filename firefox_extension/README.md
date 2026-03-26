# Firefox Extension

This temporary Firefox add-on captures LinkedIn people-search pages and messaging snippets from your real Firefox session and forwards them to the local Job Agent bridge.

Files:
- `manifest.json`: Firefox extension manifest.
- `background.js`: posts search results and message histories to the local bridge.
- `content_search.js`: parses LinkedIn people-search result pages.
- `content_messaging.js`: scrapes visible message history for connectors.
- `popup.html`, `popup.js`, `popup.css`: small settings UI for the bridge URL and auto-capture toggle.

Default bridge URL:
- `http://127.0.0.1:8765`

The Python app uses this bridge when:
- `LINKEDIN_MANUAL_REVIEW_MODE=false`
- `LINKEDIN_CAPTURE_MODE=firefox_extension`

Deployment steps are documented in the root `README.md`.
