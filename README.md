# Job Agent

This project runs a daily workflow that:

1. Uses OpenAI's agent tooling to search for AI-related product manager jobs.
2. Uses multi-round discovery across direct ATS pages plus discovery sources such as LinkedIn Jobs, Built In, and Glassdoor, then resolves each lead back to a direct ATS or company careers URL.
3. Validates each candidate against the live ATS page before keeping it.
4. Keeps only jobs that are:
   - posted within the configured recency window (default 14 days),
   - fully remote,
   - base salary at least $200k,
   - linked directly to the company's ATS rather than an aggregator.
5. Continues searching in adaptive passes until it finds the configured minimum qualifying count or exhausts the configured pass/search limits.
6. Shows a desktop control dashboard and writes `data/live-status.json` so you can monitor or stop long-running searches.
7. Writes `data/search-diagnostics-latest.json` and stores pass-by-pass rejection reasons in each run snapshot so you can see why leads were rejected.
8. Uses LinkedIn authentication that can come from:
   - email/password,
   - email/password plus TOTP secret,
   - Google sign-in credentials,
   - Google sign-in credentials plus TOTP secret,
   - `li_at` and `JSESSIONID` cookies,
   - or a saved browser session.
9. Supports three LinkedIn modes:
   - manual review mode (default): generates per-job LinkedIn people-search links focused on the company so you can review first/second-degree contacts yourself,
   - Playwright mode: attempts browser-based LinkedIn discovery of first/second-degree contacts,
   - Firefox extension mode: captures LinkedIn search and messaging data from your real Firefox session through a local bridge.
10. Drafts personalized LinkedIn messages when contact data is available and enforces that each drafted message includes the job link and the relevant LinkedIn profile link.
11. Produces two date-stamped Word documents per run and also refreshes stable latest aliases:
   - `output/linkedin_outreach_messages-YYYYMMDD-HHMMSS.docx`
   - `output/job_summary-YYYYMMDD-HHMMSS.docx`
   - latest aliases:
     - `output/linkedin_outreach_messages.docx`
     - `output/job_summary.docx`
12. Saves a structured JSON snapshot of each run in `data/` and keeps persistent run/job history logs:
   - `data/run-history.json`
   - `data/job-history.json`
13. Suppresses previously reported jobs on later runs so recurring usage focuses on newly surfaced roles instead of duplicating old outreach packages.
14. Supports a local-first non-token mode using Ollama (`LLM_PROVIDER=ollama`) with optional OpenAI fallback for low-confidence cases, plus automatic Ollama restart/retry if the local model server is killed mid-run.

## Stack

- OpenAI Agents SDK for typed agent workflows
- OpenAI GPT-5.4 for search and drafting
- Playwright for LinkedIn browser automation
- Firefox WebExtension bridge for LinkedIn capture from a real Firefox session
- Direct ATS page validation with `httpx` and `beautifulsoup4`
- `python-docx` for Word document output

The OpenAI docs currently list GPT-5.4 as the latest model in the API docs navigation, and the official Agents SDK docs show `Agent`, `Runner`, `WebSearchTool`, and typed `output_type` workflows for structured agent results:

- https://developers.openai.com/api/docs/models
- https://openai.github.io/openai-agents-python/tools/
- https://openai.github.io/openai-agents-python/agents/
- https://platform.openai.com/docs/guides/tools-web-search?api-mode=responses&lang=python

## Maintainer Docs

If you are extending or adapting the project, start here:

- [SYSTEM_OVERVIEW.md](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/SYSTEM_OVERVIEW.md)
- [CUSTOMIZING_JOB_CRITERIA.md](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/CUSTOMIZING_JOB_CRITERIA.md)
- [CROSS_PLATFORM_INSTALL.md](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/CROSS_PLATFORM_INSTALL.md)

## Setup

For a platform-specific quick start, you can also use:

- Linux/macOS: [scripts/setup_unix.sh](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/scripts/setup_unix.sh)
- Windows PowerShell: [scripts/setup_windows.ps1](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/scripts/setup_windows.ps1)

1. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   . .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   python -m pip install -e .
   python -m playwright install chromium
   ```

3. Copy the example env file and fill in secrets:

   ```bash
   cp .env.example .env
   ```

4. Set at minimum:

   - `OPENAI_API_KEY` if `LLM_PROVIDER=openai` or `USE_OPENAI_FALLBACK=true`
   - Or, for local-first non-token mode:
     - `LLM_PROVIDER=ollama`
     - `OLLAMA_BASE_URL=http://localhost:11434`
     - `OLLAMA_MODEL=qwen2.5:14b-instruct`
     - `USE_OPENAI_FALLBACK=false`
   - Optional LinkedIn configuration:
     - `LINKEDIN_MANUAL_REVIEW_MODE=true` (default): no automated LinkedIn scraping; generates review links in the output doc.
     - `LINKEDIN_MANUAL_REVIEW_MODE=false`: enables automated LinkedIn contact discovery and then you should configure one auth path below.
     - `LINKEDIN_CAPTURE_MODE=playwright` (default automated mode)
     - or `LINKEDIN_CAPTURE_MODE=firefox_extension` to use the Firefox add-on in `firefox_extension/`
     - preferred: use the manual Google bootstrap flow and saved browser session
     - `LINKEDIN_EMAIL` + `LINKEDIN_PASSWORD`
     - `LINKEDIN_EMAIL` + `LINKEDIN_PASSWORD` + `LINKEDIN_TOTP_SECRET`
     - `GOOGLE_EMAIL` + `GOOGLE_PASSWORD`
     - `GOOGLE_EMAIL` + `GOOGLE_PASSWORD` + `GOOGLE_TOTP_SECRET`
     - `LINKEDIN_LI_AT` + `LINKEDIN_JSESSIONID`
     - or use the LinkedIn bootstrap flow and saved browser session only

5. Check readiness:

   ```bash
   . .venv/bin/activate
   job-agent doctor
   ```

   Look at `browser_choice`. If it still says `Playwright bundled Chromium`, install a local Chrome/Chromium build or set `BROWSER_EXECUTABLE_PATH` to a real browser binary before trying Google sign-in again.

## LinkedIn Authentication

If `LINKEDIN_MANUAL_REVIEW_MODE=true` (default), runs do not require automated LinkedIn authentication.
You can still sign in to LinkedIn in your normal browser to open the generated people-search links.

If you switch to `LINKEDIN_MANUAL_REVIEW_MODE=false`, you can authenticate LinkedIn in one of three main ways:

1. Preferred: reuse your current Firefox LinkedIn session automatically

   If Firefox is already signed into LinkedIn, the tool now tries to import live `.linkedin.com` cookies
   from your default Firefox profile automatically. You do not need to paste `li_at` or `JSESSIONID`
   into `.env` for this path.

   Notes:
   - Leave `LINKEDIN_LI_AT` and `LINKEDIN_JSESSIONID` blank so stale env cookies do not interfere.
   - Set `LINKEDIN_MANUAL_REVIEW_MODE=false` if you want the automated LinkedIn contact-discovery step.
   - Keep Firefox signed into LinkedIn on this machine.

2. Saved browser session with LinkedIn "Continue with Google"

   ```bash
   . .venv/bin/activate
   job-agent bootstrap-linkedin-google-session
   ```

   This is the lowest-friction option and the least likely to trip bot protections because you complete the real Google and LinkedIn login flow yourself in a normal browser window.

3. Direct credentials or cookies in `.env`

   - `LINKEDIN_EMAIL` + `LINKEDIN_PASSWORD`
   - optionally `LINKEDIN_TOTP_SECRET`
   - or `GOOGLE_EMAIL` + `GOOGLE_PASSWORD`
   - optionally `GOOGLE_TOTP_SECRET`
   - or `LINKEDIN_LI_AT` + `LINKEDIN_JSESSIONID`

4. Saved browser session with LinkedIn login

   The generic fallback is:

   ```bash
   . .venv/bin/activate
   job-agent bootstrap-linkedin-session
   ```

5. Complete LinkedIn login, Google login, MFA, or captcha in the opened browser if either service asks for it.
6. Press Enter in the terminal when the browser is fully logged in.
7. The tool will save session state to `.secrets/linkedin-state.json`.

After that, normal runs can reuse the saved profile and storage state.

## Firefox Extension Mode

If you want the tool to use your real Firefox LinkedIn session instead of an automated Chromium context:

1. In `.env`, set:

   ```bash
   LINKEDIN_MANUAL_REVIEW_MODE=false
   LINKEDIN_CAPTURE_MODE=firefox_extension
   LINKEDIN_EXTENSION_BRIDGE_HOST=127.0.0.1
   LINKEDIN_EXTENSION_BRIDGE_PORT=8765
   LINKEDIN_EXTENSION_AUTO_OPEN_SEARCH_TABS=true
   ```

2. In Firefox, open:

   ```text
   about:debugging#/runtime/this-firefox
   ```

3. Click `Load Temporary Add-on...`
4. Select:

   ```text
   firefox_extension/manifest.json
   ```

5. Click the extension icon, confirm the bridge URL is `http://127.0.0.1:8765`, and leave auto-capture enabled.
6. Stay logged into LinkedIn in Firefox.
7. Run:

   ```bash
   . .venv/bin/activate
   job-agent run
   ```

During the LinkedIn phase, the app opens company people-search tabs in Firefox. The extension captures:
- first-degree people at the company
- second-degree people at the company
- visible message history for first-degree contacts and mutual connectors when LinkedIn surfaces those conversations in Messaging

Files:
- extension source: `firefox_extension/README.md`
- bridge settings: `.env.example`

### Scripted Deployment

On this machine, Firefox Release will not accept this unsigned add-on as a normal permanent install. The scripted deployment path is:

```bash
. .venv/bin/activate
job-agent deploy-firefox-extension
```

That starts a dedicated Firefox instance with:
- the extension loaded as a temporary add-on
- a dedicated profile at `.secrets/firefox-extension-host/profile`
- imported LinkedIn cookies from your default Firefox profile when available

To remove that deployed instance later:

```bash
. .venv/bin/activate
job-agent remove-firefox-extension
```

Removal details are documented in `FIREFOX_EXTENSION_REMOVAL.md`.

## Run Once

```bash
. .venv/bin/activate
job-agent run
```

## Non-Token Mode

To run without paid token usage for drafting/discovery extraction:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b-instruct
OLLAMA_KEEP_ALIVE=0m
OLLAMA_NUM_CTX=2048
OLLAMA_NUM_BATCH=16
OLLAMA_MAX_CONCURRENT_REQUESTS=1
USE_OPENAI_FALLBACK=false
```

These defaults trade speed for stability on tighter-memory Linux machines:
- `OLLAMA_KEEP_ALIVE=0m` unloads the model between requests
- `OLLAMA_NUM_CTX=2048` trims context RAM usage
- `OLLAMA_MAX_CONCURRENT_REQUESTS=1` keeps local inference strictly serialized
- if Ollama crashes or is OOM-killed, the app now attempts to restart it and retry the interrupted local-model call

For the implementation roadmap and tradeoffs, see:
- `NON_TOKEN_APPROACH.md`

## Daily Scheduling

Install or update a daily user cron job from the CLI:

```bash
. .venv/bin/activate
job-agent install-cron
```

The schedule is controlled by `DAILY_RUN_HOUR` and `DAILY_RUN_MINUTE` in `.env`.

## Desktop Dashboard

Open the desktop control center with:

```bash
. .venv/bin/activate
job-agent dashboard
```

Platform helper scripts:

- Linux/macOS: [scripts/run_dashboard_unix.sh](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/scripts/run_dashboard_unix.sh)
- Windows PowerShell: [scripts/run_dashboard_windows.ps1](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/scripts/run_dashboard_windows.ps1)

The dashboard provides:
- a live current-run panel with stage, metrics, and recent events
- a run history table with prior outputs and counts
- `Start Run` and `Kill Run` controls
- quick-open buttons for the selected message and summary `.docx` files

## Important Notes

- This project drafts messages; it does not automatically send them.
- Manual review mode is enabled by default and generates per-job company-focused LinkedIn search links in the message document.
- LinkedIn's UI changes often, so browser selectors may need occasional maintenance.
- Firefox extension mode is the most direct way to reuse your current Firefox login without copying LinkedIn cookies into `.env`.
- `job-agent deploy-firefox-extension` is the scripted deployment path for the local Firefox bridge add-on here.
- Manual Google bootstrap is the recommended auth path.
- For Google sign-in, a real local Chrome/Chromium/Edge build is preferred over Playwright's bundled Chromium. You can point the tool at one with `BROWSER_EXECUTABLE_PATH`.
- Google sign-in is supported, but Google may still trigger anti-bot checks or approval prompts that require an interactive browser session.
- LinkedIn may still challenge automated sessions with MFA or captcha even if you provide credentials or cookies. The bootstrap command exists for that case.
- Discovery is intentionally broader than validation. Salary and remote filters are enforced from the direct ATS page, not from an aggregator snippet.
- `MINIMUM_QUALIFYING_JOBS`, `TARGET_JOB_COUNT`, `MAX_ADAPTIVE_SEARCH_PASSES`, `MAX_SEARCH_ROUNDS`, `SEARCH_ROUND_QUERY_LIMIT`, `MAX_LEADS_PER_QUERY`, and `MAX_LEADS_TO_RESOLVE_PER_PASS` control how aggressively the search keeps going.
- `PER_QUERY_TIMEOUT_SECONDS` and `PER_LEAD_TIMEOUT_SECONDS` prevent one slow search or lead from stalling the whole workflow.
- `job-agent run` opens the desktop dashboard when a graphical display is available. Use `job-agent run --no-gui` to disable it.
- The live progress file is `data/live-status.json`.
- Historical runs are retained in `data/run-history.json`, and previously reported jobs are tracked in `data/job-history.json`.
- The latest rejection log is `data/search-diagnostics-latest.json`.
- The summary document reports how many second-degree intro messages were generated per company, and the message document groups drafts by job and recipient.
