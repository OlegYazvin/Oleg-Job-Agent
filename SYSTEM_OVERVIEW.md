# System Overview

This document explains how the Job Agent works end to end, with an emphasis on where to change behavior safely.

## High-Level Flow

The main workflow entry point is [workflow.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/workflow.py).

Each run does this:

1. Load settings from `.env` via [config.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/config.py).
2. Search for job leads and validate them via [job_search.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/job_search.py).
3. Reject roles that do not meet the hard filters:
   - direct ATS/company job URL
   - recent enough
   - fully remote
   - salary threshold
   - role-family match
4. Skip jobs that were already reported in prior runs using [history.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/history.py).
5. For accepted jobs:
   - generate manual LinkedIn review links, or
   - run LinkedIn discovery through Playwright or the Firefox extension bridge
6. Draft outreach messages via [drafting.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/drafting.py).
7. Produce date-stamped Word documents via [reports.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/reports.py).
8. Save run artifacts and update persistent history via [storage.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/storage.py) and [history.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/history.py).

## Important Modules

### Core runtime

- [cli.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/cli.py)
  - CLI entrypoint
  - commands like `run`, `doctor`, `dashboard`, `deploy-firefox-extension`
- [workflow.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/workflow.py)
  - orchestrates the full run
- [config.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/config.py)
  - all environment-driven configuration
- [models.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/models.py)
  - typed payloads for leads, jobs, LinkedIn contacts, messages, and manifests

### Search and validation

- [criteria.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/criteria.py)
  - centralized default role-family search profile
  - default search queries
  - AI/ML/agentic role signal tokens
  - title/seniority inference tokens
- [job_search.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/job_search.py)
  - discovery across search engines, ATSes, and job boards
  - ATS resolution
  - validation
  - adaptive search passes
  - historical dedupe against prior reported jobs
- [job_pages.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/job_pages.py)
  - ATS/job-page fetch and parsing
  - remote detection
  - salary extraction
  - posted-date extraction

### LinkedIn and message generation

- [linkedin.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/linkedin.py)
  - Playwright LinkedIn path
  - Firefox-cookie reuse
  - extension-capture result normalization
- [linkedin_extension_bridge.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/linkedin_extension_bridge.py)
  - localhost bridge used by the Firefox extension
- [firefox_extension_host.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/firefox_extension_host.py)
  - dedicated Firefox instance with temporary add-on deployment
- [drafting.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/drafting.py)
  - first-degree and second-degree message generation
  - grouped second-degree intro requests by connector

### Persistence, outputs, and GUI

- [reports.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/reports.py)
  - generates `.docx` files
  - writes date-stamped outputs plus stable latest aliases
- [storage.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/storage.py)
  - saves per-run JSON snapshots
- [history.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/history.py)
  - persistent `run-history.json`
  - persistent `job-history.json`
  - duplicate suppression memory across runs
- [status.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/status.py)
  - live status file writer
- [dashboard.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/dashboard.py)
  - desktop control center
  - run history, live status, start/kill controls

## Output Files

### Live state

- `data/live-status.json`
- `data/search-diagnostics-latest.json`

### Historical state

- `data/run-YYYYMMDD-HHMMSS.json`
- `data/run-history.json`
- `data/job-history.json`

### Documents

- `output/linkedin_outreach_messages-YYYYMMDD-HHMMSS.docx`
- `output/job_summary-YYYYMMDD-HHMMSS.docx`
- latest aliases:
  - `output/linkedin_outreach_messages.docx`
  - `output/job_summary.docx`

## How Repeated Runs Behave

This project is intended to be run repeatedly, not just once.

Important recurring-run behavior:

- accepted jobs are persisted in `data/job-history.json`
- later runs suppress already reported direct job URLs
- run-level history is kept in `data/run-history.json`
- date-stamped `.docx` outputs preserve past runs instead of overwriting them

## Local Model Resilience

Local-mode LLM calls go through [llm_provider.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/llm_provider.py) and [ollama_runtime.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/ollama_runtime.py).

Current resilience behavior:

- local requests are serialized by default
- low-memory Ollama options are applied automatically
- if Ollama is not healthy, the app tries to start it
- if Ollama dies during a request, the app can restart it and retry

## Where To Start If Something Breaks

### Search is too narrow or wrong

Start with:
- [criteria.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/criteria.py)
- [job_search.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/job_search.py)

### A good job is being rejected

Start with:
- [job_pages.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/job_pages.py)
- [job_search.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/job_search.py)

### LinkedIn capture is failing

Start with:
- [linkedin.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/linkedin.py)
- [linkedin_extension_bridge.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/linkedin_extension_bridge.py)
- [firefox_extension_host.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/firefox_extension_host.py)
- `output/firefox_extension_host.log`

### The GUI is wrong or needs improvement

Start with:
- [dashboard.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/dashboard.py)
- [status.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/status.py)
