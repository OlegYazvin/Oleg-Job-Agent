# Customizing Job Criteria

This file is meant for future coding assistants and maintainers who want to adapt the project for a different job search.

## The Main Principle

Do not hardcode new criteria by scattering new keywords across the codebase.

Instead:

1. start with [criteria.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/criteria.py)
2. adjust environment-level hard filters in [config.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/config.py) or `.env`
3. only then touch deeper search/validation logic if the new role family truly needs it

## Safe Customization Layers

### Layer 1: No code changes

You can already change these in `.env`:

- `MIN_BASE_SALARY_USD`
- `POSTED_WITHIN_DAYS`
- `MINIMUM_QUALIFYING_JOBS`
- `TARGET_JOB_COUNT`
- `MAX_ADAPTIVE_SEARCH_PASSES`
- `MAX_SEARCH_ROUNDS`
- `SEARCH_ROUND_QUERY_LIMIT`
- `MAX_LEADS_PER_QUERY`
- `MAX_LEADS_TO_RESOLVE_PER_PASS`

These change search aggressiveness and hard filters without changing the role family.

### Layer 2: Change the role family

The role-family definition lives in [criteria.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/criteria.py).

That file contains:

- `default_search_queries`
- `ai_signal_tokens`
- `ai_strong_context_tokens`
- `ai_ownership_tokens`
- `ai_low_signal_patterns`
- `senior_title_tokens`
- `title_only_salary_inference_tokens`

If you want to adapt the project for something like:

- data science leadership
- engineering management
- developer relations
- technical program management

the first place to edit is this file.

## How The Current AI PM Matching Works

The current role family says a role is relevant when:

1. it looks like a product manager role
2. there is strong AI/ML/LLM/agentic signal
3. the AI signal is not just low-signal boilerplate

Those checks are implemented in [job_search.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/job_search.py):

- `_contains_ai_signal`
- `_has_strong_ai_context`
- `_is_ai_related_product_manager_text`
- `_is_ai_related_product_manager`
- `_lead_is_ai_related_product_manager`

If you are changing the role family substantially, expect to update those functions too.

## When To Change `criteria.py` Only

Only update [criteria.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/criteria.py) when:

- titles are still broadly similar
- the discovery query language needs to change
- the keyword signals need to change
- the salary-inference seniority tokens need to change

Examples:

- switching from `AI product manager` to `ML product manager`
- emphasizing `applied ML`, `LLM platform`, or `agentic systems`
- changing seniority expectations

## When To Also Change `job_search.py`

You should also change [job_search.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/job_search.py) when:

- the job family is no longer a PM role
- the role-title grammar is different
- the remote/salary inference heuristics need a different interpretation
- title matching needs new structural logic, not just new keywords

Common places:

- `_is_ai_related_product_manager_text`
- `_lead_is_ai_related_product_manager`
- `_role_match_score`
- `_infer_salary_from_experience`
- `_matches_filters`

## Do Not Break Repeated Runs

This project is designed for recurring use.

When changing criteria:

- keep `data/job-history.json` behavior intact
- do not remove date-stamped outputs
- do not remove historical run snapshots
- preserve duplicate suppression unless the product requirement explicitly changes

If you intentionally want a new search profile to ignore old duplicates, add a profile-aware dedupe strategy rather than simply deleting the history logic.

## Recommended Change Process For Future Assistants

1. Update [criteria.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/criteria.py)
2. Update or add tests in:
   - [test_job_search.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/tests/test_job_search.py)
   - [test_job_pages.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/tests/test_job_pages.py)
   - [test_drafting.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/tests/test_drafting.py) if message behavior changes
3. Run:

```bash
. .venv/bin/activate
pytest -q
```

4. Do one real run and inspect:
   - `data/live-status.json`
   - `data/search-diagnostics-latest.json`
   - the newest `data/run-*.json`
   - the newest date-stamped `.docx` outputs

## Suggested Future Refactor If Multi-Profile Support Is Needed

If this project eventually needs multiple job families, the next clean step is:

1. allow multiple named profiles in [criteria.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/criteria.py)
2. add `SEARCH_PROFILE_NAME` to `.env`
3. make [config.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/config.py) select the active profile
4. make job-history dedupe optionally profile-aware

That would let the same app run, for example:

- `ai_pm_remote_us_200k`
- `staff_data_science_leadership`
- `engineering_manager_remote_us`

without forcing assistants to rewrite the search pipeline each time.
