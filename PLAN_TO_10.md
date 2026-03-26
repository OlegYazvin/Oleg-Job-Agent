# Plan To Reach 10 Qualifying Roles

Current baseline: the workflow now reaches 5 validated roles reliably by replaying known high-signal ATS leads and revalidating them live each run.

## Goal

Get from 5 to 10 live roles that satisfy all current rules:
- AI-related PM scope
- fully remote
- posted within 14 days
- base salary at least $200k, or a justified inference when salary is missing

## Next Steps

1. Expand the seed system from static seeds into an auto-maintained replay cache.
Keep newly accepted roles, recent near-misses, and strong direct ATS URLs from diagnostics in a scored cache. Revalidate them every run, and automatically age out stale or repeatedly invalid entries.

2. Add ATS-specific parsers for the remaining weak spots.
Prioritize Rippling, Workable, and stubborn company-career shells so posted date, remote status, and salary come from structured page data instead of brittle text fallbacks.

3. Mine more direct ATS URLs before broad search.
Run targeted `site:` discovery first across Greenhouse, Ashby, Lever, Workday, SmartRecruiters, Jobvite, Workable, iCIMS, and Rippling, then only fall back to aggregator discovery when ATS-first search is thin.

4. Add company-careers traversal for promising aggregator hits.
When LinkedIn, Built In, or Glassdoor identifies a likely role, crawl the company careers page and board APIs more aggressively to recover the exact posting instead of discarding the lead early.

5. Tighten remote validation without over-rejecting JS-shell pages.
Preserve reliable remote hints from trusted sources when the direct page is a client-rendered shell, but keep explicit office-only language as a hard override.

6. Improve salary inference ranking.
Prefer accepted roles with explicit base salary over inferred-salary roles, and treat non-US postings with missing salary as lower-confidence unless there is unusually strong compensation evidence.

7. Add a focused "top-up" run mode after the first 5 are found.
Once 5 roles are accepted, continue for a short second phase that targets only companies, ATS platforms, and role patterns most correlated with recent accepts.

8. Separate LinkedIn contact discovery from job validation.
Do not block the search pipeline on LinkedIn auth. Keep jobs/doc generation independent, then run a second LinkedIn phase with either refreshed browser state or manual-review handoff.

## Success Criteria

- 10 accepted jobs in a single run manifest
- low false-positive rate on remote/date/salary
- reproducible reruns without manual code edits
- LinkedIn phase either succeeds with valid auth or cleanly degrades to manual-review links
