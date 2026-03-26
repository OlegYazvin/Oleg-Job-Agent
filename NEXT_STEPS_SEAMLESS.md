# Next Steps To Make The Workflow More Seamless

## 1. Split Search From LinkedIn Retry

Add a `job-agent run-linkedin-phase --from-run <run-json>` command that:

- reads accepted jobs from an existing run artifact
- skips search and validation entirely
- reruns only contact discovery, message-history capture, drafting, and `.docx` generation

This removes the need to rerun the whole workflow when LinkedIn capture flakes or when you simply want fresher contact matches.

## 2. Clean Mutual Connector Names Before Drafting

Current LinkedIn search results sometimes produce noisy connector labels such as:

- `1 other`
- `Jeremy Frisch  a`
- `Tatyana Smirnova  a`

Improve normalization so drafting only keeps:

- real connector names
- canonical profile-linked names when available
- no count placeholders

This will make the generated outreach cleaner and reduce redundant messages.

## 3. Capture More Message History Context

Extend the Firefox extension flow so it can:

- open the messaging thread for each first-degree connector used in drafting
- extract the last few meaningful messages
- pass that context into local drafting

This will make more intro requests feel like natural follow-ups instead of neutral cold asks.

## 4. Add Review UI Before DOCX Export

Add a lightweight local review screen that shows, per job:

- accepted role summary
- discovered second-degree contacts
- grouped first-degree connectors
- drafted message preview

Allow:

- exclude a contact
- merge/split grouped targets
- regenerate one message

Then export the approved result to `.docx`.

## 5. Cache LinkedIn Company Results

Cache company-specific LinkedIn captures for a short TTL, for example `24h`, including:

- second-degree contacts
- connector mappings
- message-history snapshots

This will make reruns much faster and reduce repeated browsing in LinkedIn.

## 6. Add Direct Browser Handoff For Sending

After exporting the `.docx`, optionally generate:

- one-click `linkedin.com/messaging/compose` or profile URLs
- a "next message to send" queue

This would turn the workflow into:

1. find jobs
2. find connectors
3. draft grouped intros
4. walk you through sending them in Firefox with minimal hunting

## 7. Improve Accepted-Job Diversity

Right now the search is strong, but seed replay is doing a lot of the heavy lifting. To keep the pipeline fresh:

- maintain a rolling archive of accepted ATS URLs and rejection reasons
- prioritize companies that have appeared recently but not yet been accepted
- expand company-careers crawling for strong AI employers before relying on generic search results

## 8. Add Daily Delta Mode

Create a `daily-delta` mode that:

- compares today’s accepted jobs to prior run artifacts
- only drafts messages for new jobs or newly discovered connectors
- produces a short "what changed today" summary

That turns the system from a full rerun into a maintainable daily operating loop.
