# Non-Token Local-First Approach (Implement Later)

This is a roadmap for running most of the workflow without paid API tokens.

## Goal

Keep live web discovery and ATS validation, but replace OpenAI model calls with local components where possible.

## Suggested Stack (<= 200 GB)

- Runtime: `ollama`
- Primary local model: `qwen2.5:14b-instruct`
- Optional upgrade: `qwen2.5:32b-instruct` (better quality if hardware supports it)
- Reranker/embeddings (optional): `bge-m3` or equivalent small local embedding model

Estimated storage:
- `qwen2.5:14b-instruct`: ~10-20 GB class
- `qwen2.5:32b-instruct`: ~20-70 GB class depending quantization
- Tools/cache/logs: typically < 20 GB

## What To Replace First

1. Replace drafting agents with local prompt calls.
2. Replace discovery lead extraction prompts with rule-based extraction + local LLM cleanup.
3. Keep ATS page parsing and hard filters deterministic (date/remote/salary/url host checks).
4. Keep OpenAI as optional fallback for low-confidence cases only.

## Minimal Architecture

- Add a model provider abstraction:
  - `provider=openai | ollama`
  - one interface for `generate_structured(prompt, schema)`
- Use local schema-constrained JSON output where possible.
- Add confidence scoring:
  - if confidence < threshold, either reject candidate or (optionally) call fallback provider.

## Deterministic Rules To Preserve

- Only direct ATS/company-careers links (no aggregators)
- Posted within `POSTED_WITHIN_DAYS`
- Fully remote only
- Base salary >= `MIN_BASE_SALARY_USD`
- AI-related PM role only

The stricter these rules are, the less model quality variance matters.

## Implementation Phases

### Phase 1: Drafting only (lowest risk)

- Route first/second-order message drafting through local model.
- Keep existing JSON message schema.
- Validate required links are present in final body.

### Phase 2: Discovery extraction

- For each search result page, extract candidate text snippets with deterministic scraping.
- Ask local model to map snippets into `JobLead` objects.
- Continue current resolution + validation pipeline.

### Phase 3: Confidence and fallback

- Add per-candidate confidence score.
- Only use paid model fallback when:
  - role relevance is ambiguous,
  - salary parsing is ambiguous,
  - or URL resolution confidence is low.

## Operational Tips

- Run local model calls with short prompts and strict schemas.
- Cache normalized lead/resolution results by URL.
- Reduce repeated passes by excluding previously rejected URLs by reason code.
- Keep manual LinkedIn review mode enabled to avoid brittle browser automation.

## Suggested Env Additions (Future)

- `LLM_PROVIDER=ollama`
- `OLLAMA_MODEL=qwen2.5:14b-instruct`
- `USE_OPENAI_FALLBACK=false`
- `LOCAL_CONFIDENCE_THRESHOLD=0.75`

## Success Criteria

- Same hard-filter behavior as current pipeline.
- Comparable qualifying-job count on repeated runs.
- Major reduction in paid token usage.
