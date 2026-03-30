# Next Steps

## Firefox Extension

"The next high-probability fix is to target a real logged-in Firefox profile with the extension loaded, rather than the temporary WebDriver-installed add-on."

Why this is the best next step:

- The extension host is stable enough to heartbeat, restart, and report `login_required`.
- The remaining blocker is LinkedIn auth inside the Selenium-managed Firefox/add-on environment.
- In local testing, LinkedIn auth worked in plain automated Firefox with imported cookies, but failed once the temporary WebDriver-installed add-on path was involved.

Recommended implementation steps:

1. Install the extension into a real Firefox profile that is already logged into LinkedIn.
2. Add config for that profile path and prefer opening tabs in that real profile before falling back to the dedicated temporary host profile.
3. Add a doctor check that confirms both:
   - the extension is installed in the chosen Firefox profile
   - the chosen profile is authenticated to LinkedIn
4. Keep the current `login_required` fail-fast behavior so auth issues surface immediately instead of timing out.

## Ollama

Decision:

- The best path to reduce memory usage is to switch to a smaller model first.
- Lowering `num_ctx` and `num_batch` is only a secondary tuning step.

Why:

- The current model is `qwen2.5:14b-instruct`.
- Ollama reports a CPU model buffer of about `8566 MiB`.
- The service was OOM-killed around `8.1G` to `8.3G` peak usage.
- This happened even after reducing generation settings as far as:
  - `num_ctx=512`
  - `num_batch=1`
  - `num_predict=32`
- That means the model weights themselves are the main problem, not just the request context/batch settings.

Recommended Ollama path:

1. Replace `qwen2.5:14b-instruct` with a smaller model.
2. Start with `qwen2.5:7b-instruct`.
3. If that is still unstable, drop to `qwen2.5:3b-instruct`.
4. After switching models, use modest conservative settings first:
   - `OLLAMA_NUM_CTX=1024`
   - `OLLAMA_NUM_BATCH=4`
   - `OLLAMA_NUM_PREDICT=256`
5. Only increase context/batch after repeated successful runs.

Notes:

- The Ollama executable path is not the main issue.
- The local install is healthy at `/home/olegy/.local/bin/ollama`.
- The cron/scheduler path handling now respects `OLLAMA_COMMAND`, so future path overrides will work consistently.
