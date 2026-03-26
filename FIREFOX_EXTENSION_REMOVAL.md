# Firefox Extension Removal

The Job Agent Firefox bridge extension is deployed here as a dedicated Firefox host process, not as a normal signed Firefox add-on install.

## What Gets Created

- Host state file:
  - `.secrets/firefox-extension-host/state.json`
- Dedicated Firefox profile:
  - `.secrets/firefox-extension-host/profile`
- Host log:
  - `output/firefox_extension_host.log`
- Local geckodriver binary:
  - `.tools/geckodriver`

## Remove It Cleanly

Run:

```bash
. .venv/bin/activate
job-agent remove-firefox-extension
```

That:
- stops the dedicated Firefox extension host process
- removes `.secrets/firefox-extension-host/`

## Manual Cleanup

If you ever want to clean it up by hand instead:

1. Stop the host process listed in `.secrets/firefox-extension-host/state.json`
2. Delete:
   - `.secrets/firefox-extension-host/`
3. Optionally delete:
   - `.tools/geckodriver`

## What It Does Not Change

- It does not remove or alter your main Firefox profile at `~/.mozilla/firefox/altnexod.default-release`
- It does not add anything to Firefox Sync
- It does not install a signed permanent Firefox add-on into your normal profile
