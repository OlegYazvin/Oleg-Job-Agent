# Cross-Platform Install And Run

This guide is for Windows 10/11, macOS, and most Linux distros.

## What You Need

- Python 3.11 or newer
- Git
- Firefox if you want Firefox-extension LinkedIn capture
- a desktop session if you want the dashboard GUI

## Quick Start By Platform

### Linux and macOS

From the project root:

```bash
chmod +x scripts/setup_unix.sh scripts/run_dashboard_unix.sh scripts/run_once_unix.sh
./scripts/setup_unix.sh
cp .env.example .env
./scripts/run_dashboard_unix.sh
```

### Windows 10/11

From PowerShell in the project root:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
Copy-Item .env.example .env
.\scripts\run_dashboard_windows.ps1
```

## What The Setup Scripts Do

The setup scripts:

1. create `.venv`
2. install the package in editable mode
3. install dev/test dependencies
4. install Playwright Chromium

Files:

- [scripts/setup_unix.sh](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/scripts/setup_unix.sh)
- [scripts/setup_windows.ps1](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/scripts/setup_windows.ps1)

## Run Commands

### Dashboard

- Linux/macOS:

```bash
./scripts/run_dashboard_unix.sh
```

- Windows:

```powershell
.\scripts\run_dashboard_windows.ps1
```

### Run Once

- Linux/macOS:

```bash
./scripts/run_once_unix.sh
```

- Windows:

```powershell
.\scripts\run_once_windows.ps1
```

## Firefox Extension Mode On All Platforms

The Firefox extension host now auto-downloads `geckodriver` for:

- Linux x64 and ARM64
- macOS Intel and Apple Silicon
- Windows x64

That logic lives in [firefox_extension_host.py](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/src/job_agent/firefox_extension_host.py).

You still need Firefox itself installed.

## Platform Notes

### Windows

- Use PowerShell rather than Command Prompt when possible.
- If script execution is blocked, use:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

- Scheduling via Windows Task Scheduler is not yet wrapped by a dedicated CLI command. The app itself runs fine; only cron installation is Linux/macOS-centric.

### macOS

- The Unix scripts work as-is.
- If Gatekeeper prompts when launching Firefox or Python tooling, approve them in the normal macOS way.

### Linux

- The Unix scripts work as-is.
- Cron installation is already supported by `job-agent install-cron`.

## Recommended First Validation

After setup:

```bash
. .venv/bin/activate
job-agent doctor
pytest -q
```

On Windows:

```powershell
.\.venv\Scripts\Activate.ps1
job-agent doctor
pytest -q
```

## Important Files For Ongoing Use

- [README.md](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/README.md)
- [SYSTEM_OVERVIEW.md](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/SYSTEM_OVERVIEW.md)
- [CUSTOMIZING_JOB_CRITERIA.md](/home/olegy/Documents/Projects/Oleg%20Job%20Agent/CUSTOMIZING_JOB_CRITERIA.md)
