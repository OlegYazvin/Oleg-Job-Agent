$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$PythonCommand = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "py" }

if (-not (Test-Path ".venv")) {
    & $PythonCommand -3 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -e ".[dev]"
& .\.venv\Scripts\python.exe -m playwright install chromium

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next steps:"
Write-Host "  Copy-Item .env.example .env"
Write-Host "  Edit .env"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  job-agent doctor"
Write-Host "  job-agent dashboard"
