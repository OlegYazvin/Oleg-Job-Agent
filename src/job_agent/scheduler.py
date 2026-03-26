from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

from .config import Settings


CRON_MARKER = "# job-agent daily run"


def render_cron_line(settings: Settings) -> str:
    project_root = shlex.quote(str(settings.project_root))
    export_path = 'export PATH="$HOME/.local/bin:$PATH"'
    ollama_preamble = ""
    if settings.llm_provider == "ollama":
        ollama_health_url = shlex.quote(f"{settings.ollama_base_url.rstrip('/')}/api/version")
        ollama_log = shlex.quote(str(settings.output_dir / "ollama.log"))
        ollama_preamble = (
            f"if ! curl -fsS {ollama_health_url} >/dev/null; then "
            f"nohup ollama serve >> {ollama_log} 2>&1 & sleep 5; "
            f"fi && "
        )
    command = (
        f"cd {project_root} && {export_path} && . .venv/bin/activate && "
        f"{ollama_preamble}job-agent run >> output/cron.log 2>&1"
    )
    return f"{settings.daily_run_minute} {settings.daily_run_hour} * * * {command} {CRON_MARKER}"


def install_user_cron(settings: Settings) -> str:
    existing = _read_existing_crontab()
    lines = [line for line in existing.splitlines() if CRON_MARKER not in line]
    lines.append(render_cron_line(settings))
    crontab_text = "\n".join(line for line in lines if line.strip()) + "\n"
    subprocess.run(["crontab", "-"], input=crontab_text, text=True, check=True)
    return render_cron_line(settings)


def _read_existing_crontab() -> str:
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    if result.returncode != 0:
        return ""
    return result.stdout
