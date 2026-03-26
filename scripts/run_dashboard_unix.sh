#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Run scripts/setup_unix.sh first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
exec job-agent dashboard
