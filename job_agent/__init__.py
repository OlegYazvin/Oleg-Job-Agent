from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_PACKAGE_DIR = _REPO_ROOT / "src" / "job_agent"

if _SRC_PACKAGE_DIR.is_dir():
    __path__ = [str(_SRC_PACKAGE_DIR)]
else:  # pragma: no cover - fallback for non-repo packaging contexts
    __path__ = [str(Path(__file__).resolve().parent)]
