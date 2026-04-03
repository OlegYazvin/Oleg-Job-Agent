from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup

from .models import CompanyDiscoveryEntry


COMPANY_DISCOVERY_INDEX_FILENAME = "company-discovery-index.json"
KNOWN_BOARD_HOST_FRAGMENTS = (
    "jobs.ashbyhq.com",
    "boards.greenhouse.io",
    "job-boards.greenhouse.io",
    "jobs.lever.co",
    "myworkdayjobs.com",
    "careers.workday.com",
    "jobs.smartrecruiters.com",
    "smartrecruiters.com",
    "jobs.recruitee.com",
    "careers.tellent.com",
    "jobscore.com",
    "jobs.workable.com",
    "jobs.jobvite.com",
    "jobs.icims.com",
    "ats.rippling.com",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _normalize_company_key(company_name: str | None) -> str:
    return "".join(character.lower() for character in str(company_name or "") if character.isalnum())


def _load_json(path: Path, *, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def company_discovery_index_path(data_dir: Path) -> Path:
    return data_dir / COMPANY_DISCOVERY_INDEX_FILENAME


def load_company_discovery_entries(data_dir: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(company_discovery_index_path(data_dir), default={})
    if not isinstance(payload, dict):
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for raw_key, raw_entry in payload.items():
        if not isinstance(raw_entry, Mapping):
            continue
        try:
            entry = CompanyDiscoveryEntry.model_validate(raw_entry)
        except Exception:
            continue
        entries[str(raw_key)] = entry.model_dump(mode="json")
    return entries


def save_company_discovery_entries(data_dir: Path, entries: Mapping[str, Mapping[str, Any]]) -> None:
    rendered = {
        str(key): CompanyDiscoveryEntry.model_validate(value).model_dump(mode="json")
        for key, value in entries.items()
    }
    _write_json(company_discovery_index_path(data_dir), rendered)


def _dedupe_strings(values: list[str], *, limit: int) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
        if len(ordered) >= limit:
            break
    return ordered


def infer_careers_root(url: str | None) -> str | None:
    normalized = str(url or "").strip()
    if not normalized.startswith(("http://", "https://")):
        return None
    parsed = urlparse(normalized)
    host = (parsed.netloc or "").lower()
    if not host:
        return None
    if any(fragment in host for fragment in KNOWN_BOARD_HOST_FRAGMENTS):
        segments = [segment for segment in parsed.path.split("/") if segment]
        if "jobs.ashbyhq.com" in host and segments:
            return f"{parsed.scheme}://{host}/{segments[0]}"
        if "jobs.lever.co" in host and segments:
            return f"{parsed.scheme}://{host}/{segments[0]}"
        if "greenhouse.io" in host and segments:
            return f"{parsed.scheme}://{host}/{segments[0]}"
        if "ats.rippling.com" in host and segments:
            return f"{parsed.scheme}://{host}/{segments[0]}"
        return f"{parsed.scheme}://{host}"
    return f"{parsed.scheme}://{host}/careers"


def board_identifier_from_url(url: str | None) -> str | None:
    normalized = str(url or "").strip()
    if not normalized.startswith(("http://", "https://")):
        return None
    parsed = urlparse(normalized)
    host = (parsed.netloc or "").lower()
    segments = [segment for segment in parsed.path.split("/") if segment]
    if "jobs.ashbyhq.com" in host and segments:
        return f"ashby:{segments[0].lower()}"
    if "jobs.lever.co" in host and segments:
        return f"lever:{segments[0].lower()}"
    if "greenhouse.io" in host and segments:
        if segments[0] not in {"embed", "job_board"}:
            return f"greenhouse:{segments[0].lower()}"
    if "myworkdayjobs.com" in host:
        prefix = host.split(".myworkdayjobs.com", 1)[0]
        if prefix:
            return f"workday:{prefix.lower()}"
    if "careers.workday.com" in host and segments:
        return f"workday:{segments[0].lower()}"
    if "jobs.smartrecruiters.com" in host and segments:
        return f"smartrecruiters:{segments[0].lower()}"
    if ("jobs.recruitee.com" in host or "careers.tellent.com" in host) and segments:
        if segments[0] == "o" and len(segments) >= 2:
            return f"recruitee:{segments[1].lower()}"
        return f"recruitee:{segments[0].lower()}"
    if "jobscore.com" in host:
        prefix = host.split(".jobscore.com", 1)[0]
        if prefix:
            return f"jobscore:{prefix.lower()}"
    if "jobs.workable.com" in host and segments:
        return f"workable:{segments[0].lower()}"
    if "jobs.jobvite.com" in host and segments:
        return f"jobvite:{segments[0].lower()}"
    if "jobs.icims.com" in host:
        if len(segments) >= 2:
            return f"icims:{segments[1].lower()}"
        return "icims"
    if "ats.rippling.com" in host and segments:
        return f"rippling:{segments[0].lower()}"
    return None


def board_url_ats_type(url: str | None) -> str | None:
    normalized = str(url or "").strip()
    if not normalized.startswith(("http://", "https://")):
        return None
    host = (urlparse(normalized).netloc or "").lower()
    if "jobs.ashbyhq.com" in host:
        return "Ashby"
    if "jobs.lever.co" in host:
        return "Lever"
    if "greenhouse.io" in host:
        return "Greenhouse"
    if "myworkdayjobs.com" in host or "careers.workday.com" in host:
        return "Workday"
    if "smartrecruiters.com" in host:
        return "SmartRecruiters"
    if "recruitee.com" in host or "careers.tellent.com" in host:
        return "Recruitee"
    if "jobscore.com" in host:
        return "JobScore"
    if "jobs.workable.com" in host:
        return "Workable"
    if "jobvite.com" in host:
        return "Jobvite"
    if "icims.com" in host:
        return "iCIMS"
    if "ats.rippling.com" in host:
        return "Rippling"
    return None


def extract_embedded_board_urls(page_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    candidates: list[str] = []

    def maybe_add(raw_url: str | None) -> None:
        normalized = str(raw_url or "").strip()
        if not normalized.startswith(("http://", "https://")):
            return
        host = (urlparse(normalized).netloc or "").lower()
        if any(fragment in host for fragment in KNOWN_BOARD_HOST_FRAGMENTS):
            candidates.append(normalized)

    for tag in soup.find_all(["a", "iframe", "script"]):
        maybe_add(tag.get("href") or tag.get("src"))

    raw_html = html or ""
    greenhouse_embed_match = re.finditer(r"boards\.greenhouse\.io/embed/job_board/js\?for=([a-z0-9_-]+)", raw_html, re.I)
    for match in greenhouse_embed_match:
        token = match.group(1).strip()
        if token:
            candidates.append(f"https://boards.greenhouse.io/{token}")

    for raw_url in re.findall(r"https?://[^\s\"'<>]+", raw_html):
        maybe_add(raw_url)

    if page_url.startswith(("http://", "https://")):
        maybe_add(page_url)

    return _dedupe_strings(candidates, limit=12)


def upsert_company_discovery_entry(
    entries: dict[str, dict[str, Any]],
    *,
    company_name: str,
    source_url: str | None = None,
    careers_root: str | None = None,
    board_urls: list[str] | None = None,
    board_identifiers: list[str] | None = None,
    ats_types: list[str] | None = None,
    source_trust: int = 0,
    generated_at: str | None = None,
    run_id: str | None = None,
    ai_pm_candidate_delta: int = 0,
    official_board_lead_delta: int = 0,
) -> tuple[bool, int]:
    company_key = _normalize_company_key(company_name)
    if not company_key:
        return False, 0
    timestamp = generated_at or _utc_now_iso()
    was_new_company = company_key not in entries
    entry = dict(entries.get(company_key) or {})
    existing_board_identifiers = [str(item) for item in entry.get("board_identifiers", []) if str(item).strip()]
    new_board_identifiers = list(board_identifiers or [])
    for board_url in board_urls or []:
        identifier = board_identifier_from_url(board_url)
        if identifier:
            new_board_identifiers.append(identifier)
    deduped_board_identifiers = _dedupe_strings([*existing_board_identifiers, *new_board_identifiers], limit=16)
    new_board_count = len(set(deduped_board_identifiers) - set(existing_board_identifiers))
    existing = CompanyDiscoveryEntry.model_validate(
        {
            "company_key": company_key,
            "company_name": company_name,
            "first_seen_at": timestamp,
            **entry,
        }
    )
    updated = existing.model_copy(
        update={
            "company_name": company_name,
            "careers_roots": _dedupe_strings([*existing.careers_roots, *( [careers_root] if careers_root else [] )], limit=8),
            "ats_types": _dedupe_strings(
                [*existing.ats_types, *(ats_types or []), *[value for value in (board_url_ats_type(url) for url in board_urls or []) if value]],
                limit=8,
            ),
            "board_identifiers": deduped_board_identifiers,
            "board_urls": _dedupe_strings([*existing.board_urls, *(board_urls or [])], limit=16),
            "source_hosts": _dedupe_strings(
                [
                    *existing.source_hosts,
                    *(
                        [(urlparse(source_url).netloc or "").lower()]
                        if source_url and source_url.startswith(("http://", "https://"))
                        else []
                    ),
                ],
                limit=8,
            ),
            "source_trust": max(existing.source_trust, int(source_trust or 0)),
            "last_seen_at": timestamp,
            "last_successful_discovery_run": run_id or existing.last_successful_discovery_run,
            "ai_pm_candidate_count": max(0, existing.ai_pm_candidate_count + int(ai_pm_candidate_delta or 0)),
            "official_board_lead_count": max(0, existing.official_board_lead_count + int(official_board_lead_delta or 0)),
        }
    )
    entries[company_key] = updated.model_dump(mode="json")
    return was_new_company, new_board_count
