from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

from .models import CompanyDiscoveryCrawlRecord, CompanyDiscoveryEntry, CompanyDiscoveryFrontierTask


COMPANY_DISCOVERY_INDEX_FILENAME = "company-discovery-index.json"
COMPANY_DISCOVERY_FRONTIER_FILENAME = "company-discovery-frontier.json"
COMPANY_DISCOVERY_CRAWL_HISTORY_FILENAME = "company-discovery-crawl-history.json"
COMPANY_DISCOVERY_AUDIT_FILENAME = "company-discovery-audit.json"

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

BUILTIN_HOST_FRAGMENTS = (
    "builtin.com",
    "builtinnyc.com",
    "builtinsf.com",
    "builtinseattle.com",
    "builtinla.com",
    "builtinchicago.org",
    "builtinchicago.com",
    "builtincharlotte.com",
    "builtinboston.com",
    "builtincolorado.com",
)

SOCIAL_PROFILE_HOST_FRAGMENTS = (
    "linkedin.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "facebook.com",
    "youtube.com",
    "tiktok.com",
)

DIRECTORY_PORTAL_HOST_FRAGMENTS = (
    "ycombinator.com",
    "workatastartup.com",
    "wellfound.com",
    "knowledgebase.builtin.com",
    "employers.builtin.com",
    *BUILTIN_HOST_FRAGMENTS,
)

RECURSIVE_PLATFORM_HOST_FRAGMENTS = (
    "knowledgebase.builtin.com",
    "employers.builtin.com",
    *BUILTIN_HOST_FRAGMENTS,
    *SOCIAL_PROFILE_HOST_FRAGMENTS,
)

COMMON_CAREERS_PATHS = (
    "/careers",
    "/jobs",
    "/join-us",
    "/company/careers",
    "/about/careers",
    "/work-with-us",
)

DIRECTORY_SOURCE_URLS = (
    "https://www.ycombinator.com/companies",
    "https://www.workatastartup.com/companies",
    "https://wellfound.com/jobs",
    "https://www.builtin.com/jobs",
)

LOW_TRUST_SCOUT_HOST_FRAGMENTS = (
    "mediabistro.com",
    "remotejobshive.com",
    "thatstartupjob.com",
    "remoterocketship.com",
    "jobgether.com",
    "tracxn.com",
)

DEFAULT_FRONTIER_RETRY_DELAY = timedelta(hours=6)
GENERIC_DIRECTORY_COMPANY_KEYS = {
    "about",
    "career",
    "careers",
    "companies",
    "company",
    "employer",
    "employers",
    "hire",
    "hiring",
    "job",
    "jobs",
    "location",
    "locations",
    "remote",
    "role",
    "roles",
    "search",
    "team",
    "teams",
}


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


def _host_matches_fragment(host: str, fragment: str) -> bool:
    lowered_host = str(host or "").lower()
    lowered_fragment = str(fragment or "").lower()
    return lowered_host == lowered_fragment or lowered_host.endswith(f".{lowered_fragment}")


def _slug_to_company_name(slug: str, *, company_name_hint: str | None = None) -> str:
    hint = re.sub(r"\s+", " ", str(company_name_hint or "").strip())
    hint = re.sub(r"^(?:view|see|get)\s+(?:all\s+)?jobs?\s+(?:at|for)\s+", "", hint, flags=re.I)
    hint = re.sub(r"^(?:jobs?|careers?)\s+(?:at|for)\s+", "", hint, flags=re.I)
    hint = re.sub(r"\s+(?:jobs?|careers?)$", "", hint, flags=re.I)
    hint_key = _normalize_company_key(hint)
    if (
        hint
        and len(hint) <= 80
        and re.search(r"[A-Za-z]", hint)
        and not hint_key.startswith("builtin")
        and hint_key not in {"linkedin", "twitter", "instagram", "facebook", "youtube", "tiktok", "x"}
    ):
        return hint
    cleaned_slug = re.sub(r"[-_](?:\d+)$", "", str(slug or "").strip())
    words = [word for word in re.split(r"[-_]+", cleaned_slug) if word]
    return " ".join(word[:1].upper() + word[1:] for word in words)


def _directory_company_route(url: str | None) -> dict[str, str] | None:
    normalized = str(url or "").strip()
    if not normalized.startswith(("http://", "https://")):
        return None
    parsed = urlparse(normalized)
    host = (parsed.netloc or "").lower()
    segments = [segment for segment in parsed.path.split("/") if segment]
    if not segments:
        return None

    slug: str | None = None
    route_segments: list[str] | None = None
    has_jobs_segment = False

    if any(_host_matches_fragment(host, fragment) for fragment in BUILTIN_HOST_FRAGMENTS):
        if len(segments) >= 2 and segments[0] == "company":
            slug = segments[1]
            route_segments = ["company", slug]
            has_jobs_segment = "jobs" in segments[2:]
    elif _host_matches_fragment(host, "ycombinator.com"):
        if len(segments) >= 2 and segments[0] == "companies":
            slug = segments[1]
            route_segments = ["companies", slug]
            has_jobs_segment = "jobs" in segments[2:]
    elif _host_matches_fragment(host, "workatastartup.com"):
        if len(segments) >= 2 and segments[0] in {"companies", "company"}:
            slug = segments[1]
            route_segments = [segments[0], slug]
            has_jobs_segment = "jobs" in segments[2:]
    elif _host_matches_fragment(host, "wellfound.com"):
        if len(segments) >= 2 and segments[0] in {"company", "companies", "organization", "organizations"}:
            slug = segments[1]
            route_segments = [segments[0], slug]
            has_jobs_segment = "jobs" in segments[2:]

    if not slug or route_segments is None:
        return None

    cleaned_slug = re.sub(r"[-_](?:\d+)$", "", str(slug or "").strip())
    slug_key = _normalize_company_key(cleaned_slug)
    if not slug_key or slug_key in GENERIC_DIRECTORY_COMPANY_KEYS:
        return None

    base_url = f"{parsed.scheme}://{parsed.netloc}/{'/'.join(route_segments)}".rstrip("/")
    return {
        "company_url": base_url,
        "jobs_url": f"{base_url}/jobs",
        "slug": cleaned_slug,
        "task_type": "careers_root" if has_jobs_segment else "company_page",
    }


def _canonical_directory_company_url(
    url: str | None,
    *,
    preferred_task_type: str | None = None,
) -> str | None:
    route = _directory_company_route(url)
    if route is None:
        return None
    task_type = preferred_task_type if preferred_task_type in {"company_page", "careers_root"} else route["task_type"]
    if task_type == "careers_root":
        return str(route["jobs_url"]).rstrip("/")
    return str(route["company_url"]).rstrip("/")


def _directory_company_key(url: str | None) -> str:
    route = _directory_company_route(url)
    if route is None:
        return ""
    return _normalize_company_key(str(route["slug"]))


def _company_keys_conflict(expected_key: str | None, observed_key: str | None) -> bool:
    normalized_expected = str(expected_key or "").strip()
    normalized_observed = str(observed_key or "").strip()
    if not normalized_expected or not normalized_observed:
        return False
    if normalized_expected == normalized_observed:
        return False
    if normalized_expected in normalized_observed or normalized_observed in normalized_expected:
        return False
    return True


def _directory_company_candidate(
    url: str | None,
    *,
    company_name_hint: str | None = None,
) -> dict[str, str] | None:
    route = _directory_company_route(url)
    if route is None:
        return None
    company_name = _slug_to_company_name(str(route["slug"]), company_name_hint=company_name_hint)
    if not company_name:
        return None
    company_key = _normalize_company_key(company_name)
    if not company_key or company_key in GENERIC_DIRECTORY_COMPANY_KEYS:
        return None
    task_type = str(route["task_type"])
    return {
        "url": _canonical_directory_company_url(url, preferred_task_type=task_type) or str(route["company_url"]).rstrip("/"),
        "company_name": company_name,
        "task_type": task_type,
    }


def _is_blocked_external_company_homepage_host(host: str) -> bool:
    return any(_host_matches_fragment(host, fragment) for fragment in (*DIRECTORY_PORTAL_HOST_FRAGMENTS, *SOCIAL_PROFILE_HOST_FRAGMENTS))


def _is_recursive_platform_host(host: str) -> bool:
    return any(_host_matches_fragment(host, fragment) for fragment in RECURSIVE_PLATFORM_HOST_FRAGMENTS)


def is_company_discovery_seed_url(
    url: str | None,
    *,
    preferred_task_type: str | None = None,
) -> bool:
    normalized = str(url or "").strip()
    if not normalized.startswith(("http://", "https://")):
        return False
    if board_identifier_from_url(normalized):
        return True
    if _canonical_directory_company_url(normalized, preferred_task_type=preferred_task_type):
        return True
    host = (urlparse(normalized).netloc or "").lower()
    if not host:
        return False
    return not _is_blocked_external_company_homepage_host(host)


def _parse_datetime(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def company_discovery_index_path(data_dir: Path) -> Path:
    return data_dir / COMPANY_DISCOVERY_INDEX_FILENAME


def company_discovery_frontier_path(data_dir: Path) -> Path:
    return data_dir / COMPANY_DISCOVERY_FRONTIER_FILENAME


def company_discovery_crawl_history_path(data_dir: Path) -> Path:
    return data_dir / COMPANY_DISCOVERY_CRAWL_HISTORY_FILENAME


def company_discovery_audit_path(data_dir: Path) -> Path:
    return data_dir / COMPANY_DISCOVERY_AUDIT_FILENAME


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
        rendered = entry.model_dump(mode="json")
        rendered["careers_roots"] = _sanitize_company_discovery_careers_roots(
            str(rendered.get("company_name") or "").strip() or None,
            rendered.get("careers_roots") or [],
        )
        entries[str(raw_key)] = CompanyDiscoveryEntry.model_validate(rendered).model_dump(mode="json")
    return entries


def save_company_discovery_entries(data_dir: Path, entries: Mapping[str, Mapping[str, Any]]) -> None:
    rendered = {
        str(key): CompanyDiscoveryEntry.model_validate(
            {
                **dict(value),
                "careers_roots": _sanitize_company_discovery_careers_roots(
                    str((value or {}).get("company_name") or "").strip() or None,
                    (value or {}).get("careers_roots") or [],
                ),
            }
        ).model_dump(mode="json")
        for key, value in entries.items()
    }
    _write_json(company_discovery_index_path(data_dir), rendered)


def _sanitize_company_discovery_frontier_tasks(tasks: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped_tasks: dict[str, dict[str, Any]] = {}
    for raw_task in tasks:
        if not isinstance(raw_task, Mapping):
            continue
        try:
            task = CompanyDiscoveryFrontierTask.model_validate(raw_task)
        except Exception:
            continue
        rendered = task.model_dump(mode="json")
        task_type = str(rendered.get("task_type") or "")
        normalized_url = _normalize_frontier_task_url(task_type, str(rendered.get("url") or ""))
        directory_candidate = (
            _directory_company_candidate(normalized_url, company_name_hint=str(rendered.get("company_name") or "").strip() or None)
            if task_type in {"company_page", "careers_root"}
            else None
        )
        directory_company_key = _directory_company_key(normalized_url)
        discovered_from_company_key = _directory_company_key(str(rendered.get("discovered_from") or "").strip())
        if _company_keys_conflict(discovered_from_company_key, directory_company_key):
            continue
        board_identifier = (
            str(rendered.get("board_identifier") or "").strip()
            or (board_identifier_from_url(normalized_url) if task_type == "board_url" else "")
        )
        rendered["url"] = normalized_url
        rendered["board_identifier"] = board_identifier or None
        rendered["task_key"] = frontier_task_key(
            task_type,
            normalized_url,
            board_identifier=board_identifier or None,
        )
        if task_type in {"company_page", "careers_root"} and not is_company_discovery_seed_url(
            normalized_url,
            preferred_task_type=task_type,
        ):
            continue
        if directory_candidate is not None:
            rendered["company_name"] = directory_candidate["company_name"]
            rendered["company_key"] = _normalize_company_key(directory_candidate["company_name"]) or None
            rendered["source_kind"] = task_type
        task_key = str(rendered.get("task_key") or "")
        existing = deduped_tasks.get(task_key)
        if existing is None:
            deduped_tasks[task_key] = rendered
            continue
        existing["priority"] = max(int(existing.get("priority") or 0), int(rendered.get("priority") or 0))
        existing["source_trust"] = max(int(existing.get("source_trust") or 0), int(rendered.get("source_trust") or 0))
        existing["attempts"] = max(int(existing.get("attempts") or 0), int(rendered.get("attempts") or 0))
        if not existing.get("company_name") and rendered.get("company_name"):
            existing["company_name"] = rendered["company_name"]
        if not existing.get("company_key") and rendered.get("company_key"):
            existing["company_key"] = rendered["company_key"]
        if str(existing.get("status") or "") != "completed" and str(rendered.get("status") or "") == "completed":
            existing["status"] = "completed"
            existing["next_retry_at"] = None
            existing["last_error"] = None
        elif str(existing.get("status") or "") == "completed":
            existing["next_retry_at"] = None
            existing["last_error"] = None
    return list(deduped_tasks.values())


def load_company_discovery_frontier(data_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(company_discovery_frontier_path(data_dir), default=[])
    if not isinstance(payload, list):
        return []
    return _sanitize_company_discovery_frontier_tasks(payload)


def save_company_discovery_frontier(data_dir: Path, tasks: list[Mapping[str, Any]]) -> None:
    rendered = _sanitize_company_discovery_frontier_tasks(tasks)
    _write_json(company_discovery_frontier_path(data_dir), rendered)


def load_company_discovery_crawl_history(data_dir: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(company_discovery_crawl_history_path(data_dir), default={})
    if not isinstance(payload, dict):
        return {}
    records: dict[str, dict[str, Any]] = {}
    for raw_key, raw_entry in payload.items():
        if not isinstance(raw_entry, Mapping):
            continue
        try:
            record = CompanyDiscoveryCrawlRecord.model_validate(raw_entry)
        except Exception:
            continue
        records[str(raw_key)] = record.model_dump(mode="json")
    return records


def save_company_discovery_crawl_history(data_dir: Path, records: Mapping[str, Mapping[str, Any]]) -> None:
    rendered = {
        str(key): CompanyDiscoveryCrawlRecord.model_validate(value).model_dump(mode="json")
        for key, value in records.items()
    }
    _write_json(company_discovery_crawl_history_path(data_dir), rendered)


def load_company_discovery_audit(data_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(company_discovery_audit_path(data_dir), default=[])
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def save_company_discovery_audit(data_dir: Path, entries: list[Mapping[str, Any]]) -> None:
    _write_json(company_discovery_audit_path(data_dir), [dict(item) for item in entries])


def infer_careers_root(url: str | None) -> str | None:
    normalized = str(url or "").strip()
    if not normalized.startswith(("http://", "https://")):
        return None
    canonical_directory_url = _canonical_directory_company_url(normalized, preferred_task_type="careers_root")
    if canonical_directory_url:
        return canonical_directory_url
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
        if "jobs.smartrecruiters.com" in host and segments:
            return f"{parsed.scheme}://{host}/{segments[0]}"
        if "careers.smartrecruiters.com" in host and segments:
            return f"{parsed.scheme}://jobs.smartrecruiters.com/{segments[0]}"
        if ("jobs.recruitee.com" in host or "careers.tellent.com" in host) and segments:
            if segments[0] == "o" and len(segments) >= 2:
                return f"{parsed.scheme}://{host}/o/{segments[1]}"
            return f"{parsed.scheme}://{host}/{segments[0]}"
        if "jobs.workable.com" in host and segments:
            return f"{parsed.scheme}://{host}/{segments[0]}"
        if "jobs.jobvite.com" in host and segments:
            return f"{parsed.scheme}://{host}/{segments[0]}"
        if "jobs.icims.com" in host and len(segments) >= 2:
            return f"{parsed.scheme}://{host}/{segments[0]}/{segments[1]}"
        if "ats.rippling.com" in host and segments:
            return f"{parsed.scheme}://{host}/{segments[0]}"
        return f"{parsed.scheme}://{host}"
    if not is_company_discovery_seed_url(normalized, preferred_task_type="careers_root"):
        return None
    return f"{parsed.scheme}://{host}/careers"


def _normalize_board_root_url(url: str | None) -> str | None:
    normalized = str(url or "").strip().rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        return None
    if not board_identifier_from_url(normalized):
        return normalized
    return (infer_careers_root(normalized) or normalized).rstrip("/")


def _normalize_frontier_task_url(task_type: str, url: str | None) -> str:
    normalized = str(url or "").strip().rstrip("/")
    if task_type == "board_url":
        return _normalize_board_root_url(normalized) or normalized
    canonical_directory_url = _canonical_directory_company_url(normalized, preferred_task_type=task_type)
    if canonical_directory_url:
        return canonical_directory_url
    return normalized


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
    if "careers.smartrecruiters.com" in host and segments:
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


def classify_source_kind(url: str | None, *, explicit_task_type: str | None = None) -> str:
    if explicit_task_type in {"directory_source", "portfolio_source"}:
        return explicit_task_type
    normalized = str(url or "").strip()
    if not normalized.startswith(("http://", "https://")):
        return "other"
    host = (urlparse(normalized).netloc or "").lower()
    if any(fragment in host for fragment in KNOWN_BOARD_HOST_FRAGMENTS):
        return "board_url"
    if any(fragment in host for fragment in LOW_TRUST_SCOUT_HOST_FRAGMENTS):
        return "low_trust_scout"
    if host.endswith("ycombinator.com") or "workatastartup.com" in host or "wellfound.com" in host or "builtin" in host:
        return "directory_source"
    return "company_page"


def trust_score_for_url(url: str | None, *, explicit_task_type: str | None = None) -> int:
    normalized = str(url or "").strip()
    if not normalized.startswith(("http://", "https://")):
        return 0
    host = (urlparse(normalized).netloc or "").lower()
    if any(fragment in host for fragment in KNOWN_BOARD_HOST_FRAGMENTS):
        return 10
    if explicit_task_type == "portfolio_source":
        return 6
    if classify_source_kind(url, explicit_task_type=explicit_task_type) == "directory_source":
        return 5
    if any(fragment in host for fragment in LOW_TRUST_SCOUT_HOST_FRAGMENTS):
        return 2
    return 7


def extract_embedded_board_urls(page_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    candidates: list[str] = []
    attribute_markers = ("href", "src", "url", "link", "action")

    def maybe_add(raw_url: str | None) -> None:
        normalized = str(raw_url or "").strip()
        if not normalized:
            return
        resolved = urljoin(page_url, normalized)
        if not resolved.startswith(("http://", "https://")):
            return
        host = (urlparse(resolved).netloc or "").lower()
        if any(fragment in host for fragment in KNOWN_BOARD_HOST_FRAGMENTS):
            candidates.append(resolved)

    for tag in soup.find_all(True):
        for attribute_name, attribute_value in tag.attrs.items():
            lowered_name = str(attribute_name or "").strip().lower()
            if not lowered_name or not any(marker in lowered_name for marker in attribute_markers):
                continue
            raw_values = attribute_value if isinstance(attribute_value, list) else [attribute_value]
            for raw_value in raw_values:
                maybe_add(str(raw_value or ""))

    raw_html = html or ""
    greenhouse_embed_match = re.finditer(r"boards\.greenhouse\.io/embed/job_board/js\?for=([a-z0-9_-]+)", raw_html, re.I)
    for match in greenhouse_embed_match:
        token = match.group(1).strip()
        if token:
            candidates.append(f"https://boards.greenhouse.io/{token}")

    decoded_html = raw_html.replace("\\/", "/")
    for raw_url in re.findall(r"https?://[^\s\"'<>]+", decoded_html):
        maybe_add(raw_url)

    if page_url.startswith(("http://", "https://")):
        maybe_add(page_url)

    return _dedupe_strings(candidates, limit=16)


def extract_careers_page_urls(page_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    candidates: list[str] = []
    parsed_page = urlparse(page_url) if page_url.startswith(("http://", "https://")) else None
    page_host = (parsed_page.netloc or "").lower() if parsed_page else ""
    page_uses_directory_routing = _is_blocked_external_company_homepage_host(page_host)
    page_company_key = _directory_company_key(page_url)

    def maybe_add(raw_url: str | None) -> None:
        normalized = str(raw_url or "").strip()
        if not normalized:
            return
        resolved = urljoin(page_url, normalized)
        if not resolved.startswith(("http://", "https://")):
            return
        parsed = urlparse(resolved)
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").lower()
        if page_host and host != page_host:
            return
        if page_uses_directory_routing:
            candidate = _directory_company_candidate(resolved)
            if candidate is None:
                return
            candidate_company_key = _directory_company_key(resolved)
            if _company_keys_conflict(page_company_key, candidate_company_key):
                return
            candidate_url = str(candidate["url"]).rstrip("/")
            if candidate["task_type"] == "company_page":
                candidate_url = f"{candidate_url}/jobs"
            candidates.append(candidate_url.rstrip("/"))
            return
        if any(path.startswith(candidate) or candidate in path for candidate in COMMON_CAREERS_PATHS):
            candidates.append(resolved.rstrip("/"))

    for tag in soup.find_all("a"):
        maybe_add(tag.get("href"))
    return _dedupe_strings(candidates, limit=12)


def extract_company_homepage_urls(page_url: str, html: str) -> list[str]:
    soup = BeautifulSoup(html or "", "html.parser")
    candidates: list[str] = []
    page_host = (urlparse(page_url).netloc or "").lower() if page_url.startswith(("http://", "https://")) else ""
    for tag in soup.find_all("a"):
        href = str(tag.get("href") or "").strip()
        if not href:
            continue
        resolved = urljoin(page_url, href)
        if not resolved.startswith(("http://", "https://")):
            continue
        parsed = urlparse(resolved)
        host = (parsed.netloc or "").lower()
        if not host or host == page_host:
            continue
        if any(fragment in host for fragment in KNOWN_BOARD_HOST_FRAGMENTS):
            continue
        if any(fragment in host for fragment in LOW_TRUST_SCOUT_HOST_FRAGMENTS):
            continue
        if _is_blocked_external_company_homepage_host(host):
            continue
        candidates.append(f"{parsed.scheme}://{parsed.netloc}")
    return _dedupe_strings(candidates, limit=16)


def extract_directory_company_tasks(page_url: str, html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html or "", "html.parser")
    candidates_by_url: dict[str, dict[str, str]] = {}

    for tag in soup.find_all("a"):
        href = str(tag.get("href") or "").strip()
        if not href:
            continue
        resolved = urljoin(page_url, href)
        candidate = _directory_company_candidate(
            resolved,
            company_name_hint=tag.get_text(" ", strip=True),
        )
        if candidate is None:
            continue
        existing = candidates_by_url.get(candidate["url"])
        if existing is None or len(candidate["company_name"]) > len(existing["company_name"]):
            candidates_by_url[candidate["url"]] = candidate

    ordered = sorted(candidates_by_url.values(), key=lambda item: (item["task_type"] != "careers_root", item["url"]))
    return ordered[:24]


def default_careers_candidate_urls(source_url: str | None) -> list[str]:
    normalized = str(source_url or "").strip()
    if not normalized.startswith(("http://", "https://")):
        return []
    directory_candidate = _directory_company_candidate(normalized)
    if directory_candidate is not None:
        directory_url = str(directory_candidate["url"]).rstrip("/")
        if directory_candidate["task_type"] == "careers_root":
            return [directory_url]
        return [f"{directory_url}/jobs"]
    parsed = urlparse(normalized)
    if _is_blocked_external_company_homepage_host((parsed.netloc or "").lower()):
        return []
    base = f"{parsed.scheme}://{parsed.netloc}"
    candidates = [f"{base}{path}" for path in COMMON_CAREERS_PATHS]
    return _dedupe_strings(candidates, limit=8)


def _sanitize_company_discovery_careers_roots(
    company_name: str | None,
    careers_roots: list[str] | tuple[str, ...],
) -> list[str]:
    expected_company_key = _normalize_company_key(company_name)
    normalized_roots: list[str] = []
    for raw_url in careers_roots:
        normalized_url = str(raw_url or "").strip()
        if not normalized_url:
            continue
        if not is_company_discovery_seed_url(normalized_url, preferred_task_type="careers_root"):
            continue
        canonical_directory_url = _canonical_directory_company_url(
            normalized_url,
            preferred_task_type="careers_root",
        )
        if canonical_directory_url:
            candidate_company_key = _directory_company_key(canonical_directory_url)
            if _company_keys_conflict(expected_company_key, candidate_company_key):
                continue
            normalized_roots.append(canonical_directory_url)
            continue
        normalized_roots.append(normalized_url.rstrip("/"))
    return _dedupe_strings(normalized_roots, limit=12)


def frontier_task_key(
    task_type: str,
    url: str,
    *,
    board_identifier: str | None = None,
) -> str:
    normalized_url = str(url).strip().rstrip("/")
    suffix = f":{board_identifier}" if board_identifier else ""
    return f"{task_type}:{normalized_url}{suffix}"


def make_frontier_task(
    *,
    task_type: str,
    url: str,
    company_name: str | None = None,
    company_key: str | None = None,
    board_identifier: str | None = None,
    source_kind: str | None = None,
    source_trust: int | None = None,
    priority: int = 0,
    discovered_from: str | None = None,
    attempts: int = 0,
    status: str = "pending",
    next_retry_at: str | None = None,
) -> dict[str, Any]:
    normalized_url = _normalize_frontier_task_url(task_type, url)
    normalized_company_key = company_key or _normalize_company_key(company_name)
    effective_board_identifier = str(
        board_identifier or (board_identifier_from_url(normalized_url) if task_type == "board_url" else "")
    ).strip()
    return CompanyDiscoveryFrontierTask(
        task_key=frontier_task_key(task_type, normalized_url, board_identifier=effective_board_identifier or None),
        task_type=task_type,
        url=normalized_url,
        company_name=company_name,
        company_key=normalized_company_key or None,
        board_identifier=effective_board_identifier or None,
        source_kind=source_kind or classify_source_kind(normalized_url, explicit_task_type=task_type),
        source_trust=int(source_trust if source_trust is not None else trust_score_for_url(normalized_url, explicit_task_type=task_type)),
        priority=int(priority),
        attempts=int(attempts),
        status=status,
        discovered_from=discovered_from,
        next_retry_at=next_retry_at,
    ).model_dump(mode="json")


def upsert_frontier_task(
    tasks: list[dict[str, Any]],
    *,
    task_type: str,
    url: str,
    company_name: str | None = None,
    company_key: str | None = None,
    board_identifier: str | None = None,
    source_kind: str | None = None,
    source_trust: int | None = None,
    priority: int = 0,
    discovered_from: str | None = None,
    reactivate_completed: bool = False,
) -> bool:
    task = make_frontier_task(
        task_type=task_type,
        url=url,
        company_name=company_name,
        company_key=company_key,
        board_identifier=board_identifier,
        source_kind=source_kind,
        source_trust=source_trust,
        priority=priority,
        discovered_from=discovered_from,
    )
    task_key = task["task_key"]
    for existing in tasks:
        if str(existing.get("task_key") or "") != task_key:
            continue
        existing["priority"] = max(int(existing.get("priority") or 0), int(task["priority"] or 0))
        existing["source_trust"] = max(int(existing.get("source_trust") or 0), int(task["source_trust"] or 0))
        if not existing.get("company_name") and task.get("company_name"):
            existing["company_name"] = task["company_name"]
        if not existing.get("company_key") and task.get("company_key"):
            existing["company_key"] = task["company_key"]
        if existing.get("status") == "completed" and reactivate_completed:
            existing["status"] = "pending"
        return False
    tasks.append(task)
    return True


def select_frontier_tasks(
    tasks: list[dict[str, Any]],
    *,
    budget: int,
    task_types: set[str] | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    if budget <= 0:
        return []
    moment = now or datetime.now(UTC)

    def is_ready(task: dict[str, Any]) -> bool:
        if task_types and str(task.get("task_type") or "") not in task_types:
            return False
        if str(task.get("status") or "pending") != "pending":
            return False
        next_retry = _parse_datetime(str(task.get("next_retry_at") or ""))
        return next_retry is None or next_retry <= moment

    candidates = [task for task in tasks if is_ready(task)]
    candidates.sort(
        key=lambda task: (
            -int(task.get("priority") or 0),
            -int(task.get("source_trust") or 0),
            int(task.get("attempts") or 0),
            str(task.get("url") or ""),
        )
    )

    def selection_group_key(task: dict[str, Any]) -> str:
        company_key = str(task.get("company_key") or "").strip()
        if company_key:
            return f"company:{company_key}"
        board_identifier = str(task.get("board_identifier") or "").strip()
        if board_identifier:
            return f"board:{board_identifier}"
        host = (urlparse(str(task.get("url") or "")).netloc or "").lower()
        if host:
            return f"host:{host}"
        return f"task:{str(task.get('task_key') or '')}"

    grouped_candidates: dict[str, list[dict[str, Any]]] = {}
    ordered_groups: list[str] = []
    for task in candidates:
        group_key = selection_group_key(task)
        if group_key not in grouped_candidates:
            grouped_candidates[group_key] = []
            ordered_groups.append(group_key)
        grouped_candidates[group_key].append(task)

    selected: list[dict[str, Any]] = []
    active_groups = list(ordered_groups)
    while active_groups and len(selected) < budget:
        next_round_groups: list[str] = []
        for group_key in active_groups:
            queue = grouped_candidates.get(group_key) or []
            if not queue:
                continue
            selected.append(dict(queue.pop(0)))
            if queue and len(selected) < budget:
                next_round_groups.append(group_key)
            if len(selected) >= budget:
                break
        active_groups = next_round_groups
    return selected


def update_frontier_task_state(
    tasks: list[dict[str, Any]],
    *,
    task_key: str,
    success: bool,
    error: str | None = None,
    keep_pending: bool = False,
    retry_delay: timedelta = DEFAULT_FRONTIER_RETRY_DELAY,
) -> None:
    for task in tasks:
        if str(task.get("task_key") or "") != task_key:
            continue
        task["attempts"] = int(task.get("attempts") or 0) + 1
        task["last_attempted_at"] = _utc_now_iso()
        if success and not keep_pending:
            task["status"] = "completed"
            task["next_retry_at"] = None
            task["last_error"] = None
        elif success and keep_pending:
            task["status"] = "pending"
            task["next_retry_at"] = None
            task["last_error"] = None
        else:
            task["status"] = "pending"
            task["next_retry_at"] = (datetime.now(UTC) + retry_delay).isoformat(timespec="seconds")
            task["last_error"] = error
        return


def crawl_record_key(target_type: str, url: str, board_identifier: str | None = None) -> str:
    normalized_url = str(url or "").strip().rstrip("/")
    suffix = f":{board_identifier}" if board_identifier else ""
    return f"{target_type}:{normalized_url}{suffix}"


def record_crawl_result(
    records: dict[str, dict[str, Any]],
    *,
    target_type: str,
    url: str,
    company_key: str | None = None,
    board_identifier: str | None = None,
    success: bool,
    http_status: int | None = None,
    error: str | None = None,
    fresh_role_count: int = 0,
) -> None:
    key = crawl_record_key(target_type, url, board_identifier)
    existing = CompanyDiscoveryCrawlRecord.model_validate(
        {
            "record_key": key,
            "target_type": target_type,
            "url": url,
            "company_key": company_key,
            "board_identifier": board_identifier,
            **(records.get(key) or {}),
        }
    )
    updated = existing.model_copy(
        update={
            "company_key": company_key or existing.company_key,
            "board_identifier": board_identifier or existing.board_identifier,
            "attempt_count": existing.attempt_count + 1,
            "success_count": existing.success_count + (1 if success else 0),
            "failure_count": existing.failure_count + (0 if success else 1),
            "last_status": "success" if success else "failure",
            "last_http_status": http_status,
            "last_attempted_at": _utc_now_iso(),
            "last_succeeded_at": _utc_now_iso() if success else existing.last_succeeded_at,
            "last_error": None if success else error,
            "last_fresh_role_count": int(fresh_role_count),
        }
    )
    records[key] = updated.model_dump(mode="json")


def append_company_discovery_audit_entry(
    entries: list[dict[str, Any]],
    payload: Mapping[str, Any],
    *,
    limit: int = 500,
) -> None:
    rendered = dict(payload)
    rendered.setdefault("recorded_at", _utc_now_iso())
    entries.append(rendered)
    if len(entries) > limit:
        del entries[:-limit]


def source_directory_seed_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for url in DIRECTORY_SOURCE_URLS:
        tasks.append(
            make_frontier_task(
                task_type="directory_source",
                url=url,
                source_kind="directory_source",
                source_trust=trust_score_for_url(url, explicit_task_type="directory_source"),
                priority=3,
            )
        )
    return tasks


def is_low_value_company_discovery_entry(entry: Mapping[str, Any]) -> bool:
    if int(entry.get("official_board_lead_count") or 0) > 0:
        return False
    if int(entry.get("ai_pm_candidate_count") or 0) > 0:
        return False
    if int(entry.get("recent_fresh_role_count") or 0) > 0:
        return False
    if [item for item in entry.get("board_identifiers") or [] if str(item).strip()]:
        return False
    if [item for item in entry.get("board_urls") or [] if str(item).strip()]:
        return False
    source_hosts = [str(item).strip().lower() for item in entry.get("source_hosts") or [] if str(item).strip()]
    careers_root_hosts = [
        (urlparse(str(item).strip()).netloc or "").lower()
        for item in entry.get("careers_roots") or []
        if str(item).strip().startswith(("http://", "https://"))
    ]
    recursive_hosts = [host for host in [*source_hosts, *careers_root_hosts] if host]
    if not recursive_hosts:
        return False
    return all(_is_recursive_platform_host(host) for host in recursive_hosts)


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
    source_kind: str | None = None,
    board_crawl_succeeded: bool | None = None,
    fresh_role_delta: int = 0,
) -> tuple[bool, int]:
    company_key = _normalize_company_key(company_name)
    if not company_key:
        return False, 0
    timestamp = generated_at or _utc_now_iso()
    was_new_company = company_key not in entries
    entry = dict(entries.get(company_key) or {})
    existing_board_identifiers = [str(item) for item in entry.get("board_identifiers", []) if str(item).strip()]
    new_board_identifiers = list(board_identifiers or [])
    normalized_board_urls = [normalized for url in board_urls or [] if (normalized := _normalize_board_root_url(url))]
    for board_url in normalized_board_urls:
        identifier = board_identifier_from_url(board_url)
        if identifier:
            new_board_identifiers.append(identifier)
    deduped_board_identifiers = _dedupe_strings([*existing_board_identifiers, *new_board_identifiers], limit=24)
    new_board_count = len(set(deduped_board_identifiers) - set(existing_board_identifiers))
    existing = CompanyDiscoveryEntry.model_validate(
        {
            "company_key": company_key,
            "company_name": company_name,
            "first_seen_at": timestamp,
            **entry,
        }
    )
    source_key = source_kind or classify_source_kind(source_url)
    source_type_counts = dict(existing.source_type_counts)
    if source_key:
        source_type_counts[source_key] = int(source_type_counts.get(source_key) or 0) + 1
    updated = existing.model_copy(
        update={
            "company_name": company_name,
            "careers_roots": _sanitize_company_discovery_careers_roots(
                company_name,
                [*existing.careers_roots, *([careers_root] if careers_root else [])],
            ),
            "ats_types": _dedupe_strings(
                [*existing.ats_types, *(ats_types or []), *[value for value in (board_url_ats_type(url) for url in normalized_board_urls) if value]],
                limit=12,
            ),
            "board_identifiers": deduped_board_identifiers,
            "board_urls": _dedupe_strings([*existing.board_urls, *normalized_board_urls], limit=24),
            "source_hosts": _dedupe_strings(
                [
                    *existing.source_hosts,
                    *(
                        [(urlparse(source_url).netloc or "").lower()]
                        if source_url and source_url.startswith(("http://", "https://"))
                        else []
                    ),
                ],
                limit=12,
            ),
            "source_trust": max(existing.source_trust, int(source_trust or 0)),
            "last_seen_at": timestamp,
            "last_successful_discovery_run": run_id or existing.last_successful_discovery_run,
            "ai_pm_candidate_count": max(0, existing.ai_pm_candidate_count + int(ai_pm_candidate_delta or 0)),
            "official_board_lead_count": max(0, existing.official_board_lead_count + int(official_board_lead_delta or 0)),
            "source_type_counts": source_type_counts,
            "board_crawl_success_count": existing.board_crawl_success_count + (1 if board_crawl_succeeded is True else 0),
            "board_crawl_failure_count": existing.board_crawl_failure_count + (1 if board_crawl_succeeded is False else 0),
            "recent_fresh_role_count": max(0, existing.recent_fresh_role_count + int(fresh_role_delta or 0)),
            "last_crawl_status": (
                "success" if board_crawl_succeeded is True else "failure" if board_crawl_succeeded is False else existing.last_crawl_status
            ),
            "last_attempted_at": timestamp if board_crawl_succeeded is not None else existing.last_attempted_at,
            "next_retry_at": None if board_crawl_succeeded is True else existing.next_retry_at,
        }
    )
    entries[company_key] = updated.model_dump(mode="json")
    return was_new_company, new_board_count
