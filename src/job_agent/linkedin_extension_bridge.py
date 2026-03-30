from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import uuid
from urllib.parse import quote
import webbrowser

from .config import Settings, load_settings
from .firefox_extension_host import (
    enqueue_open_url,
    inspect_configured_firefox_extension_profile,
    open_url_in_firefox_profile,
    read_state as read_firefox_extension_host_state,
)


def _normalize_name(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def _normalize_profile_url(url: str | None) -> str | None:
    if not url:
        return None
    return str(url).split("?", 1)[0]


def _role_keywords_for_extension_search(role_title: str | None) -> str:
    if not role_title:
        return "product manager ai"
    stopwords = {
        "remote",
        "usa",
        "united",
        "states",
        "senior",
        "staff",
        "principal",
        "group",
        "lead",
        "sr",
    }
    tokens = []
    lowered_title = role_title.lower()
    if "product manager" in lowered_title:
        tokens.extend(["product", "manager"])
    if any(token in lowered_title for token in ("ai", "ml", "machine learning", "llm", "agent", "genai")):
        tokens.append("ai")
    for token in re.findall(r"[a-z0-9]+", lowered_title):
        if len(token) < 3 or token in stopwords:
            continue
        tokens.append(token)
    deduped = list(dict.fromkeys(tokens))
    return " ".join(deduped[:5]) or "product manager ai"


def _linkedin_people_search_url(
    company_name: str,
    keywords: str,
    network_flag: str,
    session_id: str,
    degree: str,
) -> str:
    return (
        "https://www.linkedin.com/search/results/people/"
        f"?keywords={quote(keywords)}"
        f"&network=%5B%22{network_flag}%22%5D"
        "&origin=GLOBAL_SEARCH_HEADER"
        f"&job_agent_session={quote(session_id)}"
        f"&job_agent_company={quote(company_name)}"
        f"&job_agent_degree={quote(degree)}"
    )


def build_extension_capture_urls(company_name: str, session_id: str, role_title: str | None = None) -> dict[str, list[str]]:
    query_variants = [
        f"\"{company_name}\"",
        f"\"{company_name}\" recruiter hiring talent acquisition",
        f"\"{company_name}\" {_role_keywords_for_extension_search(role_title)}",
    ]
    deduped_queries = list(dict.fromkeys(" ".join(query.split()) for query in query_variants if query.strip()))
    return {
        "1st": [_linkedin_people_search_url(company_name, query, "F", session_id, "1st") for query in deduped_queries],
        "2nd": [_linkedin_people_search_url(company_name, query, "S", session_id, "2nd") for query in deduped_queries],
    }


@dataclass(slots=True)
class ExtensionCaptureSession:
    session_id: str
    company_name: str
    role_title: str | None = None
    search_urls: dict[str, list[str]] = field(default_factory=dict)
    first_order_contacts: list[dict[str, object]] = field(default_factory=list)
    second_order_contacts: list[dict[str, object]] = field(default_factory=list)
    message_histories_by_name: dict[str, list[str]] = field(default_factory=dict)
    message_histories_by_profile_url: dict[str, list[str]] = field(default_factory=dict)
    received_degrees: set[str] = field(default_factory=set)
    login_required: bool = False
    search_results_event: asyncio.Event = field(default_factory=asyncio.Event)
    history_update_event: asyncio.Event = field(default_factory=asyncio.Event)


class LinkedInExtensionBridge:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.host = settings.linkedin_extension_bridge_host
        self.port = settings.linkedin_extension_bridge_port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, ExtensionCaptureSession] = {}

    async def __aenter__(self) -> LinkedInExtensionBridge:
        self._loop = asyncio.get_running_loop()
        handler = self._build_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, name="linkedin-extension-bridge", daemon=True)
        self._thread.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def create_session(self, company_name: str, *, role_title: str | None = None) -> ExtensionCaptureSession:
        session_id = uuid.uuid4().hex
        session = ExtensionCaptureSession(
            session_id=session_id,
            company_name=company_name,
            role_title=role_title,
            search_urls=build_extension_capture_urls(company_name, session_id, role_title),
        )
        self._sessions[session_id] = session
        return session

    async def wait_for_search_results(self, session: ExtensionCaptureSession) -> None:
        total_budget = max(1, self.settings.linkedin_extension_capture_timeout_seconds)
        end_time = asyncio.get_running_loop().time() + total_budget
        partial_result_idle_window = 5.0
        partial_deadline: float | None = None
        while True:
            if session.login_required:
                return
            if {"1st", "2nd"}.issubset(session.received_degrees):
                return
            if session.received_degrees and partial_deadline is None:
                partial_deadline = asyncio.get_running_loop().time() + partial_result_idle_window
            remaining = end_time - asyncio.get_running_loop().time()
            if partial_deadline is not None:
                remaining = min(remaining, partial_deadline - asyncio.get_running_loop().time())
            if remaining <= 0:
                if session.received_degrees:
                    return
                raise TimeoutError
            session.search_results_event.clear()
            try:
                await asyncio.wait_for(session.search_results_event.wait(), timeout=remaining)
            except TimeoutError:
                if session.received_degrees:
                    return
                raise

    async def wait_for_history_settle(self, session: ExtensionCaptureSession) -> None:
        total_budget = max(0, self.settings.linkedin_extension_history_timeout_seconds)
        if total_budget == 0:
            return
        end_time = asyncio.get_running_loop().time() + total_budget
        idle_window = 4.0
        while True:
            remaining = end_time - asyncio.get_running_loop().time()
            if remaining <= 0:
                return
            session.history_update_event.clear()
            try:
                await asyncio.wait_for(session.history_update_event.wait(), timeout=min(idle_window, remaining))
            except TimeoutError:
                return

    async def open_search_tabs(self, session: ExtensionCaptureSession, *, prime_linkedin_feed: bool = True) -> None:
        if not self.settings.linkedin_extension_auto_open_search_tabs:
            return
        configured_profile = inspect_configured_firefox_extension_profile(self.settings)
        if configured_profile.get("configured"):
            if not configured_profile.get("exists"):
                raise RuntimeError(
                    "Configured Firefox extension profile was not found. "
                    "Set FIREFOX_EXTENSION_PROFILE_DIR to the real Firefox profile you use for LinkedIn."
                )
            if not configured_profile.get("extension_installed"):
                raise RuntimeError(
                    "Configured Firefox profile does not currently have the Job Agent LinkedIn Bridge add-on loaded. "
                    "Open `about:debugging#/runtime/this-firefox` in that profile and load "
                    "`firefox_extension/manifest.json` as a temporary add-on."
                )
            if not configured_profile.get("linkedin_authenticated"):
                raise RuntimeError(
                    "Configured Firefox profile is not authenticated to LinkedIn. "
                    "Sign into LinkedIn in that profile before running the workflow."
                )
            if prime_linkedin_feed:
                await asyncio.to_thread(
                    _open_url_in_firefox,
                    "https://www.linkedin.com/feed/",
                    self.settings.project_root,
                )
                await asyncio.sleep(1.0)
        for degree in ("1st", "2nd"):
            for url in session.search_urls.get(degree, []):
                await asyncio.to_thread(_open_url_in_firefox, url, self.settings.project_root)
                await asyncio.sleep(0.5)

    async def retry_session_after_login_redirect(
        self,
        company_name: str,
        *,
        role_title: str | None = None,
    ) -> ExtensionCaptureSession:
        retry_session = await self.create_session(company_name, role_title=role_title)
        await self.open_search_tabs(retry_session, prime_linkedin_feed=True)
        return retry_session

    def _build_handler(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(HTTPStatus.NO_CONTENT)
                self._send_common_headers()
                self.end_headers()

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/health":
                    self.send_response(HTTPStatus.OK)
                    self._send_common_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": True}).encode("utf-8"))
                    return
                if self.path == "/api/linkedin-extension/live-outreach":
                    self.send_response(HTTPStatus.OK)
                    self._send_common_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps(bridge._read_live_outreach_payload()).encode("utf-8"))
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers.get("Content-Length", "0") or "0")
                raw_body = self.rfile.read(content_length)
                try:
                    payload = json.loads(raw_body.decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON payload")
                    return

                if self.path == "/api/linkedin-extension/search-results":
                    bridge._handle_search_results_from_http(payload)
                elif self.path == "/api/linkedin-extension/message-histories":
                    bridge._handle_message_histories_from_http(payload)
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return

                self.send_response(HTTPStatus.ACCEPTED)
                self._send_common_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"accepted": True}).encode("utf-8"))

            def log_message(self, format: str, *args) -> None:  # noqa: A003
                return

            def _send_common_headers(self) -> None:
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        return Handler

    def _handle_search_results_from_http(self, payload: dict[str, object]) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._apply_search_results, payload)

    def _handle_message_histories_from_http(self, payload: dict[str, object]) -> None:
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._apply_message_histories, payload)

    def _apply_search_results(self, payload: dict[str, object]) -> None:
        session_id = str(payload.get("session_id") or payload.get("sessionId") or "").strip()
        degree = str(payload.get("degree") or "").strip()
        contacts = payload.get("contacts")
        login_required = bool(payload.get("login_required") or payload.get("loginRequired"))
        session = self._sessions.get(session_id)
        if session is None or degree not in {"1st", "2nd"}:
            return
        if login_required:
            session.login_required = True
            session.search_results_event.set()
            return
        if not isinstance(contacts, list):
            return

        normalized_contacts = [contact for contact in contacts if isinstance(contact, dict)]
        if degree == "1st":
            session.first_order_contacts.extend(normalized_contacts)
        else:
            session.second_order_contacts.extend(normalized_contacts)
        session.received_degrees.add(degree)
        session.search_results_event.set()

    def _apply_message_histories(self, payload: dict[str, object]) -> None:
        session_id = str(payload.get("session_id") or payload.get("sessionId") or "").strip()
        entries = payload.get("histories")
        session = self._sessions.get(session_id)
        if session is None or not isinstance(entries, list):
            return

        updated = False
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            profile_url = _normalize_profile_url(entry.get("profile_url") or entry.get("profileUrl"))
            messages = [str(message).strip() for message in entry.get("messages", []) if str(message).strip()]
            if not messages:
                continue
            if profile_url:
                session.message_histories_by_profile_url[profile_url] = messages
                updated = True
            if name:
                session.message_histories_by_name[_normalize_name(name)] = messages
                updated = True
        if updated:
            session.history_update_event.set()

    def _read_live_outreach_payload(self) -> dict[str, object]:
        path = self.settings.data_dir / "live-outreach.json"
        if not path.exists():
            return {"generated_at": None, "items": []}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"generated_at": None, "items": []}
        return payload if isinstance(payload, dict) else {"generated_at": None, "items": []}


def _open_url_in_firefox(url: str, project_root: Path) -> None:
    settings = load_settings(project_root, require_openai=False)
    if settings.firefox_extension_profile_dir is not None:
        if open_url_in_firefox_profile(settings.firefox_extension_profile_dir, url):
            return

    firefox_binary = shutil.which("firefox")
    firefox_esr_binary = shutil.which("firefox-esr")
    flatpak_binary = shutil.which("flatpak")

    commands: list[list[str]] = []
    host_state = read_firefox_extension_host_state(project_root)
    host_profile = None
    if host_state is not None:
        try:
            host_pid = int(host_state.get("pid") or 0)
        except (TypeError, ValueError):
            host_pid = 0
        if host_pid > 0:
            try:
                os.kill(host_pid, 0)
            except OSError:
                host_pid = 0
        if host_pid > 0:
            enqueue_open_url(project_root, url)
            return
        profile_dir = host_state.get("profile_dir")
        if profile_dir:
            host_profile = str(profile_dir)
    if host_profile and firefox_binary:
        commands.append([firefox_binary, "--profile", host_profile, "--new-tab", url])
    if host_profile and firefox_esr_binary:
        commands.append([firefox_esr_binary, "--profile", host_profile, "--new-tab", url])
    if firefox_binary:
        commands.append([firefox_binary, "--new-tab", url])
    if firefox_esr_binary:
        commands.append([firefox_esr_binary, "--new-tab", url])
    if flatpak_binary:
        commands.append([flatpak_binary, "run", "org.mozilla.firefox", "--new-tab", url])

    for command in commands:
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            continue

    webbrowser.open_new_tab(url)
