from __future__ import annotations

import configparser
import contextlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
import asyncio
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time
from urllib.parse import quote, urlparse, urlunparse

import httpx
from playwright.async_api import BrowserContext, Page, TimeoutError as PlaywrightTimeoutError, async_playwright
import pyotp

from .config import Settings
from .linkedin_extension_bridge import ExtensionCaptureSession, LinkedInExtensionBridge
from .models import LinkedInContact, ManualReviewLink


@dataclass(slots=True)
class LinkedInDiscovery:
    first_order_contacts: list[LinkedInContact]
    second_order_contacts: list[LinkedInContact]


@dataclass(slots=True)
class ParsedSearchResult:
    name: str
    profile_url: str
    raw_text: str
    headline: str | None
    company_text: str | None
    mutual_connection_names: list[str]
    mutual_connection_profile_urls: dict[str, str]


class LinkedInClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @asynccontextmanager
    async def context(self, *, headless: bool | None = None) -> BrowserContext:
        async with async_playwright() as playwright:
            launch_kwargs = {
                "user_data_dir": str(self.settings.linkedin_profile_dir),
                "headless": self.settings.headless if headless is None else headless,
                "viewport": {"width": 1400, "height": 1100},
                "locale": "en-US",
                "timezone_id": self.settings.timezone,
            }
            preferred_browser = get_preferred_browser(self.settings)
            if preferred_browser.channel:
                launch_kwargs["channel"] = preferred_browser.channel
            if preferred_browser.executable_path:
                launch_kwargs["executable_path"] = preferred_browser.executable_path

            context = await playwright.chromium.launch_persistent_context(
                **launch_kwargs,
            )
            try:
                await self._apply_auth_cookies(context)
                yield context
            finally:
                await context.storage_state(path=str(self.settings.linkedin_storage_state))
                await context.close()

    async def bootstrap_session(self) -> None:
        async with self.context(headless=False) as context:
            page = await context.new_page()
            await page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
            await self._attempt_login(page)
            await page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded")
            if "feed" not in page.url:
                print("Complete any LinkedIn verification, MFA, or captcha steps in the browser, then press Enter here.")
                await asyncio.to_thread(input)
            await context.storage_state(path=str(self.settings.linkedin_storage_state))

    async def bootstrap_google_session(self) -> None:
        async with self.context(headless=False) as context:
            page = await context.new_page()
            await page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
            print(
                f"In the opened browser ({describe_browser_choice(self.settings)}), click 'Continue with Google' "
                "and finish the sign-in yourself. "
                "When you can see the LinkedIn feed, return here and press Enter."
            )
            await asyncio.to_thread(input)
            await page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded")
            if "feed" not in page.url:
                raise RuntimeError(
                    "LinkedIn Google sign-in was not fully completed. Run the command again and finish the login flow."
                )
            await context.storage_state(path=str(self.settings.linkedin_storage_state))

    async def discover_company_contacts(
        self,
        company_name: str,
        role_title: str | None = None,
        *,
        context: BrowserContext | None = None,
        extension_bridge: LinkedInExtensionBridge | None = None,
    ) -> LinkedInDiscovery:
        if self.settings.linkedin_capture_mode == "firefox_extension":
            if extension_bridge is None:
                raise RuntimeError("Firefox extension capture mode requires an active LinkedInExtensionBridge.")
            return await self._discover_company_contacts_via_extension(
                extension_bridge,
                company_name,
                role_title=role_title,
            )
        if context is None:
            async with self.context() as owned_context:
                return await self._discover_company_contacts_in_context(
                    owned_context,
                    company_name,
                    role_title=role_title,
                )
        return await self._discover_company_contacts_in_context(
            context,
            company_name,
            role_title=role_title,
        )

    async def _discover_company_contacts_via_extension(
        self,
        extension_bridge: LinkedInExtensionBridge,
        company_name: str,
        *,
        role_title: str | None = None,
    ) -> LinkedInDiscovery:
        session = await extension_bridge.create_session(company_name, role_title=role_title)
        await extension_bridge.open_search_tabs(session)
        try:
            await extension_bridge.wait_for_search_results(session)
        except TimeoutError as exc:
            raise RuntimeError(
                "No Firefox extension capture was received before timeout. "
                "Make sure Firefox is open, the LinkedIn extension is loaded, and the capture tabs were allowed to load."
            ) from exc
        if session.login_required:
            retry_session = await extension_bridge.retry_session_after_login_redirect(
                company_name,
                role_title=role_title,
            )
            try:
                await extension_bridge.wait_for_search_results(retry_session)
            except TimeoutError as exc:
                raise RuntimeError(
                    "Firefox extension capture retried after a LinkedIn login redirect, "
                    "but the retried capture still timed out before results arrived."
                ) from exc
            if retry_session.login_required:
                raise RuntimeError(
                    "Firefox extension capture was redirected to LinkedIn login before results loaded, "
                    "and the retry against the configured Firefox profile still landed on LinkedIn login. "
                    "Re-open LinkedIn in that Firefox profile and keep the temporary add-on window alive."
                )
            session = retry_session
        if session.login_required:
            raise RuntimeError(
                "Firefox extension capture was redirected to LinkedIn login before results loaded. "
                "The Firefox browser/profile handling capture is not authenticated to LinkedIn."
            )
        await extension_bridge.wait_for_history_settle(session)
        return self._build_discovery_from_extension_session(company_name, session)

    async def _discover_company_contacts_in_context(
        self,
        context: BrowserContext,
        company_name: str,
        *,
        role_title: str | None = None,
    ) -> LinkedInDiscovery:
        page = await context.new_page()
        try:
            try:
                await page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded")
            except Exception:
                await page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
            await self._ensure_logged_in(page)

            first_order = await self._search_people(page, company_name, "1st", role_title=role_title)
            second_order = await self._search_people(page, company_name, "2nd", role_title=role_title)
            connector_histories = await self._fetch_second_order_connector_histories(context, second_order)
            first_order_by_key = {_normalize_person_name(contact.name): contact for contact in first_order}
            for contact in second_order:
                ordered_connector_names = list(
                    dict.fromkeys(contact.connected_first_order_names + list(contact.connected_first_order_profile_urls))
                )
                enriched_profile_urls = dict(contact.connected_first_order_profile_urls)
                enriched_histories = dict(contact.connected_first_order_message_histories)
                for name in ordered_connector_names:
                    normalized = _normalize_person_name(name)
                    if normalized in first_order_by_key:
                        first_order_contact = first_order_by_key[normalized]
                        enriched_profile_urls.setdefault(first_order_contact.name, str(first_order_contact.profile_url))
                        if first_order_contact.message_history:
                            enriched_histories.setdefault(first_order_contact.name, first_order_contact.message_history)
                    history = connector_histories.get(normalized)
                    if history:
                        canonical_name = next(
                            (
                                existing_name
                                for existing_name in enriched_profile_urls
                                if _normalize_person_name(existing_name) == normalized
                            ),
                            name,
                        )
                        enriched_histories.setdefault(canonical_name, history)
                contact.connected_first_order_names = ordered_connector_names
                contact.connected_first_order_profile_urls = enriched_profile_urls
                contact.connected_first_order_message_histories = enriched_histories
            return LinkedInDiscovery(first_order_contacts=first_order, second_order_contacts=second_order)
        finally:
            await page.close()

    def build_manual_review_links(self, company_name: str, role_title: str) -> list[ManualReviewLink]:
        return build_manual_review_links(company_name, role_title)

    def _build_discovery_from_extension_session(
        self,
        company_name: str,
        session: ExtensionCaptureSession,
    ) -> LinkedInDiscovery:
        first_order = self._build_contacts_from_extension_capture(
            company_name,
            "1st",
            session.first_order_contacts,
            session,
        )
        second_order = self._build_contacts_from_extension_capture(
            company_name,
            "2nd",
            session.second_order_contacts,
            session,
        )
        first_order_by_key = {_normalize_person_name(contact.name): contact for contact in first_order}
        for contact in second_order:
            ordered_connector_names = list(
                dict.fromkeys(contact.connected_first_order_names + list(contact.connected_first_order_profile_urls))
            )
            enriched_histories = dict(contact.connected_first_order_message_histories)
            enriched_profile_urls = dict(contact.connected_first_order_profile_urls)
            for connector_name in ordered_connector_names:
                normalized = _normalize_person_name(connector_name)
                if normalized in first_order_by_key:
                    first_order_contact = first_order_by_key[normalized]
                    enriched_profile_urls.setdefault(first_order_contact.name, str(first_order_contact.profile_url))
                    if first_order_contact.message_history:
                        enriched_histories.setdefault(first_order_contact.name, first_order_contact.message_history)
                connector_url = enriched_profile_urls.get(connector_name)
                history = self._lookup_extension_history(session, connector_name, connector_url)
                if history:
                    enriched_histories.setdefault(connector_name, history)
            contact.connected_first_order_names = ordered_connector_names
            contact.connected_first_order_profile_urls = enriched_profile_urls
            contact.connected_first_order_message_histories = enriched_histories
        return LinkedInDiscovery(first_order_contacts=first_order, second_order_contacts=second_order)

    def _build_contacts_from_extension_capture(
        self,
        company_name: str,
        degree: str,
        raw_contacts: list[dict[str, object]],
        session: ExtensionCaptureSession,
    ) -> list[LinkedInContact]:
        contacts_by_key: dict[str, LinkedInContact] = {}
        for raw_contact in raw_contacts:
            name = _clean_person_name(str(raw_contact.get("name") or ""))
            profile_url = self._normalize_profile_url(str(raw_contact.get("profile_url") or raw_contact.get("profileUrl") or ""))
            raw_text = str(raw_contact.get("raw_text") or raw_contact.get("rawText") or "").strip()
            headline = str(raw_contact.get("headline") or "").strip() or None
            company_text = str(raw_contact.get("company_text") or raw_contact.get("companyText") or "").strip() or None
            if not name or not profile_url:
                continue
            if not _contact_appears_to_work_at_company(
                company_name,
                headline=headline,
                company_text=company_text,
            ):
                continue

            mutual_names = _dedupe_person_names(
                _clean_person_name(str(value))
                for value in raw_contact.get("mutual_connection_names", raw_contact.get("mutualConnectionNames", []))
            )
            raw_profile_urls = raw_contact.get(
                "connected_first_order_profile_urls",
                raw_contact.get("connectedFirstOrderProfileUrls", {}),
            )
            connected_profile_urls = _sanitize_connected_profile_urls(
                {
                    str(connector_name): self._normalize_profile_url(str(connector_url))
                    for connector_name, connector_url in (raw_profile_urls.items() if isinstance(raw_profile_urls, dict) else [])
                    if str(connector_name).strip() and str(connector_url).strip()
                }
            )
            connected_names = _dedupe_person_names(
                [
                    *mutual_names,
                    *(
                        _clean_person_name(str(value))
                        for value in raw_contact.get(
                            "connected_first_order_names",
                            raw_contact.get("connectedFirstOrderNames", []),
                        )
                    ),
                    *connected_profile_urls.keys(),
                ]
            )

            contact = LinkedInContact(
                name=name,
                profile_url=profile_url,
                headline=headline,
                company_text=company_text,
                connection_degree="1st" if degree == "1st" else "2nd",
                mutual_connection_names=mutual_names,
                connected_first_order_names=connected_names if degree == "2nd" else [],
                connected_first_order_profile_urls=connected_profile_urls if degree == "2nd" else {},
                connected_first_order_message_histories={},
                message_history=[],
            )
            if degree == "1st":
                history = self._lookup_extension_history(session, contact.name, str(contact.profile_url))
                if history:
                    contact.message_history = history
            else:
                for connector_name in connected_names:
                    connector_url = connected_profile_urls.get(connector_name)
                    history = self._lookup_extension_history(session, connector_name, connector_url)
                    if history:
                        contact.connected_first_order_message_histories[connector_name] = history
            existing = contacts_by_key.get(str(contact.profile_url))
            if existing is not None:
                contacts_by_key[str(contact.profile_url)] = _merge_linkedin_contacts(existing, contact)
            else:
                contacts_by_key[str(contact.profile_url)] = contact
        return list(contacts_by_key.values())

    def _lookup_extension_history(
        self,
        session: ExtensionCaptureSession,
        name: str,
        profile_url: str | None,
    ) -> list[str]:
        normalized_profile_url = self._normalize_profile_url(profile_url or "")
        if normalized_profile_url and normalized_profile_url in session.message_histories_by_profile_url:
            return session.message_histories_by_profile_url[normalized_profile_url]
        normalized_name = _normalize_person_name(name)
        return session.message_histories_by_name.get(normalized_name, [])

    async def _apply_auth_cookies(self, context: BrowserContext) -> None:
        cookies = _load_linkedin_auth_cookies(self.settings)
        if cookies:
            await context.add_cookies(cookies)

    async def _attempt_login(self, page: Page) -> None:
        if self.settings.google_email and self.settings.google_password:
            used_google = await self._attempt_google_sign_in(page)
            if used_google:
                return
        if self.settings.linkedin_email and self.settings.linkedin_password:
            email_input = page.locator('input[name="session_key"], input[name="username"]').first
            password_input = page.locator('input[name="session_password"], input[type="password"]').first
            if await email_input.count():
                await email_input.fill(self.settings.linkedin_email)
            if await password_input.count():
                await password_input.fill(self.settings.linkedin_password)
            submit = page.locator('button[type="submit"]').first
            if await submit.count():
                await submit.click()
                await page.wait_for_load_state("domcontentloaded")
        await self._handle_totp_if_prompted(page)

    async def _attempt_google_sign_in(self, page: Page) -> bool:
        button = page.locator(
            'button:has-text("Continue with Google"), a:has-text("Continue with Google"), div[role="button"]:has-text("Continue with Google")'
        ).first
        if await button.count() == 0:
            return False

        popup_page: Page | None = None
        try:
            async with page.expect_popup(timeout=5000) as popup_info:
                await button.click()
            popup_page = await popup_info.value
        except Exception:
            await button.click()
            await page.wait_for_timeout(1500)
            popup_page = page

        await self._complete_google_login(popup_page)
        if popup_page is not page:
            await page.bring_to_front()
            await page.wait_for_timeout(1500)
        return True

    async def _complete_google_login(self, page: Page) -> None:
        await page.wait_for_load_state("domcontentloaded")

        if self.settings.google_email:
            account_chip = page.locator(f'text="{self.settings.google_email}"').first
            try:
                if await account_chip.count():
                    await account_chip.click()
                    await page.wait_for_timeout(1200)
            except Exception:
                pass

        identifier_input = page.locator('input[type="email"], input[name="identifier"]').first
        if self.settings.google_email and await identifier_input.count():
            await identifier_input.fill(self.settings.google_email)
            await self._click_google_next(page)
            await page.wait_for_timeout(1200)

        password_input = page.locator('input[type="password"]').first
        if self.settings.google_password and await password_input.count():
            await password_input.fill(self.settings.google_password)
            await self._click_google_next(page)
            await page.wait_for_timeout(1200)

        await self._handle_google_totp_if_prompted(page)

    async def _click_google_next(self, page: Page) -> None:
        selectors = [
            'button:has-text("Next")',
            '#identifierNext button',
            '#passwordNext button',
            'div[role="button"]:has-text("Next")',
        ]
        for selector in selectors:
            button = page.locator(selector).first
            if await button.count():
                await button.click()
                return

    async def _handle_totp_if_prompted(self, page: Page) -> None:
        if not self.settings.linkedin_totp_secret:
            return
        totp_input = page.locator(
            'input[name="pin"], input[name="verificationCode"], input[autocomplete="one-time-code"]'
        ).first
        if await totp_input.count() == 0:
            return
        code = pyotp.TOTP(self.settings.linkedin_totp_secret).now()
        await totp_input.fill(code)
        submit = page.locator('button[type="submit"], button:has-text("Verify")').first
        if await submit.count():
            await submit.click()
            await page.wait_for_load_state("domcontentloaded")

    async def _handle_google_totp_if_prompted(self, page: Page) -> None:
        if not self.settings.google_totp_secret:
            return
        code_input = page.locator(
            'input[type="tel"], input[autocomplete="one-time-code"], input[name="totpPin"]'
        ).first
        if await code_input.count() == 0:
            return
        code = pyotp.TOTP(self.settings.google_totp_secret).now()
        await code_input.fill(code)
        await self._click_google_next(page)

    async def _ensure_logged_in(self, page: Page) -> None:
        if "feed" in page.url and not await self._is_saved_account_login_page(page):
            return
        await self._maybe_complete_saved_account_login(page)
        if "feed" in page.url and not await self._is_saved_account_login_page(page):
            return
        if "login" in page.url or "checkpoint" in page.url:
            await self._attempt_login(page)
            await self._maybe_complete_saved_account_login(page)
            await page.goto("https://www.linkedin.com/feed/", wait_until="domcontentloaded")
            await self._maybe_complete_saved_account_login(page)
        if "feed" not in page.url or await self._is_saved_account_login_page(page):
            raise RuntimeError(
                "LinkedIn session is not authenticated. Provide credentials/cookies or run "
                "`job-agent bootstrap-linkedin-session`."
            )

    async def _search_people(
        self,
        page: Page,
        company_name: str,
        degree: str,
        *,
        role_title: str | None = None,
    ) -> list[LinkedInContact]:
        network_flag = "F" if degree == "1st" else "S"
        contacts_by_url: dict[str, LinkedInContact] = {}
        for keywords in _build_linkedin_search_keywords(company_name, role_title):
            url = (
                "https://www.linkedin.com/search/results/people/"
                f"?keywords={quote(keywords)}&network=%5B%22{network_flag}%22%5D&origin=GLOBAL_SEARCH_HEADER"
            )
            await page.goto(url, wait_until="domcontentloaded")
            if await self._is_saved_account_login_page(page) or "login" in page.url or "checkpoint" in page.url:
                await self._ensure_logged_in(page)
                await page.goto(url, wait_until="domcontentloaded")

            for _ in range(self.settings.max_linkedin_pages_per_company):
                await self._load_all_results(page)
                page_contacts = await self._collect_contacts_from_search_results(page, company_name, degree)
                for contact in page_contacts:
                    key = str(contact.profile_url)
                    existing = contacts_by_url.get(key)
                    if existing is not None:
                        contacts_by_url[key] = _merge_linkedin_contacts(existing, contact)
                    else:
                        contacts_by_url[key] = contact
                if len(contacts_by_url) >= self.settings.max_linkedin_results_per_company:
                    break
                if not await self._go_to_next_results_page(page):
                    break
            if len(contacts_by_url) >= self.settings.max_linkedin_results_per_company:
                break

        return list(contacts_by_url.values())[: self.settings.max_linkedin_results_per_company]

    async def _load_all_results(self, page: Page) -> None:
        for _ in range(4):
            await page.mouse.wheel(0, 3000)
            await page.wait_for_timeout(1200)

    async def _go_to_next_results_page(self, page: Page) -> bool:
        selectors = [
            'button[aria-label="Next"]',
            'button.artdeco-pagination__button--next',
            'button[aria-label*="Next"]',
        ]
        for selector in selectors:
            button = page.locator(selector).first
            if await button.count() == 0:
                continue
            disabled = await button.get_attribute("disabled")
            aria_disabled = await button.get_attribute("aria-disabled")
            if disabled is not None or aria_disabled == "true":
                return False
            await button.click()
            await page.wait_for_timeout(2000)
            return True
        return False

    async def _collect_contacts_from_search_results(
        self, page: Page, company_name: str, degree: str
    ) -> list[LinkedInContact]:
        contacts: list[LinkedInContact] = []
        parsed_results = await self._parse_search_result_anchors(page)

        for result in parsed_results:
            if len(contacts) >= self.settings.max_linkedin_results_per_company:
                break
            if degree != self._extract_connection_degree(result.raw_text):
                continue
            if not _contact_appears_to_work_at_company(
                company_name,
                headline=result.headline,
                company_text=result.company_text,
            ):
                continue

            contact = LinkedInContact(
                name=result.name or self._extract_name_from_card_text(result.raw_text),
                profile_url=result.profile_url,
                headline=result.headline or None,
                company_text=result.company_text or None,
                connection_degree="1st" if degree == "1st" else "2nd",
                mutual_connection_names=result.mutual_connection_names,
                connected_first_order_names=_dedupe_person_names(
                    result.mutual_connection_names + list(result.mutual_connection_profile_urls)
                ),
                connected_first_order_profile_urls=_sanitize_connected_profile_urls(
                    result.mutual_connection_profile_urls
                ),
                message_history=[],
            )

            if contact.connection_degree == "1st":
                contact.message_history = await self._fetch_message_history(
                    page.context,
                    contact.name,
                    str(contact.profile_url),
                )

            contacts.append(contact)

        return contacts

    async def _fetch_second_order_connector_histories(
        self,
        context: BrowserContext,
        second_order_contacts: list[LinkedInContact],
    ) -> dict[str, list[str]]:
        connector_targets: dict[str, tuple[str, str]] = {}
        for contact in second_order_contacts:
            for connector_name, profile_url in contact.connected_first_order_profile_urls.items():
                normalized = _normalize_person_name(connector_name)
                if normalized and profile_url:
                    connector_targets.setdefault(normalized, (connector_name, profile_url))

        if not connector_targets:
            return {}

        semaphore = asyncio.Semaphore(3)

        async def fetch_one(normalized_name: str, connector_name: str, profile_url: str) -> tuple[str, list[str]]:
            async with semaphore:
                history = await self._fetch_message_history(context, connector_name, profile_url)
                return normalized_name, history

        results = await asyncio.gather(
            *(fetch_one(normalized, name, url) for normalized, (name, url) in connector_targets.items())
        )
        return {normalized: history for normalized, history in results if history}

    async def _parse_search_result_anchors(self, page: Page) -> list[ParsedSearchResult]:
        anchors = await page.eval_on_selector_all(
            'a[href*="/in/"]',
            """
            (elements) => elements.map((element) => ({
                href: element.href,
                text: (element.innerText || element.textContent || "").trim(),
            }))
            """,
        )
        results: list[ParsedSearchResult] = []
        current: dict[str, object] | None = None

        for anchor in anchors:
            href = self._normalize_profile_url(str(anchor.get("href", "")))
            text = str(anchor.get("text", "")).strip()
            if not href or not text:
                continue
            if self._extract_connection_degree(text) is not None:
                if current is not None:
                    parsed = self._finalize_parsed_search_result(current)
                    if parsed is not None:
                        results.append(parsed)
                current = {"href": href, "text": text, "mutuals": []}
                continue

            if current is None:
                continue

            current_href = str(current["href"])
            current_text = str(current["text"])
            current_name = self._extract_name_from_card_text(current_text)
            if href == current_href:
                continue
            if _normalize_person_name(text) == _normalize_person_name(current_name):
                continue
            mutuals = current["mutuals"]
            if isinstance(mutuals, list):
                mutuals.append((text, href))

        if current is not None:
            parsed = self._finalize_parsed_search_result(current)
            if parsed is not None:
                results.append(parsed)
        return results

    def _finalize_parsed_search_result(self, raw_result: dict[str, object]) -> ParsedSearchResult | None:
        raw_text = str(raw_result.get("text", "")).strip()
        profile_url = self._normalize_profile_url(str(raw_result.get("href", "")))
        if not raw_text or not profile_url:
            return None

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return None

        name = _clean_person_name(lines[0])
        degree = self._extract_connection_degree(raw_text)
        if degree is None or not name:
            return None

        filtered_lines = [line for line in lines[1:] if line != f"\u2022 {degree}" and line != degree]
        headline = filtered_lines[0] if filtered_lines else None
        company_text = None
        if headline:
            company_text = headline
        elif len(filtered_lines) > 1:
            company_text = filtered_lines[1]

        mutual_connection_names = self._extract_mutual_connection_names(raw_text)
        mutual_connection_profile_urls: dict[str, str] = {}
        for connector_name, connector_url in raw_result.get("mutuals", []):
            cleaned_connector_name = _clean_person_name(str(connector_name))
            if not cleaned_connector_name:
                continue
            mutual_connection_profile_urls[cleaned_connector_name] = self._normalize_profile_url(str(connector_url))
            if cleaned_connector_name not in mutual_connection_names:
                mutual_connection_names.append(cleaned_connector_name)

        return ParsedSearchResult(
            name=name,
            profile_url=profile_url,
            raw_text=raw_text,
            headline=headline,
            company_text=company_text,
            mutual_connection_names=_dedupe_person_names(mutual_connection_names),
            mutual_connection_profile_urls=_sanitize_connected_profile_urls(mutual_connection_profile_urls),
        )

    async def _fetch_profile_details(self, context: BrowserContext, profile_url: str) -> dict[str, str]:
        profile_page = await context.new_page()
        try:
            await profile_page.goto(profile_url, wait_until="domcontentloaded")
            await profile_page.wait_for_timeout(1200)
            await profile_page.mouse.wheel(0, 700)
            await profile_page.wait_for_timeout(400)

            headline = await self._safe_inner_text(
                profile_page.locator("div.text-body-medium.break-words").first
            )
            company = await self._safe_inner_text(
                profile_page.locator("div.pv-text-details__left-panel div.text-body-small").first
            )
            if not company:
                company = await self._safe_inner_text(profile_page.locator("main").first)
            name = await self._safe_inner_text(profile_page.locator("h1").first)
            return {
                "name": name,
                "headline": headline,
                "company_text": company[:400],
            }
        except Exception:
            return {"name": "", "headline": "", "company_text": ""}
        finally:
            await profile_page.close()

    async def _fetch_message_history(
        self, context: BrowserContext, contact_name: str, profile_url: str
    ) -> list[str]:
        history = await self._fetch_message_history_from_messaging(context, contact_name)
        if history:
            return history
        return await self._fetch_message_history_from_profile(context, profile_url)

    async def _fetch_message_history_from_messaging(
        self, context: BrowserContext, contact_name: str
    ) -> list[str]:
        page = await context.new_page()
        try:
            await page.goto("https://www.linkedin.com/messaging/", wait_until="domcontentloaded")
            await page.wait_for_timeout(1200)
            selectors = [
                'input[placeholder*="Search messages"]',
                'input[aria-label*="Search messages"]',
                'input[placeholder*="Search"]',
            ]
            search_box = None
            for selector in selectors:
                candidate = page.locator(selector).first
                if await candidate.count():
                    search_box = candidate
                    break
            if search_box is None:
                return []
            await search_box.fill(contact_name)
            await page.wait_for_timeout(1500)
            conversation = page.locator(
                f'li:has-text("{contact_name}"), div:has-text("{contact_name}")'
            ).first
            if await conversation.count() == 0:
                return []
            await conversation.click()
            await page.wait_for_timeout(1500)
            return await self._read_visible_messages(page)
        except Exception:
            return []
        finally:
            await page.close()

    async def _fetch_message_history_from_profile(self, context: BrowserContext, profile_url: str) -> list[str]:
        profile_page = await context.new_page()
        try:
            await profile_page.goto(profile_url, wait_until="domcontentloaded")
            await profile_page.wait_for_timeout(1200)
            message_button = profile_page.locator('button:has-text("Message"), a:has-text("Message")').first
            if await message_button.count() == 0:
                return []
            await message_button.click()
            await profile_page.wait_for_timeout(1800)
            return await self._read_visible_messages(profile_page)
        except Exception:
            return []
        finally:
            await profile_page.close()

    async def _load_older_messages(self, page: Page) -> None:
        selectors = [
            ".msg-s-message-list-content",
            ".msg-s-message-list",
            ".msg-overlay-conversation-bubble-list",
        ]
        for selector in selectors:
            container = page.locator(selector).first
            try:
                if await container.count() == 0:
                    continue
                for _ in range(4):
                    await container.evaluate("(node) => { node.scrollTop = 0; }")
                    await page.wait_for_timeout(450)
                return
            except Exception:
                continue

    async def _read_visible_messages(self, page: Page) -> list[str]:
        await self._load_older_messages(page)
        selectors = [
            '.msg-s-message-list__event',
            '.msg-s-message-group__messages li',
            '.msg-s-event-listitem',
        ]
        for selector in selectors:
            messages = page.locator(selector)
            count = await messages.count()
            if count == 0:
                continue
            history: list[str] = []
            start = max(0, count - 20)
            for index in range(start, count):
                text = (await messages.nth(index).inner_text()).strip()
                if text:
                    history.append(text)
            if history:
                return history
        return []

    @staticmethod
    def _extract_mutual_connection_names(text: str) -> list[str]:
        if not text:
            return []
        relevant_lines = [
            line.strip()
            for line in str(text).splitlines()
            if re.search(r"\b(mutual|shared) connections?\b", line, flags=re.IGNORECASE)
        ]
        if not relevant_lines:
            return []
        cleaned = " ".join(relevant_lines)
        cleaned = re.sub(r"\bmutual connections?\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bshared connections?\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bis\b|\bare\b", "", cleaned, flags=re.IGNORECASE)
        parts = cleaned.replace(" and ", ",").split(",")
        names: list[str] = []
        for part in parts:
            candidate = _clean_person_name(part)
            if not candidate:
                continue
            names.append(candidate)
        return _dedupe_person_names(names)

    @staticmethod
    def _extract_connection_degree(text: str) -> str | None:
        normalized = " ".join(text.split())
        if "• 1st" in normalized or re.search(r"\b1st\b", normalized):
            return "1st"
        if "• 2nd" in normalized or re.search(r"\b2nd\b", normalized):
            return "2nd"
        return None

    @staticmethod
    def _extract_name_from_card_text(text: str) -> str:
        first_line = text.splitlines()[0].strip()
        return first_line[:120] or "Unknown"

    @staticmethod
    async def _safe_inner_text(locator) -> str:
        try:
            if await locator.count() == 0:
                return ""
            return (await locator.inner_text()).strip()
        except PlaywrightTimeoutError:
            return ""

    @staticmethod
    def _normalize_profile_url(url: str) -> str:
        if url.startswith("/"):
            return f"https://www.linkedin.com{url.split('?')[0]}"
        parsed = urlparse(url)
        cleaned = parsed._replace(query="", fragment="")
        return urlunparse(cleaned)

    async def _is_saved_account_login_page(self, page: Page) -> bool:
        if "linkedin.com/feed" not in page.url and ("login" in page.url or "checkpoint" in page.url):
            return True
        selectors = [
            'button[aria-label^="Login as "]',
            'button[aria-label^="Continue as "]',
            'input[name="session_key"]',
            'input[name="username"]',
        ]
        for selector in selectors:
            locator = page.locator(selector).first
            try:
                if await locator.count():
                    return True
            except PlaywrightTimeoutError:
                continue
        return False

    async def _maybe_complete_saved_account_login(self, page: Page) -> None:
        selectors = [
            'button[aria-label^="Login as "]',
            'button[aria-label^="Continue as "]',
            'button[data-litms-control-urn*="login-submit"]',
        ]
        for selector in selectors:
            button = page.locator(selector).first
            try:
                if await button.count() == 0:
                    continue
                await button.click()
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(1500)
                return
            except Exception:
                continue


def _normalize_person_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _is_placeholder_person_name(name: str) -> bool:
    lowered = " ".join(name.lower().split())
    if not lowered:
        return True
    return bool(re.fullmatch(r"\d+\s+(?:other|others|more)", lowered))


def _clean_person_name(name: str) -> str:
    candidate = " ".join(str(name).replace("\xa0", " ").split()).strip(" ,-|")
    if not candidate:
        return ""
    candidate = re.sub(r"\b(mutual|shared) connections?\b.*$", "", candidate, flags=re.IGNORECASE).strip(" ,-|")
    candidate = re.sub(r"\bfollow\b.*$", "", candidate, flags=re.IGNORECASE).strip(" ,-|")
    if not candidate or _is_placeholder_person_name(candidate):
        return ""

    parts = candidate.split()
    if len(parts) >= 3 and parts[-1].islower() and len(parts[-1]) <= 2:
        candidate = " ".join(parts[:-1]).strip()
    return "" if _is_placeholder_person_name(candidate) else candidate


def _dedupe_person_names(values) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        candidate = _clean_person_name(str(value))
        normalized = _normalize_person_name(candidate)
        if not candidate or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        names.append(candidate)
    return names


def _sanitize_connected_profile_urls(values: dict[str, str]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for connector_name, connector_url in values.items():
        cleaned_name = _clean_person_name(connector_name)
        if not cleaned_name or not connector_url:
            continue
        sanitized.setdefault(cleaned_name, connector_url)
    return sanitized


def _merge_linkedin_contacts(existing: LinkedInContact, incoming: LinkedInContact) -> LinkedInContact:
    message_history = (
        incoming.message_history
        if len(incoming.message_history) > len(existing.message_history)
        else existing.message_history
    )
    merged_profile_urls = _sanitize_connected_profile_urls(
        {
            **existing.connected_first_order_profile_urls,
            **incoming.connected_first_order_profile_urls,
        }
    )
    merged_histories = dict(existing.connected_first_order_message_histories)
    for connector_name, history in incoming.connected_first_order_message_histories.items():
        cleaned_name = _clean_person_name(connector_name)
        if cleaned_name and history and len(history) > len(merged_histories.get(cleaned_name, [])):
            merged_histories[cleaned_name] = history

    return existing.model_copy(
        update={
            "headline": existing.headline or incoming.headline,
            "company_text": existing.company_text or incoming.company_text,
            "mutual_connection_names": _dedupe_person_names(
                existing.mutual_connection_names + incoming.mutual_connection_names
            ),
            "connected_first_order_names": _dedupe_person_names(
                existing.connected_first_order_names
                + incoming.connected_first_order_names
                + list(merged_profile_urls)
            ),
            "connected_first_order_profile_urls": merged_profile_urls,
            "connected_first_order_message_histories": merged_histories,
            "message_history": message_history,
        }
    )


def _normalize_company_name(name: str) -> str:
    lowered = re.sub(r"[^a-z0-9 ]", " ", name.lower())
    tokens = [
        token
        for token in lowered.split()
        if token not in {"inc", "llc", "ltd", "corp", "co", "company", "gmbh", "plc", "the"}
    ]
    return " ".join(tokens)


def _linkedin_people_search_url(keywords: str, network_flag: str) -> str:
    return (
        "https://www.linkedin.com/search/results/people/"
        f"?keywords={quote(keywords)}&network=%5B%22{network_flag}%22%5D&origin=GLOBAL_SEARCH_HEADER"
    )


def _role_keywords_for_linkedin_search(role_title: str | None) -> str:
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
        "pm",
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


def _build_linkedin_search_keywords(company_name: str, role_title: str | None) -> list[str]:
    keywords = [
        f"\"{company_name}\"",
        f"\"{company_name}\" recruiter hiring talent acquisition",
        f"\"{company_name}\" {_role_keywords_for_linkedin_search(role_title)}",
    ]
    return list(dict.fromkeys(" ".join(keyword.split()) for keyword in keywords if keyword.strip()))


def build_manual_review_links(company_name: str, role_title: str) -> list[ManualReviewLink]:
    role_keywords = " ".join(token for token in ("product manager", "ai", role_title or "") if token).strip()
    links = [
        ManualReviewLink(
            label="1st-degree people at company",
            url=_linkedin_people_search_url(f"\"{company_name}\"", "F"),
        ),
        ManualReviewLink(
            label="2nd-degree people at company",
            url=_linkedin_people_search_url(f"\"{company_name}\"", "S"),
        ),
        ManualReviewLink(
            label="1st-degree recruiting/hiring contacts at company",
            url=_linkedin_people_search_url(f"\"{company_name}\" recruiter hiring talent acquisition", "F"),
        ),
        ManualReviewLink(
            label="2nd-degree recruiting/hiring contacts at company",
            url=_linkedin_people_search_url(f"\"{company_name}\" recruiter hiring talent acquisition", "S"),
        ),
        ManualReviewLink(
            label="1st-degree PM/AI contacts at company",
            url=_linkedin_people_search_url(f"\"{company_name}\" {role_keywords}", "F"),
        ),
        ManualReviewLink(
            label="2nd-degree PM/AI contacts at company",
            url=_linkedin_people_search_url(f"\"{company_name}\" {role_keywords}", "S"),
        ),
    ]
    return links


def _contact_appears_to_work_at_company(
    company_name: str,
    *,
    headline: str | None = None,
    company_text: str | None = None,
) -> bool:
    return any(
        _headline_looks_like_current_employer(company_name, text)
        for text in (company_text, headline)
    )


def _headline_looks_like_current_employer(company_name: str, text: str | None) -> bool:
    if not text or not _company_matches(company_name, text):
        return False
    company_pattern = _company_search_pattern(company_name)
    if not company_pattern:
        return False
    patterns = [
        rf"\bat\s+{company_pattern}\b",
        rf"@\s*{company_pattern}\b",
        rf"^\s*{company_pattern}\b(?:\s*[\-|,:/|])?",
    ]
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in patterns)


def _company_search_pattern(company_name: str) -> str:
    tokens = _normalize_company_name(company_name).split()
    if not tokens:
        return ""
    separator_pattern = r"(?:\s|[-/&])+"
    return separator_pattern.join(re.escape(token) for token in tokens)


def _company_matches(company_name: str, *texts: str | None) -> bool:
    expected_tokens = _normalize_company_name(company_name).split()
    if not expected_tokens:
        return False
    for text in texts:
        haystack_tokens = _normalize_company_name(text or "").split()
        if not haystack_tokens or len(haystack_tokens) < len(expected_tokens):
            continue
        for index in range(len(haystack_tokens) - len(expected_tokens) + 1):
            if haystack_tokens[index : index + len(expected_tokens)] == expected_tokens:
                return True
        if haystack_tokens == expected_tokens:
            return True
    return False


@dataclass(slots=True)
class BrowserChoice:
    label: str
    channel: str | None = None
    executable_path: str | None = None


def get_preferred_browser(settings: Settings) -> BrowserChoice:
    if settings.browser_executable_path:
        return BrowserChoice(
            label=f"custom executable ({settings.browser_executable_path})",
            executable_path=settings.browser_executable_path,
        )
    if settings.browser_channel:
        return BrowserChoice(label=f"channel {settings.browser_channel}", channel=settings.browser_channel)

    candidates = [
        ("Google Chrome", shutil.which("google-chrome-stable"), None),
        ("Google Chrome", shutil.which("google-chrome"), None),
        ("Chromium", shutil.which("chromium"), None),
        ("Chromium", shutil.which("chromium-browser"), None),
        ("Chromium Flatpak", shutil.which("org.chromium.Chromium"), None),
        ("Brave", shutil.which("brave-browser"), None),
        ("Microsoft Edge", shutil.which("microsoft-edge-stable"), None),
        ("Microsoft Edge", shutil.which("microsoft-edge"), None),
        ("Google Chrome", None, "/usr/bin/google-chrome-stable"),
        ("Google Chrome", None, "/usr/bin/google-chrome"),
        ("Chromium", None, "/usr/bin/chromium"),
        ("Chromium", None, "/usr/bin/chromium-browser"),
        ("Chromium Flatpak", None, str(Path.home() / ".local/share/flatpak/exports/bin/org.chromium.Chromium")),
        ("Brave", None, "/usr/bin/brave-browser"),
        ("Microsoft Edge", None, "/usr/bin/microsoft-edge-stable"),
        ("Microsoft Edge", None, "/usr/bin/microsoft-edge"),
    ]
    for label, path_from_which, fallback_path in candidates:
        path = path_from_which or fallback_path
        if path and Path(path).exists():
            return BrowserChoice(label=label, executable_path=str(path))

    return BrowserChoice(label="Playwright bundled Chromium")


def describe_browser_choice(settings: Settings) -> str:
    choice = get_preferred_browser(settings)
    if choice.executable_path:
        return f"{choice.label} at {choice.executable_path}"
    if choice.channel:
        return choice.label
    return choice.label


def _default_firefox_profile_dir(firefox_root: Path | None = None) -> Path | None:
    root = firefox_root or (Path.home() / ".mozilla/firefox")
    profiles_ini = root / "profiles.ini"
    if not profiles_ini.exists():
        return None

    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(profiles_ini)
    except configparser.Error:
        return None

    def resolve_profile_path(raw_path: str | None, *, is_relative: bool) -> Path | None:
        if not raw_path:
            return None
        candidate = Path(raw_path)
        if is_relative:
            candidate = root / candidate
        return candidate if candidate.exists() else None

    for section in parser.sections():
        if not section.startswith("Install"):
            continue
        resolved = resolve_profile_path(
            parser.get(section, "Default", fallback=None),
            is_relative=True,
        )
        if resolved is not None:
            return resolved

    for section in parser.sections():
        if not section.startswith("Profile"):
            continue
        if parser.get(section, "Default", fallback="0") != "1":
            continue
        resolved = resolve_profile_path(
            parser.get(section, "Path", fallback=None),
            is_relative=parser.get(section, "IsRelative", fallback="1") == "1",
        )
        if resolved is not None:
            return resolved

    for section in parser.sections():
        if not section.startswith("Profile"):
            continue
        resolved = resolve_profile_path(
            parser.get(section, "Path", fallback=None),
            is_relative=parser.get(section, "IsRelative", fallback="1") == "1",
        )
        if resolved is not None:
            return resolved

    return None


def _copy_firefox_cookie_db(profile_dir: Path) -> Path | None:
    source = profile_dir / "cookies.sqlite"
    if not source.exists():
        return None

    temp_root = Path(tempfile.mkdtemp(prefix="job-agent-firefox-cookies-"))
    destination = temp_root / "cookies.sqlite"
    try:
        shutil.copy2(source, destination)
        for suffix in ("-wal", "-shm"):
            companion = profile_dir / f"cookies.sqlite{suffix}"
            if companion.exists():
                shutil.copy2(companion, temp_root / companion.name)
        return destination
    except OSError:
        shutil.rmtree(temp_root, ignore_errors=True)
        return None


def _load_linkedin_storage_state_cookies(storage_state_path: Path) -> list[dict[str, object]]:
    if not storage_state_path.exists():
        return []
    try:
        payload = json.loads(storage_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    cookies: list[dict[str, object]] = []
    for raw_cookie in payload.get("cookies", []):
        if not isinstance(raw_cookie, dict):
            continue
        domain = str(raw_cookie.get("domain") or "")
        if "linkedin.com" not in domain:
            continue
        name = str(raw_cookie.get("name") or "").strip()
        value = raw_cookie.get("value")
        if not name or value is None:
            continue
        cookie = {
            "name": name,
            "value": str(value),
            "domain": domain,
            "path": str(raw_cookie.get("path") or "/"),
            "expires": _normalize_cookie_expiry(raw_cookie.get("expires")),
            "secure": bool(raw_cookie.get("secure", False)),
            "httpOnly": bool(raw_cookie.get("httpOnly", False)),
        }
        same_site = raw_cookie.get("sameSite")
        if same_site in {"Lax", "Strict", "None"}:
            cookie["sameSite"] = same_site
        cookies.append(cookie)
    return cookies


def _load_firefox_linkedin_cookies(profile_dir: Path | None = None) -> list[dict[str, object]]:
    resolved_profile_dir = profile_dir or _default_firefox_profile_dir()
    if resolved_profile_dir is None:
        return []

    copied_db = _copy_firefox_cookie_db(resolved_profile_dir)
    if copied_db is None:
        return []

    try:
        connection = sqlite3.connect(copied_db)
        cursor = connection.cursor()
        cursor.execute(
            """
            select name, value, host, path, expiry, isSecure, isHttpOnly
            from moz_cookies
            where host like '%linkedin.com%'
            """
        )
        cookies: list[dict[str, object]] = []
        for name, value, host, path, expiry, is_secure, is_http_only in cursor.fetchall():
            if not name or value is None or not host:
                continue
            expires = _normalize_cookie_expiry(expiry)
            cookies.append(
                {
                    "name": str(name),
                    "value": str(value),
                    "domain": str(host),
                    "path": str(path or "/"),
                    "expires": expires,
                    "secure": bool(is_secure),
                    "httpOnly": bool(is_http_only),
                }
            )
        return cookies
    except sqlite3.Error:
        return []
    finally:
        try:
            connection.close()  # type: ignore[name-defined]
        except Exception:
            pass
        shutil.rmtree(copied_db.parent, ignore_errors=True)


def linkedin_cookies_authenticate(
    cookies: list[dict[str, object]],
    *,
    timeout_seconds: float = 10.0,
) -> bool:
    if not cookies:
        return False

    jar = httpx.Cookies()
    for cookie in cookies:
        name = str(cookie.get("name") or "").strip()
        value = cookie.get("value")
        domain = str(cookie.get("domain") or "").strip()
        if not name or value is None or "linkedin.com" not in domain:
            continue
        jar.set(name, str(value), domain=domain, path=str(cookie.get("path") or "/"))

    if not jar:
        return False

    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout_seconds,
            cookies=jar,
            headers={"User-Agent": "Mozilla/5.0"},
        ) as client:
            response = client.get("https://www.linkedin.com/feed/")
    except httpx.HTTPError:
        return False

    final_url = str(response.url or "")
    lowered = response.text.lower()
    if "/uas/login" in final_url or "linkedin login" in lowered:
        return False
    if "checkpoint/challenge" in final_url or "checkpoint/challenge" in lowered:
        return False
    return "linkedin.com/feed" in final_url or "voyager-web" in lowered or "feed" in final_url


def linkedin_storage_state_is_authenticated(storage_state_path: Path) -> bool:
    return linkedin_cookies_authenticate(_load_linkedin_storage_state_cookies(storage_state_path))


def sync_linkedin_cookies_to_firefox_profile(profile_dir: Path, storage_state_path: Path) -> bool:
    selected_names = {"li_at", "JSESSIONID"}
    cookies = [
        cookie
        for cookie in _load_linkedin_storage_state_cookies(storage_state_path)
        if str(cookie.get("name") or "") in selected_names
    ]
    if not cookies:
        return False

    db_path = profile_dir / "cookies.sqlite"
    if not db_path.exists():
        return False

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(db_path, timeout=2.0)
        cursor = connection.cursor()
        cursor.execute("pragma table_info(moz_cookies)")
        columns = {str(row[1]) for row in cursor.fetchall()}
        if not columns:
            return False

        base_columns = [
            "originAttributes",
            "name",
            "value",
            "host",
            "path",
            "expiry",
            "lastAccessed",
            "creationTime",
            "isSecure",
            "isHttpOnly",
            "inBrowserElement",
            "sameSite",
            "schemeMap",
        ]
        optional_columns = [column for column in ("isPartitionedAttributeSet", "updateTime") if column in columns]
        insert_columns = [column for column in base_columns if column in columns] + optional_columns
        placeholders = ", ".join("?" for _ in insert_columns)
        now_us = int(time.time() * 1_000_000)
        same_site_map = {"lax": 1, "strict": 2, "Lax": 1, "Strict": 2}

        cursor.execute("delete from moz_cookies where host like '%linkedin.com' and name in ('li_at','JSESSIONID')")
        for cookie in cookies:
            values_by_column = {
                "originAttributes": "",
                "name": str(cookie["name"]),
                "value": str(cookie["value"]),
                "host": str(cookie.get("domain") or ".www.linkedin.com"),
                "path": str(cookie.get("path") or "/"),
                "expiry": int(cookie.get("expires") or 0),
                "lastAccessed": now_us,
                "creationTime": now_us,
                "isSecure": 1 if cookie.get("secure", True) else 0,
                "isHttpOnly": 1 if cookie.get("httpOnly", False) else 0,
                "inBrowserElement": 0,
                "sameSite": same_site_map.get(str(cookie.get("sameSite") or ""), 0),
                "schemeMap": 0,
                "isPartitionedAttributeSet": 0,
                "updateTime": now_us,
            }
            cursor.execute(
                f"insert into moz_cookies ({', '.join(insert_columns)}) values ({placeholders})",
                tuple(values_by_column[column] for column in insert_columns),
            )
        connection.commit()
        return True
    except sqlite3.Error:
        return False
    finally:
        if connection is not None:
            with contextlib.suppress(Exception):
                connection.close()


def _load_env_linkedin_cookies(settings: Settings) -> list[dict[str, object]]:
    cookies: list[dict[str, object]] = []
    if settings.linkedin_li_at:
        cookies.append(
            {
                "name": "li_at",
                "value": settings.linkedin_li_at,
                "domain": ".linkedin.com",
                "path": "/",
                "secure": True,
                "httpOnly": True,
            }
        )
    if settings.linkedin_jsessionid:
        jsessionid = settings.linkedin_jsessionid
        if not jsessionid.startswith('"'):
            jsessionid = f'"{jsessionid}"'
        cookies.append(
            {
                "name": "JSESSIONID",
                "value": jsessionid,
                "domain": ".www.linkedin.com",
                "path": "/",
                "secure": True,
                "httpOnly": False,
            }
        )
    return cookies


def _load_linkedin_auth_cookies(settings: Settings) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()
    for source_cookies in (
        _load_env_linkedin_cookies(settings),
        _load_linkedin_storage_state_cookies(settings.linkedin_storage_state),
        _load_firefox_linkedin_cookies(),
    ):
        for cookie in source_cookies:
            key = (
                str(cookie.get("name") or ""),
                str(cookie.get("domain") or ""),
                str(cookie.get("path") or "/"),
            )
            if not all(key) or key in seen:
                continue
            seen.add(key)
            merged.append(cookie)
    return merged


def _normalize_cookie_expiry(value: object) -> int:
    if not isinstance(value, (int, float)):
        return -1
    expires = int(value)
    if expires <= 0:
        return -1
    if expires > 10_000_000_000:
        expires //= 1000
    return expires if expires > 0 else -1
