from __future__ import annotations

import asyncio
import json

from agents import Agent, Runner
from pydantic import BaseModel, Field

from .config import Settings
from .llm_provider import LLMProviderError, OllamaStructuredProvider
from .models import (
    FirstOrderMessage,
    JobOutreachBundle,
    JobPosting,
    LinkedInContact,
    SecondOrderIntroMessage,
)


class FirstOrderMessagesOutput(BaseModel):
    messages: list[FirstOrderMessage] = Field(default_factory=list)


class SecondOrderMessagesOutput(BaseModel):
    messages: list[SecondOrderIntroMessage] = Field(default_factory=list)


def _build_ollama_provider(settings: Settings) -> OllamaStructuredProvider:
    return OllamaStructuredProvider(settings)


def build_first_order_message_agent() -> Agent:
    return Agent(
        name="First Order Outreach Agent",
        model="gpt-5.1",
        instructions="""
You draft concise LinkedIn messages to first-degree contacts at a target company.

Requirements:
- Sound natural and specific, not spammy.
- Use any message history context if it exists.
- If the contact already works at the target company, ask for insight on the position, team, and hiring context.
- Do not make the first message a blunt referral ask.
- Include the direct job link in the actual message body.
- Include the recipient's LinkedIn profile link in the actual message body.
- Keep each draft brief enough to send in LinkedIn and avoid bullet lists in the message itself.
""".strip(),
        output_type=FirstOrderMessagesOutput,
    )


def build_second_order_message_agent() -> Agent:
    return Agent(
        name="Second Order Intro Agent",
        model="gpt-5.1",
        instructions="""
You draft concise messages asking a first-degree contact for an introduction to a second-degree contact.

Requirements:
- Sound warm, direct, and professional.
- Mention the exact job and why the second-degree contact is relevant.
- Include the direct job link in the actual message body.
- Include the second-degree contact's LinkedIn profile link in the actual message body.
- Ask for an intro only if the first-degree contact knows the target well enough.
- Keep each draft short enough for LinkedIn and avoid stiff boilerplate.
""".strip(),
        output_type=SecondOrderMessagesOutput,
    )


def _append_required_links(message_body: str, *, job_url: str, profile_url: str) -> str:
    body = message_body.strip()
    if job_url not in body:
        body = f"{body}\n\nJob link: {job_url}".strip()
    if profile_url not in body:
        body = f"{body}\nProfile link: {profile_url}".strip()
    return body


def _append_required_target_links(
    message_body: str,
    *,
    job_url: str,
    targets: list[tuple[str, str]],
) -> str:
    body = message_body.strip()
    if job_url not in body:
        body = f"{body}\n\nJob link: {job_url}".strip()
    missing_targets = [(name, url) for name, url in targets if url and url not in body]
    if missing_targets:
        target_lines = "\n".join(f"- {name}: {url}" for name, url in missing_targets)
        body = f"{body}\n\nTarget profiles:\n{target_lines}".strip()
    return body


def _normalize_name(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum())


def _natural_join(values: list[str]) -> str:
    items = [value for value in values if value]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _history_count(history: list[str]) -> int:
    return len([message for message in history if str(message).strip()])


def _contact_priority(contact: LinkedInContact) -> tuple[int, int, str]:
    history_count = _history_count(contact.message_history)
    return (
        0 if history_count >= 10 else 1,
        -history_count,
        contact.name.lower(),
    )


def _connector_group_priority(group: dict[str, object]) -> tuple[int, int, int, str]:
    history_count = _history_count(list(group["message_history"]))
    return (
        0 if history_count >= 10 else 1,
        -history_count,
        0 if group["first_order_contact_profile_url"] else 1,
        str(group["first_order_contact_name"]).lower(),
    )


def _build_second_order_connector_groups(
    second_order_contacts: list[LinkedInContact],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for second in second_order_contacts:
        connector_names = list(
            dict.fromkeys(second.connected_first_order_names + list(second.connected_first_order_profile_urls))
        )
        if not connector_names:
            continue
        for connector_name in connector_names:
            canonical_connector_name = connector_name.strip()
            if not canonical_connector_name:
                continue
            connector_key = _normalize_name(canonical_connector_name)
            if not connector_key:
                continue
            connector_profile_url = second.connected_first_order_profile_urls.get(canonical_connector_name)
            connector_history = second.connected_first_order_message_histories.get(canonical_connector_name, [])
            group = grouped.setdefault(
                connector_key,
                {
                    "first_order_contact_name": canonical_connector_name,
                    "first_order_contact_profile_url": connector_profile_url,
                    "message_history": list(connector_history),
                    "targets_by_key": {},
                },
            )
            if connector_profile_url and not group["first_order_contact_profile_url"]:
                group["first_order_contact_profile_url"] = connector_profile_url
            if connector_history and _history_count(list(connector_history)) > _history_count(list(group["message_history"])):
                group["message_history"] = list(connector_history)

            target_key = second.profile_url or _normalize_name(second.name)
            targets_by_key = group["targets_by_key"]
            if target_key not in targets_by_key:
                targets_by_key[target_key] = {
                    "name": second.name,
                    "profile_url": str(second.profile_url),
                    "headline": second.headline,
                }

    connector_groups: list[dict[str, object]] = []
    for connector_key, group in grouped.items():
        targets = list(group["targets_by_key"].values())
        if not targets:
            continue
        connector_groups.append(
            {
                "connector_key": connector_key,
                "first_order_contact_name": group["first_order_contact_name"],
                "first_order_contact_profile_url": group["first_order_contact_profile_url"],
                "message_history": group["message_history"],
                "targets": targets,
            }
        )
    connector_groups.sort(
        key=_connector_group_priority
    )
    return connector_groups


def _template_first_order_messages(job: JobPosting, contacts: list[LinkedInContact]) -> list[FirstOrderMessage]:
    messages: list[FirstOrderMessage] = []
    for contact in sorted(contacts, key=_contact_priority):
        history_reference = ""
        if contact.message_history:
            recent_excerpt = contact.message_history[-1].replace("\n", " ").strip()[:160]
            if recent_excerpt:
                history_reference = f"It was good chatting with you about \"{recent_excerpt}\" recently. "
        body = (
            f"Hi {contact.name}, {history_reference}I’m looking at the {job.role_title} role at {job.company_name} "
            f"and wanted to ask whether you’d be open to sharing any perspective on the team and what they’re looking for. "
            f"If you think it makes sense, I’d also love your advice on the best person to speak with.\n\n"
            f"Job link: {job.direct_job_url}\n"
            f"Profile link: {contact.profile_url}"
        )
        messages.append(
            FirstOrderMessage(
                contact_name=contact.name,
                contact_profile_url=str(contact.profile_url),
                subject_context=f"{job.company_name} role context",
                message_body=body,
            )
        )
    return messages


def _template_second_order_messages(
    job: JobPosting,
    first_order_contacts: list[LinkedInContact],
    second_order_contacts: list[LinkedInContact],
) -> list[SecondOrderIntroMessage]:
    messages: list[SecondOrderIntroMessage] = []
    for connector_group in _build_second_order_connector_groups(second_order_contacts):
        connector_name = str(connector_group["first_order_contact_name"])
        connector_profile_url = connector_group["first_order_contact_profile_url"]
        history = list(connector_group["message_history"])
        targets = list(connector_group["targets"])
        history_reference = ""
        if history:
            recent_excerpt = history[-1].replace("\n", " ").strip()[:160]
            if recent_excerpt:
                history_reference = f"It was good chatting with you about \"{recent_excerpt}\" recently. "
        target_names = [str(target["name"]) for target in targets]
        target_reason = ""
        if len(targets) == 1 and targets[0].get("headline"):
            target_reason = f" They look especially relevant because they’re {targets[0]['headline']}."
        body = (
            f"Hi {connector_name}, {history_reference}I’m looking at the {job.role_title} role at {job.company_name}. "
            f"I noticed you’re connected to {_natural_join(target_names)} and wanted to ask whether you’d be open to making an intro "
            f"if you know them well enough.{target_reason} Happy to send a short blurb if that helps.\n\n"
            f"Job link: {job.direct_job_url}"
        )
        messages.append(
            SecondOrderIntroMessage(
                first_order_contact_name=connector_name,
                first_order_contact_profile_url=str(connector_profile_url) if connector_profile_url else None,
                second_order_contact_names=target_names,
                second_order_contact_profile_urls=[str(target["profile_url"]) for target in targets],
                message_body=body,
            )
        )
    return messages


def _finalize_first_order_messages(
    job: JobPosting,
    contacts: list[LinkedInContact],
    messages: list[FirstOrderMessage],
) -> list[FirstOrderMessage]:
    priority_by_contact = {
        _normalize_name(contact.name): _contact_priority(contact)
        for contact in contacts
    }
    finalized: list[FirstOrderMessage] = []
    for message in messages:
        body = _append_required_links(
            message.message_body,
            job_url=str(job.direct_job_url),
            profile_url=str(message.contact_profile_url),
        )
        finalized_message = message.model_copy(update={"message_body": body})
        finalized.append(finalized_message)
    finalized.sort(
        key=lambda message: priority_by_contact.get(
            _normalize_name(message.contact_name),
            (1, 0, message.contact_name.lower()),
        )
    )
    return finalized


def _finalize_second_order_messages(
    job: JobPosting,
    first_order_contacts: list[LinkedInContact],
    second_order_contacts: list[LinkedInContact],
    messages: list[SecondOrderIntroMessage],
) -> list[SecondOrderIntroMessage]:
    connector_groups = {
        str(group["connector_key"]): group for group in _build_second_order_connector_groups(second_order_contacts)
    }
    templated_by_connector = {
        _normalize_name(message.first_order_contact_name): message
        for message in _template_second_order_messages(job, first_order_contacts, second_order_contacts)
    }
    finalized: list[SecondOrderIntroMessage] = []
    finalized_connector_keys: set[str] = set()
    for message in messages:
        first_key = _normalize_name(message.first_order_contact_name)
        if first_key not in connector_groups or first_key in finalized_connector_keys:
            continue
        connector_group = connector_groups[first_key]
        targets = [
            (str(target["name"]), str(target["profile_url"]))
            for target in connector_group["targets"]
        ]
        body = _append_required_target_links(
            message.message_body,
            job_url=str(job.direct_job_url),
            targets=targets,
        )
        finalized.append(
            message.model_copy(
                update={
                    "first_order_contact_name": str(connector_group["first_order_contact_name"]),
                    "first_order_contact_profile_url": connector_group["first_order_contact_profile_url"]
                    or message.first_order_contact_profile_url,
                    "second_order_contact_names": [name for name, _ in targets],
                    "second_order_contact_profile_urls": [url for _, url in targets],
                    "message_body": body,
                }
            )
        )
        finalized_connector_keys.add(first_key)

    for connector_key, template_message in templated_by_connector.items():
        if connector_key in finalized_connector_keys:
            continue
        finalized.append(template_message)
    finalized.sort(
        key=lambda message: _connector_group_priority(
            connector_groups.get(
                _normalize_name(message.first_order_contact_name),
                {
                    "message_history": [],
                    "first_order_contact_profile_url": message.first_order_contact_profile_url,
                    "first_order_contact_name": message.first_order_contact_name,
                },
            )
        )
    )
    return finalized


async def draft_first_order_messages(
    settings: Settings, job: JobPosting, contacts: list[LinkedInContact]
) -> list[FirstOrderMessage]:
    contacts = sorted(contacts, key=_contact_priority)
    if not contacts:
        return []

    prompt = f"""
Job:
- Company: {job.company_name}
- Title: {job.role_title}
- Job link: {job.direct_job_url}
- Salary: {job.salary_text or "Not specified"}
- Posted: {job.posted_date_text}

Contacts:
{json.dumps([contact.model_dump(mode="json") for contact in contacts], indent=2)}
""".strip()
    if settings.llm_provider == "ollama":
        provider = _build_ollama_provider(settings)
        system_prompt = """
You draft concise LinkedIn messages to first-degree contacts at a target company.
Requirements:
- Sound natural and specific, not spammy.
- Ask for quick insight on the role, team, or hiring context.
- Do not make the first message a blunt referral ask.
- Include the job link and recipient profile link in each message.
- Keep each draft short enough for LinkedIn and avoid bullet lists in the message body.
""".strip()
        try:
            output = await provider.generate_structured(
                system_prompt=system_prompt,
                user_prompt=prompt,
                schema=FirstOrderMessagesOutput,
            )
            return _finalize_first_order_messages(job, contacts, output.messages)
        except LLMProviderError:
            if settings.use_openai_fallback and settings.openai_api_key:
                agent = build_first_order_message_agent()
                result = await Runner.run(agent, prompt)
                return _finalize_first_order_messages(job, contacts, result.final_output.messages)
            return _finalize_first_order_messages(job, contacts, _template_first_order_messages(job, contacts))

    agent = build_first_order_message_agent()
    result = await Runner.run(agent, prompt)
    return _finalize_first_order_messages(job, contacts, result.final_output.messages)


async def draft_second_order_messages(
    settings: Settings,
    job: JobPosting,
    first_order_contacts: list[LinkedInContact],
    second_order_contacts: list[LinkedInContact],
) -> list[SecondOrderIntroMessage]:
    first_order_contacts = sorted(first_order_contacts, key=_contact_priority)
    second_order_contacts = [
        contact
        for contact in second_order_contacts
        if contact.connected_first_order_names or contact.connected_first_order_profile_urls
    ]
    if not second_order_contacts:
        return []

    prompt = f"""
Job:
- Company: {job.company_name}
- Title: {job.role_title}
- Job link: {job.direct_job_url}
- Salary: {job.salary_text or "Not specified"}
- Posted: {job.posted_date_text}

Known first-degree contacts:
{json.dumps([contact.model_dump(mode="json") for contact in first_order_contacts], indent=2)}

Connector groups:
{json.dumps(_build_second_order_connector_groups(second_order_contacts), indent=2)}

Write exactly one message per first-degree connector group.
Each message must cover every second-degree target listed for that connector group.
""".strip()
    if settings.llm_provider == "ollama":
        provider = _build_ollama_provider(settings)
        system_prompt = """
You draft concise messages asking a first-degree contact to introduce the sender to one or more second-degree contacts.
Requirements:
- Write one message per first-degree connector, not one message per target.
- Cover every listed second-degree target for that connector in the same message.
- Mention the exact role and why the target contacts are relevant.
- Include the job link and every target profile link in each message.
- If the chosen first-degree contact has prior message history, reference it naturally and match the tone.
- Ask for an intro only if the first-degree contact knows the target well enough.
- Keep each draft short enough for LinkedIn and avoid stiff boilerplate.
""".strip()
        try:
            output = await provider.generate_structured(
                system_prompt=system_prompt,
                user_prompt=prompt,
                schema=SecondOrderMessagesOutput,
            )
            return _finalize_second_order_messages(
                job,
                first_order_contacts,
                second_order_contacts,
                output.messages,
            )
        except LLMProviderError:
            if settings.use_openai_fallback and settings.openai_api_key:
                agent = build_second_order_message_agent()
                result = await Runner.run(agent, prompt)
                return _finalize_second_order_messages(
                    job,
                    first_order_contacts,
                    second_order_contacts,
                    result.final_output.messages,
                )
            templated = _template_second_order_messages(job, first_order_contacts, second_order_contacts)
            return _finalize_second_order_messages(job, first_order_contacts, second_order_contacts, templated)

    agent = build_second_order_message_agent()
    result = await Runner.run(agent, prompt)
    return _finalize_second_order_messages(
        job,
        first_order_contacts,
        second_order_contacts,
        result.final_output.messages,
    )


async def draft_outreach_bundle(
    settings: Settings,
    job: JobPosting,
    first_order_contacts: list[LinkedInContact],
    second_order_contacts: list[LinkedInContact],
) -> JobOutreachBundle:
    first_order_contacts = sorted(first_order_contacts, key=_contact_priority)
    first_order_messages, second_order_messages = await asyncio.gather(
        draft_first_order_messages(settings, job, first_order_contacts),
        draft_second_order_messages(settings, job, first_order_contacts, second_order_contacts),
    )
    return JobOutreachBundle(
        job=job,
        first_order_contacts=first_order_contacts,
        second_order_contacts=second_order_contacts,
        first_order_messages=first_order_messages,
        second_order_messages=second_order_messages,
    )
