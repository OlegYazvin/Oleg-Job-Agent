from pathlib import Path
import asyncio
import json

import httpx
from pydantic import BaseModel

from job_agent.config import Settings
from job_agent.llm_provider import OllamaStructuredProvider, _extract_json_payload
from job_agent.ollama_runtime import record_ollama_event


class ExampleSchema(BaseModel):
    value: int


def test_extract_json_payload_handles_plain_json() -> None:
    assert _extract_json_payload('{"value": 5}') == '{"value": 5}'


def test_extract_json_payload_handles_code_fence() -> None:
    payload = _extract_json_payload("```json\n{\"value\": 7}\n```")
    assert ExampleSchema.model_validate_json(payload).value == 7


def test_extract_json_payload_handles_wrapped_text() -> None:
    payload = _extract_json_payload("Here is result:\n{\"value\": 11}\nThanks")
    assert ExampleSchema.model_validate_json(payload).value == 11


def test_ollama_provider_builds_low_memory_payload() -> None:
    root = Path("/tmp/job-agent-tests")
    settings = Settings(
        project_root=root,
        openai_api_key="",
        linkedin_email=None,
        linkedin_password=None,
        linkedin_totp_secret=None,
        linkedin_li_at=None,
        linkedin_jsessionid=None,
        google_email=None,
        google_password=None,
        google_totp_secret=None,
        browser_executable_path=None,
        browser_channel=None,
        linkedin_profile_dir=root / ".secrets/profile",
        linkedin_storage_state=root / ".secrets/state.json",
        output_dir=root / "output",
        data_dir=root / "data",
        headless=True,
        timezone="America/Chicago",
        search_country="US",
        search_city="Chicago",
        search_region="Illinois",
        min_base_salary_usd=200000,
        posted_within_days=14,
        minimum_qualifying_jobs=10,
        target_job_count=10,
        max_adaptive_search_passes=3,
        max_search_rounds=3,
        search_round_query_limit=6,
        max_leads_per_query=6,
        max_leads_to_resolve_per_pass=60,
        reacquisition_attempt_cap=10,
        per_query_timeout_seconds=35,
        per_lead_timeout_seconds=25,
        workflow_timeout_seconds=3600,
        max_linkedin_results_per_company=25,
        max_linkedin_pages_per_company=3,
        daily_run_hour=8,
        daily_run_minute=0,
        status_heartbeat_seconds=120,
        enable_progress_gui=False,
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5:14b-instruct",
        ollama_keep_alive="0m",
        ollama_num_ctx=1024,
        ollama_num_batch=8,
        ollama_num_predict=256,
    )
    provider = OllamaStructuredProvider(settings)

    payload = provider._build_payload(
        system_prompt="Return JSON.",
        user_prompt="Test prompt",
        schema=ExampleSchema,
    )

    assert payload["keep_alive"] == "0m"
    assert payload["options"]["num_ctx"] == 1024
    assert payload["options"]["num_batch"] == 8
    assert payload["options"]["num_predict"] == 256


def test_record_ollama_event_writes_jsonl(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        openai_api_key="",
        linkedin_email=None,
        linkedin_password=None,
        linkedin_totp_secret=None,
        linkedin_li_at=None,
        linkedin_jsessionid=None,
        google_email=None,
        google_password=None,
        google_totp_secret=None,
        browser_executable_path=None,
        browser_channel=None,
        linkedin_profile_dir=tmp_path / ".secrets/profile",
        linkedin_storage_state=tmp_path / ".secrets/state.json",
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
        headless=True,
        timezone="America/Chicago",
        search_country="US",
        search_city="Chicago",
        search_region="Illinois",
        min_base_salary_usd=200000,
        posted_within_days=14,
        minimum_qualifying_jobs=10,
        target_job_count=10,
        max_adaptive_search_passes=3,
        max_search_rounds=3,
        search_round_query_limit=6,
        max_leads_per_query=6,
        max_leads_to_resolve_per_pass=60,
        reacquisition_attempt_cap=10,
        per_query_timeout_seconds=35,
        per_lead_timeout_seconds=25,
        workflow_timeout_seconds=3600,
        max_linkedin_results_per_company=25,
        max_linkedin_pages_per_company=3,
        daily_run_hour=8,
        daily_run_minute=0,
        status_heartbeat_seconds=120,
        enable_progress_gui=False,
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5:14b-instruct",
    )

    record_ollama_event(settings, "managed_process_missing", pid=1234, reason="health check failed")

    event_log = settings.output_dir / "ollama-events.jsonl"
    entries = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()]
    assert entries[-1]["event"] == "managed_process_missing"
    assert entries[-1]["pid"] == 1234


def test_ollama_provider_records_success_metrics(monkeypatch, tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        openai_api_key="",
        linkedin_email=None,
        linkedin_password=None,
        linkedin_totp_secret=None,
        linkedin_li_at=None,
        linkedin_jsessionid=None,
        google_email=None,
        google_password=None,
        google_totp_secret=None,
        browser_executable_path=None,
        browser_channel=None,
        linkedin_profile_dir=tmp_path / ".secrets/profile",
        linkedin_storage_state=tmp_path / ".secrets/state.json",
        output_dir=tmp_path / "output",
        data_dir=tmp_path / "data",
        headless=True,
        timezone="America/Chicago",
        search_country="US",
        search_city="Chicago",
        search_region="Illinois",
        min_base_salary_usd=200000,
        posted_within_days=14,
        minimum_qualifying_jobs=10,
        target_job_count=10,
        max_adaptive_search_passes=3,
        max_search_rounds=3,
        search_round_query_limit=6,
        max_leads_per_query=6,
        max_leads_to_resolve_per_pass=60,
        reacquisition_attempt_cap=10,
        per_query_timeout_seconds=35,
        per_lead_timeout_seconds=25,
        workflow_timeout_seconds=3600,
        max_linkedin_results_per_company=25,
        max_linkedin_pages_per_company=3,
        daily_run_hour=8,
        daily_run_minute=0,
        status_heartbeat_seconds=120,
        enable_progress_gui=False,
        llm_provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5:7b-instruct",
        ollama_keep_alive="5m",
        ollama_num_ctx=1024,
        ollama_num_batch=4,
        ollama_num_predict=256,
    )
    provider = OllamaStructuredProvider(settings)

    async def fake_ensure_ollama_server(_settings: Settings, *, force_restart: bool = False) -> None:
        return None

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "response": "{\"value\": 7}",
                "total_duration": 5_000_000_000,
                "load_duration": 750_000_000,
                "prompt_eval_count": 120,
                "prompt_eval_duration": 2_000_000_000,
                "eval_count": 40,
                "eval_duration": 1_500_000_000,
            }

    class FakeAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: dict[str, object]) -> FakeResponse:
            assert url == "http://localhost:11434/api/generate"
            assert json["model"] == "qwen2.5:7b-instruct"
            return FakeResponse()

    monkeypatch.setattr("job_agent.llm_provider.ensure_ollama_server", fake_ensure_ollama_server)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(
        provider.generate_structured(
            system_prompt="Return JSON.",
            user_prompt="Test prompt",
            schema=ExampleSchema,
        )
    )

    assert result.value == 7
    event_log = settings.output_dir / "ollama-events.jsonl"
    entries = [json.loads(line) for line in event_log.read_text(encoding="utf-8").splitlines()]
    success_events = [entry for entry in entries if entry["event"] == "request_success"]
    assert success_events
    assert success_events[-1]["model"] == "qwen2.5:7b-instruct"
    assert success_events[-1]["keep_alive"] == "5m"
    assert success_events[-1]["load_duration_seconds"] == 0.75
    assert success_events[-1]["prompt_eval_duration_seconds"] == 2.0
    assert success_events[-1]["eval_duration_seconds"] == 1.5
    assert success_events[-1]["cold_start"] is True
