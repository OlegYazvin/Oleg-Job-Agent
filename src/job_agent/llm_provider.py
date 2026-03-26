from __future__ import annotations

import asyncio
import json
from typing import TypeVar

import httpx
from pydantic import BaseModel

from .config import Settings
from .ollama_runtime import ensure_ollama_server, get_ollama_request_semaphore, record_ollama_event


TModel = TypeVar("TModel", bound=BaseModel)


class LLMProviderError(RuntimeError):
    pass


def _extract_json_payload(text: str) -> str:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:].strip()

    # Fast path: the whole payload is JSON.
    try:
        json.loads(candidate)
        return candidate
    except json.JSONDecodeError:
        pass

    # Fallback: grab the largest object/array-looking span.
    first_obj = candidate.find("{")
    first_arr = candidate.find("[")
    starts = [index for index in (first_obj, first_arr) if index >= 0]
    if not starts:
        raise LLMProviderError("Model output did not contain JSON.")
    start = min(starts)

    end_obj = candidate.rfind("}")
    end_arr = candidate.rfind("]")
    end = max(end_obj, end_arr)
    if end <= start:
        raise LLMProviderError("Model output contained malformed JSON.")
    return candidate[start : end + 1]


class OllamaStructuredProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.ollama_model
        self.timeout_seconds = settings.ollama_timeout_seconds

    def _build_payload(self, *, system_prompt: str, user_prompt: str, schema: type[TModel]) -> dict[str, object]:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=True)
        prompt = f"""
{system_prompt}

Return JSON only. Do not include markdown fences or extra commentary.
The JSON must validate against this schema:
{schema_json}

User request:
{user_prompt}
""".strip()
        options: dict[str, object] = {
            "temperature": 0.1,
            "num_ctx": self.settings.ollama_num_ctx,
            "num_batch": self.settings.ollama_num_batch,
        }
        if self.settings.ollama_num_predict > 0:
            options["num_predict"] = self.settings.ollama_num_predict
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.settings.ollama_keep_alive,
            "options": options,
        }

    @staticmethod
    def _is_retryable_request_failure(exc: BaseException) -> bool:
        if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.WriteError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code >= 500
        if isinstance(exc, httpx.TimeoutException):
            return True
        return False

    async def generate_structured(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[TModel],
    ) -> TModel:
        try:
            async with asyncio.timeout(self.timeout_seconds + 15):
                payload = self._build_payload(system_prompt=system_prompt, user_prompt=user_prompt, schema=schema)
                semaphore = get_ollama_request_semaphore(self.settings.ollama_max_concurrent_requests)

                async with semaphore:
                    await ensure_ollama_server(self.settings)
                    for attempt_number in range(1, self.settings.ollama_max_retries + 1):
                        try:
                            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                                response = await client.post(f"{self.base_url}/api/generate", json=payload)
                            response.raise_for_status()
                            response_json = response.json()
                            raw_text = str(response_json.get("response", "")).strip()
                            if not raw_text:
                                raise LLMProviderError("Ollama response body was empty.")
                            payload_text = _extract_json_payload(raw_text)
                            return schema.model_validate_json(payload_text)
                        except Exception as exc:  # pragma: no cover - network/runtime dependent
                            record_ollama_event(
                                self.settings,
                                "request_failure",
                                attempt_number=attempt_number,
                                model=self.model,
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                                retryable=self._is_retryable_request_failure(exc),
                            )
                            if (
                                self.settings.ollama_restart_on_failure
                                and attempt_number < self.settings.ollama_max_retries
                                and self._is_retryable_request_failure(exc)
                            ):
                                record_ollama_event(
                                    self.settings,
                                    "request_retry_restart",
                                    attempt_number=attempt_number,
                                    model=self.model,
                                )
                                await ensure_ollama_server(self.settings, force_restart=True)
                                await asyncio.sleep(self.settings.ollama_retry_backoff_seconds)
                                continue
                            if isinstance(exc, LLMProviderError):
                                raise
                            raise LLMProviderError(f"Ollama request failed: {exc}") from exc
        except TimeoutError as exc:
            record_ollama_event(
                self.settings,
                "request_outer_timeout",
                model=self.model,
                timeout_seconds=self.timeout_seconds + 15,
            )
            raise LLMProviderError(
                f"Ollama request exceeded the outer timeout of {self.timeout_seconds + 15} seconds."
            ) from exc

        raise LLMProviderError("Ollama request failed after exhausting retries.")
