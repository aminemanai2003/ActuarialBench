"""OpenRouter chat-completions adapter."""

from __future__ import annotations

import os
import time

from actuarialbench.providers.base import ProviderClient, ProviderError
from actuarialbench.schemas import ProviderResponse


class OpenRouterClient(ProviderClient):
    """Generate text through a concrete OpenRouter catalog model ID."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        task_id: str,
        run_id: str,
    ) -> ProviderResponse:
        base_url = os.environ.get(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/")
        started = time.perf_counter()
        payload, _ = self.post_json(
            f"{base_url}/chat/completions",
            {
                "model": self.config.model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": self.combined_prompt(system_prompt, user_prompt),
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
        )
        choices = payload.get("choices", [])
        if not choices:
            raise ProviderError(f"OpenRouter returned no choices: {payload.get('error', payload)}")
        choice = choices[0]
        usage = payload.get("usage", {})
        text = _content_to_text(choice.get("message", {}).get("content", ""))
        return ProviderResponse(
            model=self.config.name,
            provider="openrouter",
            task_id=task_id,
            run_id=run_id,
            text=text,
            latency_seconds=time.perf_counter() - started,
            input_tokens=_integer_or_none(usage.get("prompt_tokens")),
            output_tokens=_integer_or_none(usage.get("completion_tokens")),
            reported_cost_usd=_float_or_none(usage.get("cost")),
            finish_reason=choice.get("finish_reason"),
            api_metadata={
                "model_id": payload.get("model", self.config.model_id),
                "identity_status": self.config.identity_status,
                "prompt_transport": "combined_single_text",
                "max_tokens_supported": True,
                "generation_id": payload.get("id"),
            },
        )


def _integer_or_none(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def _float_or_none(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _content_to_text(content: object) -> str:
    """Normalize OpenAI-compatible string or content-block message payloads."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text", ""), str)
        ]
        text = "".join(parts)
        if text:
            return text
    if content is None:
        raise ProviderError("OpenRouter returned no visible text (the response may have exhausted its reasoning/output limit)")
    raise ProviderError("OpenRouter returned non-text message content")
