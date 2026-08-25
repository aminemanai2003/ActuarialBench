"""AgentRouter direct OpenAI- and Anthropic-compatible API adapters."""

from __future__ import annotations

import os
import time

from actuarialbench.providers.base import ProviderClient, ProviderError
from actuarialbench.schemas import ProviderResponse


class AgentRouterClient(ProviderClient):
    """Call the documented AgentRouter transport selected in model config."""

    def generate(self, *, system_prompt: str, user_prompt: str, max_tokens: int,
                 task_id: str, run_id: str) -> ProviderResponse:
        style = self.config.api_style or "openai"
        started = time.perf_counter()
        if style == "openai":
            payload, _ = self.post_json(
                f"{self.base_url()}/responses",
                {"model": self.config.model_id,
                 "input": self.combined_prompt(system_prompt, user_prompt),
                 "max_output_tokens": max_tokens},
            )
            text = _openai_text(payload)
            usage = payload.get("usage", {})
            finish_reason = payload.get("status")
            prompt_transport = "combined_single_text"
        elif style == "anthropic":
            payload, _ = self.post_json(
                f"{self.base_url()}/v1/messages",
                {"model": self.config.model_id, "system": system_prompt,
                 "messages": [{"role": "user", "content": user_prompt}],
                 "max_tokens": max_tokens, "temperature": 0},
            )
            text = _anthropic_text(payload)
            usage = payload.get("usage", {})
            finish_reason = payload.get("stop_reason")
            prompt_transport = "system_and_user"
        else:
            raise ProviderError(f"Unsupported AgentRouter API style: {style}")
        return ProviderResponse(
            model=self.config.name, provider="agentrouter", task_id=task_id,
            run_id=run_id, text=text,
            latency_seconds=time.perf_counter() - started,
            input_tokens=_integer_or_none(usage.get("input_tokens", usage.get("prompt_tokens"))),
            output_tokens=_integer_or_none(usage.get("output_tokens", usage.get("completion_tokens"))),
            finish_reason=finish_reason,
            api_metadata={"model_id": self.config.model_id,
                          "identity_status": self.config.identity_status,
                          "api_style": style, "base_url": self.base_url(),
                          "prompt_transport": prompt_transport,
                          "max_tokens_supported": True},
        )

    def base_url(self) -> str:
        style = self.config.api_style or "openai"
        env_name = self.config.base_url_env or "AGENTROUTER_BASE_URL"
        default = "https://agentrouter.org/v1" if style == "openai" else "https://agentrouter.org"
        return os.environ.get(env_name, default).rstrip("/")


def _openai_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str) and payload["output_text"].strip():
        return payload["output_text"].strip()
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                return content["text"]
    raise ProviderError("AgentRouter OpenAI response contained no text")


def _anthropic_text(payload: dict) -> str:
    text = "".join(item.get("text", "") for item in payload.get("content", [])
                    if item.get("type") == "text" and isinstance(item.get("text"), str)).strip()
    if not text:
        raise ProviderError("AgentRouter Anthropic response contained no text")
    return text


def _integer_or_none(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None
