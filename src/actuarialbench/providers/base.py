"""Common provider interface and HTTP utilities."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from actuarialbench.schemas import ModelConfig, ProviderResponse


class ProviderError(RuntimeError):
    """Raised when a provider request cannot produce a model response."""


class ProviderClient(ABC):
    """Lowest-common-denominator text generation interface."""

    def __init__(self, config: ModelConfig, timeout_seconds: float) -> None:
        self.config = config
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        task_id: str,
        run_id: str,
    ) -> ProviderResponse:
        """Generate one response in a fresh provider context."""

    def api_key(self) -> str:
        key = os.environ.get(self.config.api_key_env, "").strip()
        if not key:
            raise ProviderError(f"Missing required environment variable: {self.config.api_key_env}")
        return key

    @staticmethod
    def combined_prompt(system_prompt: str, user_prompt: str) -> str:
        """Use one text transport because AgentRouter has no equivalent system role."""

        return (
            "SYSTEM INSTRUCTIONS\n"
            f"{system_prompt.strip()}\n\n"
            "USER TASK\n"
            f"{user_prompt.strip()}"
        )

    def post_json(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        accept: str = "application/json",
    ) -> tuple[dict[str, Any], dict[str, str]]:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key()}",
                "Accept": accept,
                "Content-Type": "application/json",
                "User-Agent": "ActuarialBench/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
                headers = {key.lower(): value for key, value in response.headers.items()}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise ProviderError(f"HTTP {exc.code} from provider: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProviderError(f"Provider request failed: {exc}") from exc

        try:
            return json.loads(raw), headers
        except json.JSONDecodeError as exc:
            raise ProviderError("Provider returned non-JSON content") from exc
