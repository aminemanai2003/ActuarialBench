"""AgentRouter MCP adapter with explicit route metadata."""

from __future__ import annotations

import os
import re
import time
from typing import Any

from actuarialbench.providers.base import ProviderClient, ProviderError
from actuarialbench.schemas import ProviderResponse

_TASK_ID = re.compile(r"TASK_ID:\s*([0-9a-f-]+)", re.IGNORECASE)
_RESULT = re.compile(r"RESULT:\s*(.*)\Z", re.IGNORECASE | re.DOTALL)


class AgentRouterClient(ProviderClient):
    """Call one configured AgentRouter MCP tool as a model route."""

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        task_id: str,
        run_id: str,
    ) -> ProviderResponse:
        del max_tokens  # AgentRouter's documented MCP route exposes no output-token cap.
        if not self.config.route_tool or not self.config.route_field:
            raise ProviderError(f"AgentRouter route is incomplete for {self.config.name}")

        started = time.perf_counter()
        prompt = self.combined_prompt(system_prompt, user_prompt)
        result_text = self._call_tool(
            self.config.route_tool,
            {"payload": {self.config.route_field: prompt}},
        )
        if "STATUS: RUNNING" in result_text.upper():
            match = _TASK_ID.search(result_text)
            if not match:
                raise ProviderError("AgentRouter returned RUNNING without a task ID")
            result_text = self._call_tool(
                "wait_for_task",
                {"task_id": match.group(1), "max_wait_seconds": self.timeout_seconds},
            )

        upper = result_text.upper()
        if "STATUS: FAILED" in upper or "ERROR:" in upper:
            raise ProviderError(f"AgentRouter route failed: {result_text[:300]}")
        match = _RESULT.search(result_text)
        text = match.group(1).strip() if match else result_text.strip()
        return ProviderResponse(
            model=self.config.name,
            provider="agentrouter",
            task_id=task_id,
            run_id=run_id,
            text=text,
            latency_seconds=time.perf_counter() - started,
            finish_reason="completed",
            api_metadata={
                "model_id": self.config.model_id,
                "identity_status": self.config.identity_status,
                "route_tool": self.config.route_tool,
                "prompt_transport": "combined_single_text",
                "max_tokens_supported": False,
            },
        )

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        base_url = os.environ.get(
            "AGENTROUTER_BASE_URL", "https://www.agent-router.org/mcp"
        ).rstrip("/")
        payload, _ = self.post_json(
            base_url,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            accept="application/json, text/event-stream",
        )
        if "error" in payload:
            raise ProviderError(f"AgentRouter MCP error: {payload['error']}")
        result = payload.get("result", {})
        structured = result.get("structuredContent", {}).get("result")
        if isinstance(structured, str):
            return structured
        content = result.get("content", [])
        if content and isinstance(content[0].get("text"), str):
            return content[0]["text"]
        raise ProviderError("AgentRouter returned no text result")

