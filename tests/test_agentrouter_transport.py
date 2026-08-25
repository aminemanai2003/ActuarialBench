from __future__ import annotations

from actuarialbench.providers.agentrouter import AgentRouterClient
from actuarialbench.schemas import ModelConfig


def _client(style: str) -> AgentRouterClient:
    return AgentRouterClient(ModelConfig(
        name="probe", provider="agentrouter", model_id="model-x",
        api_key_env="TEST_KEY", identity_status="externally_asserted",
        api_style=style, base_url_env="TEST_BASE",
    ), 5)


def test_openai_responses_transport(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "redacted")
    monkeypatch.setenv("TEST_BASE", "https://example.test/v1")
    client = _client("openai")
    seen = {}
    monkeypatch.setattr(client, "post_json", lambda url, payload: (seen.update(url=url, payload=payload) or ({"output_text": "pong", "usage": {"input_tokens": 2, "output_tokens": 1}}, {})))
    response = client.generate(system_prompt="S", user_prompt="U", max_tokens=7, task_id="t", run_id="r")
    assert seen["url"] == "https://example.test/v1/responses"
    assert seen["payload"]["model"] == "model-x"
    assert response.text == "pong"
    assert response.output_tokens == 1


def test_anthropic_messages_transport(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "redacted")
    monkeypatch.setenv("TEST_BASE", "https://example.test")
    client = _client("anthropic")
    seen = {}
    monkeypatch.setattr(client, "post_json", lambda url, payload: (seen.update(url=url, payload=payload) or ({"content": [{"type": "text", "text": "pong"}], "usage": {"input_tokens": 3, "output_tokens": 1}}, {})))
    response = client.generate(system_prompt="S", user_prompt="U", max_tokens=7, task_id="t", run_id="r")
    assert seen["url"] == "https://example.test/v1/messages"
    assert seen["payload"]["system"] == "S"
    assert response.text == "pong"

