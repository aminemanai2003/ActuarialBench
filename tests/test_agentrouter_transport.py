from __future__ import annotations

from actuarialbench.config import load_model_configs
from actuarialbench.providers.agentrouter import (
    CLAUDE_CODE_COMPATIBLE_USER_AGENT,
    CODEX_COMPATIBLE_USER_AGENT,
    AgentRouterClient,
)
from actuarialbench.schemas import ModelConfig


def _client(style: str, reasoning_effort: str | None = None) -> AgentRouterClient:
    return AgentRouterClient(ModelConfig(
        name="probe", provider="agentrouter", model_id="model-x",
        api_key_env="TEST_KEY", identity_status="externally_asserted",
        api_style=style, base_url_env="TEST_BASE",
        reasoning_effort=reasoning_effort,
    ), 5)


def test_configured_models_use_agentrouter_compatible_transports():
    models = load_model_configs()
    assert {model.model_id for model in models} == {
        "gpt-5.6-sol",
        "deepseek-v4-flash",
        "glm-5.3",
        "claude-opus-5",
    }
    assert all(model.provider == "agentrouter" for model in models)
    assert {model.model_id: model.api_style for model in models} == {
        "gpt-5.6-sol": "openai",
        "deepseek-v4-flash": "openai",
        "glm-5.3": "openai_chat",
        "claude-opus-5": "anthropic",
    }
    assert {model.model_id: model.reasoning_effort for model in models} == {
        "gpt-5.6-sol": None,
        "deepseek-v4-flash": "low",
        "glm-5.3": "low",
        "claude-opus-5": None,
    }


def test_openai_responses_transport(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "redacted")
    monkeypatch.setenv("TEST_BASE", "https://example.test/v1")
    client = _client("openai", reasoning_effort="low")
    seen = {}
    monkeypatch.setattr(
        client,
        "post_json",
        lambda url, payload, **kwargs: (
            seen.update(url=url, payload=payload, kwargs=kwargs)
            or (
                {
                    "output": [
                        {
                            "type": "reasoning",
                            "content": [
                                {"type": "reasoning_text", "text": "thinking"}
                            ],
                        },
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "pong"}
                            ],
                        },
                    ],
                    "usage": {"input_tokens": 2, "output_tokens": 1},
                },
                {},
            )
        ),
    )
    response = client.generate(system_prompt="S", user_prompt="U", max_tokens=7, task_id="t", run_id="r")
    assert seen["url"] == "https://example.test/v1/responses"
    assert seen["payload"]["model"] == "model-x"
    assert seen["payload"]["reasoning"] == {"effort": "low"}
    assert seen["kwargs"]["user_agent"] == CODEX_COMPATIBLE_USER_AGENT
    assert response.text == "pong"
    assert response.output_tokens == 1


def test_openai_chat_completions_transport(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "redacted")
    monkeypatch.setenv("TEST_BASE", "https://example.test/v1")
    client = _client("openai_chat", reasoning_effort="low")
    seen = {}
    monkeypatch.setattr(
        client,
        "post_json",
        lambda url, payload, **kwargs: (
            seen.update(url=url, payload=payload, kwargs=kwargs)
            or (
                {
                    "choices": [
                        {
                            "message": {"content": "pong"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                },
                {},
            )
        ),
    )
    response = client.generate(
        system_prompt="S", user_prompt="U", max_tokens=7, task_id="t", run_id="r"
    )
    assert seen["url"] == "https://example.test/v1/chat/completions"
    assert seen["payload"]["model"] == "model-x"
    assert seen["payload"]["reasoning_effort"] == "low"
    assert seen["kwargs"]["user_agent"] == CODEX_COMPATIBLE_USER_AGENT
    assert response.text == "pong"
    assert response.output_tokens == 1


def test_anthropic_messages_transport(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "redacted")
    monkeypatch.setenv("TEST_BASE", "https://example.test")
    client = _client("anthropic")
    seen = {}
    monkeypatch.setattr(
        client,
        "post_json",
        lambda url, payload, **kwargs: (
            seen.update(url=url, payload=payload, kwargs=kwargs)
            or (
                {
                    "content": [{"type": "text", "text": "pong"}],
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
                {},
            )
        ),
    )
    response = client.generate(
        system_prompt="S", user_prompt="U", max_tokens=7, task_id="t", run_id="r"
    )
    assert seen["url"] == "https://example.test/v1/messages"
    assert seen["payload"]["system"] == "S"
    assert seen["kwargs"]["user_agent"] == CLAUDE_CODE_COMPATIBLE_USER_AGENT
    assert seen["kwargs"]["extra_headers"]["x-app"] == "cli"
    assert response.text == "pong"
