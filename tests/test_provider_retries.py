from __future__ import annotations

from actuarialbench.providers.base import ProviderHTTPError
from actuarialbench.runner import _generate_with_capture
from actuarialbench.tasks import build_tasks


class _RetryingProvider:
    def __init__(self):
        self.config = type("Config", (), {"provider": "test"})()
        self.calls = 0

    def generate(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise ProviderHTTPError(429, "temporarily throttled", retry_after=0)
        return type(
            "Response",
            (),
            {
                "error": None,
                "api_metadata": {},
                "experiment_id": "",
                "repetition": 0,
                "task_seed": None,
                "domain": None,
                "kind": None,
                "prompt_hash": None,
            },
        )()


def test_retryable_429_is_retried_without_exposing_key(monkeypatch) -> None:
    provider = _RetryingProvider()
    task = build_tasks(20260825)[0]
    response = _generate_with_capture(
        client=provider,
        model="test",
        task=task,
        system_prompt="system",
        max_tokens=10,
        run_id="run",
        experiment_id="experiment",
        repetition=0,
        retry_attempts=1,
        retry_delay_seconds=0,
    )
    assert provider.calls == 2
    assert response.error is None
