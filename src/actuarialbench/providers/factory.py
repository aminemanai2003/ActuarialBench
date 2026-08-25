"""Construct provider adapters from frozen model configuration."""

from actuarialbench.providers.agentrouter import AgentRouterClient
from actuarialbench.providers.base import ProviderClient
from actuarialbench.providers.openrouter import OpenRouterClient
from actuarialbench.schemas import ModelConfig


def create_provider(config: ModelConfig, timeout_seconds: float) -> ProviderClient:
    if config.provider == "agentrouter":
        return AgentRouterClient(config, timeout_seconds)
    if config.provider == "openrouter":
        return OpenRouterClient(config, timeout_seconds)
    raise ValueError(f"Unsupported provider: {config.provider}")

