"""
Providers module for Omnimancer.

This module contains AI provider implementations and the base provider interface.
Every provider registered in :class:`ProviderFactory` is also re-exported here so
the public package API stays in sync with what the engine actually supports.
"""

from .azure import AzureProvider
from .base import BaseProvider
from .bedrock import BedrockProvider
from .claude import ClaudeProvider
from .claude_code import ClaudeCodeProvider
from .cohere import CohereProvider
from .digitalocean import DigitalOceanProvider
from .factory import ProviderFactory
from .gemini import GeminiProvider
from .mistral import MistralProvider
from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .openai_compatible import OpenAICompatibleProvider
from .openrouter import OpenRouterProvider
from .perplexity import PerplexityProvider
from .vertex import VertexAIProvider
from .xai import XAIProvider

__all__ = [
    "BaseProvider",
    "ProviderFactory",
    "ClaudeProvider",
    "ClaudeCodeProvider",
    "OpenAIProvider",
    "OpenAICompatibleProvider",
    "GeminiProvider",
    "CohereProvider",
    "OllamaProvider",
    "PerplexityProvider",
    "XAIProvider",
    "MistralProvider",
    "AzureProvider",
    "VertexAIProvider",
    "BedrockProvider",
    "OpenRouterProvider",
    "DigitalOceanProvider",
]
