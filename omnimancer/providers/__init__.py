"""
Providers module for Omnimancer.

This module contains AI provider implementations and the base provider interface.
Supports multiple AI services including Claude, OpenAI, Gemini, Cohere, and Ollama.
"""

from .base import BaseProvider
from .claude import ClaudeProvider
from .openai import OpenAIProvider
from .gemini import GeminiProvider
from .cohere import CohereProvider
from .ollama import OllamaProvider
from .factory import ProviderFactory

__all__ = [
    "BaseProvider",
    "ClaudeProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "CohereProvider",
    "OllamaProvider",
    "ProviderFactory",
]
