"""
Core module for Omnimancer.

This module contains the core business logic including the chat engine,
configuration management, and session handling.
"""

from ..providers.base import BaseProvider
from ..utils.errors import (
    OmnimancerError,
    ProviderError,
)
from .models import (
    ChatContext,
    ChatMessage,
    ChatResponse,
    Config,
    EnhancedModelInfo,
    ModelInfo,
)
from .provider_registry import ProviderRegistry

__all__ = [
    "BaseProvider",
    "ChatContext",
    "ChatMessage",
    "ChatResponse",
    "Config",
    "ModelInfo",
    "EnhancedModelInfo",
    "ProviderRegistry",
    "OmnimancerError",
    "ProviderError",
]
