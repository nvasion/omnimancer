"""
Core module for Omnimancer.

This module contains the core business logic including the chat engine,
configuration management, and session handling.
"""

from .models import (
    ChatContext,
    ChatMessage,
    ChatResponse,
    Config,
    ModelInfo,
    EnhancedModelInfo,
)
from .provider_registry import ProviderRegistry
from ..utils.errors import (
    OmnimancerError,
    ProviderError,
)
from ..providers.base import BaseProvider

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