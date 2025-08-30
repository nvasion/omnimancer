"""
Omnimancer - A unified command-line interface for multiple AI language models.

This package provides a cross-platform CLI tool that allows users to interact
with various AI providers (Claude, OpenAI, etc.) through a single interface.
"""

__version__ = "0.1.0"
__author__ = "Omnimancer Team"
__description__ = "Unified CLI for multiple AI language models"

from .cli.commands import (
    Command,
    CommandType,
    SlashCommand,
)
from .core.models import (
    ChatContext,
    ChatMessage,
    ChatResponse,
    Config,
    ModelInfo,
)
from .providers.base import BaseProvider
from .utils.errors import (
    OmnimancerError,
    ProviderError,
)

__all__ = [
    "BaseProvider",
    "ChatContext",
    "ChatMessage",
    "ChatResponse",
    "Command",
    "CommandType",
    "Config",
    "ModelInfo",
    "OmnimancerError",
    "ProviderError",
    "SlashCommand",
]
