"""
Omnimancer - A unified command-line interface for multiple AI language models.

This package provides a cross-platform CLI tool that allows users to interact
with various AI providers (Claude, OpenAI, etc.) through a single interface.
"""

try:
    from importlib.metadata import version as _get_version

    __version__ = _get_version("omnimancer-cli")
except Exception:
    with open("pyproject.toml", "r") as f:
        project = f.readlines()
    version = [i for i in project if "version" in i].pop(0).strip()
    __version__ = version.split().pop().strip("\"'")
__author__ = "Omnimancer Team"
__description__ = "Unified CLI for multiple AI language models"

from .cli.commands import Command, CommandType, SlashCommand
from .core.models import ChatContext, ChatMessage, ChatResponse, Config, ModelInfo
from .providers.base import BaseProvider
from .utils.errors import OmnimancerError, ProviderError

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
