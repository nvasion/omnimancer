"""
Pytest configuration and shared fixtures for Omnimancer tests.

This module provides common test fixtures, mock factories, and configuration
for all Omnimancer test modules.
"""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from omnimancer.core.models import (
    ChatContext,
    ChatMessage,
    ChatResponse,
    Config,
    ModelInfo,
    ProviderConfig,
    ChatSettings,
    MessageRole,
)
from omnimancer.core.engine import CoreEngine
from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.chat_manager import ChatManager
from omnimancer.core.conversation_manager import ConversationManager
from omnimancer.providers.base import BaseProvider
from omnimancer.cli.interface import CommandLineInterface


# Test Configuration Fixtures


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_config():
    """Create a test configuration object."""
    return Config(
        default_provider="openai",
        providers={
            "openai": ProviderConfig(
                api_key="test-openai-key",
                model="gpt-4",
                max_tokens=4096,
                temperature=0.7,
            ),
            "claude": ProviderConfig(
                api_key="test-claude-key",
                model="claude-3-sonnet",
                max_tokens=4096,
                temperature=0.5,
            ),
        },
        chat_settings=ChatSettings(
            max_tokens=4096, temperature=0.7, context_length=4000, save_history=True
        ),
        storage_path="/tmp/omnimancer_test",
    )


@pytest.fixture
def test_provider_configs():
    """Create test provider configurations."""
    return {
        "openai": {
            "api_key": "test-openai-key",
            "model": "gpt-4",
            "max_tokens": 4096,
            "temperature": 0.7,
        },
        "claude": {
            "api_key": "test-claude-key",
            "model": "claude-3-sonnet",
            "max_tokens": 4096,
            "temperature": 0.5,
        },
    }


# Mock Provider Fixtures


@pytest.fixture
def mock_openai_provider():
    """Create a mock OpenAI provider."""
    provider = MagicMock(spec=BaseProvider)
    provider.get_provider_name.return_value = "openai"
    provider.model = "gpt-4"
    provider.send_message = AsyncMock(
        return_value=ChatResponse(
            content="Hello! How can I help you?",
            model_used="gpt-4",
            tokens_used=15,
            cost_estimate=0.0003,
        )
    )
    provider.get_available_models.return_value = [
        ModelInfo(
            name="gpt-4",
            provider="openai",
            description="GPT-4 model",
            max_tokens=8192,
            cost_per_token=0.00003,
            available=True,
        ),
        ModelInfo(
            name="gpt-3.5-turbo",
            provider="openai",
            description="GPT-3.5 Turbo model",
            max_tokens=4096,
            cost_per_token=0.000002,
            available=True,
        ),
    ]
    provider.get_model_info.return_value = ModelInfo(
        name="gpt-4",
        provider="openai",
        description="GPT-4 model",
        max_tokens=8192,
        cost_per_token=0.00003,
        available=True,
    )
    provider.validate_credentials = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def mock_claude_provider():
    """Create a mock Claude provider."""
    provider = MagicMock(spec=BaseProvider)
    provider.get_provider_name.return_value = "claude"
    provider.model = "claude-3-sonnet"
    provider.send_message = AsyncMock(
        return_value=ChatResponse(
            content="Hello! I'm Claude, how can I assist you?",
            model_used="claude-3-sonnet",
            tokens_used=12,
            cost_estimate=0.00024,
        )
    )
    provider.get_available_models.return_value = [
        ModelInfo(
            name="claude-3-sonnet",
            provider="claude",
            description="Claude 3 Sonnet model",
            max_tokens=4096,
            cost_per_token=0.00002,
            available=True,
        ),
        ModelInfo(
            name="claude-3-haiku",
            provider="claude",
            description="Claude 3 Haiku model",
            max_tokens=4096,
            cost_per_token=0.000005,
            available=True,
        ),
    ]
    provider.get_model_info.return_value = ModelInfo(
        name="claude-3-sonnet",
        provider="claude",
        description="Claude 3 Sonnet model",
        max_tokens=4096,
        cost_per_token=0.00002,
        available=True,
    )
    provider.validate_credentials = AsyncMock(return_value=True)
    return provider


# Chat Context and Message Fixtures


@pytest.fixture
def sample_chat_messages():
    """Create sample chat messages for testing."""
    return [
        ChatMessage(
            role=MessageRole.USER,
            content="Hello, how are you?",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            model_used="",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Hello! I'm doing well, thank you for asking. How can I help you today?",
            timestamp=datetime(2024, 1, 1, 12, 0, 5),
            model_used="gpt-4",
        ),
        ChatMessage(
            role=MessageRole.USER,
            content="Can you explain quantum computing?",
            timestamp=datetime(2024, 1, 1, 12, 1, 0),
            model_used="",
        ),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Quantum computing is a type of computation that harnesses quantum mechanics...",
            timestamp=datetime(2024, 1, 1, 12, 1, 10),
            model_used="gpt-4",
        ),
    ]


@pytest.fixture
def sample_chat_context(sample_chat_messages):
    """Create a sample chat context with messages."""
    return ChatContext(
        messages=sample_chat_messages,
        current_model="gpt-4",
        session_id="test-session-123",
        max_context_length=4000,
    )


# Core Component Fixtures


@pytest.fixture
def mock_config_manager(test_config, temp_dir):
    """Create a mock configuration manager."""
    config_manager = MagicMock(spec=ConfigManager)
    config_manager.get_config.return_value = test_config
    config_manager.get_storage_path.return_value = temp_dir
    config_manager.set_default_provider = MagicMock()
    config_manager.update_provider_settings = MagicMock()
    config_manager.save_config = MagicMock()
    return config_manager


@pytest.fixture
def mock_chat_manager(sample_chat_context):
    """Create a mock chat manager."""
    chat_manager = MagicMock(spec=ChatManager)
    chat_manager.get_current_context.return_value = sample_chat_context
    chat_manager.add_user_message = MagicMock()
    chat_manager.add_assistant_message = MagicMock()
    chat_manager.clear_context = MagicMock()
    chat_manager.set_current_model = MagicMock()
    return chat_manager


@pytest.fixture
def mock_conversation_manager(temp_dir):
    """Create a mock conversation manager."""
    conv_manager = MagicMock(spec=ConversationManager)
    conv_manager.save_conversation.return_value = "conversation_20240101_120000.json"
    conv_manager.load_conversation.return_value = MagicMock()
    conv_manager.list_conversations.return_value = [
        {
            "filename": "conversation1.json",
            "created_at": "2024-01-01T12:00:00",
            "message_count": 5,
            "current_model": "gpt-4",
            "session_id": "session1",
        }
    ]
    conv_manager.delete_conversation.return_value = True
    conv_manager.get_conversation_info.return_value = {
        "filename": "test.json",
        "message_count": 5,
        "current_model": "gpt-4",
        "created_at": "2024-01-01T12:00:00",
    }
    return conv_manager


@pytest.fixture
def mock_engine(
    mock_config_manager,
    mock_chat_manager,
    mock_conversation_manager,
    mock_openai_provider,
    mock_claude_provider,
):
    """Create a mock core engine with all dependencies."""
    engine = MagicMock(spec=CoreEngine)
    engine.config_manager = mock_config_manager
    engine.chat_manager = mock_chat_manager
    engine.conversation_manager = mock_conversation_manager

    # Provider management
    engine.providers = {"openai": mock_openai_provider, "claude": mock_claude_provider}
    engine.current_provider = mock_openai_provider
    engine.register_provider = MagicMock()
    engine.set_current_provider = MagicMock()

    # Chat functionality
    engine.send_message = AsyncMock(
        return_value=ChatResponse(
            content="Test response",
            model_used="gpt-4",
            tokens_used=10,
            cost_estimate=0.0001,
        )
    )

    # Model management
    engine.get_available_models.return_value = [
        ModelInfo(
            name="gpt-4",
            provider="openai",
            description="GPT-4",
            max_tokens=8192,
            cost_per_token=0.00003,
            available=True,
        ),
        ModelInfo(
            name="claude-3-sonnet",
            provider="claude",
            description="Claude 3",
            max_tokens=4096,
            cost_per_token=0.00002,
            available=True,
        ),
    ]
    engine.get_current_model_info.return_value = ModelInfo(
        name="gpt-4",
        provider="openai",
        description="GPT-4",
        max_tokens=8192,
        cost_per_token=0.00003,
        available=True,
    )
    engine.switch_model = AsyncMock(return_value=True)

    # Conversation management
    engine.get_conversation_summary.return_value = {
        "message_count": 5,
        "current_model": "gpt-4",
        "session_id": "test-session",
    }
    engine.clear_conversation = MagicMock()
    engine.save_conversation = MagicMock(return_value="saved_conversation.json")
    engine.load_conversation = MagicMock()
    engine.list_conversations = MagicMock(return_value=[])
    engine.delete_conversation = MagicMock(return_value=True)
    engine.get_conversation_info = MagicMock(
        return_value={
            "filename": "test.json",
            "message_count": 5,
            "current_model": "gpt-4",
        }
    )

    # Initialization
    engine.initialize_providers = AsyncMock()
    engine.validate_current_provider = AsyncMock(return_value=True)

    return engine


@pytest.fixture
def mock_cli_interface(mock_engine):
    """Create a mock CLI interface."""
    return CommandLineInterface(mock_engine)


# Response Factory Functions


def create_chat_response(
    content: str = "Test response",
    model_used: str = "gpt-4",
    tokens_used: int = 10,
    cost_estimate: Optional[float] = 0.0001,
    error: Optional[str] = None,
) -> ChatResponse:
    """Factory function to create ChatResponse objects."""
    return ChatResponse(
        content=content,
        model_used=model_used,
        tokens_used=tokens_used,
        cost_estimate=cost_estimate,
        error=error,
    )


def create_model_info(
    name: str = "gpt-4",
    provider: str = "openai",
    description: str = "Test model",
    max_tokens: int = 4096,
    cost_per_token: float = 0.00002,
    available: bool = True,
) -> ModelInfo:
    """Factory function to create ModelInfo objects."""
    return ModelInfo(
        name=name,
        provider=provider,
        description=description,
        max_tokens=max_tokens,
        cost_per_token=cost_per_token,
        available=available,
    )


def create_chat_message(
    role: MessageRole = MessageRole.USER,
    content: str = "Test message",
    timestamp: Optional[datetime] = None,
    model_used: str = "",
) -> ChatMessage:
    """Factory function to create ChatMessage objects."""
    if timestamp is None:
        timestamp = datetime.now()

    return ChatMessage(
        role=role, content=content, timestamp=timestamp, model_used=model_used
    )


# Error Response Factories


@pytest.fixture
def api_error_response():
    """Create an API error response."""
    return ChatResponse(
        content="",
        model_used="gpt-4",
        tokens_used=0,
        error="API request failed: Rate limit exceeded",
    )


@pytest.fixture
def network_error_response():
    """Create a network error response."""
    return ChatResponse(
        content="",
        model_used="gpt-4",
        tokens_used=0,
        error="Network error: Connection timeout",
    )


@pytest.fixture
def auth_error_response():
    """Create an authentication error response."""
    return ChatResponse(
        content="",
        model_used="gpt-4",
        tokens_used=0,
        error="Authentication failed: Invalid API key",
    )


# Provider Mock Factories


class MockProviderFactory:
    """Factory for creating mock providers with different behaviors."""

    @staticmethod
    def create_working_provider(
        name: str = "test",
        model: str = "test-model",
        response_content: str = "Test response",
    ) -> MagicMock:
        """Create a mock provider that works normally."""
        provider = MagicMock(spec=BaseProvider)
        provider.get_provider_name.return_value = name
        provider.model = model
        provider.send_message = AsyncMock(
            return_value=create_chat_response(
                content=response_content, model_used=model
            )
        )
        provider.get_available_models.return_value = [
            create_model_info(name=model, provider=name)
        ]
        provider.get_model_info.return_value = create_model_info(
            name=model, provider=name
        )
        provider.validate_credentials = AsyncMock(return_value=True)
        return provider

    @staticmethod
    def create_failing_provider(
        name: str = "failing",
        model: str = "failing-model",
        error_message: str = "Provider error",
    ) -> MagicMock:
        """Create a mock provider that always fails."""
        provider = MagicMock(spec=BaseProvider)
        provider.get_provider_name.return_value = name
        provider.model = model
        provider.send_message = AsyncMock(
            return_value=create_chat_response(
                content="", model_used=model, error=error_message
            )
        )
        provider.get_available_models.side_effect = Exception(error_message)
        provider.get_model_info.side_effect = Exception(error_message)
        provider.validate_credentials = AsyncMock(return_value=False)
        return provider

    @staticmethod
    def create_rate_limited_provider(
        name: str = "rate_limited", model: str = "rate-limited-model"
    ) -> MagicMock:
        """Create a mock provider that returns rate limit errors."""
        provider = MagicMock(spec=BaseProvider)
        provider.get_provider_name.return_value = name
        provider.model = model
        provider.send_message = AsyncMock(
            return_value=create_chat_response(
                content="", model_used=model, error="Rate limit exceeded"
            )
        )
        provider.get_available_models.return_value = [
            create_model_info(name=model, provider=name)
        ]
        provider.get_model_info.return_value = create_model_info(
            name=model, provider=name
        )
        provider.validate_credentials = AsyncMock(return_value=True)
        return provider


@pytest.fixture
def mock_provider_factory():
    """Provide the MockProviderFactory for tests."""
    return MockProviderFactory


# Pytest Configuration


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "network: mark test as requiring network access")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if "slow" in item.nodeid or "load" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Mark network tests
        if "network" in item.nodeid or "api" in item.nodeid:
            item.add_marker(pytest.mark.network)


# Async Test Utilities


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Test Data Generators


def generate_conversation_data(message_count: int = 5) -> List[Dict]:
    """Generate test conversation data."""
    conversations = []
    for i in range(message_count):
        conversations.append(
            {
                "filename": f"conversation_{i}.json",
                "created_at": f"2024-01-0{i+1}T12:00:00",
                "message_count": (i + 1) * 2,
                "current_model": "gpt-4" if i % 2 == 0 else "claude-3-sonnet",
                "session_id": f"session-{i}",
            }
        )
    return conversations


def generate_model_list(provider_count: int = 2) -> List[ModelInfo]:
    """Generate a list of test models."""
    models = []
    providers = ["openai", "claude", "anthropic", "google"][:provider_count]

    for provider in providers:
        if provider == "openai":
            models.extend(
                [
                    create_model_info("gpt-4", provider, "GPT-4 model", 8192, 0.00003),
                    create_model_info(
                        "gpt-3.5-turbo", provider, "GPT-3.5 Turbo", 4096, 0.000002
                    ),
                ]
            )
        elif provider == "claude":
            models.extend(
                [
                    create_model_info(
                        "claude-3-sonnet", provider, "Claude 3 Sonnet", 4096, 0.00002
                    ),
                    create_model_info(
                        "claude-3-haiku", provider, "Claude 3 Haiku", 4096, 0.000005
                    ),
                ]
            )

    return models
