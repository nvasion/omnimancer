"""
Claude-code provider implementation for Omnimancer.

This module provides the Claude-code provider implementation for local Claude-code integration
with support for opus and sonnet modes with free access.
"""

import os
import subprocess
from datetime import datetime
from typing import List


from ..core.models import ChatContext, ChatResponse, EnhancedModelInfo
from ..utils.errors import (
    AuthenticationError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
)
from .base import BaseProvider


class ClaudeCodeProvider(BaseProvider):
    """
    Claude-code provider implementation for local Claude-code integration.

    Provides integration with existing Claude-code installations,
    supporting opus and sonnet modes with free local access.
    """

    def __init__(self, api_key: str = "local", model: str = "", **kwargs):
        """
        Initialize Claude-code provider.

        Args:
            api_key: Not used for Claude-code (local installation), defaults to "local"
            model: Claude-code model mode (e.g., 'claude-code-sonnet', 'claude-code-opus', 'claude-code-haiku')
            **kwargs: Additional configuration including Claude-code specific settings
        """
        super().__init__(api_key or "local", model or "claude-code-sonnet", **kwargs)

        # Claude-code specific configuration
        self.claude_code_mode = kwargs.get(
            "claude_code_mode", self._extract_mode_from_model()
        )
        self.claude_code_path = kwargs.get("claude_code_path", "claude")
        self.working_directory = kwargs.get("working_directory", os.getcwd())

        # Standard parameters (not all may be supported by claude-code)
        self.max_tokens = kwargs.get("max_tokens", 4096)
        self.temperature = kwargs.get("temperature", 0.7)

        # Validate Claude-code installation
        self._validate_installation()

    def _extract_mode_from_model(self) -> str:
        """
        Extract mode from model name.

        Returns:
            Mode string (opus or sonnet)
        """
        if "opus" in self.model.lower():
            return "opus"
        elif "sonnet" in self.model.lower():
            return "sonnet"
        else:
            return "sonnet"  # Default to sonnet

    def _validate_installation(self):
        """
        Validate that Claude-code is installed and accessible.

        Raises:
            ProviderError: If Claude-code is not found or not working
        """
        try:
            # Try to run claude with version command
            result = subprocess.run(
                [self.claude_code_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                # Try alternative command
                result = subprocess.run(
                    [self.claude_code_path, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode != 0:
                    raise ProviderError(
                        f"Claude command not found or not working at: {self.claude_code_path}"
                    )

        except subprocess.TimeoutExpired:
            raise ProviderError("Claude command timed out")
        except FileNotFoundError:
            raise ProviderError(f"Claude executable not found: {self.claude_code_path}")
        except Exception as e:
            raise ProviderError(f"Error validating Claude installation: {e}")

    async def send_message(self, message: str, context: ChatContext) -> ChatResponse:
        """
        Send a message to Claude-code.

        Args:
            message: User message
            context: Conversation context

        Returns:
            ChatResponse with Claude-code's reply
        """

        try:
            # Prepare the conversation for Claude-code
            conversation = self._prepare_conversation(message, context)
            result = await self._execute_claude_code(conversation)
            response = self._handle_claude_code_response(result)
            return response

        except Exception as e:
            raise ProviderError(f"Error communicating with Claude-code: {e}")

    async def validate_credentials(self) -> bool:
        """
        Validate Claude installation by running a test command.

        Returns:
            True if Claude is working
        """
        try:
            # Test with a simple message
            test_result = await self._execute_claude_code("Hi")
            return test_result.returncode == 0 and test_result.stdout.strip() != ""
        except Exception:
            return False

    def _prepare_conversation(self, message: str, context: ChatContext) -> str:
        """
        Prepare conversation for Claude-code.

        Args:
            message: Current user message
            context: Conversation context

        Returns:
            Formatted conversation string
        """
        conversation_parts = []

        # Add context messages
        for msg in context.messages:
            if msg.role.value == "user":
                conversation_parts.append(f"Human: {msg.content}")
            elif msg.role.value == "assistant":
                conversation_parts.append(f"Assistant: {msg.content}")
            elif msg.role.value == "system":
                conversation_parts.append(f"System: {msg.content}")

        # Add current message
        conversation_parts.append(f"Human: {message}")

        return "\n\n".join(conversation_parts)

    async def _execute_claude_code(
        self, conversation: str
    ) -> subprocess.CompletedProcess:
        """
        Execute Claude with the conversation.

        Args:
            conversation: Formatted conversation string

        Returns:
            Subprocess result
        """
        # Build command arguments - use --print for non-interactive mode
        cmd = [self.claude_code_path, "--print"]

        # Add model if supported
        if hasattr(self, "claude_code_mode") and self.claude_code_mode:
            if self.claude_code_mode == "opus":
                cmd.extend(["--model", "opus"])
            elif self.claude_code_mode == "sonnet":
                cmd.extend(["--model", "sonnet"])

        # Add the conversation as the prompt
        cmd.append(conversation)

        # Execute the command
        try:
            # Run in an executor to avoid blocking
            import asyncio

            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout for longer responses
                    cwd=self.working_directory,
                ),
            )

            return result

        except subprocess.TimeoutExpired:
            raise ProviderError("Claude command timed out")
        except Exception as e:
            raise ProviderError(f"Error executing Claude: {e}")

    def _handle_claude_code_response(
        self, result: subprocess.CompletedProcess
    ) -> ChatResponse:
        """
        Handle Claude-code command result.

        Args:
            result: Subprocess result from Claude-code

        Returns:
            ChatResponse object

        Raises:
            ProviderError: If Claude-code returned an error
        """
        if result.returncode == 0:
            # Successful execution
            output = result.stdout.strip()

            if not output:
                raise ProviderError("Empty response from Claude-code")

            # Claude-code might include metadata or formatting
            # Try to extract just the assistant's response
            response_content = self._extract_response_content(output)

            return ChatResponse(
                content=response_content,
                model_used=self.model,
                tokens_used=0,  # Claude-code doesn't provide token counts
                timestamp=datetime.now(),
            )
        else:
            # Error occurred
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"

            # Check for common error patterns
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                raise AuthenticationError(
                    f"Claude-code authentication error: {error_msg}"
                )
            elif "rate limit" in error_msg.lower():
                raise RateLimitError(f"Claude-code rate limit: {error_msg}")
            elif "not found" in error_msg.lower():
                raise ModelNotFoundError(f"Claude-code model not found: {error_msg}")
            else:
                raise ProviderError(f"Claude-code error: {error_msg}")

    def _extract_response_content(self, output: str) -> str:
        """
        Extract the actual response content from Claude output.

        Args:
            output: Raw output from Claude --print

        Returns:
            Cleaned response content
        """
        # Claude --print gives us clean output directly
        # Just return the output stripped of whitespace
        return output.strip()

    def get_model_info(self) -> EnhancedModelInfo:
        """
        Get information about the current Claude-code model.
        """
        model_configs = {
            "claude-code-opus": {
                "description": "Claude-code Opus - Local free access to Claude Opus",
                "max_tokens": 200000,
                "swe_score": 84.9,
                "supports_tools": False,
                "supports_multimodal": True,
            },
            "claude-code-sonnet": {
                "description": "Claude-code Sonnet - Local free access to Claude Sonnet",
                "max_tokens": 200000,
                "swe_score": 73.0,
                "supports_tools": False,
                "supports_multimodal": True,
            },
        }

        config = model_configs.get(
            self.model,
            {
                "description": f"Claude-code model {self.model}",
                "max_tokens": 200000,
                "swe_score": 75.0,
                "supports_tools": False,
                "supports_multimodal": True,
            },
        )

        enhanced_info = EnhancedModelInfo(
            name=self.model,
            provider="claude-code",
            description=config["description"],
            max_tokens=config["max_tokens"],
            cost_per_million_input=0.0,  # Free local access
            cost_per_million_output=0.0,  # Free local access
            swe_score=config["swe_score"],
            available=True,
            supports_tools=config["supports_tools"],
            supports_multimodal=config["supports_multimodal"],
            latest_version=self.model == "claude-code-opus",
            context_window=config["max_tokens"],
            is_free=True,  # Local installation is free
            release_date=datetime(2024, 3, 1),  # Approximate availability
        )

        # Update SWE rating based on score
        enhanced_info.update_swe_rating()

        return enhanced_info

    def get_available_models(self) -> List[EnhancedModelInfo]:
        """
        Get list of available Claude-code models.
        """
        models = [
            EnhancedModelInfo(
                name="claude-code-opus",
                provider="claude-code",
                description="Claude-code Opus - Local free access to Claude Opus",
                max_tokens=200000,
                cost_per_million_input=0.0,
                cost_per_million_output=0.0,
                swe_score=84.9,
                available=True,
                supports_tools=False,
                supports_multimodal=True,
                latest_version=True,
                context_window=200000,
                is_free=True,
                release_date=datetime(2024, 3, 1),
            ),
            EnhancedModelInfo(
                name="claude-code-sonnet",
                provider="claude-code",
                description="Claude-code Sonnet - Local free access to Claude Sonnet",
                max_tokens=200000,
                cost_per_million_input=0.0,
                cost_per_million_output=0.0,
                swe_score=73.0,
                available=True,
                supports_tools=False,
                supports_multimodal=True,
                context_window=200000,
                is_free=True,
                release_date=datetime(2024, 3, 1),
            ),
        ]

        # Update SWE ratings for all models
        for model in models:
            model.update_swe_rating()

        return models

    def supports_tools(self) -> bool:
        """
        Check if Claude-code provider supports tool calling.

        Returns:
            False - Claude-code typically doesn't support tool calling
        """
        return False

    def supports_multimodal(self) -> bool:
        """
        Check if Claude-code provider supports multimodal inputs.

        Returns:
            True - Claude models support multimodal inputs
        """
        return True

    def supports_streaming(self) -> bool:
        """
        Check if Claude-code provider supports streaming responses.

        Returns:
            False - Local command execution doesn't support streaming
        """
        return False
