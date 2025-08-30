"""
Configuration module for the agent status display system.

This module provides configuration management utilities for controlling
the behavior of the status tracking and display system.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .status_core import (
    EventType,
    StatusDisplayConfig,
    StatusDisplayLevel,
    StatusUpdateFrequency,
)


class StatusConfigManager:
    """Manager for status system configuration."""

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to configuration file (optional)
        """
        self.config_file = config_file or self._get_default_config_path()
        self.config = StatusDisplayConfig()

        # Try to load existing configuration
        self._load_config()

    def _get_default_config_path(self) -> Path:
        """Get default configuration file path."""
        config_dir = Path.home() / ".omnimancer" / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "agent_status_config.json"

    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, "r") as f:
                    config_data = json.load(f)

                # Update configuration from loaded data
                self._update_config(config_data)

        except Exception as e:
            # If loading fails, use defaults
            print(f"Warning: Failed to load status config: {e}")

    def _update_config(self, data: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in data.items():
            if hasattr(self.config, key):
                # Handle enum fields specially
                if key == "display_level" and isinstance(value, str):
                    try:
                        value = StatusDisplayLevel(value)
                    except ValueError:
                        continue
                elif key == "update_frequency" and isinstance(value, (str, float)):
                    try:
                        if isinstance(value, str):
                            value = StatusUpdateFrequency(value)
                        else:
                            # Find closest enum value
                            for freq in StatusUpdateFrequency:
                                if abs(freq.value - value) < 0.01:
                                    value = freq
                                    break
                    except ValueError:
                        continue
                elif key in [
                    "filtered_event_types",
                    "important_event_types",
                ] and isinstance(value, list):
                    try:
                        value = {EventType(event_type) for event_type in value}
                    except ValueError:
                        continue
                elif key == "colors" and isinstance(value, dict):
                    # Merge colors instead of replacing
                    self.config.colors.update(value)
                    continue

                setattr(self.config, key, value)

    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            config_data = self._config_to_dict()

            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.config_file, "w") as f:
                json.dump(config_data, f, indent=2, default=str)

        except Exception as e:
            print(f"Warning: Failed to save status config: {e}")

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from enum import Enum

        config = {}
        for key, value in self.config.__dict__.items():
            if isinstance(value, Enum):
                config[key] = value.value
            elif isinstance(value, set):
                config[key] = [
                    item.value if isinstance(item, Enum) else item for item in value
                ]
            elif isinstance(value, Path):
                config[key] = str(value)
            else:
                config[key] = value
        return config

    def update_display_level(self, level: StatusDisplayLevel) -> None:
        """Update display level."""
        self.config.display_level = level
        self.save_config()

    def update_update_frequency(self, frequency: StatusUpdateFrequency) -> None:
        """Update update frequency."""
        self.config.update_frequency = frequency
        self.save_config()

    def toggle_status_display(self) -> bool:
        """Toggle status display on/off."""
        self.config.enabled = not self.config.enabled
        self.save_config()
        return self.config.enabled

    def set_compact_mode(self, compact: bool) -> None:
        """Set compact display mode."""
        self.config.compact_mode = compact
        self.save_config()

    def add_filtered_event_type(self, event_type: EventType) -> None:
        """Add event type to filter (hide from display)."""
        self.config.filtered_event_types.add(event_type)
        self.save_config()

    def remove_filtered_event_type(self, event_type: EventType) -> None:
        """Remove event type from filter."""
        self.config.filtered_event_types.discard(event_type)
        self.save_config()

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self.config = StatusDisplayConfig()
        self.save_config()

    def get_effective_config(self) -> Dict[str, Any]:
        """Get the effective configuration as a dictionary."""
        return self._config_to_dict()


# Global configuration instance
_global_config_manager: Optional[StatusConfigManager] = None


def get_status_config_manager(
    config_file: Optional[Path] = None,
) -> StatusConfigManager:
    """
    Get the global status configuration manager.

    Args:
        config_file: Optional custom config file path

    Returns:
        Status configuration manager instance
    """
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = StatusConfigManager(config_file)
    return _global_config_manager


def load_config_from_env() -> StatusDisplayConfig:
    """
    Load status configuration from environment variables.

    Returns:
        Configuration with values from environment variables
    """
    config = StatusDisplayConfig()

    # Load from environment variables with omnimancer_STATUS_ prefix
    if os.getenv("omnimancer_STATUS_ENABLED"):
        config.enabled = (
            os.getenv("omnimancer_STATUS_ENABLED", "true").lower() == "true"
        )

    if os.getenv("omnimancer_STATUS_LEVEL"):
        try:
            config.display_level = StatusDisplayLevel(
                os.getenv("omnimancer_STATUS_LEVEL")
            )
        except ValueError:
            pass

    if os.getenv("omnimancer_STATUS_UPDATE_FREQ"):
        try:
            freq_value = float(os.getenv("omnimancer_STATUS_UPDATE_FREQ"))
            for freq in StatusUpdateFrequency:
                if abs(freq.value - freq_value) < 0.01:
                    config.update_frequency = freq
                    break
        except ValueError:
            pass

    if os.getenv("omnimancer_STATUS_AUTO_START"):
        config.auto_start_display = (
            os.getenv("omnimancer_STATUS_AUTO_START", "false").lower() == "true"
        )

    if os.getenv("omnimancer_STATUS_MAX_OPS"):
        try:
            config.max_visible_operations = int(os.getenv("omnimancer_STATUS_MAX_OPS"))
        except ValueError:
            pass

    if os.getenv("omnimancer_STATUS_COMPACT"):
        config.compact_mode = (
            os.getenv("omnimancer_STATUS_COMPACT", "false").lower() == "true"
        )

    return config
