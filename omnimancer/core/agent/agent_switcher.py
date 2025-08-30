"""
Agent Switcher for managing persona transitions and state preservation.

This module provides the switching mechanism that preserves session state
and handles transitions between personas.
"""

import hashlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple


from .config import AgentConfig
from .persona import AgentPersona, PersonaManager, PersonaStatus
from .status_core import AgentStatus
from .status_manager import UnifiedStatusManager as AgentStatusManager
from .status_manager import (
    get_status_manager,
)

logger = logging.getLogger(__name__)


class SwitchState(Enum):
    """States for the switching process."""

    IDLE = "idle"
    PREPARING = "preparing"
    SWITCHING = "switching"
    VALIDATING = "validating"
    COMPLETE = "complete"
    ERROR = "error"
    ROLLBACK = "rollback"


class SwitchValidationError(Exception):
    """Exception raised when switch validation fails."""

    pass


class SwitchOperationError(Exception):
    """Exception raised when switch operation fails."""

    pass


@dataclass
class SessionState:
    """
    Session state that needs to be preserved during persona switches.
    """

    # Core conversation state
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_context: Dict[str, Any] = field(default_factory=dict)

    # User preferences and settings
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    active_tools: List[str] = field(default_factory=list)
    disabled_tools: List[str] = field(default_factory=list)

    # Session metadata
    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)

    # Persona-specific data
    persona_data: Dict[str, Any] = field(default_factory=dict)

    # Operation state
    pending_operations: List[Dict[str, Any]] = field(default_factory=list)
    active_operations: List[Dict[str, Any]] = field(default_factory=list)

    # Security and permissions
    permission_overrides: Dict[str, Any] = field(default_factory=dict)
    approval_history: List[Dict[str, Any]] = field(default_factory=list)

    def update_timestamp(self) -> None:
        """Update the last modified timestamp."""
        self.last_modified = datetime.now()

    def to_json(self) -> str:
        """Serialize state to JSON."""
        data = asdict(self)
        # Convert datetime objects to ISO format
        data["created_at"] = self.created_at.isoformat()
        data["last_modified"] = self.last_modified.isoformat()
        return json.dumps(data, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "SessionState":
        """Deserialize state from JSON."""
        data = json.loads(json_str)
        # Convert ISO format strings back to datetime objects
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "last_modified" in data:
            data["last_modified"] = datetime.fromisoformat(data["last_modified"])
        return cls(**data)

    def to_pickle(self) -> bytes:
        """Serialize state to pickle format."""
        return pickle.dumps(self)

    @classmethod
    def from_pickle(cls, pickle_data: bytes) -> "SessionState":
        """Deserialize state from pickle format."""
        return pickle.loads(pickle_data)

    def get_hash(self) -> str:
        """Get a hash of the current state for validation."""
        # Create a copy without timestamps for consistent hashing
        data = asdict(self)
        # Remove timestamp fields that shouldn't affect hash
        data.pop("created_at", None)
        data.pop("last_modified", None)

        # Sort for consistent ordering
        state_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()


@dataclass
class SwitchContext:
    """Context for a persona switch operation."""

    from_persona: Optional[AgentPersona]
    to_persona: AgentPersona
    session_state: SessionState
    validation_checks: List[str] = field(default_factory=list)
    transition_hooks: List[Callable] = field(default_factory=list)
    rollback_state: Optional[SessionState] = None
    error_message: Optional[str] = None
    switch_reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


class AgentSwitcher:
    """
    Manages persona switching with state preservation and validation.

    This class handles the complex process of switching between agent personas
    while preserving session state and ensuring clean transitions.
    """

    def __init__(
        self,
        persona_manager: PersonaManager,
        agent_config: Optional[AgentConfig] = None,
        status_manager: Optional[AgentStatusManager] = None,
        state_storage_path: Optional[Path] = None,
    ):
        """
        Initialize the AgentSwitcher.

        Args:
            persona_manager: PersonaManager instance for managing personas
            agent_config: Agent configuration instance
            status_manager: Status manager for tracking agent state
            state_storage_path: Path for persistent state storage
        """
        self.persona_manager = persona_manager
        self.agent_config = agent_config
        self.status_manager = status_manager or get_status_manager()
        self.state_storage_path = (
            state_storage_path or Path.home() / ".omnimancer" / "session_states"
        )

        # Ensure storage directory exists
        self.state_storage_path.mkdir(parents=True, exist_ok=True)

        # Current state
        self.current_state = SwitchState.IDLE
        self.current_session_state: Optional[SessionState] = None
        self.switch_history: List[SwitchContext] = []

        # Thread safety
        self._lock = Lock()

        # Hooks and validators
        self._pre_switch_hooks: List[Callable] = []
        self._post_switch_hooks: List[Callable] = []
        self._validation_hooks: List[Callable] = []

        # Register default hooks
        self._register_default_hooks()

    def _register_default_hooks(self) -> None:
        """Register default transition and validation hooks."""
        # Default pre-switch hooks
        self._pre_switch_hooks.append(self._save_current_state)

        # Default post-switch hooks
        self._post_switch_hooks.append(self._restore_state)
        self._post_switch_hooks.append(self._update_status_manager)

        # Default validation hooks
        self._validation_hooks.append(self._validate_persona_available)
        self._validation_hooks.append(self._validate_no_active_operations)
        self._validation_hooks.append(self._validate_state_integrity)

    def switch_persona(
        self, target_persona_id: str, reason: str = "", force: bool = False
    ) -> Tuple[bool, str]:
        """
        Switch to a target persona with state preservation.

        Args:
            target_persona_id: ID of the persona to switch to
            reason: Reason for the switch (for logging)
            force: Force switch even if validation fails

        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            try:
                # Set state to preparing
                self.current_state = SwitchState.PREPARING

                # Get target persona
                target_persona = self.persona_manager.get_persona(target_persona_id)
                if not target_persona:
                    raise SwitchValidationError(
                        f"Persona not found: {target_persona_id}"
                    )

                # Get current persona
                current_persona = self.persona_manager.active_persona

                # Create switch context
                context = SwitchContext(
                    from_persona=current_persona,
                    to_persona=target_persona,
                    session_state=self._capture_current_state(),
                    switch_reason=reason,
                )

                # Validation phase
                if not force:
                    self.current_state = SwitchState.VALIDATING
                    validation_result = self._validate_switch(context)
                    if not validation_result[0]:
                        self.current_state = SwitchState.ERROR
                        context.error_message = validation_result[1]
                        self.switch_history.append(context)
                        return validation_result

                # Pre-switch phase
                self.current_state = SwitchState.SWITCHING
                for hook in self._pre_switch_hooks:
                    try:
                        hook(context)
                    except Exception as e:
                        logger.error(f"Pre-switch hook failed: {e}")
                        if not force:
                            self._rollback(context)
                            return (False, f"Pre-switch hook failed: {e}")

                # Perform the switch
                if current_persona:
                    self.persona_manager.deactivate_persona()

                success = self.persona_manager.activate_persona(target_persona_id)
                if not success:
                    self._rollback(context)
                    return (
                        False,
                        f"Failed to activate persona: {target_persona_id}",
                    )

                # Post-switch phase
                for hook in self._post_switch_hooks:
                    try:
                        hook(context)
                    except Exception as e:
                        logger.error(f"Post-switch hook failed: {e}")
                        # Post-switch failures are logged but don't rollback

                # Complete
                self.current_state = SwitchState.COMPLETE
                self.switch_history.append(context)

                logger.info(f"Successfully switched to persona: {target_persona_id}")
                return (True, f"Switched to {target_persona.name}")

            except Exception as e:
                logger.error(f"Persona switch failed: {e}")
                self.current_state = SwitchState.ERROR
                return (False, str(e))
            finally:
                # Reset to idle after a short delay
                self.current_state = SwitchState.IDLE

    def _capture_current_state(self) -> SessionState:
        """Capture the current session state."""
        if self.current_session_state:
            return self.current_session_state

        # Create new session state
        state = SessionState()

        # Capture from active persona if available
        active_persona = self.persona_manager.active_persona
        if active_persona:
            state.persona_data = {
                "persona_id": active_persona.id,
                "session_data": active_persona._session_data.copy(),
            }

        # TODO: Capture conversation history from the main engine
        # This would need integration with the main conversation loop

        return state

    def _validate_switch(self, context: SwitchContext) -> Tuple[bool, str]:
        """
        Validate that a switch can be performed.

        Args:
            context: Switch context

        Returns:
            Tuple of (valid, message)
        """
        for validator in self._validation_hooks:
            try:
                result = validator(context)
                if not result[0]:
                    context.validation_checks.append(f"Failed: {validator.__name__}")
                    return result
                context.validation_checks.append(f"Passed: {validator.__name__}")
            except Exception as e:
                logger.error(f"Validation hook error: {e}")
                return (False, f"Validation error: {e}")

        return (True, "All validations passed")

    def _validate_persona_available(self, context: SwitchContext) -> Tuple[bool, str]:
        """Validate that target persona is available."""
        if context.to_persona.status not in [
            PersonaStatus.AVAILABLE,
            PersonaStatus.ACTIVE,
        ]:
            return (
                False,
                f"Persona {context.to_persona.id} is not available (status: {context.to_persona.status})",
            )
        return (True, "Persona available")

    def _validate_no_active_operations(
        self, context: SwitchContext
    ) -> Tuple[bool, str]:
        """Validate no active operations are in progress."""
        if context.session_state.active_operations:
            return (
                False,
                f"Cannot switch: {len(context.session_state.active_operations)} operations in progress",
            )
        return (True, "No active operations")

    def _validate_state_integrity(self, context: SwitchContext) -> Tuple[bool, str]:
        """Validate state integrity."""
        try:
            # Test serialization/deserialization
            json_str = context.session_state.to_json()
            restored = SessionState.from_json(json_str)

            # Verify hash matches
            if context.session_state.get_hash() != restored.get_hash():
                return (False, "State integrity check failed")

            return (True, "State integrity verified")
        except Exception as e:
            return (False, f"State validation error: {e}")

    def _save_current_state(self, context: SwitchContext) -> None:
        """Save current state before switch."""
        # Create rollback state
        context.rollback_state = context.session_state

        # Persist to disk if we have a from_persona
        if context.from_persona:
            state_file = (
                self.state_storage_path / f"{context.from_persona.id}_state.json"
            )
            try:
                with open(state_file, "w") as f:
                    f.write(context.session_state.to_json())
                logger.debug(f"Saved state for persona: {context.from_persona.id}")
            except Exception as e:
                logger.error(f"Failed to save state: {e}")

    def _restore_state(self, context: SwitchContext) -> None:
        """Restore state after switch."""
        # Check if we have saved state for the target persona
        state_file = self.state_storage_path / f"{context.to_persona.id}_state.json"

        if state_file.exists():
            try:
                with open(state_file, "r") as f:
                    saved_state = SessionState.from_json(f.read())

                # Merge with current session state
                context.session_state.persona_data = saved_state.persona_data
                context.session_state.user_preferences.update(
                    saved_state.user_preferences
                )

                # Restore to persona
                if "session_data" in saved_state.persona_data:
                    for key, value in saved_state.persona_data["session_data"].items():
                        context.to_persona.set_session_data(key, value)

                logger.debug(f"Restored state for persona: {context.to_persona.id}")
            except Exception as e:
                logger.error(f"Failed to restore state: {e}")

        # Update current session state
        self.current_session_state = context.session_state

    def _update_status_manager(self, context: SwitchContext) -> None:
        """Update status manager after switch."""
        if self.status_manager:
            try:
                import asyncio

                # Try to get running loop
                try:
                    loop = asyncio.get_running_loop()

                    # Update old persona status
                    if context.from_persona:
                        loop.create_task(
                            self.status_manager.set_agent_status(
                                agent_id=f"persona_{context.from_persona.id}",
                                status=AgentStatus.DISABLED,
                            )
                        )

                    # Update new persona status
                    loop.create_task(
                        self.status_manager.set_agent_status(
                            agent_id=f"persona_{context.to_persona.id}",
                            status=AgentStatus.ENABLED,
                            metadata={
                                "switch_reason": context.switch_reason,
                                "from_persona": (
                                    context.from_persona.id
                                    if context.from_persona
                                    else None
                                ),
                            },
                        )
                    )
                except RuntimeError:
                    # No event loop running, skip async status updates
                    logger.debug("No event loop available for status update")

            except Exception as e:
                # Catch all exceptions to prevent status update failures from affecting the switch
                logger.debug(f"Status manager update failed: {e}")

    def _rollback(self, context: SwitchContext) -> None:
        """Rollback a failed switch."""
        self.current_state = SwitchState.ROLLBACK

        try:
            # Restore original persona if we had one
            if context.from_persona:
                self.persona_manager.activate_persona(context.from_persona.id)

            # Restore original state
            if context.rollback_state:
                self.current_session_state = context.rollback_state

            logger.info("Successfully rolled back persona switch")
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
        finally:
            self.current_state = SwitchState.IDLE

    def add_pre_switch_hook(self, hook: Callable) -> None:
        """Add a pre-switch hook."""
        self._pre_switch_hooks.append(hook)

    def add_post_switch_hook(self, hook: Callable) -> None:
        """Add a post-switch hook."""
        self._post_switch_hooks.append(hook)

    def add_validation_hook(self, hook: Callable) -> None:
        """Add a validation hook."""
        self._validation_hooks.append(hook)

    def get_switch_history(self) -> List[SwitchContext]:
        """Get the switch history."""
        return self.switch_history.copy()

    def get_current_state(self) -> SwitchState:
        """Get the current switch state."""
        return self.current_state

    def can_switch(self, target_persona_id: str) -> Tuple[bool, str]:
        """
        Check if a switch to target persona is possible.

        Args:
            target_persona_id: ID of target persona

        Returns:
            Tuple of (can_switch, reason)
        """
        # Check if persona exists
        target_persona = self.persona_manager.get_persona(target_persona_id)
        if not target_persona:
            return (False, f"Persona not found: {target_persona_id}")

        # Check if already active
        if self.persona_manager.active_persona == target_persona:
            return (False, "Persona already active")

        # Check if switch is in progress
        if self.current_state != SwitchState.IDLE:
            return (
                False,
                f"Switch in progress (state: {self.current_state.value})",
            )

        # Create mock context for validation
        mock_context = SwitchContext(
            from_persona=self.persona_manager.active_persona,
            to_persona=target_persona,
            session_state=self._capture_current_state(),
        )

        # Run validations
        return self._validate_switch(mock_context)

    def handle_active_operation_switch(
        self, target_persona_id: str, operation_handler: Callable
    ) -> Tuple[bool, str]:
        """
        Handle switching during an active operation.

        This is an edge case handler that attempts to gracefully handle
        switches when operations are in progress.

        Args:
            target_persona_id: Target persona ID
            operation_handler: Callable to handle active operations

        Returns:
            Tuple of (success, message)
        """
        try:
            # Capture current state (within lock)
            with self._lock:
                current_state = self._capture_current_state()

                if current_state.active_operations:
                    logger.warning(
                        f"Attempting switch with {len(current_state.active_operations)} active operations"
                    )

                    # Call the operation handler
                    handled = operation_handler(current_state.active_operations)
                    if not handled:
                        return (False, "Could not handle active operations")

                    # Clear active operations after handling
                    current_state.active_operations.clear()

            # Proceed with normal switch (outside of lock to avoid deadlock)
            return self.switch_persona(
                target_persona_id,
                reason="Switch with operation handling",
                force=False,
            )

        except Exception as e:
            logger.error(f"Failed to handle active operation switch: {e}")
            return (False, str(e))

    def export_session_state(self, format: str = "json") -> Optional[str]:
        """
        Export current session state.

        Args:
            format: Export format ('json' or 'pickle')

        Returns:
            Exported state string or None
        """
        if not self.current_session_state:
            self.current_session_state = self._capture_current_state()

        try:
            if format == "json":
                return self.current_session_state.to_json()
            elif format == "pickle":
                import base64

                return base64.b64encode(self.current_session_state.to_pickle()).decode()
            else:
                logger.error(f"Unknown export format: {format}")
                return None
        except Exception as e:
            logger.error(f"Failed to export session state: {e}")
            return None

    def import_session_state(self, state_data: str, format: str = "json") -> bool:
        """
        Import session state.

        Args:
            state_data: State data string
            format: Import format ('json' or 'pickle')

        Returns:
            Success status
        """
        try:
            if format == "json":
                self.current_session_state = SessionState.from_json(state_data)
            elif format == "pickle":
                import base64

                pickle_data = base64.b64decode(state_data.encode())
                self.current_session_state = SessionState.from_pickle(pickle_data)
            else:
                logger.error(f"Unknown import format: {format}")
                return False

            return True
        except Exception as e:
            logger.error(f"Failed to import session state: {e}")
            return False


# Global instance for easy access
_global_agent_switcher: Optional[AgentSwitcher] = None


def get_agent_switcher(
    persona_manager: Optional[PersonaManager] = None,
) -> AgentSwitcher:
    """Get the global agent switcher instance."""
    global _global_agent_switcher
    if _global_agent_switcher is None:
        from .persona import get_persona_manager

        pm = persona_manager or get_persona_manager()
        _global_agent_switcher = AgentSwitcher(pm)
    return _global_agent_switcher


def set_agent_switcher(switcher: AgentSwitcher) -> None:
    """Set the global agent switcher instance."""
    global _global_agent_switcher
    _global_agent_switcher = switcher
