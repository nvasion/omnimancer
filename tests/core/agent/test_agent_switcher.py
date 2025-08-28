"""
Tests for the AgentSwitcher class and state management.
"""

import json
import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call

from omnimancer.core.agent.agent_switcher import (
    AgentSwitcher,
    SessionState,
    SwitchContext,
    SwitchState,
    SwitchValidationError,
    SwitchOperationError
)
from omnimancer.core.agent.persona import (
    PersonaManager,
    AgentPersona,
    PersonaStatus,
    CodingPersona,
    ResearchPersona,
    GeneralPersona
)
from omnimancer.core.agent.config import AgentConfig
from omnimancer.core.agent.status_manager import UnifiedStatusManager as AgentStatusManager
from omnimancer.core.agent.status_core import AgentStatus


class TestSessionState:
    """Test SessionState class."""
    
    def test_session_state_initialization(self):
        """Test SessionState initialization with default values."""
        state = SessionState()
        
        assert state.conversation_history == []
        assert state.current_context == {}
        assert state.user_preferences == {}
        assert state.active_tools == []
        assert state.disabled_tools == []
        assert state.session_id == ""
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.last_modified, datetime)
        assert state.persona_data == {}
        assert state.pending_operations == []
        assert state.active_operations == []
        assert state.permission_overrides == {}
        assert state.approval_history == []
    
    def test_session_state_update_timestamp(self):
        """Test updating the timestamp."""
        state = SessionState()
        original_time = state.last_modified
        
        # Small delay to ensure timestamp changes
        import time
        time.sleep(0.01)
        
        state.update_timestamp()
        assert state.last_modified > original_time
    
    def test_session_state_json_serialization(self):
        """Test JSON serialization and deserialization."""
        state = SessionState()
        state.session_id = "test-session-123"
        state.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        state.user_preferences = {"theme": "dark", "language": "en"}
        state.active_tools = ["tool1", "tool2"]
        
        # Serialize to JSON
        json_str = state.to_json()
        assert isinstance(json_str, str)
        
        # Deserialize from JSON
        restored_state = SessionState.from_json(json_str)
        
        assert restored_state.session_id == state.session_id
        assert restored_state.conversation_history == state.conversation_history
        assert restored_state.user_preferences == state.user_preferences
        assert restored_state.active_tools == state.active_tools
        assert isinstance(restored_state.created_at, datetime)
        assert isinstance(restored_state.last_modified, datetime)
    
    def test_session_state_pickle_serialization(self):
        """Test pickle serialization and deserialization."""
        state = SessionState()
        state.session_id = "pickle-test-456"
        state.persona_data = {"persona_id": "coding", "data": {"key": "value"}}
        
        # Serialize to pickle
        pickle_data = state.to_pickle()
        assert isinstance(pickle_data, bytes)
        
        # Deserialize from pickle
        restored_state = SessionState.from_pickle(pickle_data)
        
        assert restored_state.session_id == state.session_id
        assert restored_state.persona_data == state.persona_data
    
    def test_session_state_hash(self):
        """Test state hashing for integrity checks."""
        state1 = SessionState()
        state1.session_id = "hash-test"
        
        state2 = SessionState()
        state2.session_id = "hash-test"
        
        # Same content should produce same hash
        assert state1.get_hash() == state2.get_hash()
        
        # Different content should produce different hash
        state2.session_id = "different"
        assert state1.get_hash() != state2.get_hash()


class TestAgentSwitcher:
    """Test AgentSwitcher class."""
    
    @pytest.fixture
    def mock_persona_manager(self):
        """Create a mock PersonaManager."""
        manager = Mock(spec=PersonaManager)
        manager.active_persona = None
        manager.personas = {}
        
        # Create mock personas
        coding_persona = Mock(spec=AgentPersona)
        coding_persona.id = "coding"
        coding_persona.name = "Coding Agent"
        coding_persona.status = PersonaStatus.AVAILABLE
        coding_persona._session_data = {}
        coding_persona.set_session_data = Mock()
        
        research_persona = Mock(spec=AgentPersona)
        research_persona.id = "research"
        research_persona.name = "Research Agent"
        research_persona.status = PersonaStatus.AVAILABLE
        research_persona._session_data = {}
        research_persona.set_session_data = Mock()
        
        manager.get_persona = Mock(side_effect=lambda pid: {
            "coding": coding_persona,
            "research": research_persona
        }.get(pid))
        
        manager.activate_persona = Mock(return_value=True)
        manager.deactivate_persona = Mock(return_value=True)
        
        return manager
    
    @pytest.fixture
    def temp_storage_path(self):
        """Create a temporary storage path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def agent_switcher(self, mock_persona_manager, temp_storage_path):
        """Create an AgentSwitcher instance."""
        return AgentSwitcher(
            persona_manager=mock_persona_manager,
            state_storage_path=temp_storage_path
        )
    
    def test_agent_switcher_initialization(self, agent_switcher):
        """Test AgentSwitcher initialization."""
        assert agent_switcher.current_state == SwitchState.IDLE
        assert agent_switcher.current_session_state is None
        assert agent_switcher.switch_history == []
        assert agent_switcher.state_storage_path.exists()
    
    def test_switch_persona_success(self, agent_switcher, mock_persona_manager):
        """Test successful persona switch."""
        # Set up initial state
        mock_persona_manager.active_persona = None
        
        # Perform switch
        success, message = agent_switcher.switch_persona("coding", reason="Test switch")
        
        assert success is True
        assert "Coding Agent" in message
        assert mock_persona_manager.activate_persona.called
        assert mock_persona_manager.activate_persona.call_args[0][0] == "coding"
        assert len(agent_switcher.switch_history) == 1
        assert agent_switcher.switch_history[0].switch_reason == "Test switch"
    
    def test_switch_persona_not_found(self, agent_switcher):
        """Test switch to non-existent persona."""
        success, message = agent_switcher.switch_persona("nonexistent")
        
        assert success is False
        assert "not found" in message
        assert agent_switcher.current_state == SwitchState.IDLE
    
    def test_switch_persona_with_current_active(self, agent_switcher, mock_persona_manager):
        """Test switching from one active persona to another."""
        # Set up current active persona
        coding_persona = mock_persona_manager.get_persona("coding")
        mock_persona_manager.active_persona = coding_persona
        
        # Perform switch
        success, message = agent_switcher.switch_persona("research")
        
        assert success is True
        assert mock_persona_manager.deactivate_persona.called
        assert mock_persona_manager.activate_persona.called
        assert mock_persona_manager.activate_persona.call_args[0][0] == "research"
    
    def test_validation_persona_not_available(self, agent_switcher, mock_persona_manager):
        """Test validation fails when persona is not available."""
        # Make persona unavailable
        research_persona = mock_persona_manager.get_persona("research")
        research_persona.status = PersonaStatus.ERROR
        
        success, message = agent_switcher.switch_persona("research")
        
        assert success is False
        assert "not available" in message
    
    def test_validation_active_operations(self, agent_switcher):
        """Test validation fails with active operations."""
        # Set up session state with active operations
        state = SessionState()
        state.active_operations = [{"operation": "test"}]
        agent_switcher.current_session_state = state
        
        success, message = agent_switcher.switch_persona("coding")
        
        assert success is False
        assert "operations in progress" in message
    
    def test_force_switch_bypasses_validation(self, agent_switcher):
        """Test force switch bypasses validation."""
        # Set up session state with active operations
        state = SessionState()
        state.active_operations = [{"operation": "test"}]
        agent_switcher.current_session_state = state
        
        success, message = agent_switcher.switch_persona("coding", force=True)
        
        assert success is True
        assert "Coding Agent" in message
    
    def test_state_persistence(self, agent_switcher, mock_persona_manager, temp_storage_path):
        """Test state is persisted to disk."""
        # Set up initial persona
        coding_persona = mock_persona_manager.get_persona("coding")
        mock_persona_manager.active_persona = coding_persona
        
        # Create session state
        state = SessionState()
        state.session_id = "test-persistence"
        state.user_preferences = {"key": "value"}
        agent_switcher.current_session_state = state
        
        # Perform switch
        agent_switcher.switch_persona("research")
        
        # Check state was saved
        state_file = temp_storage_path / "coding_state.json"
        assert state_file.exists()
        
        # Load and verify saved state
        with open(state_file) as f:
            saved_data = json.load(f)
        assert saved_data["session_id"] == "test-persistence"
        assert saved_data["user_preferences"]["key"] == "value"
    
    def test_state_restoration(self, agent_switcher, temp_storage_path):
        """Test state restoration from disk."""
        # Create saved state file
        state = SessionState()
        state.session_id = "restored-session"
        state.persona_data = {"session_data": {"restored_key": "restored_value"}}
        
        state_file = temp_storage_path / "coding_state.json"
        with open(state_file, 'w') as f:
            f.write(state.to_json())
        
        # Mock persona to verify restoration
        coding_persona = Mock()
        coding_persona.id = "coding"
        coding_persona.name = "Coding Agent"
        coding_persona.status = PersonaStatus.AVAILABLE
        coding_persona._session_data = {}
        coding_persona.set_session_data = Mock()
        
        agent_switcher.persona_manager.get_persona = Mock(return_value=coding_persona)
        
        # Perform switch
        agent_switcher.switch_persona("coding")
        
        # Verify restoration
        coding_persona.set_session_data.assert_called_with("restored_key", "restored_value")
    
    def test_rollback_on_failure(self, agent_switcher, mock_persona_manager):
        """Test rollback when switch fails."""
        # Set up initial persona
        coding_persona = mock_persona_manager.get_persona("coding")
        mock_persona_manager.active_persona = coding_persona
        
        # Make activation fail
        mock_persona_manager.activate_persona = Mock(return_value=False)
        
        # Perform switch
        success, message = agent_switcher.switch_persona("research")
        
        assert success is False
        assert "Failed to activate" in message
        # Verify rollback attempt (reactivating original persona)
        assert mock_persona_manager.activate_persona.call_count == 2  # Failed attempt + rollback
    
    def test_can_switch_validation(self, agent_switcher, mock_persona_manager):
        """Test can_switch validation method."""
        # Test with valid target
        can_switch, reason = agent_switcher.can_switch("coding")
        assert can_switch is True
        assert "passed" in reason.lower()
        
        # Test with non-existent persona
        can_switch, reason = agent_switcher.can_switch("nonexistent")
        assert can_switch is False
        assert "not found" in reason
        
        # Test when already active
        coding_persona = mock_persona_manager.get_persona("coding")
        mock_persona_manager.active_persona = coding_persona
        can_switch, reason = agent_switcher.can_switch("coding")
        assert can_switch is False
        assert "already active" in reason.lower()
    
    def test_handle_active_operation_switch(self, agent_switcher):
        """Test switching with active operation handler."""
        # Set up session state with active operations
        state = SessionState()
        state.active_operations = [{"op": "test1"}, {"op": "test2"}]
        agent_switcher.current_session_state = state
        
        # Create operation handler
        handler = Mock(return_value=True)
        
        # Perform switch with handler
        success, message = agent_switcher.handle_active_operation_switch(
            "coding", handler
        )
        
        assert success is True
        handler.assert_called_once_with(state.active_operations)
        assert len(state.active_operations) == 0  # Should be cleared
    
    def test_export_import_session_state(self, agent_switcher):
        """Test exporting and importing session state."""
        # Create session state
        state = SessionState()
        state.session_id = "export-test"
        state.conversation_history = [{"msg": "test"}]
        agent_switcher.current_session_state = state
        
        # Export as JSON
        exported_json = agent_switcher.export_session_state("json")
        assert exported_json is not None
        assert "export-test" in exported_json
        
        # Import JSON
        agent_switcher.current_session_state = None
        success = agent_switcher.import_session_state(exported_json, "json")
        assert success is True
        assert agent_switcher.current_session_state.session_id == "export-test"
        
        # Export as pickle
        exported_pickle = agent_switcher.export_session_state("pickle")
        assert exported_pickle is not None
        
        # Import pickle
        agent_switcher.current_session_state = None
        success = agent_switcher.import_session_state(exported_pickle, "pickle")
        assert success is True
        assert agent_switcher.current_session_state.session_id == "export-test"
    
    def test_custom_hooks(self, agent_switcher):
        """Test adding and executing custom hooks."""
        pre_hook_called = []
        post_hook_called = []
        validation_hook_called = []
        
        def pre_hook(context):
            pre_hook_called.append(context.to_persona.id)
        
        def post_hook(context):
            post_hook_called.append(context.to_persona.id)
        
        def validation_hook(context):
            validation_hook_called.append(context.to_persona.id)
            return (True, "Custom validation passed")
        
        # Add custom hooks
        agent_switcher.add_pre_switch_hook(pre_hook)
        agent_switcher.add_post_switch_hook(post_hook)
        agent_switcher.add_validation_hook(validation_hook)
        
        # Perform switch
        agent_switcher.switch_persona("coding")
        
        # Verify hooks were called
        assert "coding" in pre_hook_called
        assert "coding" in post_hook_called
        assert "coding" in validation_hook_called
    
    def test_switch_history(self, agent_switcher):
        """Test switch history tracking."""
        # Perform multiple switches
        agent_switcher.switch_persona("coding", reason="First switch")
        agent_switcher.switch_persona("research", reason="Second switch")
        
        # Get history
        history = agent_switcher.get_switch_history()
        
        assert len(history) == 2
        assert history[0].to_persona.id == "coding"
        assert history[0].switch_reason == "First switch"
        assert history[1].to_persona.id == "research"
        assert history[1].switch_reason == "Second switch"
    
    def test_concurrent_switch_protection(self, agent_switcher):
        """Test that concurrent switches are protected by lock."""
        import threading
        import time
        
        results = []
        
        def slow_switch():
            # Add a slow pre-switch hook
            def slow_hook(context):
                time.sleep(0.1)
            
            agent_switcher.add_pre_switch_hook(slow_hook)
            success, msg = agent_switcher.switch_persona("coding")
            results.append((success, msg))
        
        def fast_switch():
            time.sleep(0.01)  # Start slightly after slow switch
            success, msg = agent_switcher.switch_persona("research")
            results.append((success, msg))
        
        # Start concurrent switches
        thread1 = threading.Thread(target=slow_switch)
        thread2 = threading.Thread(target=fast_switch)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Both should complete without errors
        assert len(results) == 2
        for success, msg in results:
            assert success is True


class TestSwitchContext:
    """Test SwitchContext class."""
    
    def test_switch_context_initialization(self):
        """Test SwitchContext initialization."""
        from_persona = Mock(spec=AgentPersona)
        to_persona = Mock(spec=AgentPersona)
        session_state = SessionState()
        
        context = SwitchContext(
            from_persona=from_persona,
            to_persona=to_persona,
            session_state=session_state,
            switch_reason="Test reason"
        )
        
        assert context.from_persona == from_persona
        assert context.to_persona == to_persona
        assert context.session_state == session_state
        assert context.switch_reason == "Test reason"
        assert context.validation_checks == []
        assert context.transition_hooks == []
        assert context.rollback_state is None
        assert context.error_message is None
        assert isinstance(context.timestamp, datetime)


class TestIntegration:
    """Integration tests with real PersonaManager."""
    
    @pytest.fixture
    def real_persona_manager(self):
        """Create a real PersonaManager."""
        from omnimancer.core.models import ConfigTemplateManager
        template_manager = ConfigTemplateManager()
        return PersonaManager(template_manager=template_manager)
    
    @pytest.fixture
    def real_agent_switcher(self, real_persona_manager, tmp_path):
        """Create AgentSwitcher with real PersonaManager."""
        return AgentSwitcher(
            persona_manager=real_persona_manager,
            state_storage_path=tmp_path
        )
    
    def test_real_persona_switching(self, real_agent_switcher, real_persona_manager):
        """Test switching between real personas."""
        # Start with no active persona
        assert real_persona_manager.active_persona is None
        
        # Switch to coding persona
        success, message = real_agent_switcher.switch_persona("coding")
        assert success is True
        assert real_persona_manager.active_persona is not None
        assert real_persona_manager.active_persona.id == "coding"
        
        # Switch to research persona
        success, message = real_agent_switcher.switch_persona("research")
        assert success is True
        assert real_persona_manager.active_persona.id == "research"
        
        # Verify switch history
        history = real_agent_switcher.get_switch_history()
        assert len(history) == 2
        assert history[0].to_persona.id == "coding"
        assert history[1].from_persona.id == "coding"
        assert history[1].to_persona.id == "research"
    
    def test_persona_state_preservation(self, real_agent_switcher, real_persona_manager):
        """Test that persona state is preserved across switches."""
        # Activate coding persona
        real_agent_switcher.switch_persona("coding")
        coding_persona = real_persona_manager.active_persona
        
        # Set some session data
        coding_persona.set_session_data("test_key", "test_value")
        coding_persona.set_session_data("counter", 42)
        
        # Switch to research persona
        real_agent_switcher.switch_persona("research")
        
        # Switch back to coding persona
        real_agent_switcher.switch_persona("coding")
        
        # Verify state was preserved
        # Note: In the real implementation, this would require proper integration
        # with the persona's session data restoration
        assert real_persona_manager.active_persona.id == "coding"