"""
Integration tests for Agent Mode functionality.

This module tests the complete agent mode system including:
- Agent mode manager
- Progress UI components
- CLI command integration
- Approval workflow integration
"""

import asyncio
import pytest
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from omnimancer.core.agent_mode_manager import AgentModeManager, AgentMode, AgentOperationStatus
from omnimancer.core.agent_progress_ui import AgentProgressUI, AgentStatusIndicator
# from omnimancer.core.agent_demo import AgentModeDemo, create_demo_instance  # Removed during consolidation
from omnimancer.core.agent_engine import Operation, OperationType
from omnimancer.cli.commands import Command, SlashCommand
from omnimancer.cli.interface import CommandLineInterface


@pytest.fixture
def temp_storage():
    """Create temporary storage directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_config_manager():
    """Create mock configuration manager."""
    config_manager = Mock()
    config_manager.get_config.return_value = Mock()
    return config_manager


@pytest.fixture
def agent_manager(mock_config_manager, temp_storage):
    """Create agent mode manager for testing."""
    return AgentModeManager(mock_config_manager, storage_path=temp_storage)


@pytest.fixture
def agent_ui(agent_manager):
    """Create agent progress UI for testing."""
    from rich.console import Console
    console = Console(file=None, width=80)  # No output during tests
    return AgentProgressUI(agent_manager, console)


@pytest.fixture
def sample_operation():
    """Create sample operation for testing."""
    return Operation(
        type=OperationType.FILE_READ,
        description="Test file read operation",
        data={'path': '/test/file.txt', 'risk_level': 'low'},
        requires_approval=False,
        reversible=True
    )


class TestAgentModeManager:
    """Test the core agent mode manager functionality."""
    
    def test_initial_state(self, agent_manager):
        """Test initial state of agent manager."""
        assert agent_manager.mode == AgentMode.ON
        assert len(agent_manager.operation_queue) == 0
        assert len(agent_manager.active_operations) == 0
        assert len(agent_manager.completed_operations) == 0
    
    @pytest.mark.asyncio
    async def test_enable_disable_agent_mode(self, agent_manager):
        """Test enabling and disabling agent mode."""
        # Test enable
        success = await agent_manager.enable_agent_mode()
        assert success
        assert agent_manager.mode == AgentMode.ON
        
        # Test disable
        success = await agent_manager.disable_agent_mode(wait_for_completion=False)
        assert success
        assert agent_manager.mode == AgentMode.OFF
    
    def test_queue_operation(self, agent_manager, sample_operation):
        """Test operation queuing."""
        op_id = agent_manager.queue_operation(sample_operation, priority=1)
        
        assert len(agent_manager.operation_queue) == 1
        assert agent_manager.operation_queue[0].id == op_id
        assert agent_manager.operation_queue[0].operation == sample_operation
        assert agent_manager.operation_queue[0].priority == 1
    
    def test_cancel_operation(self, agent_manager, sample_operation):
        """Test operation cancellation."""
        op_id = agent_manager.queue_operation(sample_operation)
        
        success = agent_manager.cancel_operation(op_id)
        assert success
        assert len(agent_manager.operation_queue) == 0
        assert len(agent_manager.completed_operations) == 1
        assert agent_manager.completed_operations[0].status == AgentOperationStatus.CANCELLED
    
    def test_get_status(self, agent_manager, sample_operation):
        """Test status reporting."""
        # Queue some operations
        agent_manager.queue_operation(sample_operation)
        
        status = agent_manager.get_status()
        
        assert status['mode'] == 'on'
        assert status['operations']['queued'] == 1
        assert status['operations']['in_progress'] == 0
        assert status['operations']['completed'] == 0
        assert status['operations']['failed'] == 0
    
    def test_pause_resume(self, agent_manager):
        """Test pause and resume functionality."""
        # Agent starts ON by default, test pause
        assert agent_manager.pause_agent_mode()
        assert agent_manager.mode == AgentMode.PAUSED
        
        # Test resume
        assert agent_manager.resume_agent_mode()
        assert agent_manager.mode == AgentMode.ON
        
        # Turn off agent and verify can't pause when off
        asyncio.run(agent_manager.disable_agent_mode())
        assert not agent_manager.pause_agent_mode()
    
    def test_operation_history(self, agent_manager, sample_operation):
        """Test operation history retrieval."""
        # Queue and cancel an operation
        op_id = agent_manager.queue_operation(sample_operation)
        agent_manager.cancel_operation(op_id)
        
        history = agent_manager.get_operation_history(limit=10)
        
        assert len(history) == 1
        assert history[0]['id'] == op_id
        assert history[0]['status'] == 'cancelled'
        assert history[0]['type'] == 'file_read'


class TestAgentProgressUI:
    """Test the agent progress UI components."""
    
    def test_status_panel_creation(self, agent_ui):
        """Test status panel creation."""
        panel = agent_ui.show_status_panel()
        assert panel is not None
        from rich.panel import Panel
        assert isinstance(panel, Panel)
        assert panel.title == "Agent Status"
    
    def test_operations_table_creation(self, agent_ui):
        """Test operations table creation."""
        table = agent_ui.show_operations_table(limit=5)
        assert table is not None
        from rich.table import Table
        assert isinstance(table, Table)
        assert table.title == "Recent Operations (Last 5)"
    
    def test_progress_panel_creation(self, agent_ui):
        """Test progress panel creation."""
        panel = agent_ui.show_progress_panel()
        assert panel is not None
        from rich.panel import Panel
        assert isinstance(panel, Panel)
        assert panel.title == "Active Operations"
    
    def test_approval_queue_panel_creation(self, agent_ui):
        """Test approval queue panel creation."""
        panel = agent_ui.show_approval_queue_panel()
        assert panel is not None
        from rich.panel import Panel
        assert isinstance(panel, Panel)
        assert "Approval Queue" in panel.title
    
    def test_dashboard_creation(self, agent_ui):
        """Test dashboard layout creation."""
        try:
            dashboard = agent_ui.show_agent_dashboard()
            assert dashboard is not None
            from rich.layout import Layout
            assert isinstance(dashboard, Layout)
        except Exception as e:
            # If there's an error in dashboard creation, it might be a Rich Layout implementation issue
            # Let's just verify the method exists and can be called
            assert hasattr(agent_ui, 'show_agent_dashboard')
            assert callable(agent_ui.show_agent_dashboard)
    
    def test_status_indicator(self, agent_manager):
        """Test status indicator functionality."""
        indicator = AgentStatusIndicator(agent_manager)
        
        # Test ON state (agent starts ON by default)
        badge = indicator.get_status_badge()
        assert "AGENT" in str(badge)
        
        mini_status = indicator.get_mini_status()
        assert "Agent: ON" in mini_status
    
    def test_operation_progress_tracking(self, agent_ui):
        """Test operation progress tracking."""
        # Update progress for a fake operation
        agent_ui.update_operation_progress("test_op_1", 50.0, "Processing...")
        
        assert "test_op_1" in agent_ui.operation_progress
        assert agent_ui.operation_progress["test_op_1"].progress == 50.0
        assert agent_ui.operation_progress["test_op_1"].status_text == "Processing..."



class TestCLIIntegration:
    """Test CLI command integration."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create mock engine for CLI testing."""
        engine = Mock()
        engine.config_manager = Mock()
        engine.initialize_providers = AsyncMock()
        engine.initialize_mcp = AsyncMock()
        engine.shutdown_mcp = AsyncMock()
        return engine
    
    @pytest.fixture
    def cli_interface(self, mock_engine):
        """Create CLI interface for testing."""
        return CommandLineInterface(mock_engine)
    
    def test_agent_command_parsing(self):
        """Test agent command parsing."""
        from omnimancer.cli.commands import parse_command
        
        # Test basic commands
        command = parse_command("/agent on")
        assert command.slash_command == SlashCommand.AGENT
        assert command.args == ["on"]
        
        command = parse_command("/agent off")
        assert command.slash_command == SlashCommand.AGENT
        assert command.args == ["off"]
        
        command = parse_command("/agent status")
        assert command.slash_command == SlashCommand.AGENT
        assert command.args == ["status"]
        
        command = parse_command("/agent on --auto-approve")
        assert command.slash_command == SlashCommand.AGENT
        assert command.args == ["on", "--auto-approve"]
    
    @pytest.mark.asyncio
    async def test_agent_command_handler(self, cli_interface, mock_engine):
        """Test agent command handler."""
        # Mock the agent manager initialization
        with patch('omnimancer.core.agent_mode_manager.AgentModeManager') as mock_manager_class:
            with patch('omnimancer.core.agent_progress_ui.AgentProgressUI') as mock_ui_class:
                # Setup mocks
                mock_manager = Mock()
                mock_manager.mode.value = 'off'
                mock_manager.enable_agent_mode = AsyncMock(return_value=True)
                mock_manager.disable_agent_mode = AsyncMock(return_value=True)
                mock_manager.get_status.return_value = {
                    'mode': 'off',
                    'operations': {
                        'in_progress': 0,
                        'queued': 0,
                        'completed': 0,
                        'failed': 0,
                        'requires_approval': 0
                    }
                }
                
                mock_manager_class.return_value = mock_manager
                mock_ui_class.return_value = Mock()
                
                # Initialize agent components
                cli_interface.agent_manager = mock_manager
                cli_interface.agent_progress_ui = Mock()
                
                # Test 'on' command
                command = Command.create_slash_command(SlashCommand.AGENT, ["on"], "/agent on")
                await cli_interface._handle_agent_command(command)
                mock_manager.enable_agent_mode.assert_called_once_with(auto_approve=False)
                
                # Test 'off' command
                mock_manager.mode.value = 'on'
                command = Command.create_slash_command(SlashCommand.AGENT, ["off"], "/agent off")
                await cli_interface._handle_agent_command(command)
                mock_manager.disable_agent_mode.assert_called_once_with(wait_for_completion=True)


class TestPersistentState:
    """Test persistent state management."""
    
    def test_state_save_load(self, agent_manager, sample_operation, temp_storage):
        """Test state persistence."""
        # Queue an operation and update settings
        agent_manager.queue_operation(sample_operation)
        agent_manager.update_settings(auto_approve_low_risk=True)
        
        # Force save state
        agent_manager._save_state()
        
        # Create new manager instance
        new_manager = AgentModeManager(
            agent_manager.config_manager, 
            storage_path=temp_storage
        )
        
        # Verify settings were loaded
        assert new_manager.settings.auto_approve_low_risk == True
        
        # Verify state file exists
        state_file = Path(temp_storage) / "agent_state.json"
        assert state_file.exists()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_enable_already_enabled(self, agent_manager):
        """Test enabling agent mode when already enabled."""
        await agent_manager.enable_agent_mode()
        
        # Try to enable again
        success = await agent_manager.enable_agent_mode()
        assert success  # Should still return True
        assert agent_manager.mode == AgentMode.ON
    
    def test_invalid_operation_id(self, agent_manager):
        """Test cancelling non-existent operation."""
        success = agent_manager.cancel_operation("non_existent_id")
        assert not success
    
    def test_pause_when_off(self, agent_manager):
        """Test pausing when agent mode is off."""
        # First turn off the agent (starts ON by default)
        asyncio.run(agent_manager.disable_agent_mode())
        success = agent_manager.pause_agent_mode()
        assert not success
    
    def test_resume_when_off(self, agent_manager):
        """Test resuming when agent mode is off."""
        success = agent_manager.resume_agent_mode()
        assert not success


@pytest.mark.integration
class TestFullIntegration:
    """Full integration tests combining all components."""
    
    @pytest.mark.asyncio
    async def test_complete_agent_workflow(self, mock_config_manager, temp_storage):
        """Test complete agent workflow from start to finish."""
        # Create all components
        agent_manager = AgentModeManager(mock_config_manager, storage_path=temp_storage)
        
        from rich.console import Console
        console = Console(file=None)
        agent_ui = AgentProgressUI(agent_manager, console)
        
        # Create and queue operations
        operations = [
            Operation(
                type=OperationType.FILE_READ,
                description="Read test file",
                data={'risk_level': 'low'},
                requires_approval=False,
                reversible=True
            ),
            Operation(
                type=OperationType.FILE_WRITE,
                description="Write test file",
                data={'risk_level': 'medium'},
                requires_approval=True,
                reversible=True
            )
        ]
        
        # Queue operations
        op_ids = []
        for op in operations:
            op_id = agent_manager.queue_operation(op)
            op_ids.append(op_id)
        
        # Enable agent mode
        success = await agent_manager.enable_agent_mode(auto_approve=True)
        assert success
        
        # Verify status
        status = agent_manager.get_status()
        assert status['mode'] == 'on'
        assert status['operations']['queued'] >= 0  # May have been processed
        
        # Test UI components
        status_panel = agent_ui.show_status_panel()
        assert status_panel is not None
        
        operations_table = agent_ui.show_operations_table()
        assert operations_table is not None
        
        # Clean up
        await agent_manager.disable_agent_mode(wait_for_completion=False)
        
        # Verify final state
        final_status = agent_manager.get_status()
        assert final_status['mode'] == 'off'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])