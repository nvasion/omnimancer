"""
Tests for the Proposed Changes Display Integration module.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
from rich.console import Console

from omnimancer.core.agent.proposed_changes_integration import (
    ProposedChangesIntegration,
    ProposedChange,
    ChangeSet,
    ChangeDisplayMode
)
from omnimancer.core.agent.approval_manager import ChangeType
from omnimancer.core.security.approval_workflow import RiskLevel


class TestProposedChangesIntegration:
    """Test suite for ProposedChangesIntegration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.console = Mock(spec=Console)
        self.file_system_manager = Mock()
        self.approval_manager = Mock()
        
        self.integration = ProposedChangesIntegration(
            file_system_manager=self.file_system_manager,
            approval_manager=self.approval_manager,
            console=self.console
        )
    
    def create_test_change(self, file_path="/test/file.py", operation_type=ChangeType.FILE_MODIFY):
        """Create a test ProposedChange."""
        return ProposedChange(
            file_path=file_path,
            operation_type=operation_type,
            original_content="original content",
            modified_content="modified content",
            risk_level=RiskLevel.LOW,
            change_summary="Test change"
        )
    
    def create_test_change_set(self, num_changes=3):
        """Create a test ChangeSet."""
        changes = [self.create_test_change(f"/test/file{i}.py") for i in range(num_changes)]
        return ChangeSet(
            id="test-changeset-001",
            description="Test change set",
            changes=changes
        )
    
    @pytest.mark.asyncio
    async def test_fetch_proposed_changes(self):
        """Test fetching proposed changes."""
        # Mock approval manager response
        mock_operations = [
            {
                'file_path': '/test/file1.py',
                'type': 'modify',
                'original_content': 'old',
                'modified_content': 'new',
                'risk_level': 'low'
            },
            {
                'file_path': '/test/file2.py',
                'type': 'create',
                'modified_content': 'new file',
                'risk_level': 'medium'
            }
        ]
        
        self.approval_manager.get_pending_operations = AsyncMock(return_value=mock_operations)
        
        change_set = await self.integration.fetch_proposed_changes("op-001")
        
        assert change_set.id == "op-001"
        assert len(change_set.changes) == 2
        assert change_set.changes[0].file_path == '/test/file1.py'
        assert change_set.changes[1].operation_type == ChangeType.FILE_CREATE
    
    @pytest.mark.asyncio
    async def test_fetch_proposed_changes_with_filter(self):
        """Test fetching proposed changes with file path filter."""
        mock_operations = [
            {'file_path': '/test/file1.py', 'type': 'modify'},
            {'file_path': '/test/file2.py', 'type': 'create'},
            {'file_path': '/test/file3.py', 'type': 'delete'}
        ]
        
        self.approval_manager.get_pending_operations = AsyncMock(return_value=mock_operations)
        
        change_set = await self.integration.fetch_proposed_changes(
            "op-002",
            file_paths=['/test/file1.py', '/test/file3.py']
        )
        
        assert len(change_set.changes) == 2
        file_paths = [c.file_path for c in change_set.changes]
        assert '/test/file1.py' in file_paths
        assert '/test/file3.py' in file_paths
        assert '/test/file2.py' not in file_paths
    
    @pytest.mark.asyncio
    async def test_display_proposed_changes_summary_mode(self):
        """Test displaying proposed changes in summary mode."""
        change_set = self.create_test_change_set(3)
        
        result = await self.integration.display_proposed_changes(
            change_set,
            display_mode=ChangeDisplayMode.SUMMARY,
            interactive=False
        )
        
        assert result['displayed'] is True
        assert result['interactive'] is False
        assert self.console.print.called
    
    @pytest.mark.asyncio
    async def test_display_proposed_changes_interactive(self):
        """Test interactive display of proposed changes."""
        change_set = self.create_test_change_set(2)
        
        with patch.object(self.integration, '_get_change_approval') as mock_approval:
            mock_approval.return_value = {'approved': True, 'all_changes': True}
            
            result = await self.integration.display_proposed_changes(
                change_set,
                display_mode=ChangeDisplayMode.UNIFIED,
                interactive=True
            )
            
            assert mock_approval.called
            assert result['approved'] is True
    
    @pytest.mark.asyncio
    async def test_display_inline_changes(self):
        """Test inline display of changes."""
        file_path = "/test/file.py"
        original = "line 1\nline 2\nline 3"
        modified = "line 1\nmodified line 2\nline 3\nnew line 4"
        
        await self.integration.display_inline_changes(file_path, original, modified)
        
        assert self.console.print.called
        # Verify that a panel was printed
        call_args = self.console.print.call_args[0][0]
        assert hasattr(call_args, 'title')  # Panel has a title attribute
    
    @pytest.mark.asyncio
    async def test_display_side_by_side_changes(self):
        """Test side-by-side display of changes."""
        file_path = "/test/file.py"
        original = "def foo():\n    pass"
        modified = "def foo():\n    print('hello')"
        
        with patch.object(self.integration.diff_renderer, 'render_side_by_side_diff') as mock_render:
            await self.integration.display_side_by_side_changes(file_path, original, modified)
            
            assert mock_render.called
            file_change = mock_render.call_args[0][0]
            assert file_change.file_path == file_path
            assert file_change.old_content == original
            assert file_change.new_content == modified
    
    @pytest.mark.asyncio
    async def test_apply_proposed_changes_success(self):
        """Test successful application of proposed changes."""
        change_set = self.create_test_change_set(2)
        change_set.approved = True
        self.integration.pending_changes["test-id"] = change_set
        
        # Mock file system operations
        self.file_system_manager.modify_file = AsyncMock(return_value={'success': True})
        
        result = await self.integration.apply_proposed_changes("test-id")
        
        assert result['success'] is True
        assert len(result['applied']) == 2
        assert len(result['failed']) == 0
        assert change_set.applied is True
    
    @pytest.mark.asyncio
    async def test_apply_proposed_changes_not_approved(self):
        """Test applying changes that haven't been approved."""
        change_set = self.create_test_change_set(2)
        change_set.approved = False
        self.integration.pending_changes["test-id"] = change_set
        
        result = await self.integration.apply_proposed_changes("test-id")
        
        assert result['success'] is False
        assert result['error'] == 'Changes not approved'
    
    @pytest.mark.asyncio
    async def test_apply_proposed_changes_partial_failure(self):
        """Test partial failure when applying changes."""
        change_set = self.create_test_change_set(3)
        change_set.approved = True
        self.integration.pending_changes["test-id"] = change_set
        
        # Mock file system operations with one failure
        results = [
            {'success': True},
            {'success': False, 'error': 'Permission denied'},
            {'success': True}
        ]
        self.file_system_manager.modify_file = AsyncMock(side_effect=results)
        
        result = await self.integration.apply_proposed_changes("test-id")
        
        assert result['success'] is False
        assert len(result['applied']) == 2
        assert len(result['failed']) == 1
        assert result['failed'][0][1] == 'Permission denied'
    
    @pytest.mark.asyncio
    async def test_apply_selected_changes(self):
        """Test applying only selected changes."""
        change_set = self.create_test_change_set(4)
        change_set.approved = True
        self.integration.pending_changes["test-id"] = change_set
        
        self.file_system_manager.modify_file = AsyncMock(return_value={'success': True})
        
        # Apply only changes at indices 0 and 2
        result = await self.integration.apply_proposed_changes("test-id", selected_changes=[0, 2])
        
        assert result['success'] is True
        assert len(result['applied']) == 2
    
    def test_get_change_statistics(self):
        """Test getting statistics for a change set."""
        changes = [
            ProposedChange(
                file_path="/test/file1.py",
                operation_type=ChangeType.FILE_CREATE,
                risk_level=RiskLevel.LOW,
                line_changes={'added': 10, 'removed': 0}
            ),
            ProposedChange(
                file_path="/test/file2.py",
                operation_type=ChangeType.FILE_MODIFY,
                risk_level=RiskLevel.MEDIUM,
                line_changes={'added': 5, 'removed': 3}
            ),
            ProposedChange(
                file_path="/test/file1.py",  # Same file as first
                operation_type=ChangeType.FILE_MODIFY,
                risk_level=RiskLevel.LOW,
                line_changes={'added': 2, 'removed': 1}
            )
        ]
        
        change_set = ChangeSet(
            id="test",
            description="Test",
            changes=changes
        )
        
        stats = self.integration.get_change_statistics(change_set)
        
        assert stats['total_changes'] == 3
        assert stats['files_affected'] == 2  # Only 2 unique files
        assert stats['operations']['file_create'] == 1
        assert stats['operations']['file_modify'] == 2
        assert stats['risk_distribution']['low'] == 2
        assert stats['risk_distribution']['medium'] == 1
        assert stats['total_lines_added'] == 17
        assert stats['total_lines_removed'] == 4
    
    def test_calculate_line_changes(self):
        """Test line change calculation."""
        original = "line 1\nline 2\nline 3"
        modified = "line 1\nmodified line 2\nline 3\nline 4"
        
        changes = self.integration._calculate_line_changes(original, modified)
        
        assert changes['added'] == 2  # modified line 2 + line 4
        assert changes['removed'] == 1  # original line 2
    
    def test_calculate_total_risk(self):
        """Test total risk calculation."""
        changes = [
            ProposedChange(file_path="/test/file1.py", operation_type=ChangeType.FILE_CREATE,
                         risk_level=RiskLevel.LOW),
            ProposedChange(file_path="/test/file2.py", operation_type=ChangeType.FILE_MODIFY,
                         risk_level=RiskLevel.MEDIUM),
            ProposedChange(file_path="/test/file3.py", operation_type=ChangeType.FILE_DELETE,
                         risk_level=RiskLevel.HIGH),
        ]
        
        total_risk = self.integration._calculate_total_risk(changes)
        
        # (1.0 + 3.0 + 7.0) / 3 = 3.67
        assert abs(total_risk - 3.67) < 0.01
    
    def test_create_risk_distribution_bar(self):
        """Test risk distribution bar creation."""
        risk_distribution = {
            'LOW': 5,
            'MEDIUM': 3,
            'HIGH': 1,
            'CRITICAL': 0
        }
        
        bar = self.integration._create_risk_distribution_bar(risk_distribution)
        
        assert 'LOW: 5' in bar
        assert 'MEDIUM: 3' in bar
        assert 'HIGH: 1' in bar
        assert 'CRITICAL' not in bar  # Should not show zero counts
    
    def test_detect_language(self):
        """Test language detection from file path."""
        test_cases = [
            ("/test/file.py", "python"),
            ("/test/file.js", "javascript"),
            ("/test/file.ts", "typescript"),
            ("/test/file.java", "java"),
            ("/test/file.unknown", "text")
        ]
        
        for file_path, expected_lang in test_cases:
            detected = self.integration._detect_language(file_path)
            assert detected == expected_lang
    
    @pytest.mark.asyncio
    async def test_error_handling_fetch_changes(self):
        """Test error handling when fetching changes."""
        self.approval_manager.get_pending_operations = AsyncMock(
            side_effect=Exception("API error")
        )
        
        change_set = await self.integration.fetch_proposed_changes("op-error")
        
        assert change_set.id == "op-error"
        assert change_set.description == "Error"
        assert len(change_set.changes) == 0
    
    @pytest.mark.asyncio
    async def test_error_handling_display_changes(self):
        """Test error handling when displaying changes."""
        change_set = self.create_test_change_set(2)
        
        with patch.object(self.integration, '_display_change_set_header',
                         side_effect=Exception("Display error")):
            result = await self.integration.display_proposed_changes(change_set)
            
            assert result['displayed'] is False
            assert 'error' in result
            assert "Display error" in result['error']