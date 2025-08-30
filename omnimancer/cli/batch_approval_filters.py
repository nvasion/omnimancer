"""
Batch Approval Filtering and Sorting Logic for Omnimancer CLI.

This module provides filtering and sorting capabilities to handle large
batches of approval requests efficiently with complex query support.
"""

import fnmatch
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..core.agent.approval_manager import BatchApprovalRequest, ChangePreview
from ..core.agent.types import Operation, OperationType

logger = logging.getLogger(__name__)


class FilterOperator(Enum):
    """Filter operators for query expressions."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class SortDirection(Enum):
    """Sort direction options."""

    ASCENDING = "asc"
    DESCENDING = "desc"


class SortBy(Enum):
    """Sort criteria options."""

    TIMESTAMP = "timestamp"
    RISK_LEVEL = "risk"
    ACTION_TYPE = "type"
    TARGET = "target"
    STATUS = "status"
    ALPHABETICAL = "alpha"


@dataclass
class FilterCriteria:
    """Represents a single filter criterion."""

    field: str
    operator: str
    value: str
    negated: bool = False


@dataclass
class SortCriteria:
    """Represents sort configuration."""

    sort_by: SortBy
    direction: SortDirection = SortDirection.ASCENDING
    secondary_sort: Optional["SortCriteria"] = None


class BatchFilter(ABC):
    """Abstract base class for batch filters."""

    @abstractmethod
    def matches(
        self, operation: Operation, preview: Optional[ChangePreview] = None
    ) -> bool:
        """Check if operation matches this filter."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of this filter."""
        pass


class ActionTypeFilter(BatchFilter):
    """Filter operations by action type."""

    def __init__(self, operation_types: List[OperationType]):
        """
        Initialize action type filter.

        Args:
            operation_types: List of operation types to match
        """
        self.operation_types = set(operation_types)

    def matches(
        self, operation: Operation, preview: Optional[ChangePreview] = None
    ) -> bool:
        """Check if operation type matches."""
        return operation.type in self.operation_types

    def get_description(self) -> str:
        """Get description of this filter."""
        types = [op.value for op in self.operation_types]
        return f"Type: {', '.join(types)}"


class RiskLevelFilter(BatchFilter):
    """Filter operations by risk level."""

    def __init__(self, risk_levels: List[str]):
        """
        Initialize risk level filter.

        Args:
            risk_levels: List of risk levels to match (low, medium, high, critical)
        """
        self.risk_levels = set(level.lower() for level in risk_levels)

    def matches(
        self, operation: Operation, preview: Optional[ChangePreview] = None
    ) -> bool:
        """Check if risk level matches."""
        if not preview or not preview.risk_assessment:
            return "unknown" in self.risk_levels

        risk_level = self._extract_risk_level(preview.risk_assessment)
        return risk_level in self.risk_levels

    def _extract_risk_level(self, risk_assessment: str) -> str:
        """Extract risk level from assessment string."""
        risk_lower = risk_assessment.lower()
        if "critical" in risk_lower:
            return "critical"
        elif "high" in risk_lower:
            return "high"
        elif "medium" in risk_lower:
            return "medium"
        elif "low" in risk_lower:
            return "low"
        else:
            return "unknown"

    def get_description(self) -> str:
        """Get description of this filter."""
        return f"Risk: {', '.join(self.risk_levels)}"


class TargetPatternFilter(BatchFilter):
    """Filter operations by target pattern using glob matching."""

    def __init__(self, patterns: List[str]):
        """
        Initialize target pattern filter.

        Args:
            patterns: List of glob patterns to match against targets
        """
        self.patterns = patterns

    def matches(
        self, operation: Operation, preview: Optional[ChangePreview] = None
    ) -> bool:
        """Check if operation target matches any pattern."""
        target = self._extract_target(operation)
        if not target:
            return False

        return any(
            fnmatch.fnmatch(target.lower(), pattern.lower())
            for pattern in self.patterns
        )

    def _extract_target(self, operation: Operation) -> Optional[str]:
        """Extract target from operation data."""
        if operation.data:
            # Try common target fields
            for field in ["path", "url", "command", "target"]:
                if field in operation.data:
                    return str(operation.data[field])

        return operation.description

    def get_description(self) -> str:
        """Get description of this filter."""
        return f"Target: {', '.join(self.patterns)}"


class StatusFilter(BatchFilter):
    """Filter operations by approval status."""

    def __init__(self, statuses: List[str], batch_request: BatchApprovalRequest):
        """
        Initialize status filter.

        Args:
            statuses: List of statuses to match (pending, approved)
            batch_request: Batch request for status lookup
        """
        self.statuses = set(status.lower() for status in statuses)
        self.batch_request = batch_request

    def matches(
        self, operation: Operation, preview: Optional[ChangePreview] = None
    ) -> bool:
        """Check if operation status matches."""
        # Find operation index in batch
        try:
            op_index = self.batch_request.operations.index(operation)
            is_approved = op_index in self.batch_request.approved_operations

            current_status = "approved" if is_approved else "pending"
            return current_status in self.statuses
        except ValueError:
            return False

    def get_description(self) -> str:
        """Get description of this filter."""
        return f"Status: {', '.join(self.statuses)}"


class CustomQueryFilter(BatchFilter):
    """Filter operations using custom query expressions."""

    def __init__(self, query: str, batch_request: BatchApprovalRequest):
        """
        Initialize custom query filter.

        Args:
            query: Query expression (e.g., "risk:high AND type:file_write")
            batch_request: Batch request for context
        """
        self.query = query
        self.batch_request = batch_request
        self.parsed_filters = self._parse_query(query)

    def matches(
        self, operation: Operation, preview: Optional[ChangePreview] = None
    ) -> bool:
        """Check if operation matches the query."""
        return self._evaluate_filters(self.parsed_filters, operation, preview)

    def _parse_query(self, query: str) -> List[Any]:
        """Parse query into filter components."""
        # Simple query parser - supports basic expressions like:
        # "risk:high", "type:file_write", "target:*.py"
        # "risk:high AND type:file_write"
        # "NOT status:approved"

        # Tokenize the query
        tokens = re.findall(
            r"\w+:\w+|\w+:\*?\.\w+|\w+:\*|AND|OR|NOT|\(|\)",
            query,
            re.IGNORECASE,
        )

        filters = []
        current_filter = None
        operator = None
        negated = False

        for token in tokens:
            token_upper = token.upper()

            if token_upper == "AND":
                if current_filter:
                    filters.append(current_filter)
                operator = FilterOperator.AND
                current_filter = None
            elif token_upper == "OR":
                if current_filter:
                    filters.append(current_filter)
                operator = FilterOperator.OR
                current_filter = None
            elif token_upper == "NOT":
                negated = True
            elif ":" in token:
                # Field:value expression
                field, value = token.split(":", 1)
                filter_obj = self._create_field_filter(field.lower(), value.lower())
                if filter_obj:
                    if negated:
                        filter_obj = NegatedFilter(filter_obj)
                        negated = False

                    if operator:
                        filters.append((operator, filter_obj))
                        operator = None
                    else:
                        current_filter = filter_obj

        if current_filter:
            filters.append(current_filter)

        return filters

    def _create_field_filter(self, field: str, value: str) -> Optional[BatchFilter]:
        """Create appropriate filter based on field and value."""
        if field == "risk":
            return RiskLevelFilter([value])
        elif field == "type":
            # Map common type names to OperationType
            type_map = {
                "file_read": OperationType.FILE_READ,
                "file_write": OperationType.FILE_WRITE,
                "file_delete": OperationType.FILE_DELETE,
                "command": OperationType.COMMAND_EXECUTE,
                "web": OperationType.WEB_REQUEST,
                "mcp": OperationType.MCP_TOOL_CALL,
            }
            op_type = type_map.get(value)
            if op_type:
                return ActionTypeFilter([op_type])
        elif field == "target":
            return TargetPatternFilter([value])
        elif field == "status":
            return StatusFilter([value], self.batch_request)

        return None

    def _evaluate_filters(
        self,
        filters: List[Any],
        operation: Operation,
        preview: Optional[ChangePreview],
    ) -> bool:
        """Evaluate parsed filters against operation."""
        if not filters:
            return True

        result = True

        for item in filters:
            if isinstance(item, tuple) and len(item) == 2:
                # Operator and filter
                op, filter_obj = item
                filter_result = filter_obj.matches(operation, preview)

                if op == FilterOperator.AND:
                    result = result and filter_result
                elif op == FilterOperator.OR:
                    result = result or filter_result
            elif isinstance(item, BatchFilter):
                # Direct filter
                filter_result = item.matches(operation, preview)
                result = result and filter_result

        return result

    def get_description(self) -> str:
        """Get description of this filter."""
        return f"Query: {self.query}"


class NegatedFilter(BatchFilter):
    """Wrapper for negated filters."""

    def __init__(self, base_filter: BatchFilter):
        """Initialize negated filter."""
        self.base_filter = base_filter

    def matches(
        self, operation: Operation, preview: Optional[ChangePreview] = None
    ) -> bool:
        """Return negated result of base filter."""
        return not self.base_filter.matches(operation, preview)

    def get_description(self) -> str:
        """Get description of negated filter."""
        return f"NOT ({self.base_filter.get_description()})"


class BatchFilterManager:
    """Manages multiple filters and applies them to batch requests."""

    def __init__(self):
        """Initialize batch filter manager."""
        self.active_filters: List[BatchFilter] = []

    def add_filter(self, batch_filter: BatchFilter):
        """Add a filter to the active filters."""
        self.active_filters.append(batch_filter)

    def remove_filter(self, index: int) -> bool:
        """Remove filter by index."""
        if 0 <= index < len(self.active_filters):
            del self.active_filters[index]
            return True
        return False

    def clear_filters(self):
        """Remove all active filters."""
        self.active_filters.clear()

    def apply_filters(
        self, batch_request: BatchApprovalRequest
    ) -> Tuple[List[Operation], List[ChangePreview]]:
        """
        Apply active filters to batch request.

        Args:
            batch_request: Batch request to filter

        Returns:
            Tuple of filtered operations and previews
        """
        if not self.active_filters:
            return batch_request.operations, batch_request.previews

        filtered_ops = []
        filtered_previews = []

        for i, (operation, preview) in enumerate(
            zip(batch_request.operations, batch_request.previews)
        ):
            if self._matches_all_filters(operation, preview):
                filtered_ops.append(operation)
                filtered_previews.append(preview)

        return filtered_ops, filtered_previews

    def _matches_all_filters(
        self, operation: Operation, preview: Optional[ChangePreview]
    ) -> bool:
        """Check if operation matches all active filters."""
        return all(f.matches(operation, preview) for f in self.active_filters)

    def get_filter_descriptions(self) -> List[str]:
        """Get descriptions of all active filters."""
        return [f.get_description() for f in self.active_filters]

    def get_filter_count(self) -> int:
        """Get number of active filters."""
        return len(self.active_filters)


class BatchSorter:
    """Handles sorting of batch operations."""

    def __init__(self):
        """Initialize batch sorter."""
        self.sort_functions = {
            SortBy.TIMESTAMP: self._sort_by_timestamp,
            SortBy.RISK_LEVEL: self._sort_by_risk,
            SortBy.ACTION_TYPE: self._sort_by_action_type,
            SortBy.TARGET: self._sort_by_target,
            SortBy.STATUS: self._sort_by_status,
            SortBy.ALPHABETICAL: self._sort_by_alphabetical,
        }

    def sort_batch(
        self,
        operations: List[Operation],
        previews: List[ChangePreview],
        criteria: SortCriteria,
        batch_request: Optional[BatchApprovalRequest] = None,
    ) -> Tuple[List[Operation], List[ChangePreview]]:
        """
        Sort operations and previews according to criteria.

        Args:
            operations: Operations to sort
            previews: Corresponding previews
            criteria: Sort criteria
            batch_request: Optional batch request for additional context

        Returns:
            Tuple of sorted operations and previews
        """
        if not operations:
            return operations, previews

        # Create list of tuples for sorting
        items = list(zip(operations, previews))

        # Get sort function
        sort_func = self.sort_functions.get(criteria.sort_by)
        if not sort_func:
            logger.warning(f"Unknown sort criteria: {criteria.sort_by}")
            return operations, previews

        # Sort items
        sorted_items = sort_func(items, criteria.direction, batch_request)

        # Apply secondary sort if specified
        if criteria.secondary_sort:
            secondary_func = self.sort_functions.get(criteria.secondary_sort.sort_by)
            if secondary_func:
                # Stable sort to preserve primary ordering
                sorted_items = secondary_func(
                    sorted_items,
                    criteria.secondary_sort.direction,
                    batch_request,
                )

        # Unzip back to separate lists
        if sorted_items:
            sorted_ops, sorted_previews = zip(*sorted_items)
            return list(sorted_ops), list(sorted_previews)
        else:
            return [], []

    def _sort_by_timestamp(
        self,
        items: List[Tuple[Operation, ChangePreview]],
        direction: SortDirection,
        batch_request: Optional[BatchApprovalRequest],
    ) -> List[Tuple[Operation, ChangePreview]]:
        """Sort by timestamp (using batch creation time as proxy)."""
        # Since operations don't have individual timestamps, use batch timestamp
        # In practice, this would sort by when each operation was added
        batch_request.created_at if batch_request else datetime.now()

        # For now, maintain original order since all operations have same timestamp
        return items if direction == SortDirection.ASCENDING else list(reversed(items))

    def _sort_by_risk(
        self,
        items: List[Tuple[Operation, ChangePreview]],
        direction: SortDirection,
        batch_request: Optional[BatchApprovalRequest],
    ) -> List[Tuple[Operation, ChangePreview]]:
        """Sort by risk level."""
        risk_order = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
            "unknown": 0,
        }

        def get_risk_score(item):
            operation, preview = item
            if preview and preview.risk_assessment:
                risk_level = self._extract_risk_level(preview.risk_assessment)
                return risk_order.get(risk_level, 0)
            return 0

        reverse = direction == SortDirection.DESCENDING
        return sorted(items, key=get_risk_score, reverse=reverse)

    def _sort_by_action_type(
        self,
        items: List[Tuple[Operation, ChangePreview]],
        direction: SortDirection,
        batch_request: Optional[BatchApprovalRequest],
    ) -> List[Tuple[Operation, ChangePreview]]:
        """Sort by action type."""

        def get_type_name(item):
            operation, preview = item
            return operation.type.value

        reverse = direction == SortDirection.DESCENDING
        return sorted(items, key=get_type_name, reverse=reverse)

    def _sort_by_target(
        self,
        items: List[Tuple[Operation, ChangePreview]],
        direction: SortDirection,
        batch_request: Optional[BatchApprovalRequest],
    ) -> List[Tuple[Operation, ChangePreview]]:
        """Sort by operation target."""

        def get_target(item):
            operation, preview = item
            if operation.data:
                for field in ["path", "url", "command", "target"]:
                    if field in operation.data:
                        return str(operation.data[field]).lower()
            return operation.description.lower() if operation.description else ""

        reverse = direction == SortDirection.DESCENDING
        return sorted(items, key=get_target, reverse=reverse)

    def _sort_by_status(
        self,
        items: List[Tuple[Operation, ChangePreview]],
        direction: SortDirection,
        batch_request: Optional[BatchApprovalRequest],
    ) -> List[Tuple[Operation, ChangePreview]]:
        """Sort by approval status."""

        def get_status_score(item):
            if not batch_request:
                return 0

            operation, preview = item
            try:
                op_index = batch_request.operations.index(operation)
                is_approved = op_index in batch_request.approved_operations
                return 1 if is_approved else 0
            except ValueError:
                return 0

        reverse = direction == SortDirection.DESCENDING
        return sorted(items, key=get_status_score, reverse=reverse)

    def _sort_by_alphabetical(
        self,
        items: List[Tuple[Operation, ChangePreview]],
        direction: SortDirection,
        batch_request: Optional[BatchApprovalRequest],
    ) -> List[Tuple[Operation, ChangePreview]]:
        """Sort alphabetically by description."""

        def get_description(item):
            operation, preview = item
            return operation.description.lower() if operation.description else ""

        reverse = direction == SortDirection.DESCENDING
        return sorted(items, key=get_description, reverse=reverse)

    def _extract_risk_level(self, risk_assessment: str) -> str:
        """Extract risk level from assessment string."""
        risk_lower = risk_assessment.lower()
        if "critical" in risk_lower:
            return "critical"
        elif "high" in risk_lower:
            return "high"
        elif "medium" in risk_lower:
            return "medium"
        elif "low" in risk_lower:
            return "low"
        else:
            return "unknown"


class BatchPaginator:
    """Handles pagination of filtered and sorted results."""

    def __init__(self, page_size: int = 10):
        """
        Initialize paginator.

        Args:
            page_size: Number of items per page
        """
        self.page_size = page_size

    def paginate(
        self,
        operations: List[Operation],
        previews: List[ChangePreview],
        page: int = 0,
    ) -> Tuple[List[Operation], List[ChangePreview], Dict[str, Any]]:
        """
        Paginate operations and previews.

        Args:
            operations: Operations to paginate
            previews: Corresponding previews
            page: Page number (0-based)

        Returns:
            Tuple of (page_operations, page_previews, pagination_info)
        """
        total_items = len(operations)
        total_pages = (total_items + self.page_size - 1) // self.page_size

        # Ensure page is within bounds
        page = max(0, min(page, total_pages - 1))

        start_idx = page * self.page_size
        end_idx = min(start_idx + self.page_size, total_items)

        page_operations = operations[start_idx:end_idx]
        page_previews = previews[start_idx:end_idx]

        pagination_info = {
            "current_page": page,
            "total_pages": total_pages,
            "page_size": self.page_size,
            "total_items": total_items,
            "start_index": start_idx,
            "end_index": end_idx,
            "has_previous": page > 0,
            "has_next": page < total_pages - 1,
        }

        return page_operations, page_previews, pagination_info


def create_quick_filters() -> Dict[str, BatchFilter]:
    """Create commonly used quick filters."""
    return {
        "high_risk": RiskLevelFilter(["high", "critical"]),
        "file_ops": ActionTypeFilter(
            [OperationType.FILE_WRITE, OperationType.FILE_DELETE]
        ),
        "read_only": ActionTypeFilter([OperationType.FILE_READ]),
        "commands": ActionTypeFilter([OperationType.COMMAND_EXECUTE]),
        "web_requests": ActionTypeFilter([OperationType.WEB_REQUEST]),
        "python_files": TargetPatternFilter(["*.py"]),
        "config_files": TargetPatternFilter(
            ["*.conf", "*.config", "*.json", "*.yaml", "*.yml"]
        ),
    }


def parse_filter_expression(
    expression: str, batch_request: BatchApprovalRequest
) -> Optional[BatchFilter]:
    """
    Parse a filter expression into a BatchFilter.

    Args:
        expression: Filter expression string
        batch_request: Batch request for context

    Returns:
        BatchFilter instance or None if parsing failed
    """
    try:
        return CustomQueryFilter(expression, batch_request)
    except Exception as e:
        logger.error(f"Failed to parse filter expression '{expression}': {e}")
        return None


def create_default_sort_criteria() -> SortCriteria:
    """Create default sort criteria (by risk level, descending)."""
    return SortCriteria(
        sort_by=SortBy.RISK_LEVEL,
        direction=SortDirection.DESCENDING,
        secondary_sort=SortCriteria(
            sort_by=SortBy.ACTION_TYPE, direction=SortDirection.ASCENDING
        ),
    )
