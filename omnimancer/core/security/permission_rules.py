"""Config-driven permission rules layered over the approval workflow.

A :class:`PermissionRuleEngine` turns a user's
:class:`~omnimancer.core.models.PermissionsConfig` into a decision for a given
operation:

* ``DENY``  — refuse without prompting.
* ``ASK``   — always prompt, even if the operation would otherwise be
  auto-approved or remembered.
* ``ALLOW`` — auto-approve without prompting (and authorize sensitive
  project-local writes such as ``.env``).
* ``DEFAULT`` — no rule matched; fall back to the normal approval workflow.

Precedence is ``deny > ask > allow > default`` so the safest matching rule wins
regardless of ordering in the config.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import List, Optional

from ..models import PermissionRule, PermissionsConfig

logger = logging.getLogger(__name__)


class PermissionDecision(str, Enum):
    """Outcome of evaluating permission rules for an operation."""

    DENY = "deny"
    ASK = "ask"
    ALLOW = "allow"
    DEFAULT = "default"


class PermissionRuleEngine:
    """Evaluates :class:`PermissionsConfig` rules against an operation."""

    def __init__(self, config: Optional[PermissionsConfig]) -> None:
        self._config = (
            config if isinstance(config, PermissionsConfig) else PermissionsConfig()
        )

    def evaluate(self, tool: str, target: str = "") -> PermissionDecision:
        """Return the decision for an operation.

        Args:
            tool: OperationType value (e.g. ``"file_write"``).
            target: The operation's path/command/url, tested against matchers.
        """
        if not self._config.enabled:
            return PermissionDecision.DEFAULT

        # Precedence: deny > ask > allow.
        if self._matches_any(self._config.always_deny, tool, target):
            return PermissionDecision.DENY
        if self._matches_any(self._config.always_ask, tool, target):
            return PermissionDecision.ASK
        if self._matches_any(self._config.always_allow, tool, target):
            return PermissionDecision.ALLOW
        return PermissionDecision.DEFAULT

    @staticmethod
    def _matches_any(rules: List[PermissionRule], tool: str, target: str) -> bool:
        return any(PermissionRuleEngine._matches(rule, tool, target) for rule in rules)

    @staticmethod
    def _matches(rule: PermissionRule, tool: str, target: str) -> bool:
        if rule.tool != "*" and rule.tool != tool:
            return False
        if rule.matcher:
            try:
                if not re.search(rule.matcher, target):
                    return False
            except re.error as e:
                logger.warning(
                    "Permission rule has an invalid matcher %r: %s",
                    rule.matcher,
                    e,
                )
                return False
        return True
