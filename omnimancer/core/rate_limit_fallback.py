"""
Rate-limit / quota fallback handler for Omnimancer.

When a provider returns a 429 (Too Many Requests) or quota-exceeded error this
module decides:
  1. Whether the error warrants a fallback attempt (based on FallbackConfig).
  2. Which provider to fall back to (fallback_order or first available).
  3. Whether the switch is approved (auto mode or interactive callback).

The *interactive* approval path is wired up by the CLI: it registers an
``ApprovalCallback`` that shows a Rich prompt and returns the user's answer.
In headless / non-interactive mode no callback is registered and the handler
falls back only when ``auto_fallback=True``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Awaitable, Callable, List, Optional

if TYPE_CHECKING:
    from .models import FallbackConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callback type
# ---------------------------------------------------------------------------

# async (current_provider, next_provider, error_msg) -> bool
# Returns True  → proceed with the fallback switch.
# Returns False → stay on the current provider (surface the error).
ApprovalCallback = Callable[[str, str, str], Awaitable[bool]]

# async (current_provider, next_provider) -> None
# Called *after* a switch has taken place (for auto-fallback notifications).
SwitchNotificationCallback = Callable[[str, str], Awaitable[None]]


# ---------------------------------------------------------------------------
# Patterns used to recognise rate-limit and quota errors
# ---------------------------------------------------------------------------

_RATE_LIMIT_PATTERNS: tuple[str, ...] = (
    "rate limit",
    "ratelimit",
    "rate_limit",
    "429",
    "too many requests",
    "requests per minute",
    "tokens per minute",
    "capacity",
)

_QUOTA_PATTERNS: tuple[str, ...] = (
    "quota exceeded",
    "quota_exceeded",
    "usage limit",
    "insufficient_quota",
    "billing",
    "payment required",
    "402",
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RateLimitFallbackHandler:
    """Encapsulates all logic for deciding *if* and *how* to fall back.

    This class is intentionally stateless with respect to providers — it
    operates on names (strings) and delegates switching to the engine.  This
    makes it easy to unit-test without spinning up real providers.

    Parameters
    ----------
    fallback_order:
        Ordered list of provider names to consider.  May be empty — in that
        case the handler falls back to *any* available provider that is not
        the failing one.
    auto_fallback:
        When True the switch is performed immediately without asking.
    fallback_on_rate_limit:
        Enable fallback for HTTP 429 / rate-limit errors.
    fallback_on_quota:
        Enable fallback for quota-exceeded errors.
    """

    def __init__(
        self,
        fallback_order: Optional[List[str]] = None,
        auto_fallback: bool = False,
        fallback_on_rate_limit: bool = True,
        fallback_on_quota: bool = False,
    ) -> None:
        self.fallback_order: List[str] = list(fallback_order or [])
        self.auto_fallback = auto_fallback
        self.fallback_on_rate_limit = fallback_on_rate_limit
        self.fallback_on_quota = fallback_on_quota

        self._approval_callback: Optional[ApprovalCallback] = None
        self._switch_notification_callback: Optional[SwitchNotificationCallback] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def update_from_config(self, config: FallbackConfig) -> None:
        """Sync settings from a :class:`FallbackConfig` instance."""
        from .models import FallbackConfig  # local import to avoid circular

        if not isinstance(config, FallbackConfig):
            return
        self.fallback_order = list(config.fallback_order)
        self.auto_fallback = config.auto_fallback
        self.fallback_on_rate_limit = config.fallback_on_rate_limit
        self.fallback_on_quota = config.fallback_on_quota

    def set_approval_callback(self, callback: ApprovalCallback) -> None:
        """Register the interactive-approval callback (used by the CLI)."""
        self._approval_callback = callback

    def set_switch_notification_callback(
        self, callback: SwitchNotificationCallback
    ) -> None:
        """Register a callback that is fired *after* an auto-switch completes."""
        self._switch_notification_callback = callback

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    def is_rate_limit_error(self, error: str) -> bool:
        """Return True if *error* looks like an HTTP 429 / rate-limit error."""
        lowered = error.lower()
        return any(p in lowered for p in _RATE_LIMIT_PATTERNS)

    def is_quota_error(self, error: str) -> bool:
        """Return True if *error* looks like a quota / billing error."""
        lowered = error.lower()
        return any(p in lowered for p in _QUOTA_PATTERNS)

    def should_fallback(self, error: str) -> bool:
        """Return True if this error should trigger a fallback attempt."""
        if self.fallback_on_rate_limit and self.is_rate_limit_error(error):
            return True
        if self.fallback_on_quota and self.is_quota_error(error):
            return True
        return False

    # ------------------------------------------------------------------
    # Provider selection
    # ------------------------------------------------------------------

    def get_next_provider(
        self,
        current_provider: str,
        available_providers: List[str],
    ) -> Optional[str]:
        """Return the name of the next provider to try, or None.

        The selection algorithm:

        1. If ``fallback_order`` is set, walk through it starting *after*
           the current provider's position.  The first entry that is in
           ``available_providers`` wins.  If the current provider is not in
           the list, the first available entry is used.

        2. If ``fallback_order`` is empty, return the first entry in
           ``available_providers`` that is different from ``current_provider``.
        """
        candidates = [p for p in available_providers if p != current_provider]
        if not candidates:
            return None

        if not self.fallback_order:
            return candidates[0]

        # Try to find current provider's position in the order list and
        # walk forward from there.
        try:
            start = self.fallback_order.index(current_provider)
            ordered_after = self.fallback_order[start + 1 :]
        except ValueError:
            # Current provider not in the order list — start from the beginning.
            ordered_after = self.fallback_order

        for name in ordered_after:
            if name in candidates:
                return name

        # Wrap around: nothing useful after current position, try from start.
        for name in self.fallback_order:
            if name in candidates:
                return name

        # Fallback_order exists but none of its entries are available — pick any.
        return candidates[0]

    # ------------------------------------------------------------------
    # Approval
    # ------------------------------------------------------------------

    async def request_approval(
        self,
        current_provider: str,
        next_provider: str,
        error: str,
    ) -> bool:
        """Ask whether the fallback switch should proceed.

        * ``auto_fallback=True``  → always returns True, fires the notification
          callback to inform the user.
        * ``auto_fallback=False`` → calls the registered interactive callback.
          If no callback is registered, returns False (safe default — do not
          silently switch in non-interactive environments).
        """
        if self.auto_fallback:
            if self._switch_notification_callback:
                try:
                    await self._switch_notification_callback(
                        current_provider, next_provider
                    )
                except Exception as exc:
                    logger.warning("Switch notification callback failed: %s", exc)
            return True

        if self._approval_callback:
            try:
                return await self._approval_callback(
                    current_provider, next_provider, error
                )
            except Exception as exc:
                logger.warning("Fallback approval callback failed: %s", exc)
                return False

        # No callback registered and not auto → do not fall back silently.
        logger.debug(
            "Fallback not attempted: auto_fallback=False and no approval "
            "callback registered."
        )
        return False
