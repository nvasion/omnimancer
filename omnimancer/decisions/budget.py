"""Persistent, conservative reservations for opt-in TypeSafe evaluations.

Reservations are never refunded: even a timeout could have been billed. This
limits maximum possible spend across runs rather than claiming invoice totals.
The 2026-09-19 price for jev-1.13.0 is $0.042/M input tokens, output free.
Its 64K request ceiling costs less than the $0.003 reserved for each attempt.
"""

import os
import sqlite3
import stat
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterator

MODEL = "jev-1.13.0"
RESERVATION_NANO_USD = 3_000_000
NANO_USD = 1_000_000_000


class BudgetExhausted(RuntimeError):
    """No request may start without sufficient reserved budget."""


class BudgetLedger:
    """An SQLite transaction is the cross-process request reservation boundary."""

    def __init__(self, path: Path, budget_usd: float) -> None:
        try:
            amount = Decimal(str(budget_usd))
            if (
                isinstance(budget_usd, bool)
                or not amount.is_finite()
                or not 0 < amount <= 1_000_000
            ):
                raise ValueError("Budget must be finite and positive")
            limit = int(amount * NANO_USD)
            if limit < 1:
                raise ValueError("Budget is too small")
        except (InvalidOperation, TypeError):
            raise ValueError("Invalid budget") from None
        supplied = Path(path).absolute()
        self.path = supplied.parent.resolve() / supplied.name
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "CREATE TABLE IF NOT EXISTS limits "
                "(id INTEGER PRIMARY KEY CHECK(id=1), nano_usd INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts "
                "(id INTEGER PRIMARY KEY, reserved INTEGER NOT NULL, actual INTEGER)"
            )
            db.execute("INSERT OR IGNORE INTO limits VALUES (1, ?)", (limit,))
            # Reopening cannot quietly reset or raise an experiment's cap.
            db.execute("UPDATE limits SET nano_usd=MIN(nano_usd, ?)", (limit,))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._prepare_private_file()
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _prepare_private_file(self) -> None:
        """Reject shared/symlink storage before SQLite can open or modify it.

        A private parent plus trusted ancestors prevents another local account
        from swapping the filename between validation and SQLite's own open.
        Root-owned sticky temporary directories are safe ancestors. This is a
        spend guard for cooperative runs, not protection against the owner
        deliberately modifying their own ledger or using another API client.
        """
        if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
            raise ValueError("Live budgets require POSIX private ledger storage")
        uid = os.getuid()
        parent = self.path.parent
        info = parent.stat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != uid
            or stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise ValueError("Ledger directory must be private (0700) and owned")
        for ancestor in parent.parents:
            info = ancestor.stat()
            writable = stat.S_IMODE(info.st_mode) & 0o022
            if info.st_uid not in (0, uid) or (
                writable and not info.st_mode & stat.S_ISVTX
            ):
                raise ValueError("Ledger needs a trusted private directory path")
        try:
            descriptor = os.open(
                self.path,
                os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
            )
        except OSError:
            raise ValueError("Ledger must be a private regular file") from None
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != uid
                or info.st_nlink != 1
                or stat.S_IMODE(info.st_mode) & 0o077
            ):
                raise ValueError("Ledger must be a private owned regular file")
        finally:
            os.close(descriptor)

    def reserve(self, model: str) -> int:
        """Commit a worst-case charge BEFORE sending a single API attempt."""
        if model != MODEL:
            raise ValueError("No verified price/context bound for requested model")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            limit = db.execute("SELECT nano_usd FROM limits WHERE id=1").fetchone()[0]
            spent = db.execute(
                "SELECT COALESCE(SUM(reserved), 0) FROM attempts"
            ).fetchone()[0]
            if spent + RESERVATION_NANO_USD > limit:
                raise BudgetExhausted("Experiment budget exhausted before request")
            row = db.execute(
                "INSERT INTO attempts (reserved) VALUES (?)", (RESERVATION_NANO_USD,)
            )
            assert row.lastrowid is not None
            return row.lastrowid

    def record_usage(self, attempt: int, input_tokens: int) -> None:
        """Record known usage without freeing the conservative reservation."""
        if type(input_tokens) is not int or not 0 <= input_tokens <= 65_536:
            raise ValueError("Usage exceeds the verified model bound")
        with self._connect() as db:
            row = db.execute(
                "UPDATE attempts SET actual=? WHERE id=? AND actual IS NULL",
                (input_tokens * 42, attempt),
            )
            if row.rowcount != 1:
                raise ValueError("Unknown or already settled reservation")

    def summary(self) -> dict:
        """Only portable numeric metadata; no ledger path or raw request data."""
        with self._connect() as db:
            limit = db.execute("SELECT nano_usd FROM limits WHERE id=1").fetchone()[0]
            count, reserved, actual, settled = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(reserved),0), "
                "COALESCE(SUM(actual),0), COUNT(actual) FROM attempts"
            ).fetchone()
        return {
            "cap_usd": limit / NANO_USD,
            "reserved_usd": reserved / NANO_USD,
            "estimated_usage_usd": actual / NANO_USD,
            "attempts": count,
            "unsettled_attempts": count - settled,
        }
