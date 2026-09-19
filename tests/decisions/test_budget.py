from concurrent.futures import ThreadPoolExecutor

import pytest

from omnimancer.decisions.budget import BudgetExhausted, BudgetLedger


def test_reservations_survive_reopen_and_failures(tmp_path):
    path = tmp_path / "ledger.sqlite"
    first = BudgetLedger(path, 0.006)
    first.reserve("jev-1.13.0")
    second = BudgetLedger(path, 0.006)
    second.reserve("jev-1.13.0")
    with pytest.raises(BudgetExhausted):
        first.reserve("jev-1.13.0")
    assert second.summary()["reserved_usd"] == 0.006
    assert second.summary()["unsettled_attempts"] == 2


def test_cannot_raise_existing_cap_or_use_unknown_price(tmp_path):
    path = tmp_path / "ledger.sqlite"
    BudgetLedger(path, 0.003).reserve("jev-1.13.0")
    budget = BudgetLedger(path, 5)
    with pytest.raises(BudgetExhausted):
        budget.reserve("jev-1.13.0")
    with pytest.raises(ValueError):
        budget.reserve("future-model")


def test_usage_is_separate_from_conservative_reservation(tmp_path):
    budget = BudgetLedger(tmp_path / "ledger.sqlite", 1)
    attempt = budget.reserve("jev-1.13.0")
    budget.record_usage(attempt, 1000)
    assert budget.summary()["estimated_usage_usd"] == 0.000042
    assert budget.summary()["reserved_usd"] == 0.003
    with pytest.raises(ValueError):
        budget.record_usage(attempt, 1000)


def test_parallel_reservations_cannot_overspend(tmp_path):
    budget = BudgetLedger(tmp_path / "ledger.sqlite", 0.006)

    def attempt(_):
        try:
            budget.reserve("jev-1.13.0")
            return True
        except BudgetExhausted:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(20))) == 2
    assert budget.summary()["attempts"] == 2


@pytest.mark.parametrize("amount", [0, -1, float("nan"), float("inf"), True])
def test_invalid_budget_rejected(tmp_path, amount):
    with pytest.raises(ValueError):
        BudgetLedger(tmp_path / "ledger.sqlite", amount)


@pytest.mark.parametrize("tokens", [-1, True, 1.5, 65537])
def test_invalid_usage_rejected(tmp_path, tokens):
    budget = BudgetLedger(tmp_path / "ledger.sqlite", 1)
    attempt = budget.reserve("jev-1.13.0")
    with pytest.raises(ValueError):
        budget.record_usage(attempt, tokens)


def test_ledger_rejects_symlinks_without_modifying_target(tmp_path):
    target = tmp_path / "unrelated"
    import sqlite3

    with sqlite3.connect(target) as db:
        db.execute("CREATE TABLE unrelated (value TEXT)")
        db.execute("INSERT INTO unrelated VALUES ('preserve me')")
    before = target.read_bytes()
    link = tmp_path / "ledger.sqlite"
    link.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        BudgetLedger(link, 5)
    assert target.read_bytes() == before


def test_ledger_requires_private_directory(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    shared.chmod(0o777)
    with pytest.raises(ValueError, match="private"):
        BudgetLedger(shared / "ledger.sqlite", 5)
    assert not (shared / "ledger.sqlite").exists()


def test_ledger_rejects_precreated_shared_file(tmp_path):
    path = tmp_path / "ledger.sqlite"
    path.touch(mode=0o666)
    path.chmod(0o666)
    with pytest.raises(ValueError, match="private"):
        BudgetLedger(path, 5)
    assert path.read_bytes() == b""


def test_ledger_allocates_private_file(tmp_path):
    import stat

    path = tmp_path / "ledger.sqlite"
    BudgetLedger(path, 5)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
