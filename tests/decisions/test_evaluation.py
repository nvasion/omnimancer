import json

import httpx
import pytest

from omnimancer.decisions.budget import BudgetLedger
from omnimancer.decisions.evaluation import (
    Case,
    evaluate,
    load_cases,
    validate_local_endpoint,
)
from omnimancer.decisions.routing import RouteTarget, RoutingPolicy


def policy():
    return RoutingPolicy(
        targets={
            "fast": RouteTarget(provider="fast", model="fast-model", criteria="simple"),
            "deep": RouteTarget(
                provider="deep", model="deep-model", criteria="complex"
            ),
        }
    )


def test_reject_duplicate_case_ids(tmp_path):
    case = dict(
        id="one", split="test", category="simple", prompt="Fix label", expected="fast"
    )
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([case, case]))
    with pytest.raises(ValueError):
        load_cases(path)


@pytest.mark.parametrize(
    "url",
    [
        "https://cloud.example/v1",
        "http://127.0.0.1.evil/v1",
        "http://user:password@localhost/v1",
        "file:///tmp/a",
    ],
)
def test_local_tasks_cannot_use_cloud_or_credentials(url):
    with pytest.raises(ValueError):
        validate_local_endpoint(url)


@pytest.mark.asyncio
async def test_seeded_order_and_budget_stop_before_network(tmp_path):
    cases = [
        Case(
            id=f"case-{i}",
            split="test",
            category="simple",
            prompt="Change the exact label.",
            expected="fast",
        )
        for i in range(4)
    ]
    calls = []

    def transport(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": {
                    "route": {
                        "type": "choice",
                        "choice": "fast",
                        "confidence": 1.0,
                        "probabilities": {"fast": 1.0, "deep": 0.0},
                    }
                },
                "usage": {"input_tokens": 100, "output_tokens": 10},
            },
        )

    budget = BudgetLedger(tmp_path / "ledger.sqlite", 0.003)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        report = await evaluate(
            cases,
            policy(),
            live=True,
            budget=budget,
            seed=42,
            repetitions=1,
            client=client,
            api_key="synthetic-key",
        )
    assert len(calls) == 1
    assert [r.status for r in report.records if r.arm == "jev"].count(
        "budget_exhausted"
    ) == 3
    assert report.manifest.planned_jev_decisions == 4
    assert len(report.records) == 12
    one = await evaluate(cases, policy(), seed=42, repetitions=1)
    two = await evaluate(cases, policy(), seed=42, repetitions=1)
    assert [r.case_id for r in one.records] == [r.case_id for r in two.records]
    assert one.manifest.fixture_sha256 == two.manifest.fixture_sha256


@pytest.mark.asyncio
async def test_offline_never_calls_endpoint():
    case = Case(
        id="one", split="test", category="simple", prompt="Fix label", expected="fast"
    )

    def forbidden(request):
        raise AssertionError("Unexpected network")

    async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as client:
        report = await evaluate([case], policy(), client=client, repetitions=1)
    assert not report.manifest.live
    assert [r.status for r in report.records if r.arm == "jev"] == ["offline"]


@pytest.mark.asyncio
@pytest.mark.parametrize("update", [{"mode": "shadow"}, {"model": "jev-1.12.0"}])
async def test_evaluation_rejects_policy_outside_measured_contract(update):
    invalid = policy().model_copy(update=update)
    with pytest.raises(ValueError, match="pinned route policy"):
        await evaluate([], invalid)
