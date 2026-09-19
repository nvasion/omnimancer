"""Regression checks against the documented TypeSafe wire and billing contract."""

import copy

import httpx
import pytest
from pydantic import ValidationError

from omnimancer.decisions.routing import RoutingPolicy, classify_route

POLICY = {
    "targets": {
        "fast": {"provider": "local", "model": "small", "criteria": "simple"},
        "deep": {"provider": "local", "model": "large", "criteria": "complex"},
    }
}
RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "route": {
            "type": "choice",
            "choice": "fast",
            "confidence": 1.0,
            "probabilities": {"fast": 1.0, "deep": 0.0},
        }
    },
    "usage": {"input_tokens": 1000, "output_tokens": 1000},
}


@pytest.mark.asyncio
async def test_output_tokens_are_free():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=RESPONSE)
        )
    ) as client:
        decision = await classify_route(
            "Fix a typo", RoutingPolicy(**POLICY), api_key="synthetic", client=client
        )
    assert decision.estimated_cost_usd == pytest.approx(0.000042)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["extra_answer", "model_newline", "excess_usage", "wrong_model"]
)
async def test_rejects_invalid_contract(mutation):
    response = copy.deepcopy(RESPONSE)
    if mutation == "extra_answer":
        response["answers"]["unexpected"] = {"raw": "private"}
    elif mutation == "model_newline":
        response["model"] += "\n"
    elif mutation == "wrong_model":
        response["model"] = "jev-1.12.0"
    else:
        response["usage"]["input_tokens"] = 65537
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=response)
        )
    ) as client:
        decision = await classify_route(
            "Fix a typo", RoutingPolicy(**POLICY), api_key="synthetic", client=client
        )
    assert decision.status == "invalid_response"
    assert decision.model is None
    assert decision.target is None


def test_rejects_newline_in_target_label():
    policy = copy.deepcopy(POLICY)
    policy["targets"]["fast\n"] = policy["targets"].pop("fast")
    with pytest.raises(ValidationError):
        RoutingPolicy(**policy)
