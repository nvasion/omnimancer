"""Reproducible synthetic routing evaluation; live inference is opt-in."""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import Field

from .budget import MODEL, BudgetExhausted, BudgetLedger
from .provenance import implementation_digest, policy_digest, source_provenance
from .report import (
    BudgetSummary,
    CaseResult,
    EvaluationReport,
    Manifest,
    PublicModel,
    render_report,
)
from .routing import RoutingDecision, RoutingPolicy, classify_route


class Case(PublicModel):
    id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
    split: Literal["development", "test"]
    category: Literal["simple", "complex", "ambiguous", "adversarial"]
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]
    expected: Literal["fast", "deep"]


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def read_json(path: Path, maximum: int = 1_000_000) -> Any:
    with path.open("rb") as file:
        data = file.read(maximum + 1)
    if len(data) > maximum:
        raise ValueError("Input file exceeds size limit")
    return json.loads(data)


def load_cases(path: Path) -> list[Case]:
    raw = read_json(path)
    if not isinstance(raw, list) or not 1 <= len(raw) <= 256:
        raise ValueError("Expected 1–256 synthetic cases")
    cases = [Case.model_validate(item) for item in raw]
    if len({c.id for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    return cases


def validate_local_endpoint(endpoint: str) -> str:
    value = urlsplit(endpoint)
    if (
        value.scheme != "http"
        or value.hostname not in ("localhost", "127.0.0.1", "::1")
        or value.username
        or value.password
        or value.query
        or value.fragment
        or value.path.rstrip("/") != "/v1"
    ):
        raise ValueError("Task replay requires a keyless loopback HTTP /v1 endpoint")
    return endpoint.rstrip("/")


def heuristic(prompt: str) -> str:
    words = (
        "security",
        "authentication",
        "authorization",
        "architecture",
        "concurren",
        "deadlock",
        "race condition",
        "intermittent",
        "investigate",
        "unknown cause",
        "multi-file",
        "distributed",
        "migration",
        "cryptograph",
        "ambiguous",
        "algorithm",
    )
    return "deep" if any(word in prompt.lower() for word in words) else "fast"


async def evaluate(
    cases: list[Case],
    policy: RoutingPolicy,
    *,
    live: bool = False,
    budget: BudgetLedger | None = None,
    repetitions: int = 3,
    seed: int = 19,
    client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
) -> EvaluationReport:
    if set(policy.targets) != {"fast", "deep"}:
        raise ValueError("This evaluation requires fast and deep target labels")
    if policy.mode != "route" or policy.model != MODEL:
        raise ValueError("This evaluation requires the pinned route policy")
    if not 1 <= repetitions <= 20:
        raise ValueError("Repetitions must be between 1 and 20")
    if live and budget is None:
        raise ValueError("Live evaluation requires a persistent budget ledger")
    fixture = [case.model_dump() for case in cases]
    manifest = Manifest(
        date=datetime.now(timezone.utc).isoformat(),
        live=live,
        seed=seed,
        repetitions=repetitions,
        planned_jev_decisions=len(cases) * repetitions,
        fixture_sha256=canonical_hash(fixture),
        policy_sha256=policy_digest(policy),
        implementation_sha256=implementation_digest(),
        confidence_threshold=policy.min_confidence,
        python_version=platform.python_version(),
        httpx_version=httpx.__version__,
        **source_provenance(),
    )
    report = EvaluationReport(manifest=manifest)
    order = [(case, repeat) for repeat in range(repetitions) for case in cases]
    random.Random(seed).shuffle(order)
    owned = live and client is None
    if owned:
        client = httpx.AsyncClient(follow_redirects=False)
    first_call = True
    try:
        for case, repeat in order:
            common = dict(
                case_id=case.id,
                split=case.split,
                category=case.category,
                prompt=case.prompt,
                expected=case.expected,
                repeat=repeat,
            )
            for arm in ("baseline", "heuristic"):
                started = time.perf_counter()
                choice = "deep" if arm == "baseline" else heuristic(case.prompt)
                report.records.append(
                    CaseResult.model_validate(
                        {
                            **common,
                            "arm": arm,
                            "choice": choice,
                            "target": choice,
                            "status": arm,
                            "elapsed_ms": (time.perf_counter() - started) * 1000,
                        }
                    )
                )
            decision = RoutingDecision(status="offline")
            cold = False
            if live:
                assert budget is not None
                try:
                    attempt = budget.reserve(policy.model)
                except BudgetExhausted:
                    decision = RoutingDecision(status="budget_exhausted")
                else:
                    cold = first_call
                    first_call = False
                    decision = await classify_route(
                        case.prompt, policy, client=client, api_key=api_key
                    )
                    if (
                        decision.input_tokens is not None
                        and decision.model == "jev-1.13.0"
                    ):
                        budget.record_usage(attempt, decision.input_tokens)
            metadata = decision.as_dict()
            metadata["target"] = decision.target or "deep"
            metadata["probabilities"] = decision.probabilities or {}
            report.records.append(
                CaseResult.model_validate(
                    {**common, "arm": "jev", "cold_connection": cold, **metadata}
                )
            )
    finally:
        if owned and client is not None:
            await client.aclose()
    if budget:
        report.budget = BudgetSummary.model_validate(budget.summary())
    return report


def save_report(report: EvaluationReport, output: Path) -> None:
    """Write only validated public fields and escaped HTML beside the JSON."""
    report = EvaluationReport.model_validate(report.model_dump())
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    html_path = output.with_suffix(".html")
    for path, content in (
        (json_path, report.model_dump_json(indent=2)),
        (html_path, render_report(report)),
    ):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=".omnimancer-report-",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(content + "\n")
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=1.0)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    if args.live and not args.ledger:
        parser.error("--live requires a persistent --ledger path")
    if args.live and not os.environ.get("TYPESAFE_API_KEY"):
        parser.error("TYPESAFE_API_KEY is required for live inference")
    cases = load_cases(args.cases)
    policy = RoutingPolicy.model_validate(read_json(args.policy, 65536))
    budget = BudgetLedger(args.ledger, args.budget_usd) if args.live else None
    report = asyncio.run(
        evaluate(
            cases,
            policy,
            live=args.live,
            budget=budget,
            repetitions=args.repetitions,
            seed=args.seed,
        )
    )
    save_report(report, args.output)
    print(f"Wrote {len(report.records)} predictions; live={args.live}.")


if __name__ == "__main__":
    main()
