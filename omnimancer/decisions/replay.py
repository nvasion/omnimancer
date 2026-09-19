"""Opt-in paired repairs through actual headless Omnimancer subprocesses.

Only keyless loopback workers are supported. Engine retries and provider fallback
are disabled. The provider's internal timeout/context-refit retries have no public
off switch; the process deadline bounds them and local requests have no API charge.
Reported turns are visible model turns, not a count of internal HTTP attempts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import secrets
import shutil
import signal
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, field_validator

from ..core.agent.types import OperationType
from ..core.models import (
    Config,
    EnhancedModelInfo,
    EventsConfig,
    FallbackConfig,
    HooksConfig,
    MCPConfig,
    PermissionsConfig,
    ProviderConfig,
)
from .budget import MODEL, BudgetExhausted, BudgetLedger
from .evaluation import canonical_hash, read_json, save_report
from .provenance import implementation_digest, policy_digest
from .report import BudgetSummary, EvaluationReport, PublicModel, TaskResult
from .routing import RoutingPolicy

PUBLIC_MODEL = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
CHECK_STARTED = b"OMN_REPLAY_CHECK_STARTED\n"
ROUTING_STATUSES = {
    "hook_blocked",
    "routed",
    "shadow",
    "switch_failed",
    "target_unavailable",
    "selected",
    "low_confidence",
    "missing_key",
    "invalid_state",
    "timeout",
    "http_error",
    "invalid_response",
    "network_error",
    "policy_unreadable",
    "policy_invalid",
    "policy_too_large",
    "insufficient_targets",
    "explicit_selection",
    "resume",
}
STOPS = {"done", "nudge_exhausted", "max_iterations", "repeat_abort"}


class Task(PublicModel):
    id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
    expected: Literal["fast", "deep"]
    prompt: Annotated[str, Field(min_length=1, max_length=4000)]
    files: dict[str, str]
    checks: Annotated[str, Field(min_length=1, max_length=16000)]

    @field_validator("files")
    @classmethod
    def isolated_solution(cls, value: dict[str, str]) -> dict[str, str]:
        # Avoid module-name injection during `python -m omnimancer` startup.
        if set(value) != {"solution.py"} or len(value["solution.py"]) > 64000:
            raise ValueError("Replay fixtures must contain only bounded solution.py")
        return value


class Workers(PublicModel):
    fast: Annotated[str, Field(pattern=PUBLIC_MODEL)] = "gemma4:12b"
    deep: Annotated[str, Field(pattern=PUBLIC_MODEL)] = "qwen3.8:27b"


@dataclass
class ProcessResult:
    returncode: int | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    elapsed_ms: float = 0
    timed_out: bool = False
    output_limited: bool = False


def validate_endpoint(endpoint: str) -> str:
    value = urlsplit(endpoint)
    # Access port to reject malformed ports before any subprocess is started.
    port = value.port
    if (
        value.scheme != "http"
        or value.hostname not in ("127.0.0.1", "::1", "localhost")
        or value.username is not None
        or value.password is not None
        or value.query
        or value.fragment
        or value.path.rstrip("/") != "/v1"
        or (port is not None and port == 0)
    ):
        raise ValueError("Replay requires a keyless loopback HTTP /v1 endpoint")
    # Use numeric loopback, so local resolver overrides cannot redirect localhost.
    host = "[::1]" if value.hostname == "::1" else "127.0.0.1"
    return f"http://{host}{':' + str(port) if port is not None else ''}/v1"


def load_tasks(path: Path) -> list[Task]:
    data = read_json(path)
    if not isinstance(data, list) or not 1 <= len(data) <= 12:
        raise ValueError("Expected one to twelve synthetic tasks")
    tasks = [Task.model_validate(value) for value in data]
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    return tasks


def build_config(
    workspace: Path, home: Path, endpoint: str, workers: Workers, timeout: float = 120
) -> dict:
    endpoint = validate_endpoint(endpoint)
    names = ["solution.py", "./solution.py", str(workspace.resolve() / "solution.py")]
    allowed = "(?:" + "|".join(re.escape(name) for name in names) + ")"
    deny = [
        {"tool": operation.value}
        for operation in OperationType
        if operation not in (OperationType.FILE_READ, OperationType.FILE_WRITE)
    ]
    deny.extend(
        {"tool": tool, "matcher": rf"\A(?!{allowed}\Z)[\s\S]*\Z"}
        for tool in ("file_read", "file_write")
    )
    config = Config(
        default_provider="deep",
        storage_path=str(home / "storage"),
        providers={
            label: ProviderConfig(
                provider_type="openai-compatible",
                model=model,
                api_key="",
                base_url=endpoint,
                auth_type="none",
                timeout=timeout,
                max_tokens=4096,
                temperature=0,
                max_retries=0,
                health_check_enabled=False,
            )
            for label, model in workers.model_dump().items()
        },
        custom_models=[
            EnhancedModelInfo(
                name=model,
                provider=label,
                description="Synthetic replay local worker",
                max_tokens=32768,
                cost_per_million_input=0,
                cost_per_million_output=0,
                supports_tools=True,
                available=True,
                is_free=True,
            )
            for label, model in workers.model_dump().items()
        ],
        permissions=PermissionsConfig.model_validate(
            {
                "always_deny": deny,
                "always_allow": [
                    {"tool": tool, "matcher": rf"\A{allowed}\Z"}
                    for tool in ("file_read", "file_write")
                ],
            }
        ),
        mcp=MCPConfig(),
        hooks=HooksConfig(enabled=False),
        events=EventsConfig(enabled=False),
        fallback=FallbackConfig(
            fallback_order=[],
            auto_fallback=False,
            fallback_on_rate_limit=False,
            fallback_on_quota=False,
        ),
        provider_fallback_enabled=False,
        provider_health_check_enabled=False,
        auto_update_check=False,
        telemetry_enabled=False,
    )
    return config.model_dump(mode="json")


def acceptance_environment() -> dict[str, str]:
    """Construct a clean environment rather than trying to enumerate secrets."""
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}


def worker_environment(home: Path, *, api_key: str | None = None) -> dict[str, str]:
    environment = acceptance_environment()
    environment.update(
        {
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / "config"),
            "XDG_CACHE_HOME": str(home / "cache"),
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "PYTHONNOUSERSITE": "1",
            "PYTHONIOENCODING": "utf-8",
            "OMNIMANCER_RATE_LIMIT_RETRIES": "0",
            "OMNIMANCER_CHECKPOINT": "0",
            "OMNIMANCER_EVENTS": "0",
            "OMNIMANCER_MAX_ITERATIONS": "8",
        }
    )
    if api_key is not None:
        environment["TYPESAFE_API_KEY"] = api_key
    return environment


async def run_process(
    argv: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    max_output_bytes: int = 1024 * 1024,
) -> ProcessResult:
    """Bound output and lifetime, always killing the complete process group."""
    if os.name != "posix":
        raise ValueError("Replay process isolation requires POSIX")
    started = time.perf_counter()
    result = ProcessResult()
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError:
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    class OutputLimit(Exception):
        pass

    async def read(stream: asyncio.StreamReader, field: str) -> None:
        buffer = bytearray()
        while chunk := await stream.read(8192):
            remaining = max_output_bytes - len(buffer)
            buffer.extend(chunk[:remaining])
            setattr(result, field, bytes(buffer))
            if len(chunk) > remaining:
                raise OutputLimit()

    assert process.stdout is not None and process.stderr is not None
    readers = [
        asyncio.create_task(read(process.stdout, "stdout")),
        asyncio.create_task(read(process.stderr, "stderr")),
    ]
    try:
        await asyncio.wait_for(asyncio.gather(process.wait(), *readers), timeout)
    except asyncio.TimeoutError:
        result.timed_out = True
    except OutputLimit:
        result.output_limited = True
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for reader in readers:
            reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        await process.wait()
    result.returncode = process.returncode
    result.elapsed_ms = (time.perf_counter() - started) * 1000
    return result


async def run_acceptance(workspace: Path, checks: Path, *, timeout: float = 5) -> str:
    """Run generated code with no host home, network, or writable source mount."""
    executable = shutil.which("bwrap")
    if executable is None or os.name != "posix":
        return "error"
    try:
        with checks.open(encoding="utf-8") as source_file:
            source = source_file.read(65537)
    except (OSError, UnicodeError):
        return "error"
    if len(source) > 65536:
        return "error"
    completion = "OMN_REPLAY_COMPLETED_" + secrets.token_hex(24)
    # Completion evidence is emitted only after every check. This detects early
    # interpreter exits; it is not an adversarial anti-cheating mechanism.
    with tempfile.TemporaryDirectory(prefix="omn-acceptance-") as directory:
        instrumented = Path(directory) / "checks.py"
        instrumented.write_text(
            "print('OMN_REPLAY_CHECK_STARTED', flush=True)\n"
            + source
            + "\n"
            + f"print({completion!r}, flush=True)\n",
            encoding="utf-8",
        )
        argv = [
            executable,
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--ro-bind",
            "/usr",
            "/usr",
        ]
        for directory in ("/lib", "/lib64"):
            if Path(directory).exists():
                argv.extend(["--ro-bind", directory, directory])
        argv.extend(
            [
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--ro-bind",
                str(workspace),
                "/work",
                "--ro-bind",
                str(instrumented),
                "/checks.py",
                "--chdir",
                "/work",
                "--clearenv",
                "--setenv",
                "HOME",
                "/tmp",
                "--setenv",
                "PYTHONDONTWRITEBYTECODE",
                "1",
                "--",
                "/usr/bin/python3",
                "-I",
                "-S",
                "/checks.py",
            ]
        )
        result = await run_process(
            argv, cwd=workspace, env=acceptance_environment(), timeout=timeout
        )
    if result.timed_out:
        return "timeout"
    if result.output_limited or CHECK_STARTED not in result.stdout:
        return "error"
    completed = completion.encode() in result.stdout.splitlines()
    return "pass" if result.returncode == 0 and completed else "fail"


def paired_order(tasks: list[Task], seed: int) -> list[tuple[Task, str]]:
    randomizer = random.Random(seed)
    ordered = list(tasks)
    randomizer.shuffle(ordered)
    pairs: list[tuple[Task, str]] = []
    for task in ordered:
        arms = ["baseline", "jev"]
        randomizer.shuffle(arms)
        pairs.extend((task, arm) for arm in arms)
    return pairs


def _number(value: object, *, integer: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Invalid numeric result")
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 100_000_000
    ):
        raise ValueError("Invalid numeric result")
    return value


def _metadata(process: ProcessResult, arm: str, workers: Workers) -> tuple[dict, dict]:
    value = json.loads(process.stdout)
    if (
        not isinstance(value, dict)
        or value.get("type") != "result"
        or type(value.get("is_error")) is not bool
    ):
        raise ValueError("Invalid worker output")
    routing = value.get("routing", {}) if arm == "jev" else {}
    if not isinstance(routing, dict) or (
        arm == "jev" and routing.get("status") not in ROUTING_STATUSES
    ):
        raise ValueError("Invalid routing output")
    target = routing.get("applied_target") or "deep"
    if (
        target not in ("fast", "deep")
        or value.get("provider") != target
        or value.get("model") != getattr(workers, target)
    ):
        raise ValueError("Worker identity mismatch")
    usage = value.get("usage") or {}
    tools = value.get("tool_calls") or []
    if not isinstance(usage, dict) or not isinstance(tools, list):
        raise ValueError("Invalid worker usage")
    stop = value.get("stop_cause")
    return {
        "target": target,
        "routing_status": routing.get("status", "baseline"),
        "routing_ms": _number(routing.get("elapsed_ms", 0)),
        "turns": _number(value.get("num_turns", 0), integer=True),
        "tool_calls": len(tools),
        "input_tokens": _number(usage.get("input_tokens", 0), integer=True),
        "output_tokens": _number(usage.get("output_tokens", 0), integer=True),
        "stop_cause": (
            stop if stop in STOPS and not value["is_error"] else "provider_error"
        ),
    }, routing


async def _one(
    task: Task,
    arm: str,
    *,
    endpoint: str,
    workers: Workers,
    policy: RoutingPolicy,
    ledger: BudgetLedger,
    api_key: str,
    timeout: float,
    max_turns: int,
    check_timeout: float,
) -> TaskResult:
    started = time.perf_counter()
    row = dict(
        case_id=task.id,
        arm=arm,
        expected=task.expected,
        target="deep",
        success=False,
        elapsed_ms=0,
        worker_ms=0,
        stop_cause="provider_error",
        check="error",
        routing_status="baseline" if arm == "baseline" else "invalid_output",
    )
    attempt = None
    with tempfile.TemporaryDirectory(prefix="omn-replay-") as directory:
        root = Path(directory)
        workspace, home = root / "work", root / "home"
        workspace.mkdir()
        home.mkdir()
        (workspace / ".git").mkdir()
        (workspace / "solution.py").write_text(task.files["solution.py"])
        config, policy_path = root / "config.json", root / "routing.json"
        config.write_text(
            json.dumps(build_config(workspace, home, endpoint, workers, timeout))
        )
        policy_path.write_text(policy.model_dump_json())
        argv = [
            sys.executable,
            "-m",
            "omnimancer",
            "-p",
            task.prompt,
            "--config",
            str(config),
            "--output-format",
            "json",
            "--no-approval",
            "--max-iterations",
            str(max_turns),
        ]
        if arm == "jev":
            try:
                attempt = ledger.reserve(MODEL)
            except BudgetExhausted:
                row["routing_status"] = "budget_exhausted"
                row["elapsed_ms"] = (time.perf_counter() - started) * 1000
                return TaskResult.model_validate(row)
            argv.extend(["--routing-policy", str(policy_path)])
        process = await run_process(
            argv,
            cwd=workspace,
            env=worker_environment(home, api_key=api_key if arm == "jev" else None),
            timeout=timeout,
        )
        row["worker_ms"] = process.elapsed_ms
        if process.timed_out:
            row["stop_cause"] = "timeout"
        elif process.output_limited:
            row["stop_cause"] = "invalid_output"
        else:
            try:
                metadata, routing = _metadata(process, arm, workers)
                row.update(metadata)
                row["worker_ms"] = max(0, process.elapsed_ms - metadata["routing_ms"])
                if (
                    attempt is not None
                    and routing.get("model") == MODEL
                    and type(routing.get("input_tokens")) is int
                ):
                    ledger.record_usage(attempt, routing["input_tokens"])
            except (ValueError, TypeError, KeyError):
                row["stop_cause"] = "invalid_output"
        # Checks do not exist anywhere visible to the worker until it has exited.
        checks = root / "acceptance.py"
        checks.write_text(
            "import resource, sys\n"
            "resource.setrlimit(resource.RLIMIT_AS, (536870912, 536870912))\n"
            "resource.setrlimit(resource.RLIMIT_CPU, "
            f"({math.ceil(check_timeout)}, {math.ceil(check_timeout) + 1}))\n"
            "resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))\n"
            "resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))\n"
            "sys.path.insert(0, '/work')\n"
            "print('OMN_REPLAY_CHECK_STARTED', flush=True)\n" + task.checks
        )
        row["check"] = await run_acceptance(workspace, checks, timeout=check_timeout)
        # The acceptance outcome measures the fixture state at termination. A
        # passing repair can coexist with a timeout/provider-error stop cause.
        row["success"] = row["check"] == "pass"
    row["elapsed_ms"] = (time.perf_counter() - started) * 1000
    return TaskResult.model_validate(row)


async def replay_tasks(
    tasks: list[Task],
    report: EvaluationReport,
    *,
    policy: RoutingPolicy,
    endpoint: str,
    ledger: BudgetLedger,
    api_key: str,
    workers: Workers | None = None,
    seed: int = 19,
    timeout: float = 120,
    max_turns: int = 8,
    check_timeout: float = 5,
    output: Path | None = None,
) -> EvaluationReport:
    endpoint = validate_endpoint(endpoint)
    if not report.manifest.live:
        raise ValueError("Replay requires a live classification report")
    if not report.manifest.policy_sha256:
        raise ValueError("Classification policy provenance is required")
    if report.manifest.implementation_sha256 != implementation_digest():
        raise ValueError("Replay implementation differs from classification")
    if not 0 < timeout <= 120 or not 1 <= max_turns <= 8 or not 0 < check_timeout <= 10:
        raise ValueError("Replay limits exceed the bounded experiment")
    if not 1 <= len(tasks) <= 12 or not api_key or ledger.summary()["cap_usd"] > 5:
        raise ValueError(
            "Replay needs tasks, a routing key, and a budget of at most $5"
        )
    if (
        set(policy.targets) != {"fast", "deep"}
        or policy.mode != "route"
        or policy.model != MODEL
    ):
        raise ValueError("Replay requires the frozen fast/deep route policy")
    workers = workers or Workers(
        fast=policy.targets["fast"].model, deep=policy.targets["deep"].model
    )
    if any(
        target.provider != label or target.model != getattr(workers, label)
        for label, target in policy.targets.items()
    ):
        raise ValueError("Replay worker aliases/models must match the frozen policy")
    policy_hash = policy_digest(policy)
    if report.manifest.policy_sha256 != policy_hash:
        raise ValueError(
            "Replay policy/classifier differs from classification manifest"
        )
    if workers.fast == workers.deep:
        raise ValueError("Replay requires two different worker model names")
    if report.tasks and (
        report.manifest.fast_worker != workers.fast
        or report.manifest.deep_worker != workers.deep
    ):
        raise ValueError("Cannot mix worker model pairs in one task report")
    order = paired_order(tasks, seed)
    report.manifest.fast_worker = workers.fast
    report.manifest.deep_worker = workers.deep
    report.manifest.planned_task_runs += len(order)
    report.manifest.task_policy_sha256 = policy_hash
    report.manifest.task_sha256 = canonical_hash(
        {
            "previous_task_sha256": report.manifest.task_sha256,
            "tasks": [task.model_dump() for task in tasks],
            "workers": workers.model_dump(),
            "seed": seed,
            "timeout": timeout,
            "max_turns": max_turns,
            "check_timeout": check_timeout,
            "policy": policy.model_dump(),
            "order": [(task.id, arm) for task, arm in order],
        }
    )
    for task, arm in order:
        report.tasks.append(
            await _one(
                task,
                arm,
                endpoint=endpoint,
                workers=workers,
                policy=policy,
                ledger=ledger,
                api_key=api_key,
                timeout=timeout,
                max_turns=max_turns,
                check_timeout=check_timeout,
            )
        )
        report.budget = BudgetSummary.model_validate(ledger.summary())
        if output is not None:
            save_report(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--budget-usd", type=float, default=5)
    parser.add_argument("--fast-model")
    parser.add_argument("--deep-model")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    key = os.environ.get("TYPESAFE_API_KEY")
    if not args.live or not key:
        parser.error("Replay requires --live and TYPESAFE_API_KEY in the environment")
    if not 0 < args.budget_usd <= 5:
        parser.error("Replay budget must be positive and at most $5")
    try:
        tasks = load_tasks(args.tasks)
        report = EvaluationReport.model_validate(read_json(args.report, 4_000_000))
        policy = RoutingPolicy.model_validate(read_json(args.policy, 65536))
        if set(policy.targets) != {"fast", "deep"}:
            raise ValueError("Replay policy requires fast/deep labels")
        workers = Workers(
            fast=args.fast_model or policy.targets["fast"].model,
            deep=args.deep_model or policy.targets["deep"].model,
        )
        endpoint = validate_endpoint(args.endpoint)
        ledger = BudgetLedger(args.ledger, args.budget_usd)
        report = asyncio.run(
            replay_tasks(
                tasks,
                report,
                policy=policy,
                output=args.output,
                endpoint=endpoint,
                ledger=ledger,
                api_key=key,
                workers=workers,
                seed=args.seed,
                timeout=args.timeout,
                max_turns=args.max_turns,
            )
        )
        save_report(report, args.output)
    except (ValueError, OSError):
        parser.error("Replay input, budget, or output could not be validated")
    print(f"Recorded {len(report.tasks)} task results; failures are retained.")


if __name__ == "__main__":
    main()
