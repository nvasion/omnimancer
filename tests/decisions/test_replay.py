"""Bounded paired replay contracts; no live classifier or worker requests."""

import asyncio
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.models import Config
from omnimancer.core.security.permission_rules import (
    PermissionDecision,
    PermissionRuleEngine,
)
from omnimancer.decisions.budget import BudgetLedger
from omnimancer.decisions.report import EvaluationReport, Manifest


@pytest.fixture
def replay():
    return importlib.import_module("omnimancer.decisions.replay")


def synthetic_policy(replay):
    return replay.RoutingPolicy(
        targets={
            "fast": {
                "provider": "fast",
                "model": "gemma4:12b",
                "criteria": "Frozen synthetic fast criterion",
            },
            "deep": {
                "provider": "deep",
                "model": "qwen3.8:27b",
                "criteria": "Frozen synthetic deep criterion",
            },
        }
    )


def classification_report(replay, **changes):
    from omnimancer.decisions.provenance import implementation_digest, policy_digest

    metadata = {
        "live": True,
        "implementation_sha256": implementation_digest(),
        "policy_sha256": policy_digest(synthetic_policy(replay)),
    }
    metadata.update(changes)
    return EvaluationReport(manifest=Manifest(**metadata))


@pytest.fixture
def task(replay):
    return replay.Task(
        id="synthetic-repair",
        expected="fast",
        prompt="Repair solution.py using Read/Edit/Write only.",
        files={"solution.py": "def answer():\n    return 0\n"},
        checks="import solution\nassert solution.answer() == 42\n",
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1/v1",
        "http://public.test/v1",
        "http://user:secret@127.0.0.1/v1",
        "http://127.0.0.1/v1?key=secret",
        "http://127.0.0.1/v2",
        "http://127.0.0.1:bad/v1",
    ],
)
def test_cloud_or_credentialed_endpoints_rejected(replay, endpoint):
    with pytest.raises(ValueError):
        replay.validate_endpoint(endpoint)


@pytest.mark.parametrize(
    "model", ["../secret", "/private/path", "user@host", "bad\nmodel"]
)
def test_worker_model_identity_is_public_and_bounded(replay, model):
    with pytest.raises(ValueError):
        replay.Workers(fast=model)


def test_config_only_allows_fixture_files_and_denies_other_tools(replay, tmp_path):
    workspace = tmp_path / "work"
    workspace.mkdir()
    config = replay.build_config(
        workspace, tmp_path / "home", "http://127.0.0.1:9000/v1", replay.Workers()
    )
    parsed = Config.model_validate(config)
    assert set(parsed.providers) == {"fast", "deep"}
    assert parsed.default_provider == "deep"
    assert not parsed.mcp.servers
    assert not parsed.hooks.tool_use_request
    assert parsed.fallback.fallback_on_rate_limit is False
    assert all(
        p.api_key in ("", None) and p.auth_type == "none"
        for p in parsed.providers.values()
    )
    assert all(
        m.cost_per_million_input == m.cost_per_million_output == 0
        for m in parsed.custom_models
    )
    rules = PermissionRuleEngine(parsed.permissions)
    for name in ("solution.py", "./solution.py", str(workspace / "solution.py")):
        assert rules.evaluate("file_read", name) == PermissionDecision.ALLOW
        assert rules.evaluate("file_write", name) == PermissionDecision.ALLOW
    for name in (
        "../acceptance.py",
        "/etc/passwd",
        "../config.json",
        "solution.py\n",
        "other.py",
    ):
        assert rules.evaluate("file_read", name) == PermissionDecision.DENY
        assert rules.evaluate("file_write", name) == PermissionDecision.DENY
    for tool in (
        "command_execute",
        "web_request",
        "mcp_tool_call",
        "file_delete",
        "workflow_step",
    ):
        assert rules.evaluate(tool, "anything") == PermissionDecision.DENY


@pytest.mark.asyncio
async def test_real_agent_and_tool_handler_deny_command_network_and_hidden_reads(
    replay, tmp_path, monkeypatch
):
    from omnimancer.cli.tool_handler import ToolHandler
    from omnimancer.core.agent_engine import AgentEngine
    from omnimancer.core.models import ToolCall

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    config = replay.build_config(
        tmp_path, tmp_path / "home", "http://127.0.0.1/v1", replay.Workers()
    )
    manager = ConfigManager(str(tmp_path / "config.json"))
    manager.config = Config.model_validate(config)
    engine = AgentEngine(manager)
    handler = ToolHandler(engine)
    for name, args in (
        ("Bash", {"command": "touch forbidden"}),
        ("WebFetch", {"url": "http://public.test", "prompt": "fetch"}),
        ("Read", {"file_path": "../acceptance.py"}),
    ):
        result = await handler.execute_tool_call(
            ToolCall(id="test", name=name, arguments=args)
        )
        assert "permission rule" in result.error
    assert not (tmp_path / "forbidden").exists()


def test_environment_drops_secrets_and_provider_overrides(
    replay, tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "PRIVATE")
    monkeypatch.setenv("TYPESAFE_API_KEY", "PRIVATE")
    monkeypatch.setenv("OMNIMANCER_FAST_BASE_URL", "http://public.test")
    monkeypatch.setenv("HTTP_PROXY", "http://public.test")
    environment = replay.worker_environment(tmp_path, api_key="synthetic-router-key")
    assert environment["TYPESAFE_API_KEY"] == "synthetic-router-key"
    assert environment["OMNIMANCER_RATE_LIMIT_RETRIES"] == "0"
    assert all("PRIVATE" not in value for value in environment.values())
    assert "HTTP_PROXY" not in environment
    assert "OMNIMANCER_FAST_BASE_URL" not in environment
    assert "TYPESAFE_API_KEY" not in replay.acceptance_environment()
    assert "OPENAI_API_KEY" not in replay.acceptance_environment()


@pytest.mark.asyncio
async def test_subprocess_timeout_kills_descendants(replay, tmp_path):
    marker = tmp_path / "survivor"
    script = (
        "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',"
        "'import time,pathlib; time.sleep(0.3); "
        'pathlib.Path(sys.argv[1]).write_text("bad")\','
        "sys.argv[1]]); time.sleep(30)"
    )
    # The grandchild uses a complete script with its own sys import.
    script = script.replace("import time,pathlib;", "import time,pathlib,sys;")
    result = await replay.run_process(
        [sys.executable, "-c", script, str(marker)],
        cwd=tmp_path,
        env=replay.acceptance_environment(),
        timeout=0.1,
    )
    assert result.timed_out
    await asyncio.sleep(0.4)
    assert not marker.exists()


@pytest.mark.asyncio
async def test_subprocess_output_is_bounded(replay, tmp_path):
    result = await replay.run_process(
        [sys.executable, "-c", "import sys; sys.stdout.write('x'*2000000)"],
        cwd=tmp_path,
        env=replay.acceptance_environment(),
        timeout=2,
        max_output_bytes=1024,
    )
    assert result.output_limited
    assert len(result.stdout) <= 1024


@pytest.mark.asyncio
async def test_acceptance_is_sandboxed_secret_free_and_reports_failures(
    replay, tmp_path, monkeypatch
):
    workspace = tmp_path / "work"
    workspace.mkdir()
    checks = tmp_path / "hidden.py"
    checks.write_text("assert False")
    monkeypatch.setenv("TYPESAFE_API_KEY", "PRIVATE")
    monkeypatch.setattr(replay.shutil, "which", lambda name: "/usr/bin/bwrap")

    async def boundary(argv, *, cwd, env, timeout, **kwargs):
        assert "--unshare-all" in argv
        assert "--ro-bind" in argv
        assert str(workspace) in argv
        wrapper = Path(argv[argv.index("/checks.py") - 1])
        assert "assert False" in wrapper.read_text()
        assert "TYPESAFE_API_KEY" not in env
        assert "PRIVATE" not in json.dumps(env)
        return replay.ProcessResult(
            returncode=1,
            stdout=replay.CHECK_STARTED + b"PRIVATE stdout",
            stderr=b"PRIVATE stderr",
            elapsed_ms=1,
        )

    monkeypatch.setattr(replay, "run_process", boundary)
    assert await replay.run_acceptance(workspace, checks, timeout=2) == "fail"


@pytest.mark.asyncio
async def test_missing_acceptance_sandbox_fails_closed(replay, tmp_path, monkeypatch):
    monkeypatch.setattr(replay.shutil, "which", lambda name: None)
    assert (
        await replay.run_acceptance(tmp_path, tmp_path / "checks.py", timeout=2)
        == "error"
    )


def test_paired_order_is_reproducible_and_complete(replay, task):
    second = task.model_copy(update={"id": "another"})
    first = replay.paired_order([task, second], 19)
    assert first == replay.paired_order([task, second], 19)
    assert {(t.id, arm) for t, arm in first} == {
        (identifier, arm)
        for identifier in (task.id, second.id)
        for arm in ("baseline", "jev")
    }
    assert first[0][0].id == first[1][0].id


@pytest.mark.asyncio
async def test_budget_reservation_precedes_child_and_failed_checks_stay_in_denominator(
    replay, task, tmp_path, monkeypatch
):
    ledger = BudgetLedger(tmp_path / "ledger.sqlite", 0.003)
    second = task.model_copy(update={"id": "second-task"})
    calls = []

    async def child(argv, *, cwd, env, timeout, **kwargs):
        assert argv[1:3] == ["-m", "omnimancer"]
        assert "--no-approval" in argv and "--max-iterations" in argv
        config_path = Path(argv[argv.index("--config") + 1])
        assert config_path.parent != cwd
        assert not (config_path.parent / "acceptance.py").exists()
        assert (cwd / ".git").is_dir()
        config = json.loads(config_path.read_text())
        assert "PRIVATE" not in json.dumps(config)
        jev = "--routing-policy" in argv
        if jev:
            assert ledger.summary()["attempts"] == 1
            assert env["TYPESAFE_API_KEY"] == "synthetic-key"
        else:
            assert "TYPESAFE_API_KEY" not in env
        calls.append(jev)
        result = {
            "type": "result",
            "is_error": False,
            "num_turns": 1,
            "tool_calls": [],
            "usage": {"input_tokens": 7, "output_tokens": 2},
            "stop_cause": "done",
            "model": "qwen3.8:27b",
            "provider": "deep",
        }
        if jev:
            result["routing"] = {
                "status": "low_confidence",
                "applied_target": None,
                "model": "jev-1.13.0",
                "input_tokens": 100,
                "elapsed_ms": 1,
            }
        return replay.ProcessResult(
            returncode=0,
            stdout=json.dumps(result).encode(),
            stderr=b"PRIVATE",
            elapsed_ms=10,
        )

    async def acceptance(workspace, checks, *, timeout):
        assert checks.parent != workspace
        assert "assert solution.answer() == 42" in checks.read_text()
        return "fail"

    monkeypatch.setattr(replay, "run_process", child)
    monkeypatch.setattr(replay, "run_acceptance", acceptance)
    report = classification_report(replay)
    await replay.replay_tasks(
        [task, second],
        report,
        endpoint="http://127.0.0.1/v1",
        ledger=ledger,
        policy=synthetic_policy(replay),
        api_key="synthetic-key",
    )
    assert len(report.tasks) == 4
    assert sum(calls) == 1
    assert len(calls) == 3
    assert not any(result.success for result in report.tasks)
    assert (
        sum(result.routing_status == "budget_exhausted" for result in report.tasks) == 1
    )
    assert ledger.summary()["unsettled_attempts"] == 0
    assert report.manifest.task_sha256 is not None
    assert "PRIVATE" not in report.model_dump_json()
    assert str(tmp_path) not in report.model_dump_json()


@pytest.mark.asyncio
async def test_invalid_worker_output_is_categorical_data(
    replay, task, tmp_path, monkeypatch
):
    async def child(*args, **kwargs):
        return replay.ProcessResult(
            returncode=0,
            stdout=b"PRIVATE invalid json",
            stderr=b"PRIVATE exception",
            elapsed_ms=1,
        )

    async def acceptance(*args, **kwargs):
        return "fail"

    monkeypatch.setattr(replay, "run_process", child)
    monkeypatch.setattr(replay, "run_acceptance", acceptance)
    report = classification_report(replay)
    await replay.replay_tasks(
        [task],
        report,
        endpoint="http://127.0.0.1/v1",
        policy=synthetic_policy(replay),
        ledger=BudgetLedger(tmp_path / "ledger", 0.003),
        api_key="synthetic-key",
    )
    assert len(report.tasks) == 2
    assert all(result.stop_cause == "invalid_output" for result in report.tasks)
    assert "PRIVATE" not in report.model_dump_json()


@pytest.mark.asyncio
@pytest.mark.parametrize("arm", ["baseline", "jev"])
async def test_timeout_preserves_outcome_without_inventing_metadata(
    replay, task, tmp_path, monkeypatch, arm
):
    async def child(*args, **kwargs):
        return replay.ProcessResult(
            returncode=-9, stdout=b"", stderr=b"", elapsed_ms=120000, timed_out=True
        )

    async def acceptance(*args, **kwargs):
        return "pass"

    monkeypatch.setattr(replay, "run_process", child)
    monkeypatch.setattr(replay, "run_acceptance", acceptance)
    result = await replay._one(
        task,
        arm,
        endpoint="http://127.0.0.1/v1",
        workers=replay.Workers(),
        policy=synthetic_policy(replay),
        ledger=BudgetLedger(tmp_path / "ledger", 0.003),
        api_key="synthetic-key",
        timeout=120,
        max_turns=8,
        check_timeout=2,
    )
    assert result.success is True
    assert result.stop_cause == "timeout"
    assert result.target == ("deep" if arm == "baseline" else None)
    assert result.routing_ms == (0 if arm == "baseline" else None)
    assert result.worker_ms == (120000 if arm == "baseline" else None)
    assert result.turns is None
    assert result.tool_calls is None
    assert result.input_tokens is None
    assert result.output_tokens is None


def test_task_schema_rejects_escape_and_private_report_fields(replay, task):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        replay.Task.model_validate(
            {**task.model_dump(), "files": {"../escape.py": "bad"}}
        )
    with pytest.raises(ValidationError):
        replay.Task.model_validate({**task.model_dump(), "api_key": "PRIVATE"})


@pytest.mark.asyncio
async def test_actual_headless_subprocess_repairs_fixture_through_http_boundary(
    replay, task, tmp_path, monkeypatch
):
    bootstrap = tmp_path / "http-boundary"
    bootstrap.mkdir()
    (bootstrap / "sitecustomize.py").write_text("""
import json
import httpx
original = httpx.AsyncClient
turn = 0
def respond(request):
    global turn
    turn += 1
    payload = json.loads(request.content)
    if turn == 1:
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "repair", "type": "function", "function": {
                "name": "Write", "arguments": json.dumps({
                    "file_path": "solution.py",
                    "content": "def answer():\\n    return 42\\n"
                })
            }
        }]}
        reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": "DONE"}
        reason = "stop"
    return httpx.Response(200, json={
        "id": "synthetic", "model": payload["model"],
        "choices": [{"message": message, "finish_reason": reason}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    })
def client(*args, **kwargs):
    kwargs["transport"] = httpx.MockTransport(respond)
    return original(*args, **kwargs)
httpx.AsyncClient = client
""")
    original = replay.worker_environment

    def environment(home, *, api_key=None):
        values = original(home, api_key=api_key)
        values["PYTHONPATH"] = str(bootstrap) + os.pathsep + values["PYTHONPATH"]
        return values

    async def acceptance(workspace, checks, *, timeout):
        assert (
            workspace / "solution.py"
        ).read_text() == "def answer():\n    return 42\n"
        assert checks.parent != workspace
        return "pass"

    monkeypatch.setattr(replay, "worker_environment", environment)
    monkeypatch.setattr(replay, "run_acceptance", acceptance)
    result = await replay._one(
        task,
        "baseline",
        endpoint="http://127.0.0.1/v1",
        workers=replay.Workers(),
        policy=synthetic_policy(replay),
        ledger=BudgetLedger(tmp_path / "ledger", 0.003),
        api_key="synthetic-key",
        timeout=8,
        max_turns=8,
        check_timeout=2,
    )
    assert result.success is True
    assert result.turns == 2
    assert result.tool_calls == 1
    assert result.target == "deep"


@pytest.mark.asyncio
async def test_actual_acceptance_sandbox_hides_host_files_and_credentials(
    replay, tmp_path, monkeypatch
):
    if replay.shutil.which("bwrap") is None:
        pytest.skip("bubblewrap unavailable")
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "solution.py").write_text("def answer():\n    return 42\n")
    sentinel = tmp_path / "private-sentinel"
    sentinel.write_text("PRIVATE")
    checks = tmp_path / "checks.py"
    checks.write_text(
        "import os, sys\nfrom pathlib import Path\n"
        "sys.path.insert(0, '/work')\n"
        "print('OMN_REPLAY_CHECK_STARTED', flush=True)\n"
        "import solution\nassert solution.answer() == 42\n"
        "assert 'TYPESAFE_API_KEY' not in os.environ\n"
        f"assert not Path({str(sentinel)!r}).exists()\n"
        "assert os.readlink('/proc/self/ns/net') != "
        f"{os.readlink('/proc/self/ns/net')!r}\n"
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-secret")
    assert await replay.run_acceptance(workspace, checks, timeout=3) == "pass"


@pytest.mark.asyncio
@pytest.mark.parametrize("early_exit", ["sys.exit(0)", "os._exit(0)"])
async def test_acceptance_import_early_exit_cannot_pass(replay, tmp_path, early_exit):
    if replay.shutil.which("bwrap") is None:
        pytest.skip("bubblewrap unavailable")
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "solution.py").write_text("import sys, os\n" + early_exit + "\n")
    checks = tmp_path / "checks.py"
    checks.write_text(
        "import sys\nsys.path.insert(0, '/work')\n"
        "print('OMN_REPLAY_CHECK_STARTED', flush=True)\n"
        "import solution\nassert False, 'Checks must actually execute'\n"
    )
    assert await replay.run_acceptance(workspace, checks, timeout=3) == "fail"


@pytest.mark.asyncio
async def test_timeout_remains_reported_when_written_fix_passes(
    replay, task, tmp_path, monkeypatch
):
    async def child(*args, **kwargs):
        return replay.ProcessResult(
            returncode=-9, stdout=b"PRIVATE", elapsed_ms=100, timed_out=True
        )

    async def acceptance(*args, **kwargs):
        return "pass"

    monkeypatch.setattr(replay, "run_process", child)
    monkeypatch.setattr(replay, "run_acceptance", acceptance)
    report = classification_report(replay)
    await replay.replay_tasks(
        [task],
        report,
        endpoint="http://127.0.0.1/v1",
        policy=synthetic_policy(replay),
        ledger=BudgetLedger(tmp_path / "ledger", 0.003),
        api_key="synthetic-key",
    )
    assert all(result.stop_cause == "timeout" for result in report.tasks)
    assert all(result.success for result in report.tasks)
    assert "PRIVATE" not in report.model_dump_json()


def test_cli_requires_live_opt_in_without_launching(replay, monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay",
            "--tasks",
            "missing",
            "--policy",
            "missing-policy",
            "--report",
            "missing",
            "--output",
            str(tmp_path / "out"),
            "--ledger",
            str(tmp_path / "ledger"),
            "--endpoint",
            "http://127.0.0.1/v1",
        ],
    )
    with pytest.raises(SystemExit) as error:
        replay.main()
    assert error.value.code == 2
    assert not (tmp_path / "ledger").exists()


def test_cli_appends_validated_tasks_and_saves_json_and_html(
    replay, task, tmp_path, monkeypatch
):
    tasks = tmp_path / "tasks.json"
    tasks.write_text(json.dumps([task.model_dump()]))
    policy = tmp_path / "policy.json"
    policy.write_text(synthetic_policy(replay).model_dump_json())
    previous = tmp_path / "previous.json"
    previous.write_text(classification_report(replay).model_dump_json())

    async def child(argv, **kwargs):
        data = {
            "type": "result",
            "is_error": False,
            "provider": "deep",
            "model": "qwen3.8:27b",
            "stop_cause": "done",
            "num_turns": 1,
            "tool_calls": [],
            "usage": {},
        }
        if "--routing-policy" in argv:
            data["routing"] = {
                "status": "low_confidence",
                "model": "jev-1.13.0",
                "input_tokens": 10,
            }
        return replay.ProcessResult(
            returncode=0, stdout=json.dumps(data).encode(), elapsed_ms=1
        )

    async def acceptance(*args, **kwargs):
        return "fail"

    monkeypatch.setattr(replay, "run_process", child)
    monkeypatch.setattr(replay, "run_acceptance", acceptance)
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-key")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay",
            "--tasks",
            str(tasks),
            "--policy",
            str(policy),
            "--report",
            str(previous),
            "--output",
            str(tmp_path / "output"),
            "--ledger",
            str(tmp_path / "ledger"),
            "--endpoint",
            "http://127.0.0.1/v1",
            "--live",
        ],
    )
    replay.main()
    result = EvaluationReport.model_validate_json(
        (tmp_path / "output.json").read_text()
    )
    assert len(result.tasks) == 2
    assert result.manifest.planned_task_runs == 2
    assert result.manifest.fast_worker == "gemma4:12b"
    assert result.manifest.deep_worker == "qwen3.8:27b"
    assert result.budget.attempts == 1
    assert (tmp_path / "output.html").exists()
    assert "synthetic-key" not in (tmp_path / "output.html").read_text()


@pytest.mark.asyncio
async def test_changed_frozen_policy_rejected_before_budget(replay, task, tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger", 0.003)
    report = classification_report(replay, policy_sha256="a" * 64)
    with pytest.raises(ValueError):
        await replay.replay_tasks(
            [task],
            report,
            policy=synthetic_policy(replay),
            endpoint="http://127.0.0.1/v1",
            ledger=ledger,
            api_key="synthetic",
        )
    assert ledger.summary()["attempts"] == 0


@pytest.mark.asyncio
async def test_completed_rows_checkpointed_before_next_child(
    replay, task, tmp_path, monkeypatch
):
    output = tmp_path / "checkpoint.json"
    report = classification_report(replay)
    policy = synthetic_policy(replay)
    seen = []

    async def one(task, arm, **kwargs):
        assert kwargs["policy"].model_dump() == policy.model_dump()
        if seen:
            saved = EvaluationReport.model_validate_json(output.read_text())
            assert len(saved.tasks) == 1
            assert saved.manifest.planned_task_runs == 2
        seen.append(arm)
        return replay.TaskResult(
            case_id=task.id,
            arm=arm,
            expected="fast",
            target="deep",
            success=False,
            elapsed_ms=1,
            worker_ms=1,
            stop_cause="provider_error",
            check="fail",
        )

    monkeypatch.setattr(replay, "_one", one)
    await replay.replay_tasks(
        [task],
        report,
        policy=policy,
        endpoint="http://127.0.0.1/v1",
        ledger=BudgetLedger(tmp_path / "ledger", 0.003),
        api_key="synthetic",
        output=output,
    )
    assert len(EvaluationReport.model_validate_json(output.read_text()).tasks) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid",
    [
        {"live": False},
        {"policy_sha256": None},
        {"implementation_sha256": None},
        {"implementation_sha256": "0" * 64},
    ],
)
async def test_unverified_classification_report_cannot_launch_replay(
    replay, task, tmp_path, monkeypatch, invalid
):
    report = classification_report(replay, **invalid)
    ledger = BudgetLedger(tmp_path / "ledger", 0.003)
    children = []

    async def child(task, arm, **kwargs):
        children.append(arm)
        return replay.TaskResult(
            case_id=task.id,
            arm=arm,
            expected="fast",
            target="deep",
            success=False,
            elapsed_ms=1,
            worker_ms=1,
            stop_cause="provider_error",
            check="fail",
        )

    monkeypatch.setattr(replay, "_one", child)
    with pytest.raises(ValueError):
        await replay.replay_tasks(
            [task],
            report,
            policy=synthetic_policy(replay),
            endpoint="http://127.0.0.1/v1",
            ledger=ledger,
            api_key="synthetic-key",
        )
    assert not children
    assert ledger.summary()["attempts"] == 0
