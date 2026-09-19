"""Routing integration contracts using real engines and isolated HTTP traffic."""

import asyncio
import builtins
import copy
import importlib
import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path

import httpx
import pytest

from omnimancer.core.config_manager import ConfigManager
from omnimancer.core.engine import CoreEngine
from omnimancer.core.models import Config, EnhancedModelInfo, ModelInfo, ProviderConfig
from omnimancer.providers.azure import AzureProvider
from omnimancer.providers.openai import OpenAIProvider
from omnimancer.providers.openai_compatible import OpenAICompatibleProvider


@pytest.fixture
def engine(tmp_path, monkeypatch):
    """Real CoreEngine; initialize only its two synthetic, in-memory providers."""
    monkeypatch.setenv("OMNIMANCER_CHECKPOINT", "0")
    monkeypatch.setenv("TYPESAFE_API_KEY", "synthetic-router-key")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    config = Config(
        default_provider="local",
        providers={
            "local": ProviderConfig(model="small", provider_type="openai-compatible"),
            "remote": ProviderConfig(
                model="previous", provider_type="openai-compatible"
            ),
        },
        storage_path=str(tmp_path / "storage"),
        events={"enabled": False},
    )
    path = tmp_path / "config.json"
    path.write_text(config.model_dump_json())
    manager = ConfigManager(str(path))
    manager.config = config
    core = CoreEngine(manager)
    for name, model, catalog in (
        ("local", "small", ["small", "large"]),
        ("remote", "previous", ["large", "previous"]),
    ):
        provider = OpenAICompatibleProvider("", model, base_url="http://worker.test/v1")
        provider._catalog_models = [
            ModelInfo(label, name, "synthetic", 4096, 0.0, available=True)
            for label in catalog
        ]
        core.providers[name] = provider
    core.current_provider = core.providers["local"]
    core.chat_manager.set_current_model("small")
    core.chat_manager.add_user_message("existing synthetic context")
    return core


@pytest.fixture
def policy_path(tmp_path):
    path = tmp_path / "routes.json"
    path.write_text(
        json.dumps(
            {
                "targets": {
                    "fast": {
                        "provider": "local",
                        "model": "small",
                        "criteria": "simple",
                    },
                    "deep": {
                        "provider": "remote",
                        "model": "large",
                        "criteria": "complex",
                    },
                }
            }
        )
    )
    return path


@pytest.fixture
def http_boundary(monkeypatch):
    """All HTTP is in memory, including worker requests and classifier calls."""
    requests = []
    response = {
        "model": "jev-1.13.0",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "deep",
                "confidence": 0.95,
                "probabilities": {"fast": 0.05, "deep": 0.95},
            }
        },
        "usage": {"input_tokens": 360, "output_tokens": 31},
    }

    def handle(request):
        requests.append(request)
        if request.url.host == "worker.test":
            return httpx.Response(
                200,
                json={
                    "id": "test-completion",
                    "model": json.loads(request.content)["model"],
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "DONE"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 7,
                        "completion_tokens": 2,
                        "total_tokens": 9,
                    },
                },
            )
        return httpx.Response(200, json=response)

    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handle)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    return requests, response


async def route(engine, prompt, policy_path, **kwargs):
    integration = importlib.import_module("omnimancer.decisions.integration")
    return await integration.route_headless(engine, prompt, policy_path, **kwargs)


@pytest.mark.asyncio
async def test_route_changes_outgoing_worker_model_without_persistence(
    engine, policy_path, http_boundary
):
    config_path = engine.config_manager.config_path
    before = config_path.read_bytes()
    config_before = engine.config_manager.get_config().model_dump()
    metadata = await route(engine, "synthetic complex task", policy_path)
    assert metadata["status"] == "routed"
    assert metadata["target"] == metadata["applied_target"] == "deep"
    assert metadata["input_tokens"] == 360
    assert metadata["estimated_cost_usd"] > 0
    assert engine.current_provider is engine.providers["remote"]
    response = await engine.current_provider.send_message(
        "synthetic worker task", engine.chat_manager.current_context
    )
    assert response.is_success
    requests, _ = http_boundary
    worker_request = next(r for r in requests if r.url.host == "worker.test")
    assert json.loads(worker_request.content)["model"] == "large"
    assert engine.providers["local"].model == "small"
    assert config_path.read_bytes() == before
    assert engine.config_manager.get_config().model_dump() == config_before
    payload = json.dumps(metadata)
    for private in (
        "synthetic complex task",
        "worker.test",
        "synthetic-router-key",
        str(policy_path),
    ):
        assert private not in payload


@pytest.mark.asyncio
async def test_shadow_preserves_models_and_context(engine, policy_path, http_boundary):
    policy = json.loads(policy_path.read_text())
    policy["mode"] = "shadow"
    policy_path.write_text(json.dumps(policy))
    context = engine.chat_manager.current_context
    before = copy.deepcopy(context)
    metadata = await route(engine, "synthetic task", policy_path)
    assert metadata["status"] == "shadow"
    assert metadata["target"] == "deep"
    assert metadata["applied_target"] is None
    assert engine.current_provider is engine.providers["local"]
    assert engine.providers["remote"].model == "previous"
    assert engine.chat_manager.current_context is context
    assert context == before


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["false", "raise", "timeout", "cancel"])
@pytest.mark.parametrize("same_provider", [False, True])
async def test_failed_switch_restores_source_destination_and_context(
    engine, policy_path, http_boundary, monkeypatch, failure, same_provider
):
    if same_provider:
        policy = json.loads(policy_path.read_text())
        policy["targets"]["deep"]["provider"] = "local"
        policy_path.write_text(json.dumps(policy))
    original = engine.current_provider
    context = engine.chat_manager.current_context
    context_before = copy.deepcopy(context)

    async def partial_switch(provider, model):
        engine.providers[provider].model = model
        original.model = "mutated-source"
        engine.current_provider = engine.providers[provider]
        context.messages[0].content = "mutated conversation"
        engine.chat_manager.set_current_model(model)
        engine.chat_manager._initialize_context()
        if failure == "raise":
            raise RuntimeError("SECRET endpoint credential /private/path")
        if failure == "timeout":
            raise TimeoutError("SECRET endpoint credential /private/path")
        if failure == "cancel":
            raise asyncio.CancelledError()
        return False

    monkeypatch.setattr(engine, "switch_model", partial_switch)
    if failure == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await route(engine, "synthetic task", policy_path)
    else:
        metadata = await route(engine, "synthetic task", policy_path)
        assert metadata["status"] == "switch_failed"
        assert metadata["target"] == "deep"
        assert metadata["applied_target"] is None
        assert "SECRET" not in json.dumps(metadata)
    assert engine.current_provider is original
    assert original.model == "small"
    assert engine.providers["remote"].model == "previous"
    assert engine.chat_manager.current_context is context
    assert context == context_before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs,status",
    [
        ({"provider": "local"}, "explicit_selection"),
        ({"model": "small"}, "explicit_selection"),
        ({"base_url": "http://worker.test/v1"}, "explicit_selection"),
        ({"resume": "synthetic-session"}, "resume"),
    ],
)
async def test_overrides_bypass_even_unreadable_policy(
    engine, tmp_path, http_boundary, kwargs, status
):
    metadata = await route(engine, "synthetic task", tmp_path / "missing", **kwargs)
    assert metadata["status"] == status
    assert metadata["applied_target"] is None
    assert not http_boundary[0]
    assert engine.current_provider is engine.providers["local"]


@pytest.mark.asyncio
async def test_disabled_does_not_load_policy_or_query(engine, http_boundary):
    assert await route(engine, "synthetic task", None) is None
    assert not http_boundary[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content,status",
    [
        (None, "policy_unreadable"),
        (b"x" * 65537, "policy_too_large"),
        (b"\xff", "policy_invalid"),
        (b'{"api_key": "PRIVATE"}', "policy_invalid"),
        (b"[]", "policy_invalid"),
        (b"{broken /private/path", "policy_invalid"),
    ],
)
async def test_policy_failures_are_bounded_categorical_fallbacks(
    engine, tmp_path, http_boundary, content, status
):
    path = tmp_path / "policy"
    if content is not None:
        path.write_bytes(content)
    metadata = await route(engine, "synthetic task", path)
    assert metadata["status"] == status
    assert metadata["target"] is None
    assert not http_boundary[0]
    assert "/private/path" not in json.dumps(metadata)
    assert str(path) not in json.dumps(metadata)
    assert "PRIVATE" not in json.dumps(metadata)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid", ["unknown", "unsupported_model", "azure", "subclass", "unavailable"]
)
async def test_invalid_target_excluded_before_query(
    engine, policy_path, http_boundary, invalid
):
    policy = json.loads(policy_path.read_text())
    if invalid == "unknown":
        policy["targets"]["deep"]["provider"] = "missing"
    elif invalid == "unsupported_model":
        policy["targets"]["deep"]["model"] = "unadvertised"
    elif invalid == "unavailable":
        engine.providers["remote"]._catalog_models[0].available = False
    elif invalid == "azure":
        engine.providers["remote"] = AzureProvider(
            "synthetic", "large", azure_endpoint="https://azure.test"
        )
    else:

        class UnknownProvider(OpenAICompatibleProvider):
            pass

        engine.providers["remote"] = UnknownProvider("", "large")
    policy_path.write_text(json.dumps(policy))
    metadata = await route(engine, "synthetic task", policy_path)
    assert metadata["status"] == "insufficient_targets"
    assert not http_boundary[0]
    assert engine.current_provider is engine.providers["local"]


@pytest.mark.asyncio
async def test_filtered_choices_do_not_include_excluded_target(
    engine, policy_path, http_boundary
):
    policy = json.loads(policy_path.read_text())
    policy["targets"]["excluded"] = {
        "provider": "missing",
        "model": "missing",
        "criteria": "excluded private criterion",
    }
    policy_path.write_text(json.dumps(policy))
    metadata = await route(engine, "synthetic task", policy_path)
    assert metadata["status"] == "routed"
    assert len(http_boundary[0]) == 1
    assert b"excluded" not in http_boundary[0][0].content
    assert b"remote" not in http_boundary[0][0].content
    assert b"worker.test" not in http_boundary[0][0].content


@pytest.mark.asyncio
async def test_openai_and_custom_model_target_can_be_selected(
    engine, policy_path, http_boundary
):
    engine.providers["remote"] = OpenAIProvider("synthetic", "previous")
    engine.config_manager.config.custom_models = [
        EnhancedModelInfo(
            name="large",
            provider="remote",
            description="synthetic",
            max_tokens=4096,
            cost_per_million_input=0,
            cost_per_million_output=0,
        )
    ]
    metadata = await route(engine, "synthetic task", policy_path)
    assert metadata["status"] == "routed"
    assert engine.current_provider.model == "large"


@pytest.mark.asyncio
async def test_low_confidence_keeps_baseline_and_raw_recommendation(
    engine, policy_path, http_boundary
):
    answer = http_boundary[1]["answers"]["route"]
    answer["confidence"] = 0.6
    answer["probabilities"] = {"fast": 0.4, "deep": 0.6}
    metadata = await route(engine, "synthetic task", policy_path)
    assert metadata["status"] == "low_confidence"
    assert metadata["target"] is None
    assert metadata["choice"] == "deep"
    assert metadata["applied_target"] is None
    assert engine.current_provider is engine.providers["local"]


@pytest.mark.parametrize("output_format", ["json", "stream-json"])
@pytest.mark.parametrize("error", [False, True])
def test_routing_metadata_in_success_error_and_stream_output(output_format, error):
    from omnimancer.cli.headless import HeadlessOutputEmitter, OutputFormat

    metadata = {
        "status": "routed",
        "applied_target": "deep",
        "input_tokens": 360,
        "estimated_cost_usd": 0.0001,
    }
    emitter = HeadlessOutputEmitter(
        OutputFormat(output_format), "session", routing=metadata
    )
    emitter._stdout = StringIO()
    emitter._stderr = StringIO()
    if error:
        emitter.emit_error("synthetic error")
    else:
        emitter.emit_result("DONE", "large", {"input_tokens": 7}, 0.03, "stop")
    output = json.loads(emitter._stdout.getvalue())
    assert output["routing"] == metadata
    if not error:
        assert output["usage"]["input_tokens"] == 7
        assert output["total_cost_usd"] == 0.03


@pytest.mark.asyncio
async def test_headless_disabled_never_imports_routing(
    engine, monkeypatch, capsys, http_boundary
):
    from omnimancer.cli.headless import run_headless

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if "decisions" in name:
            raise AssertionError("disabled routing was imported")
        return real_import(name, *args, **kwargs)

    async def initialized(self):
        self.providers = engine.providers
        self.current_provider = self.providers["local"]
        self._initialize_agent_engine()

    monkeypatch.setattr(CoreEngine, "initialize_providers", initialized)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = await run_headless(
        "synthetic task",
        config_path=str(engine.config_manager.config_path),
        output_format="json",
    )
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert "routing" not in output
    assert output["model"] == "small"
    assert all(request.url.host == "worker.test" for request in http_boundary[0])


@pytest.mark.asyncio
async def test_headless_uses_routing_before_worker_and_reports_separate_usage(
    engine, policy_path, http_boundary, monkeypatch, capsys
):
    from omnimancer.cli.headless import run_headless

    async def initialized(self):
        self.providers = engine.providers
        self.current_provider = self.providers["local"]
        self._initialize_agent_engine()

    monkeypatch.setattr(CoreEngine, "initialize_providers", initialized)
    result = await run_headless(
        "synthetic task",
        config_path=str(engine.config_manager.config_path),
        output_format="json",
        routing_policy=str(policy_path),
    )
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["routing"]["status"] == "routed"
    assert output["routing"]["applied_target"] == "deep"
    assert output["usage"]["input_tokens"] == 7
    assert output["model"] == "large"
    assert output["provider"] == "remote"
    assert len(http_boundary[0]) == 2


def test_cli_rejects_routing_policy_in_interactive_mode(monkeypatch, capsys):
    from omnimancer.cli.interface import main

    monkeypatch.setattr(
        sys, "argv", ["omn", "--routing-policy", "synthetic-policy.json"]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "headless" in capsys.readouterr().err.lower()


def test_cli_passes_routing_policy_to_headless(monkeypatch, capsys):
    from omnimancer.cli import headless
    from omnimancer.cli.interface import main

    async def inspect_run(**kwargs):
        assert kwargs["routing_policy"] == "synthetic-policy.json"
        assert kwargs["prompt"] == "synthetic task"
        return 0

    monkeypatch.setattr(headless, "run_headless", inspect_run)
    monkeypatch.setattr(sys, "stdin", StringIO(""))
    monkeypatch.setattr(
        sys,
        "argv",
        ["omn", "-p", "synthetic task", "--routing-policy", "synthetic-policy.json"],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 0


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes require POSIX")
def test_policy_named_pipe_is_rejected_without_blocking(tmp_path):
    path = tmp_path / "not-a-file"
    os.mkfifo(path)
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from omnimancer.decisions.integration import _load_policy; "
            "print(_load_policy(sys.argv[1])[1])",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=2,
        check=True,
    )
    assert process.stdout.strip() == "policy_unreadable"


@pytest.mark.asyncio
@pytest.mark.parametrize("override", ["provider", "model", "base_url", "resume"])
async def test_headless_overrides_reach_bypass_before_policy_io(
    engine, http_boundary, monkeypatch, capsys, tmp_path, override
):
    from omnimancer.cli.headless import run_headless

    async def initialized(self):
        self.providers = engine.providers
        self.current_provider = self.providers["local"]
        self._initialize_agent_engine()

    monkeypatch.setattr(CoreEngine, "initialize_providers", initialized)
    values = {
        "provider": "local",
        "model": "small",
        "base_url": "http://worker.test/v1",
        "resume": "missing-session",
    }
    result = await run_headless(
        "synthetic task",
        output_format="json",
        config_path=str(engine.config_manager.config_path),
        routing_policy=str(tmp_path / "missing-policy"),
        **{override: values[override]},
    )
    output = json.loads(capsys.readouterr().out)
    assert output["routing"]["status"] == (
        "resume" if override == "resume" else "explicit_selection"
    )
    assert result == (1 if override == "resume" else 0)
    assert all(request.url.host == "worker.test" for request in http_boundary[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["deny", "read_only", "hook"])
async def test_routing_preserves_tool_permission_and_hook_gates(
    engine, policy_path, http_boundary, tmp_path, gate
):
    from omnimancer.cli.headless import HeadlessRunner
    from omnimancer.core.agent.types import Operation, OperationType
    from omnimancer.core.models import HookCommand, PermissionRule

    config = engine.config_manager.get_config()
    if gate == "deny":
        config.permissions.always_deny = [PermissionRule(tool="command_execute")]
    if gate == "hook":
        config.hooks.tool_use_request = [
            HookCommand(name="synthetic-veto", command="exit 3", blocking=True)
        ]
    engine._initialize_agent_engine()
    agent = engine.agent_engine
    assert agent is not None
    if gate == "read_only":
        agent.set_read_only(True)
    HeadlessRunner._enable_auto_approval(agent)
    config_before = config.model_dump()
    file_before = engine.config_manager.config_path.read_bytes()
    metadata = await route(engine, "synthetic task", policy_path)
    assert metadata["status"] == "routed"
    target = tmp_path / "must-not-exist.txt"
    result = await asyncio.wait_for(
        agent.execute_with_approval(
            Operation(
                type=OperationType.COMMAND_EXECUTE,
                description="synthetic write",
                data={"command": "touch", "args": [str(target)]},
                requires_approval=True,
            )
        ),
        timeout=2,
    )
    assert result.success is False
    assert ("hook" if gate == "hook" else "permission rule") in result.error
    assert not target.exists()
    assert config.model_dump() == config_before
    assert engine.config_manager.config_path.read_bytes() == file_before


@pytest.mark.asyncio
async def test_pre_send_hook_blocks_classifier_egress(
    engine, policy_path, http_boundary
):
    from omnimancer.core.models import HookCommand, HooksConfig

    engine.config_manager.config.hooks = HooksConfig(
        pre_send_message=[
            HookCommand(
                name="egress-veto", command="exit 1", blocking=True, matcher="protected"
            )
        ]
    )
    metadata = await route(engine, "protected synthetic task", policy_path)
    assert metadata["status"] == "hook_blocked"
    assert not http_boundary[0]
    assert engine.current_provider is engine.providers["local"]


@pytest.mark.asyncio
async def test_fresh_compatible_configured_models_are_targets(
    engine, policy_path, http_boundary
):
    for provider in engine.providers.values():
        provider._catalog_models = []
    policy = json.loads(policy_path.read_text())
    policy["targets"]["deep"]["model"] = "previous"
    policy_path.write_text(json.dumps(policy))
    metadata = await route(engine, "complex task", policy_path)
    assert metadata["status"] == "routed"
    assert len(http_boundary[0]) == 1
    assert engine.current_provider.model == "previous"
