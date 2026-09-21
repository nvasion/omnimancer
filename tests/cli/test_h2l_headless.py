"""Headless H2L surface: flags, config merge, events, result, exit codes (PRD §8)."""

import json
import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.cli.h2l_headless import H2LOptions, resolve_h2l_config
from omnimancer.cli.headless import HeadlessOutputEmitter, HeadlessRunner, OutputFormat
from omnimancer.core.models import Config, EventsConfig, ProviderConfig
from omnimancer.h2l.models import (
    H2LModelRef,
    Plan,
    RunSummary,
    Story,
    StoryOutcome,
    StoryState,
    UsageTotals,
    Verdict,
    WorkerReport,
)
from omnimancer.h2l.planner import PlanResult
from omnimancer.h2l.refs import H2LConfigError

HIGH = H2LModelRef(provider="claude", model="opus")
LOW = H2LModelRef(provider="gateway", model="qwen")


def _config(h2l=None):
    return Config(
        default_provider="claude",
        providers={
            "claude": ProviderConfig(api_key="k", model="sonnet"),
            "gateway": ProviderConfig(
                api_key="", model="qwen", provider_type="openai-compatible"
            ),
        },
        storage_path="/tmp/omn-h2l-headless",
        events=EventsConfig(enabled=False),
        h2l=h2l,
    )


class TestResolveConfig:
    def test_config_block_alone(self):
        cfg = resolve_h2l_config(
            _config({"high": "claude:opus", "low": ["gateway:qwen"]}), H2LOptions()
        )
        assert cfg.high == HIGH and cfg.low == [LOW]
        assert cfg.max_parallel == 2

    def test_flags_override_field_by_field(self):
        cfg = resolve_h2l_config(
            _config({"high": "claude:opus", "low": ["gateway:qwen"], "max_retries": 5}),
            H2LOptions(
                high="claude:haiku",
                low=["gateway:a", "gateway:b"],
                parallel=3,
                threshold=60,
                escalation="block",
            ),
        )
        assert cfg.high.model == "haiku"
        assert [r.model for r in cfg.low] == ["a", "b"]
        assert cfg.max_parallel == 3
        assert cfg.max_retries == 5  # untouched by flags
        assert cfg.pass_threshold == 60
        assert cfg.escalation == "block"

    def test_flags_alone_without_block(self):
        cfg = resolve_h2l_config(
            _config(None), H2LOptions(high="claude:opus", low=["gateway:qwen"])
        )
        assert cfg.high == HIGH

    def test_bare_refs_use_default_provider(self):
        cfg = resolve_h2l_config(_config(None), H2LOptions(high="opus", low=["qwen"]))
        assert cfg.high.provider == "claude"
        assert cfg.low[0].provider == "claude"

    def test_missing_high(self):
        with pytest.raises(H2LConfigError) as exc:
            resolve_h2l_config(_config(None), H2LOptions(low=["gateway:qwen"]))
        assert "H2L: no high model configured (set h2l.high or pass --high)" in str(
            exc.value
        )

    def test_missing_low(self):
        with pytest.raises(H2LConfigError) as exc:
            resolve_h2l_config(_config(None), H2LOptions(high="claude:opus"))
        assert "no low models configured" in str(exc.value)

    def test_headless_escalation_ask_becomes_high(self):
        cfg = resolve_h2l_config(
            _config(
                {"high": "claude:opus", "low": ["gateway:qwen"], "escalation": "ask"}
            ),
            H2LOptions(),
        )
        assert cfg.escalation == "high"
        assert cfg.plan_approval is False


PLAN = Plan(
    goal="g",
    design="d",
    stories=[
        Story(
            id="S1",
            title="one",
            instructions="i",
            files=["/repo/a.py"],
            acceptance=["ok"],
        ),
        Story(
            id="S2",
            title="two",
            instructions="i",
            files=["/repo/b.py"],
            acceptance=["ok"],
        ),
    ],
)


def _summary(states, stop_cause="done"):
    outcomes = [
        StoryOutcome(
            id=s.id,
            title=s.title,
            state=st,
            attempts=1,
            score=90 if st == StoryState.PASSED else None,
            model="gateway:qwen",
            reason=None if st == StoryState.PASSED else st.value,
        )
        for s, st in zip(PLAN.stories, states)
    ]
    return RunSummary(
        goal="g",
        stories=outcomes,
        usage_by_tier={
            "high": UsageTotals(input_tokens=10, output_tokens=2, cost_usd=0.1),
            "low": UsageTotals(input_tokens=20, output_tokens=4, cost_usd=0.01),
        },
        stop_cause=stop_cause,
    )


class FakeRun:
    """Stands in for H2LRun: drives the presenter the way the real run does."""

    summary = _summary([StoryState.PASSED, StoryState.PASSED])
    plan_result = PlanResult(plan=PLAN)
    instances: list = []
    run_called = False

    def __init__(self, engine, config, presenter, cwd, session_id, tiers=None):
        self.config = config
        self.presenter = presenter
        self.plan = None
        self.usage_by_tier = {"high": UsageTotals(), "low": UsageTotals()}
        FakeRun.instances.append(self)

    async def plan_goal(self, goal):
        return FakeRun.plan_result

    async def run(self, goal):
        FakeRun.run_called = True
        result = await self.plan_goal(goal)
        if result.plan is None:
            summary = RunSummary(
                goal=goal, stop_cause="plan_failed", error=result.error
            )
            await self.presenter.on_done(summary)
            return summary
        self.plan = result.plan
        await self.presenter.on_plan(result.plan)
        for story in result.plan.stories:
            await self.presenter.on_story_start(
                story, LOW, 1, f"h2l-{story.id}-abcd1234"
            )
            await self.presenter.on_report(
                story,
                WorkerReport(
                    story_id=story.id,
                    worker=LOW,
                    diff="+x",
                    verify_exit=0,
                    files_changed=story.files,
                ),
            )
            await self.presenter.on_verdict(
                story, Verdict(story_id=story.id, passed=True, score=90, judge="model")
            )
        await self.presenter.on_done(FakeRun.summary)
        return FakeRun.summary


def _engine(h2l=None):
    engine = MagicMock()
    engine.runtime_identity.return_value = ("claude", "sonnet")
    engine.config_manager.get_config.return_value = _config(h2l)
    engine.agent_engine = MagicMock()
    return engine


def _runner(engine, fmt, no_approval=True, options=None):
    runner = HeadlessRunner(
        engine=engine,
        output_format=fmt,
        no_approval=no_approval,
        h2l=options or H2LOptions(enabled=True),
    )
    out, err = StringIO(), StringIO()
    runner._emitter._stdout = out
    runner._emitter._stderr = err
    return runner, out, err


def _lines(buf):
    return [json.loads(line) for line in buf.getvalue().splitlines() if line.strip()]


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeRun.instances = []
    FakeRun.run_called = False
    FakeRun.summary = _summary([StoryState.PASSED, StoryState.PASSED])
    FakeRun.plan_result = PlanResult(plan=PLAN)
    with (
        patch("omnimancer.cli.h2l_headless.H2LRun", FakeRun),
        patch("omnimancer.cli.h2l_headless.validate_tiers", return_value=MagicMock()),
    ):
        yield


BLOCK = {"high": "claude:opus", "low": ["gateway:qwen"]}


class TestHeadlessRun:
    async def test_stream_json_events_and_result(self):
        runner, out, _ = _runner(_engine(BLOCK), OutputFormat.STREAM_JSON)
        code = await runner.run("goal")
        assert code == 0
        lines = _lines(out)
        subtypes = [(line["type"], line.get("subtype")) for line in lines]
        assert subtypes[0] == ("system", "init")
        assert ("h2l", "plan") in subtypes
        assert subtypes.count(("h2l", "story_start")) == 2
        assert subtypes.count(("h2l", "story_report")) == 2
        assert subtypes.count(("h2l", "story_verdict")) == 2
        assert ("h2l", "done") in subtypes
        assert subtypes[-1] == ("result", "success")
        result = lines[-1]
        assert result["h2l"]["passed"] == 2
        assert result["h2l"]["usage_by_tier"]["high"]["input_tokens"] == 10
        assert result["stop_cause"] == "done"

    async def test_event_key_sets_locked(self):
        runner, out, _ = _runner(_engine(BLOCK), OutputFormat.STREAM_JSON)
        await runner.run("goal")
        by_subtype = {}
        for line in _lines(out):
            if line["type"] == "h2l":
                by_subtype.setdefault(line["subtype"], set(line))
        assert by_subtype["plan"] == {
            "type",
            "subtype",
            "session_id",
            "plan",
            "model_used",
        }
        assert by_subtype["story_start"] == {
            "type",
            "subtype",
            "session_id",
            "story_id",
            "attempt",
            "provider",
            "model",
            "agent_id",
        }
        assert by_subtype["story_report"] == {
            "type",
            "subtype",
            "session_id",
            "story_id",
            "attempt",
            "success",
            "files_changed",
            "verify_exit",
        }
        assert by_subtype["story_verdict"] == {
            "type",
            "subtype",
            "session_id",
            "story_id",
            "attempt",
            "score",
            "passed",
            "failures",
            "judge",
        }
        assert by_subtype["done"] == {
            "type",
            "subtype",
            "session_id",
            "passed",
            "blocked",
            "skipped",
            "cancelled",
        }

    async def test_blocked_and_skipped_events_and_exit_3(self):
        FakeRun.summary = _summary([StoryState.BLOCKED, StoryState.SKIPPED], "partial")
        runner, out, _ = _runner(_engine(BLOCK), OutputFormat.STREAM_JSON)
        code = await runner.run("goal")
        assert code == 3
        lines = _lines(out)
        blocked = [line for line in lines if line.get("subtype") == "story_blocked"]
        skipped = [line for line in lines if line.get("subtype") == "story_skipped"]
        assert set(blocked[0]) == {
            "type",
            "subtype",
            "session_id",
            "story_id",
            "reason",
        }
        assert skipped[0]["story_id"] == "S2"

    async def test_rate_limited_exit_4(self):
        FakeRun.summary = _summary(
            [StoryState.PENDING, StoryState.PENDING], "rate_limited"
        )
        runner, out, _ = _runner(_engine(BLOCK), OutputFormat.JSON)
        code = await runner.run("goal")
        assert code == 4
        assert _lines(out)[-1]["stop_cause"] == "rate_limited"

    async def test_plan_failed_exit_1(self):
        FakeRun.plan_result = PlanResult(error="plan_failed: no submit")
        runner, out, err = _runner(_engine(BLOCK), OutputFormat.JSON)
        code = await runner.run("goal")
        assert code == 1
        blob = _lines(out)[-1]
        assert blob["is_error"] is True
        assert blob["stop_cause"] == "plan_failed"
        assert blob["model"] == "claude:opus"  # the tier, not the session model
        assert "no submit" in err.getvalue()

    async def test_plan_iterations_flag_reaches_config(self):
        runner, _, _ = _runner(
            _engine(BLOCK),
            OutputFormat.JSON,
            options=H2LOptions(enabled=True, plan_iterations=45),
        )
        await runner.run("goal")
        assert FakeRun.instances[0].config.planner_max_iterations == 45

    async def test_config_error_exit_1(self):
        runner, out, err = _runner(_engine(None), OutputFormat.JSON)
        code = await runner.run("goal")
        assert code == 1
        assert "no high model configured" in err.getvalue()
        assert not FakeRun.run_called

    async def test_without_skip_permissions_is_plan_only(self):
        runner, out, _ = _runner(
            _engine(BLOCK), OutputFormat.STREAM_JSON, no_approval=False
        )
        code = await runner.run("goal")
        assert code == 0
        lines = _lines(out)
        assert any(line.get("subtype") == "plan" for line in lines)
        assert lines[-1]["subtype"] == "h2l_plan_only"
        assert lines[-1]["h2l"]["plan"]["stories"][0]["id"] == "S1"
        assert not FakeRun.run_called

    async def test_init_names_the_high_tier_not_the_session_model(self):
        runner, out, _ = _runner(_engine(BLOCK), OutputFormat.STREAM_JSON)
        await runner.run("goal")
        init = _lines(out)[0]
        assert init["subtype"] == "init"
        assert init["model"] == "claude:opus"

    async def test_planner_bash_follows_auto_approval(self):
        runner, _, _ = _runner(_engine(BLOCK), OutputFormat.JSON, no_approval=False)
        await runner.run("goal")
        assert FakeRun.instances[0].planner_bash is False
        runner, _, _ = _runner(_engine(BLOCK), OutputFormat.JSON, no_approval=True)
        await runner.run("goal")
        assert FakeRun.instances[1].planner_bash is True

    async def test_plan_only_reports_served_model(self):
        FakeRun.plan_result = PlanResult(plan=PLAN, model_used="qwen3.8-max")
        runner, out, _ = _runner(
            _engine(BLOCK),
            OutputFormat.JSON,
            options=H2LOptions(enabled=True, plan_only=True),
        )
        await runner.run("goal")
        assert _lines(out)[-1]["h2l"]["model_used"] == "qwen3.8-max"

    async def test_plan_only_flag(self):
        runner, out, _ = _runner(
            _engine(BLOCK),
            OutputFormat.JSON,
            options=H2LOptions(enabled=True, plan_only=True),
        )
        code = await runner.run("goal")
        assert code == 0
        assert _lines(out)[-1]["subtype"] == "h2l_plan_only"
        assert not FakeRun.run_called

    async def test_text_mode_prints_summary_to_stdout(self):
        runner, out, err = _runner(_engine(BLOCK), OutputFormat.TEXT)
        code = await runner.run("goal")
        assert code == 0
        assert "S1" in out.getvalue() and "passed" in out.getvalue()
        assert "S1" in err.getvalue()  # progress lines go to stderr

    async def test_json_result_keys_include_h2l(self):
        runner, out, _ = _runner(_engine(BLOCK), OutputFormat.JSON)
        await runner.run("goal")
        blob = _lines(out)[-1]
        assert "h2l" in blob
        assert blob["usage"]["input_tokens"] == 30

    async def test_flags_reach_run_config(self):
        runner, _, _ = _runner(
            _engine(BLOCK),
            OutputFormat.JSON,
            options=H2LOptions(enabled=True, low=["gateway:other"], parallel=4),
        )
        await runner.run("goal")
        assert FakeRun.instances[0].config.low[0].model == "other"
        assert FakeRun.instances[0].config.max_parallel == 4

    async def test_notify_payload_flags_h2l(self):
        runner, _, _ = _runner(_engine(BLOCK), OutputFormat.JSON)
        await runner.run("goal")
        assert runner._turn_notifier.build_payload()["h2l"] is True


class TestWaitHeartbeat:
    async def test_stream_json_waiting_event_keys(self):
        from omnimancer.cli.h2l_headless import HeadlessPresenter

        emitter = HeadlessOutputEmitter(OutputFormat.STREAM_JSON, "s")
        buf = StringIO()
        emitter._stdout = buf
        await HeadlessPresenter(emitter, text_mode=False).on_wait(
            "planner digitalocean:qwen3.8-max", 61.4
        )
        line = json.loads(buf.getvalue())
        assert set(line) == {"type", "subtype", "session_id", "who", "seconds"}
        assert line["subtype"] == "waiting" and line["seconds"] == 61

    async def test_text_mode_heartbeat_on_stderr(self):
        from omnimancer.cli.h2l_headless import HeadlessPresenter

        emitter = HeadlessOutputEmitter(OutputFormat.TEXT, "s")
        out, err = StringIO(), StringIO()
        emitter._stdout, emitter._stderr = out, err
        await HeadlessPresenter(emitter, text_mode=True).on_wait("planner x", 30.0)
        assert out.getvalue() == ""
        assert "waiting on planner x (30s)" in err.getvalue()


class TestPlannerProgress:
    async def test_stream_json_planner_tool_event_keys(self):
        from omnimancer.cli.h2l_headless import HeadlessPresenter
        from omnimancer.core.models import ToolCall, ToolResult

        emitter = HeadlessOutputEmitter(OutputFormat.STREAM_JSON, "s")
        buf = StringIO()
        emitter._stdout = buf
        presenter = HeadlessPresenter(emitter, text_mode=False)
        await presenter.on_planner_tool_call(
            ToolCall(name="Read", arguments={"file_path": "/r/a.py"}),
            ToolResult(content="x"),
        )
        line = json.loads(buf.getvalue())
        assert set(line) == {
            "type",
            "subtype",
            "session_id",
            "name",
            "arguments",
            "error",
        }
        assert line["subtype"] == "planner_tool"
        assert line["name"] == "Read"

    async def test_submit_plan_is_summarized_and_error_comes_first(self):
        from omnimancer.cli.h2l_headless import HeadlessPresenter
        from omnimancer.core.models import ToolCall, ToolResult

        emitter = HeadlessOutputEmitter(OutputFormat.STREAM_JSON, "s")
        buf = StringIO()
        emitter._stdout = buf
        presenter = HeadlessPresenter(emitter, text_mode=False)
        await presenter.on_planner_tool_call(
            ToolCall(
                name="submit_plan",
                arguments={
                    "goal": "g" * 500,
                    "design": "d" * 5000,
                    "stories": [{"id": "S1", "instructions": "x" * 9000}, {"id": "S2"}],
                },
            ),
            ToolResult(content="", error="H2L: dependency cycle: S1 -> S2 -> S1"),
        )
        raw = buf.getvalue()
        line = json.loads(raw)
        assert line["arguments"] == {"goal": "g" * 120, "stories": ["S1", "S2"]}
        assert line["error"] == "H2L: dependency cycle: S1 -> S2 -> S1"
        # A log cut at 300 characters still shows the rejection reason.
        assert "dependency cycle" in raw[:300]
        assert raw.index('"error"') < raw.index('"arguments"')

    async def test_text_mode_shows_rejection_reason(self):
        from omnimancer.cli.h2l_headless import HeadlessPresenter
        from omnimancer.core.models import ToolCall, ToolResult

        emitter = HeadlessOutputEmitter(OutputFormat.TEXT, "s")
        out, err = StringIO(), StringIO()
        emitter._stdout, emitter._stderr = out, err
        await HeadlessPresenter(emitter, text_mode=True).on_planner_tool_call(
            ToolCall(name="submit_plan", arguments={"stories": [{"id": "S1"}]}),
            ToolResult(content="", error="H2L: story S1 depends on unknown story S9"),
        )
        assert "submit_plan 1 stories — rejected: H2L: story S1" in err.getvalue()

    async def test_text_mode_writes_progress_to_stderr(self):
        from omnimancer.cli.h2l_headless import HeadlessPresenter
        from omnimancer.core.models import ToolCall, ToolResult

        emitter = HeadlessOutputEmitter(OutputFormat.TEXT, "s")
        out, err = StringIO(), StringIO()
        emitter._stdout, emitter._stderr = out, err
        presenter = HeadlessPresenter(emitter, text_mode=True)
        await presenter.on_planner_tool_call(
            ToolCall(name="Grep", arguments={"pattern": "route"}),
            ToolResult(content="", error="nope"),
        )
        assert out.getvalue() == ""
        assert "planner › Grep route — error: nope" in err.getvalue()


class TestEmitterAdditions:
    def test_result_without_h2l_keeps_existing_keys(self):
        emitter = HeadlessOutputEmitter(OutputFormat.STREAM_JSON, "s")
        buf = StringIO()
        emitter._stdout = buf
        emitter.emit_result("hi", "m", {}, 0.0, "end_turn")
        assert "h2l" not in json.loads(buf.getvalue())

    def test_emit_h2l_text_mode_writes_nothing_to_stdout(self):
        emitter = HeadlessOutputEmitter(OutputFormat.TEXT, "s")
        buf = StringIO()
        emitter._stdout = buf
        emitter.emit_h2l("plan", {"plan": {}})
        assert buf.getvalue() == ""


class TestCliFlags:
    def _invoke(self, argv):
        from omnimancer.cli.interface import main

        rh = AsyncMock(return_value=0)
        with (
            patch.object(sys, "argv", ["omn", *argv]),
            patch("omnimancer.cli.headless.run_headless", rh),
            patch("sys.stdin") as stdin,
        ):
            stdin.isatty.return_value = True
            with pytest.raises(SystemExit) as exc:
                main()
        return exc.value.code, rh

    def test_h2l_options_passed_through(self):
        code, rh = self._invoke(
            [
                "-p",
                "goal",
                "--h2l",
                "--high",
                "claude:opus",
                "--low",
                "gateway:a",
                "--low",
                "gateway:b",
                "--h2l-parallel",
                "2",
                "--h2l-retries",
                "1",
                "--h2l-threshold",
                "70",
                "--h2l-escalation",
                "block",
            ]
        )
        assert code == 0
        options = rh.call_args.kwargs["h2l"]
        assert options.enabled is True
        assert options.high == "claude:opus"
        assert options.low == ["gateway:a", "gateway:b"]
        assert options.parallel == 2 and options.retries == 1
        assert options.threshold == 70 and options.escalation == "block"

    def test_plan_iterations_flag(self):
        code, rh = self._invoke(
            [
                "-p",
                "goal",
                "--h2l",
                "--high",
                "claude:opus",
                "--low",
                "gateway:a",
                "--h2l-plan-iterations",
                "40",
            ]
        )
        assert code == 0
        assert rh.call_args.kwargs["h2l"].plan_iterations == 40

    def test_plain_run_has_h2l_disabled(self):
        code, rh = self._invoke(["-p", "goal"])
        assert code == 0
        assert rh.call_args.kwargs["h2l"].enabled is False

    def test_h2l_requires_prompt(self):
        code, rh = self._invoke(["--h2l"])
        assert code == 2
        rh.assert_not_called()

    def test_h2l_flags_require_h2l(self):
        code, rh = self._invoke(["-p", "goal", "--high", "claude:opus"])
        assert code == 2
        rh.assert_not_called()
