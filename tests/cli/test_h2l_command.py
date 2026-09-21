"""The /h2l slash command and its TUI presenter (PRD §7)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.cli.commands import Command, SlashCommand, parse_command
from omnimancer.cli.h2l_command import TUIPresenter, handle_h2l_command
from omnimancer.core.models import Config, EventsConfig, ProviderConfig
from omnimancer.h2l.models import (
    H2LModelRef,
    Plan,
    RunSummary,
    Story,
    StoryOutcome,
    StoryState,
    Verdict,
    WorkerReport,
)
from omnimancer.h2l.planner import PlanResult

LOW = H2LModelRef(provider="gateway", model="qwen")
BLOCK = {"high": "claude:opus", "low": ["gateway:qwen"]}


def _config(h2l=BLOCK):
    return Config(
        default_provider="claude",
        providers={
            "claude": ProviderConfig(api_key="k", model="sonnet"),
            "gateway": ProviderConfig(
                api_key="", model="qwen", provider_type="openai-compatible"
            ),
        },
        storage_path="/tmp/omn-h2l-cmd",
        events=EventsConfig(enabled=False),
        h2l=h2l,
    )


PLAN = Plan(
    goal="g",
    design="d",
    stories=[
        Story(
            id="S1",
            title="one",
            instructions="first",
            files=["/r/a.py"],
            acceptance=["ok"],
        ),
        Story(
            id="S2",
            title="two",
            instructions="second",
            files=["/r/b.py"],
            acceptance=["ok"],
        ),
    ],
)


def _summary():
    return RunSummary(
        goal="g",
        stories=[
            StoryOutcome(
                id="S1",
                title="one",
                state=StoryState.PASSED,
                attempts=1,
                score=90,
                model="gateway:qwen",
            ),
            StoryOutcome(
                id="S2",
                title="two",
                state=StoryState.BLOCKED,
                attempts=2,
                reason="retries_exhausted",
            ),
        ],
        stop_cause="partial",
    )


class FakeRun:
    instances: list = []

    def __init__(self, engine, config, presenter, cwd, session_id, tiers=None):
        self.config = config
        self.presenter = presenter
        self.plan = None
        self.run_goal = None
        self.ran_plan = None
        FakeRun.instances.append(self)

    async def plan_goal(self, goal):
        return PlanResult(plan=PLAN)

    async def run(self, goal):
        self.run_goal = goal
        summary = _summary()
        await self.presenter.on_done(summary)
        return summary

    async def run_plan(self, plan):
        self.ran_plan = plan
        summary = _summary()
        await self.presenter.on_done(summary)
        return summary


class _CLI:
    def __init__(self, config):
        self.engine = MagicMock()
        self.engine.config_manager.get_config.return_value = config
        self.engine.runtime_identity.return_value = ("claude", "sonnet")
        self.console = MagicMock()
        self.errors, self.infos, self.successes = [], [], []
        self._h2l_inputs = []

    def _show_error(self, m):
        self.errors.append(m)

    def _show_info(self, m):
        self.infos.append(m)

    def _show_success(self, m):
        self.successes.append(m)

    async def _h2l_input(self, prompt):
        return self._h2l_inputs.pop(0)


def _cmd(*args):
    return Command.create_slash_command(SlashCommand.H2L, list(args), "/h2l")


@pytest.fixture(autouse=True)
def _fake_run():
    FakeRun.instances = []
    with (
        patch("omnimancer.cli.h2l_command.H2LRun", FakeRun),
        patch("omnimancer.cli.h2l_command.validate_tiers", return_value=MagicMock()),
    ):
        yield


class TestParsing:
    def test_h2l_is_a_slash_command(self):
        cmd = parse_command("/h2l add rate limiting to the handler")
        assert cmd.slash_command == SlashCommand.H2L
        assert cmd.args == ["add", "rate", "limiting", "to", "the", "handler"]


class TestCommand:
    async def test_not_configured(self):
        cli = _CLI(_config(h2l=None))
        await handle_h2l_command(cli, _cmd("do", "it"))
        assert cli.errors and "H2L is not configured" in cli.errors[0]

    async def test_goal_runs_and_posts_one_summary_message(self):
        cli = _CLI(_config())
        await handle_h2l_command(cli, _cmd("add", "rate", "limiting"))
        assert FakeRun.instances[0].run_goal == "add rate limiting"
        cli.engine.chat_manager.add_assistant_message.assert_called_once()
        text = cli.engine.chat_manager.add_assistant_message.call_args[0][0]
        assert "S1" in text and "passed" in text and "S2" in text
        cli.engine.chat_manager.add_user_message.assert_not_called()

    async def test_plan_then_run(self):
        cli = _CLI(_config())
        await handle_h2l_command(cli, _cmd("plan", "the", "goal"))
        assert cli._h2l_pending_plan is not None
        assert cli._h2l_pending_plan.stories[0].id == "S1"
        cli.console.print.assert_called()  # the story table
        await handle_h2l_command(cli, _cmd("run"))
        assert FakeRun.instances[-1].ran_plan is cli._h2l_pending_plan or (
            FakeRun.instances[-1].ran_plan.stories[0].id == "S1"
        )

    async def test_run_without_pending_plan(self):
        cli = _CLI(_config())
        await handle_h2l_command(cli, _cmd("run"))
        assert cli.errors and "No pending plan" in cli.errors[0]

    async def test_status_without_plan(self):
        cli = _CLI(_config())
        await handle_h2l_command(cli, _cmd("status"))
        assert cli.infos

    async def test_bare_command_shows_usage(self):
        cli = _CLI(_config())
        await handle_h2l_command(cli, _cmd())
        assert cli.infos and "Usage" in cli.infos[0]

    async def test_config_show(self):
        cli = _CLI(_config())
        await handle_h2l_command(cli, _cmd("config"))
        cli.console.print.assert_called()

    async def test_config_set_persists(self):
        cli = _CLI(_config())
        config = cli.engine.config_manager.get_config()
        await handle_h2l_command(
            cli, _cmd("config", "set", "high", "claude:haiku", "max_parallel", "3")
        )
        assert config.h2l.high.model == "haiku"
        assert config.h2l.max_parallel == 3
        cli.engine.config_manager.save_config.assert_called_once()
        assert cli.successes

    async def test_config_set_planner_budget_and_off(self):
        cli = _CLI(_config())
        config = cli.engine.config_manager.get_config()
        await handle_h2l_command(
            cli, _cmd("config", "set", "planner_max_iterations", "40")
        )
        assert config.h2l.planner_max_iterations == 40
        await handle_h2l_command(
            cli, _cmd("config", "set", "planner_max_iterations", "off")
        )
        assert config.h2l.planner_max_iterations is None

    async def test_config_set_low_list(self):
        cli = _CLI(_config())
        config = cli.engine.config_manager.get_config()
        await handle_h2l_command(
            cli, _cmd("config", "set", "low", "gateway:a,gateway:b")
        )
        assert [r.model for r in config.h2l.low] == ["a", "b"]

    async def test_config_set_unknown_field(self):
        cli = _CLI(_config())
        await handle_h2l_command(cli, _cmd("config", "set", "nope", "1"))
        assert cli.errors
        cli.engine.config_manager.save_config.assert_not_called()

    async def test_config_set_creates_block_when_both_tiers_given(self):
        cli = _CLI(_config(h2l=None))
        config = cli.engine.config_manager.get_config()
        await handle_h2l_command(
            cli, _cmd("config", "set", "high", "claude:opus", "low", "gateway:qwen")
        )
        assert config.h2l is not None
        assert config.h2l.high.model == "opus"

    async def test_config_set_without_block_needs_both(self):
        cli = _CLI(_config(h2l=None))
        await handle_h2l_command(cli, _cmd("config", "set", "high", "claude:opus"))
        assert cli.errors and "low" in cli.errors[0]


class TestTUIPresenter:
    def _presenter(self, inputs, editor=None):
        cli = _CLI(_config())
        cli._h2l_inputs = list(inputs)
        presenter = TUIPresenter(cli)
        if editor is not None:
            presenter._edit_text = AsyncMock(return_value=editor)
        return presenter, cli

    async def test_accept(self):
        presenter, _ = self._presenter(["a"])
        decision = await presenter.approve_plan(PLAN)
        assert decision.accepted is True
        assert [s.id for s in decision.plan.stories] == ["S1", "S2"]

    async def test_drop_then_accept(self):
        presenter, _ = self._presenter(["d S2", "a"])
        decision = await presenter.approve_plan(PLAN)
        assert decision.accepted is True
        assert [s.id for s in decision.plan.stories] == ["S1"]

    async def test_cancel(self):
        presenter, _ = self._presenter(["c"])
        decision = await presenter.approve_plan(PLAN)
        assert decision.accepted is False

    async def test_edit_replaces_instructions(self):
        presenter, _ = self._presenter(["e S1", "a"], editor="rewritten steps")
        decision = await presenter.approve_plan(PLAN)
        assert decision.plan.stories[0].instructions == "rewritten steps"

    async def test_invalid_input_reprompts(self):
        presenter, cli = self._presenter(["zzz", "d S9", "a"])
        decision = await presenter.approve_plan(PLAN)
        assert decision.accepted is True
        assert len(cli.errors) >= 2

    @pytest.mark.parametrize(
        "answer,expected", [("h", "high"), ("b", "block"), ("r", "retry")]
    )
    async def test_escalation_choices(self, answer, expected):
        presenter, _ = self._presenter([answer])
        outcome = StoryOutcome(id="S1", title="one", attempts=3)
        assert await presenter.on_escalation(PLAN.stories[0], outcome) == expected

    async def test_planner_progress_prints(self):
        from omnimancer.core.models import ToolCall, ToolResult

        presenter, cli = self._presenter([])
        await presenter.on_planner_tool_call(
            ToolCall(name="Read", arguments={"file_path": "/r/a.py"}),
            ToolResult(content="x"),
        )
        printed = cli.console.print.call_args[0][0]
        assert "planner" in printed and "Read" in printed and "/r/a.py" in printed

    async def test_progress_and_summary_print(self):
        presenter, cli = self._presenter([])
        await presenter.on_story_start(PLAN.stories[0], LOW, 1, "h2l-S1-abcd")
        await presenter.on_report(
            PLAN.stories[0],
            WorkerReport(story_id="S1", worker=LOW, diff="+x", verify_exit=0),
        )
        await presenter.on_verdict(
            PLAN.stories[0],
            Verdict(
                story_id="S1",
                passed=False,
                failures=["missing"],
                feedback="add it",
                judge="model",
            ),
        )
        await presenter.on_done(_summary())
        assert cli.console.print.call_count >= 4
