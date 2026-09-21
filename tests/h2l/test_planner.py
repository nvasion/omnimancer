"""Planner: high model explores, submits a validated plan via a tool (PRD §3)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from omnimancer.core.models import ChatResponse, ToolCall, ToolResult
from omnimancer.h2l.planner import SUBMIT_PLAN_TOOL, Planner


def _resp(content="", tool_calls=None, error=None):
    return ChatResponse(
        content=content,
        model_used="m",
        tokens_used=0,
        tool_calls=tool_calls,
        error=error,
    )


def _provider(responses):
    provider = MagicMock()
    provider.model = "claude-opus-5"
    provider.send_message_with_tools = AsyncMock(side_effect=responses)
    provider.supports_native_tool_history.return_value = False
    return provider


def _plan_args(tmp_path: Path, **overrides):
    args = {
        "goal": "g",
        "design": "d",
        "stories": [
            {
                "id": "S1",
                "title": "t",
                "instructions": "i",
                "files": [str(tmp_path / "a.py")],
                "acceptance": ["ok"],
            }
        ],
    }
    args.update(overrides)
    return args


def _submit(args):
    return _resp(tool_calls=[ToolCall(name="submit_plan", arguments=args)])


def _planner(tmp_path, provider, th=None):
    agent_engine = MagicMock()
    planner = Planner(provider, agent_engine, cwd=tmp_path, max_iterations=6)
    handler = th or MagicMock()
    if th is None:
        handler.execute_tool_call = AsyncMock(return_value=ToolResult(content="ok"))
    planner._tool_handler_factory = lambda e: handler
    return planner, handler


class TestPlanner:
    async def test_submit_then_self_check_second_submission_wins(self, tmp_path):
        first = _plan_args(tmp_path)
        second = _plan_args(tmp_path, design="revised")
        provider = _provider([_submit(first), _submit(second)])
        planner, _ = _planner(tmp_path, provider)
        result = await planner.plan("do the thing")
        assert result.plan is not None
        assert result.plan.design == "revised"
        assert result.error is None
        # The self-check message was sent as the second turn.
        second_message = provider.send_message_with_tools.call_args_list[1][0][0]
        assert "Re-read every story" in second_message

    async def test_first_message_has_prompt_and_goal(self, tmp_path):
        provider = _provider([_submit(_plan_args(tmp_path)), _resp("fine as is")])
        planner, _ = _planner(tmp_path, provider)
        await planner.plan("add rate limiting")
        message = provider.send_message_with_tools.call_args_list[0][0][0]
        assert message.startswith("SYSTEM: You are the principal engineer")
        assert "Goal: add rate limiting" in message
        assert "Working Directory" in message

    async def test_self_check_without_resubmission_keeps_first(self, tmp_path):
        provider = _provider([_submit(_plan_args(tmp_path)), _resp("looks right")])
        planner, _ = _planner(tmp_path, provider)
        result = await planner.plan("g")
        assert result.plan is not None
        assert result.plan.stories[0].id == "S1"

    async def test_tools_offered_are_read_only_plus_submit(self, tmp_path):
        provider = _provider([_submit(_plan_args(tmp_path)), _resp("ok")])
        planner, _ = _planner(tmp_path, provider)
        await planner.plan("g")
        tools = provider.send_message_with_tools.call_args_list[0][0][2]
        assert sorted(t.name for t in tools) == [
            "Bash",
            "Glob",
            "Grep",
            "Read",
            "add_story",
            "submit_plan",
        ]
        assert SUBMIT_PLAN_TOOL in tools

    async def test_write_and_edit_refused(self, tmp_path):
        provider = _provider(
            [
                _resp(
                    tool_calls=[
                        ToolCall(
                            name="Write", arguments={"file_path": "x", "content": ""}
                        )
                    ]
                ),
                _submit(_plan_args(tmp_path)),
                _resp("ok"),
            ]
        )
        planner, handler = _planner(tmp_path, provider)
        result = await planner.plan("g")
        handler.execute_tool_call.assert_not_awaited()
        assert result.plan is not None
        assert (
            result.tool_log[0].error == "H2L: the planner may not modify files (Write)"
        )

    async def test_read_tools_go_through_handler(self, tmp_path):
        provider = _provider(
            [
                _resp(tool_calls=[ToolCall(name="Read", arguments={"file_path": "x"})]),
                _submit(_plan_args(tmp_path)),
                _resp("ok"),
            ]
        )
        planner, handler = _planner(tmp_path, provider)
        await planner.plan("g")
        handler.execute_tool_call.assert_awaited_once()

    async def test_invalid_plan_rejected_back_to_model(self, tmp_path):
        bad = _plan_args(tmp_path)
        bad["stories"][0]["files"] = ["relative.py"]
        provider = _provider([_submit(bad), _submit(_plan_args(tmp_path)), _resp("ok")])
        planner, _ = _planner(tmp_path, provider)
        result = await planner.plan("g")
        assert result.plan is not None
        assert result.plan.stories[0].files == [str(tmp_path / "a.py")]
        assert "not an absolute path" in result.tool_log[0].error

    async def test_schema_error_rejected_back_to_model(self, tmp_path):
        provider = _provider(
            [
                _submit({"goal": "g", "stories": "nope"}),
                _submit(_plan_args(tmp_path)),
                _resp("ok"),
            ]
        )
        planner, _ = _planner(tmp_path, provider)
        result = await planner.plan("g")
        assert result.plan is not None
        assert "submit_plan rejected" in result.tool_log[0].error

    async def test_no_submission_is_plan_failed(self, tmp_path):
        provider = _provider([_resp("I would start by...")] * 10)
        planner, _ = _planner(tmp_path, provider)
        result = await planner.plan("g")
        assert result.plan is None
        assert result.error is not None
        assert "plan_failed" in result.error
        assert "stopped without submitting" in result.error
        # Iterations are not the user's problem; don't send them there.
        assert "iterations" not in result.error

    async def test_stories_added_one_per_call_then_submitted(self, tmp_path):
        # The live failure: one huge submit_plan was cut off at the output
        # token cap and arrived with no stories. One story per call cannot be.
        def add(id, deps=None):
            return _resp(
                tool_calls=[
                    ToolCall(
                        name="add_story",
                        arguments={
                            "id": id,
                            "title": f"t{id}",
                            "instructions": "i",
                            "files": [str(tmp_path / f"{id}.py")],
                            "acceptance": ["ok"],
                            "depends_on": deps or [],
                        },
                    )
                ]
            )

        close = _resp(
            tool_calls=[
                ToolCall(name="submit_plan", arguments={"goal": "g", "design": "d"})
            ]
        )
        provider = _provider(
            [add("S1"), add("S2", ["S1"]), add("S3"), close, _resp("OK")]
        )
        planner, _ = _planner(tmp_path, provider)
        result = await planner.plan("g")
        assert result.error is None
        assert [s.id for s in result.plan.stories] == ["S1", "S2", "S3"]
        assert result.plan.stories[1].depends_on == ["S1"]
        assert result.plan.design == "d"

    async def test_bad_story_rejected_immediately_with_every_problem(self, tmp_path):
        bad = _resp(
            tool_calls=[
                ToolCall(
                    name="add_story",
                    arguments={
                        "id": "S1",
                        "title": "t",
                        "instructions": "i",
                        "files": ["relative.py", "/etc/hosts"],
                        "acceptance": ["ok"],
                    },
                )
            ]
        )
        provider = _provider([bad, _submit(_plan_args(tmp_path)), _resp("OK")])
        planner, _ = _planner(tmp_path, provider)
        result = await planner.plan("g")
        error = result.tool_log[0].error
        assert "not an absolute path" in error
        assert "outside the working directory" in error
        assert result.plan is not None

    async def test_add_story_with_same_id_replaces(self, tmp_path):
        def add(title):
            return _resp(
                tool_calls=[
                    ToolCall(
                        name="add_story",
                        arguments={
                            "id": "S1",
                            "title": title,
                            "instructions": "i",
                            "files": [str(tmp_path / "a.py")],
                            "acceptance": ["ok"],
                        },
                    )
                ]
            )

        close = _resp(
            tool_calls=[
                ToolCall(name="submit_plan", arguments={"goal": "g", "design": "d"})
            ]
        )
        provider = _provider([add("first"), add("second"), close, _resp("OK")])
        planner, _ = _planner(tmp_path, provider)
        result = await planner.plan("g")
        assert [s.title for s in result.plan.stories] == ["second"]
        assert "replaced" in result.tool_log[1].result

    async def test_empty_submission_points_at_add_story(self, tmp_path):
        empty = _resp(
            tool_calls=[
                ToolCall(
                    name="submit_plan",
                    arguments={"goal": "g", "design": "d", "stories": []},
                )
            ]
        )
        provider = _provider([empty, _submit(_plan_args(tmp_path)), _resp("OK")])
        planner, _ = _planner(tmp_path, provider)
        result = await planner.plan("g")
        assert "add_story" in result.tool_log[0].error
        assert result.plan is not None

    async def test_failure_names_the_rejection_and_truncation(self, tmp_path):
        # Exactly the live failure: submissions rejected after being cut off.
        empty = _resp(
            tool_calls=[
                ToolCall(name="submit_plan", arguments={"goal": "g", "stories": []})
            ]
        )
        empty.stop_reason = "length"
        provider = _provider([empty] * 40)
        planner, _ = _planner(tmp_path, provider)  # budget of 6
        result = await planner.plan("g")
        assert result.plan is None
        assert "submission(s) rejected" in result.error
        assert "plan has no stories" in result.error
        assert "cut off at the token limit" in result.error
        assert "high_max_tokens" in result.error

    async def test_self_check_fixes_one_story_via_add_story(self, tmp_path):
        fix = _resp(
            tool_calls=[
                ToolCall(
                    name="add_story",
                    arguments={
                        "id": "S1",
                        "title": "fixed",
                        "instructions": "better",
                        "files": [str(tmp_path / "a.py")],
                        "acceptance": ["ok"],
                    },
                )
            ]
        )
        close = _resp(
            tool_calls=[
                ToolCall(name="submit_plan", arguments={"goal": "g", "design": "d2"})
            ]
        )
        provider = _provider([_submit(_plan_args(tmp_path)), fix, close])
        planner, _ = _planner(tmp_path, provider)
        result = await planner.plan("g")
        assert result.plan.stories[0].title == "fixed"
        assert result.plan.design == "d2"

    async def test_exhausted_budget_gets_a_closing_submit_phase(self, tmp_path):
        # The live failure: 20 exploration turns, never submitted. The
        # closing phase offers submit_plan only and the plan lands.
        reads = [
            _resp(tool_calls=[ToolCall(name="Read", arguments={"file_path": f"/f{i}"})])
            for i in range(6)
        ]
        provider = _provider(reads + [_submit(_plan_args(tmp_path)), _resp("ok")])
        planner, _ = _planner(tmp_path, provider)  # max_iterations=6
        result = await planner.plan("g")
        assert result.plan is not None
        closing_call = provider.send_message_with_tools.call_args_list[6]
        assert "exploration budget is used up" in closing_call[0][0]
        assert [t.name for t in closing_call[0][2]] == ["add_story", "submit_plan"]

    async def test_budget_stated_in_first_message_only_when_set(self, tmp_path):
        provider = _provider([_submit(_plan_args(tmp_path)), _resp("ok")])
        planner, _ = _planner(tmp_path, provider)  # explicit budget of 6
        await planner.plan("g")
        first = provider.send_message_with_tools.call_args_list[0][0][0]
        assert "Exploration budget: 6 turns" in first

        provider = _provider([_submit(_plan_args(tmp_path)), _resp("ok")])
        agent_engine = MagicMock()
        planner = Planner(provider, agent_engine, cwd=tmp_path)  # default
        handler = MagicMock()
        handler.execute_tool_call = AsyncMock(return_value=ToolResult(content="ok"))
        planner._tool_handler_factory = lambda e: handler
        await planner.plan("g")
        first = provider.send_message_with_tools.call_args_list[0][0][0]
        assert "Exploration budget" not in first

    async def test_default_planner_is_unbounded(self, tmp_path):
        reads = [
            _resp(tool_calls=[ToolCall(name="Read", arguments={"file_path": f"/f{i}"})])
            for i in range(35)
        ]
        provider = _provider(reads + [_submit(_plan_args(tmp_path)), _resp("ok")])
        agent_engine = MagicMock()
        planner = Planner(provider, agent_engine, cwd=tmp_path)
        handler = MagicMock()
        handler.execute_tool_call = AsyncMock(return_value=ToolResult(content="ok"))
        planner._tool_handler_factory = lambda e: handler
        result = await planner.plan("g")
        assert result.plan is not None
        assert result.iterations == 37
        assert result.error is None

    async def test_provider_error_is_plan_failed(self, tmp_path):
        provider = _provider([_resp(error="boom")])
        planner, _ = _planner(tmp_path, provider)
        result = await planner.plan("g")
        assert result.plan is None
        assert "boom" in result.error

    async def test_parent_conversation_untouched(self, tmp_path):
        provider = _provider([_submit(_plan_args(tmp_path)), _resp("ok")])
        agent_engine = MagicMock()
        planner = Planner(provider, agent_engine, cwd=tmp_path)
        handler = MagicMock()
        handler.execute_tool_call = AsyncMock(return_value=ToolResult(content="ok"))
        planner._tool_handler_factory = lambda e: handler
        await planner.plan("g")
        agent_engine.chat_manager.add_user_message.assert_not_called()
        agent_engine.send_message_with_tools.assert_not_called()

    async def test_usage_reported(self, tmp_path):
        r1 = _submit(_plan_args(tmp_path))
        r1.input_tokens, r1.output_tokens = 100, 20
        r2 = _resp("ok")
        r2.input_tokens, r2.output_tokens = 50, 5
        planner, _ = _planner(tmp_path, _provider([r1, r2]))
        result = await planner.plan("g")
        assert result.usage.input_tokens == 150
        assert result.usage.output_tokens == 25

    async def test_bash_withheld_when_not_allowed(self, tmp_path):
        provider = _provider(
            [
                _resp(tool_calls=[ToolCall(name="Bash", arguments={"command": "ls"})]),
                _submit(_plan_args(tmp_path)),
                _resp("ok"),
            ]
        )
        agent_engine = MagicMock()
        planner = Planner(provider, agent_engine, cwd=tmp_path, allow_bash=False)
        handler = MagicMock()
        handler.execute_tool_call = AsyncMock(return_value=ToolResult(content="ok"))
        planner._tool_handler_factory = lambda e: handler
        result = await planner.plan("g")
        tools = provider.send_message_with_tools.call_args_list[0][0][2]
        assert "Bash" not in [t.name for t in tools]
        handler.execute_tool_call.assert_not_awaited()
        assert "Bash is not available" in result.tool_log[0].error
        assert result.plan is not None

    async def test_served_model_reported(self, tmp_path):
        r1 = _submit(_plan_args(tmp_path))
        r1.model_used = "qwen3.8-max"
        r2 = _resp("ok")
        r2.model_used = "qwen3.8-max"
        planner, _ = _planner(tmp_path, _provider([r1, r2]))
        result = await planner.plan("g")
        assert result.model_used == "qwen3.8-max"

    async def test_progress_observer_sees_every_call(self, tmp_path):
        # A long planning phase must never look idle: every exploration call
        # and the submission itself reach the observer.
        provider = _provider(
            [
                _resp(tool_calls=[ToolCall(name="Read", arguments={"file_path": "x"})]),
                _submit(_plan_args(tmp_path)),
                _resp("ok"),
            ]
        )
        seen = []

        async def observe(tc, result):
            seen.append((tc.name, bool(result.error)))

        agent_engine = MagicMock()
        planner = Planner(provider, agent_engine, cwd=tmp_path, on_tool_call=observe)
        handler = MagicMock()
        handler.execute_tool_call = AsyncMock(return_value=ToolResult(content="ok"))
        planner._tool_handler_factory = lambda e: handler
        await planner.plan("g")
        assert seen == [("Read", False), ("submit_plan", False)]
