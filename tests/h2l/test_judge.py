"""Judge: deterministic checks first, then the high model grades the diff (PRD §5)."""

from unittest.mock import AsyncMock, MagicMock

from omnimancer.core.models import ChatResponse, ToolCall
from omnimancer.h2l.judge import SUBMIT_VERDICT_TOOL, Judge
from omnimancer.h2l.models import H2LModelRef, Story, WorkerReport

REF = H2LModelRef(provider="gateway", model="qwen3-8b")


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


def _verdict(score, passed, failures=None, feedback="fb"):
    return _resp(
        tool_calls=[
            ToolCall(
                name="submit_verdict",
                arguments={
                    "score": score,
                    "passed": passed,
                    "failures": failures or [],
                    "feedback": feedback,
                },
            )
        ]
    )


STORY = Story(
    id="S1",
    title="t",
    instructions="i",
    files=["/repo/a.py"],
    acceptance=["a.py greets", "no other file changes"],
    verify="pytest -q",
)


def _report(**kw):
    base = dict(
        story_id="S1",
        worker=REF,
        attempt=1,
        success=True,
        narrative="I did great work, trust me",
        diff="--- a/a.py\n+++ b/a.py\n-x\n+y\n",
        verify_output="1 passed",
        verify_exit=0,
    )
    base.update(kw)
    return WorkerReport(**base)


class TestDeterministic:
    async def test_verify_failure_short_circuits(self):
        provider = _provider([])
        verdict = await Judge(provider, pass_threshold=80).check(
            STORY, _report(verify_exit=2, verify_output="FAILED")
        )
        assert verdict.passed is False
        assert verdict.judge == "deterministic"
        assert verdict.failures == ["verify failed (exit 2)"]
        provider.send_message_with_tools.assert_not_called()

    async def test_out_of_scope_files_short_circuits(self):
        provider = _provider([])
        verdict = await Judge(provider, 80).check(
            STORY, _report(files_outside_scope=["/repo/keep.py"])
        )
        assert verdict.failures == ["modified files outside story scope: /repo/keep.py"]
        provider.send_message_with_tools.assert_not_called()

    async def test_empty_diff_short_circuits(self):
        provider = _provider([])
        verdict = await Judge(provider, 80).check(STORY, _report(diff=""))
        assert verdict.failures == ["no changes made"]
        provider.send_message_with_tools.assert_not_called()

    async def test_worker_failure_short_circuits(self):
        provider = _provider([])
        verdict = await Judge(provider, 80).check(
            STORY, _report(success=False, error="provider boom")
        )
        assert verdict.failures == ["worker failed: provider boom"]
        provider.send_message_with_tools.assert_not_called()

    async def test_order_verify_before_scope(self):
        verdict = await Judge(_provider([]), 80).check(
            STORY, _report(verify_exit=1, files_outside_scope=["/repo/k.py"])
        )
        assert verdict.failures == ["verify failed (exit 1)"]

    async def test_feedback_is_actionable_text(self):
        verdict = await Judge(_provider([]), 80).check(STORY, _report(verify_exit=1))
        assert "pytest -q" in verdict.feedback


class TestModelStage:
    async def test_prompt_contains_diff_and_verify_but_not_narrative(self):
        provider = _provider([_verdict(90, True)])
        await Judge(provider, 80).check(STORY, _report())
        message = provider.send_message_with_tools.call_args[0][0]
        assert message.startswith("SYSTEM: You are reviewing")
        assert "+y" in message
        assert "1 passed" in message
        assert "a.py greets" in message
        assert "trust me" not in message
        tools = provider.send_message_with_tools.call_args[0][2]
        assert tools == [SUBMIT_VERDICT_TOOL]

    async def test_pass_requires_flag_and_threshold(self):
        v = await Judge(_provider([_verdict(80, True)]), 80).check(STORY, _report())
        assert v.passed is True and v.score == 80 and v.judge == "model"

    async def test_score_below_threshold_fails_even_if_flagged_passed(self):
        v = await Judge(_provider([_verdict(79, True)]), 80).check(STORY, _report())
        assert v.passed is False
        assert any("threshold" in f for f in v.failures)

    async def test_flag_false_fails_even_above_threshold(self):
        v = await Judge(
            _provider([_verdict(95, False, ["missing greeting"])]), 80
        ).check(STORY, _report())
        assert v.passed is False
        assert v.failures == ["missing greeting"]
        assert v.feedback == "fb"

    async def test_no_verdict_is_a_failed_verdict_with_error(self):
        provider = _provider([_resp("looks fine")] * 5)
        v = await Judge(provider, 80, max_iterations=2).check(STORY, _report())
        assert v.passed is False
        assert v.error is not None
        assert v.judge == "model"

    async def test_invalid_verdict_rejected_back_to_model(self):
        bad = _resp(
            tool_calls=[ToolCall(name="submit_verdict", arguments={"score": 500})]
        )
        provider = _provider([bad, _verdict(85, True)])
        v = await Judge(provider, 80).check(STORY, _report())
        assert v.passed is True
        assert provider.send_message_with_tools.call_count == 2

    async def test_provider_error_is_failed_verdict(self):
        v = await Judge(_provider([_resp(error="boom")]), 80).check(STORY, _report())
        assert v.passed is False
        assert "boom" in (v.error or "")

    async def test_usage_and_attempt_carried(self):
        r = _verdict(90, True)
        r.input_tokens, r.output_tokens = 7, 3
        v = await Judge(_provider([r]), 80).check(STORY, _report(attempt=3))
        assert v.attempt == 3
        assert v.usage.input_tokens == 7 and v.usage.output_tokens == 3

    async def test_large_diff_truncated_in_prompt(self):
        provider = _provider([_verdict(90, True)])
        await Judge(provider, 80).check(STORY, _report(diff="x" * 100_000))
        message = provider.send_message_with_tools.call_args[0][0]
        assert len(message) < 60_000
        assert "truncated" in message
