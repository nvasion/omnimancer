"""WorkerRunner: isolated, scoped, literal execution of one story (PRD §4)."""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from omnimancer.core.agent.status_core import EventType
from omnimancer.core.models import ChatResponse, ToolCall, ToolResult
from omnimancer.events import emitter as fleet_events
from omnimancer.h2l.models import H2LModelRef, Story
from omnimancer.h2l.worker import WorkerRunner

REF = H2LModelRef(provider="gateway", model="qwen3-8b")


def _resp(content="", tool_calls=None, error=None):
    return ChatResponse(
        content=content,
        model_used="m",
        tokens_used=0,
        tool_calls=tool_calls,
        error=error,
    )


def _provider(responses, native=False):
    provider = MagicMock()
    provider.model = "qwen3-8b"
    provider.send_message_with_tools = AsyncMock(side_effect=responses)
    provider.supports_native_tool_history.return_value = native
    return provider


def _story(tmp_path: Path, **kw):
    base = dict(
        id="S1",
        title="Add greeting",
        instructions="Edit the file.",
        files=[str(tmp_path / "a.py")],
        acceptance=["a.py greets"],
    )
    base.update(kw)
    return Story(**base)


def _tool_handler(results=None):
    th = MagicMock()
    th.execute_tool_call = AsyncMock(
        side_effect=results or [ToolResult(content="ok")] * 20
    )
    return th


def _runner(tmp_path, th=None):
    agent_engine = MagicMock()
    runner = WorkerRunner(agent_engine, cwd=tmp_path)
    runner._tool_handler_factory = lambda engine: th or _tool_handler()
    return runner


class TestWorkerRunner:
    async def test_first_message_carries_prompt_story_and_feedback(self, tmp_path):
        provider = _provider([_resp("done")])
        runner = _runner(tmp_path)
        report = await runner.run(
            _story(tmp_path),
            REF,
            provider,
            feedback="fix the import",
            attempt=2,
            worker_tools=["Edit", "Write", "Bash"],
            allow_read=True,
            max_iterations=5,
            run_id="h2l-run",
        )
        message = provider.send_message_with_tools.call_args[0][0]
        assert message.startswith("SYSTEM: You are a worker")
        assert "Add greeting" in message
        assert "Edit the file." in message
        assert str(tmp_path / "a.py") in message
        assert "Feedback from review of attempt 1:" in message
        assert "fix the import" in message
        assert report.attempt == 2
        assert report.narrative == "done"
        assert report.success is True

    async def test_tool_definitions_scoped_to_worker_set(self, tmp_path):
        provider = _provider([_resp("done")])
        runner = _runner(tmp_path)
        await runner.run(
            _story(tmp_path),
            REF,
            provider,
            worker_tools=["Edit", "Bash"],
            allow_read=False,
            max_iterations=5,
            run_id="r",
        )
        tools = provider.send_message_with_tools.call_args[0][2]
        assert sorted(t.name for t in tools) == ["Bash", "Edit"]

    async def test_read_added_when_allowed(self, tmp_path):
        provider = _provider([_resp("done")])
        runner = _runner(tmp_path)
        await runner.run(
            _story(tmp_path),
            REF,
            provider,
            worker_tools=["Edit"],
            allow_read=True,
            max_iterations=5,
            run_id="r",
        )
        tools = provider.send_message_with_tools.call_args[0][2]
        assert sorted(t.name for t in tools) == ["Edit", "Read"]

    async def test_hallucinated_tool_refused_before_gate(self, tmp_path):
        provider = _provider(
            [
                _resp(tool_calls=[ToolCall(name="WebFetch", arguments={"url": "x"})]),
                _resp("done"),
            ]
        )
        th = _tool_handler()
        runner = _runner(tmp_path, th)
        report = await runner.run(
            _story(tmp_path),
            REF,
            provider,
            worker_tools=["Edit"],
            allow_read=True,
            max_iterations=5,
            run_id="r",
        )
        th.execute_tool_call.assert_not_awaited()
        assert (
            report.tool_log[0].error
            == "H2L: tool WebFetch not permitted for this story"
        )

    async def test_out_of_scope_read_refused(self, tmp_path):
        outside = str(tmp_path / "secret.py")
        provider = _provider(
            [
                _resp(
                    tool_calls=[ToolCall(name="Read", arguments={"file_path": outside})]
                ),
                _resp("done"),
            ]
        )
        th = _tool_handler()
        runner = _runner(tmp_path, th)
        report = await runner.run(
            _story(tmp_path),
            REF,
            provider,
            worker_tools=["Edit"],
            allow_read=True,
            max_iterations=5,
            run_id="r",
        )
        th.execute_tool_call.assert_not_awaited()
        assert "not in this story's files" in report.tool_log[0].error

    async def test_out_of_scope_edit_refused(self, tmp_path):
        outside = str(tmp_path / "other.py")
        provider = _provider(
            [
                _resp(
                    tool_calls=[
                        ToolCall(
                            name="Edit",
                            arguments={
                                "file_path": outside,
                                "old_string": "a",
                                "new_string": "b",
                            },
                        )
                    ]
                ),
                _resp("done"),
            ]
        )
        th = _tool_handler()
        runner = _runner(tmp_path, th)
        report = await runner.run(
            _story(tmp_path),
            REF,
            provider,
            worker_tools=["Edit"],
            allow_read=False,
            max_iterations=5,
            run_id="r",
        )
        th.execute_tool_call.assert_not_awaited()
        assert "not in this story's files" in report.tool_log[0].error

    async def test_in_scope_edit_reaches_gate(self, tmp_path):
        target = str(tmp_path / "a.py")
        provider = _provider(
            [
                _resp(
                    tool_calls=[
                        ToolCall(
                            name="Edit",
                            arguments={
                                "file_path": target,
                                "old_string": "a",
                                "new_string": "b",
                            },
                        )
                    ]
                ),
                _resp("done"),
            ]
        )
        th = _tool_handler()
        runner = _runner(tmp_path, th)
        await runner.run(
            _story(tmp_path),
            REF,
            provider,
            worker_tools=["Edit"],
            allow_read=False,
            max_iterations=5,
            run_id="r",
        )
        th.execute_tool_call.assert_awaited_once()

    async def test_cancelled_aborts_and_marks_report(self, tmp_path):
        provider = _provider(
            [
                _resp(tool_calls=[ToolCall(name="Bash", arguments={"command": "ls"})]),
                _resp("x"),
            ]
        )
        th = _tool_handler([ToolResult(content="", error="cancelled", cancelled=True)])
        runner = _runner(tmp_path, th)
        report = await runner.run(
            _story(tmp_path),
            REF,
            provider,
            worker_tools=["Bash"],
            allow_read=False,
            max_iterations=5,
            run_id="r",
        )
        assert report.cancelled is True
        assert report.success is False

    async def test_provider_error_marks_report(self, tmp_path):
        provider = _provider([_resp(error="boom")])
        report = await _runner(tmp_path).run(
            _story(tmp_path),
            REF,
            provider,
            worker_tools=["Edit"],
            allow_read=False,
            max_iterations=5,
            run_id="r",
        )
        assert report.success is False
        assert report.error == "boom"

    async def test_parent_provider_never_touched(self, tmp_path):
        parent = MagicMock()
        parent.model = "session-model"
        worker_provider = _provider([_resp("done")])
        agent_engine = MagicMock()
        agent_engine.current_provider = parent
        runner = WorkerRunner(agent_engine, cwd=tmp_path)
        runner._tool_handler_factory = lambda e: _tool_handler()
        await runner.run(
            _story(tmp_path),
            REF,
            worker_provider,
            worker_tools=["Edit"],
            allow_read=False,
            max_iterations=5,
            run_id="r",
        )
        assert parent.model == "session-model"
        parent.send_message_with_tools.assert_not_called()

    async def test_verify_runs_through_gate_and_records_exit(self, tmp_path):
        provider = _provider([_resp("done")])
        th = _tool_handler(
            [
                ToolResult(
                    content="", error="Command exited with code 2.\nstdout:\nFAILED"
                )
            ]
        )
        runner = _runner(tmp_path, th)
        report = await runner.run(
            _story(tmp_path, verify="pytest -q"),
            REF,
            provider,
            worker_tools=["Edit"],
            allow_read=False,
            max_iterations=5,
            run_id="r",
        )
        call = th.execute_tool_call.call_args[0][0]
        assert call.name == "Bash"
        assert call.arguments["command"] == "pytest -q"
        assert report.verify_exit == 2
        assert "FAILED" in (report.verify_output or "")

    async def test_verify_success_exit_zero(self, tmp_path):
        provider = _provider([_resp("done")])
        th = _tool_handler([ToolResult(content="3 passed")])
        report = await _runner(tmp_path, th).run(
            _story(tmp_path, verify="pytest -q"),
            REF,
            provider,
            worker_tools=["Edit"],
            allow_read=False,
            max_iterations=5,
            run_id="r",
        )
        assert report.verify_exit == 0
        assert report.verify_output == "3 passed"

    async def test_emits_lifecycle_events_with_story_identity(self, tmp_path):
        provider = _provider([_resp("done")])
        calls = []

        async def _spy(event_type, data, *a, **k):
            calls.append(
                (
                    event_type,
                    data,
                    fleet_events.current_agent_id(),
                    fleet_events.current_parent_id(),
                )
            )

        with patch.object(fleet_events, "emit_event", _spy):
            await _runner(tmp_path).run(
                _story(tmp_path),
                REF,
                provider,
                worker_tools=["Edit"],
                allow_read=False,
                max_iterations=5,
                run_id="h2l-sess",
                attempt=2,
            )
        starts = [c for c in calls if c[0] == EventType.SESSION_START]
        ends = [c for c in calls if c[0] == EventType.SESSION_END]
        assert len(starts) == 1 and len(ends) == 1
        assert starts[0][1]["story_id"] == "S1"
        assert starts[0][1]["attempt"] == 2
        assert starts[0][1]["model"] == "qwen3-8b"
        assert starts[0][1]["provider"] == "gateway"
        assert starts[0][2].startswith("h2l-S1-")
        assert starts[0][3] == "h2l-sess"
        assert ends[0][1]["status"] == 0


@pytest.mark.skipif(shutil.which("git") is None, reason="requires the git binary")
class TestWorkerGitCapture:
    def _repo(self, tmp_path: Path) -> Path:
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@t"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "keep.py").write_text("k = 1\n")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)
        return tmp_path

    async def test_diff_scoped_and_outside_scope_detected(self, tmp_path):
        repo = self._repo(tmp_path)

        async def _mutate(tc):
            # Simulate the gate performing the worker's edit plus a stray write.
            (repo / "a.py").write_text("x = 2\n")
            (repo / "keep.py").write_text("k = 2\n")
            (repo / "new.txt").write_text("hi\n")
            return ToolResult(content="Edited")

        th = MagicMock()
        th.execute_tool_call = AsyncMock(side_effect=_mutate)
        provider = _provider(
            [
                _resp(
                    tool_calls=[
                        ToolCall(
                            name="Edit",
                            arguments={
                                "file_path": str(repo / "a.py"),
                                "old_string": "1",
                                "new_string": "2",
                            },
                        )
                    ]
                ),
                _resp("done"),
            ]
        )
        runner = _runner(repo, th)
        report = await runner.run(
            _story(repo, files=[str(repo / "a.py")]),
            REF,
            provider,
            worker_tools=["Edit"],
            allow_read=False,
            max_iterations=5,
            run_id="r",
        )
        assert "-x = 1" in report.diff and "+x = 2" in report.diff
        assert "k = 2" not in report.diff
        assert sorted(report.files_outside_scope) == [
            str(repo / "keep.py"),
            str(repo / "new.txt"),
        ]
        assert report.files_changed == [str(repo / "a.py")]

    async def test_new_in_scope_file_appears_in_diff(self, tmp_path):
        repo = self._repo(tmp_path)
        target = repo / "fresh.py"

        async def _mutate(tc):
            target.write_text("print('new')\n")
            return ToolResult(content="Wrote")

        th = MagicMock()
        th.execute_tool_call = AsyncMock(side_effect=_mutate)
        provider = _provider(
            [
                _resp(
                    tool_calls=[
                        ToolCall(
                            name="Write",
                            arguments={"file_path": str(target), "content": "x"},
                        )
                    ]
                ),
                _resp("done"),
            ]
        )
        report = await _runner(repo, th).run(
            _story(repo, files=[str(target)]),
            REF,
            provider,
            worker_tools=["Write"],
            allow_read=False,
            max_iterations=5,
            run_id="r",
        )
        assert "+print('new')" in report.diff
        assert report.files_outside_scope == []

    async def test_pre_existing_dirty_file_not_blamed_on_worker(self, tmp_path):
        repo = self._repo(tmp_path)
        (repo / "keep.py").write_text("k = 9\n")  # dirty before the worker runs
        provider = _provider([_resp("done")])
        report = await _runner(repo).run(
            _story(repo, files=[str(repo / "a.py")]),
            REF,
            provider,
            worker_tools=["Edit"],
            allow_read=False,
            max_iterations=5,
            run_id="r",
        )
        assert report.files_outside_scope == []
        assert report.diff == ""
