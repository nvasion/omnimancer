"""H2LRun: DAG scheduling, pool assignment, retries, escalation, events (PRD §6)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from omnimancer.core.agent.types import Operation, OperationType
from omnimancer.events import emitter as fleet_events
from omnimancer.h2l.models import (
    H2LConfig,
    H2LModelRef,
    Plan,
    Story,
    StoryState,
    UsageTotals,
    Verdict,
    WorkerReport,
)
from omnimancer.h2l.orchestrator import H2LRun, PlanDecision
from omnimancer.h2l.planner import PlanResult
from omnimancer.h2l.refs import ResolvedTiers

HIGH = H2LModelRef(provider="claude", model="opus")
LOW_A = H2LModelRef(provider="gateway", model="a")
LOW_B = H2LModelRef(provider="gateway", model="b")


def _story(tmp_path, id, deps=None, files=None):
    return Story(
        id=id,
        title=f"title {id}",
        instructions="do",
        files=files or [str(tmp_path / f"{id}.py")],
        acceptance=["ok"],
        depends_on=deps or [],
    )


def _config(**kw):
    base = dict(high=HIGH, low=[LOW_A, LOW_B], max_parallel=1, max_retries=1)
    base.update(kw)
    return H2LConfig(**base)


def _tiers():
    high = MagicMock(name="high")
    high.model = "opus"
    low_a = MagicMock(name="low_a")
    low_a.model = "a"
    low_b = MagicMock(name="low_b")
    low_b.model = "b"
    return ResolvedTiers(high=high, low=[low_a, low_b])


class RecordingPresenter:
    def __init__(self, decision=None, escalate="block"):
        self.events = []
        self._decision = decision
        self._escalate = escalate

    async def approve_plan(self, plan):
        self.events.append(("approve_plan", len(plan.stories)))
        return self._decision or PlanDecision(accepted=True, plan=plan)

    async def on_plan(self, plan, model_used=""):
        self.events.append(("plan", len(plan.stories)))

    async def on_story_start(self, story, ref, attempt, agent_id=None):
        self.events.append(("start", story.id, ref.label, attempt))
        assert agent_id is None or agent_id.startswith(f"h2l-{story.id}-")

    async def on_report(self, story, report):
        self.events.append(("report", story.id, report.success))

    async def on_verdict(self, story, verdict):
        self.events.append(("verdict", story.id, verdict.passed))

    async def on_escalation(self, story, outcome):
        self.events.append(("escalation", story.id))
        return self._escalate

    async def on_done(self, summary):
        self.events.append(("done", summary.passed))


class FakeWorker:
    """Scripted worker: returns reports in order, records calls."""

    def __init__(self, reports=None, delay=0.0):
        self.calls = []
        self.reports = list(reports or [])
        self.delay = delay
        self.active = 0
        self.max_active = 0

    async def run(self, story, ref, provider, **kw):
        self.calls.append(
            (
                story.id,
                ref.label,
                kw.get("feedback"),
                kw.get("attempt"),
                sorted(kw["worker_tools"]),
            )
        )
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.reports:
                report = self.reports.pop(0)
            else:
                report = WorkerReport(
                    story_id=story.id, worker=ref, diff="+x", verify_exit=0
                )
            report.story_id = story.id
            report.worker = ref
            report.attempt = kw.get("attempt", 1)
            return report
        finally:
            self.active -= 1


class FakeJudge:
    def __init__(self, verdicts=None):
        self.verdicts = list(verdicts or [])
        self.calls = []

    async def check(self, story, report):
        self.calls.append((story.id, report.attempt))
        if self.verdicts:
            v = self.verdicts.pop(0)
        else:
            v = Verdict(story_id=story.id, passed=True, score=90, judge="model")
        v.story_id = story.id
        v.attempt = report.attempt
        return v


def _run(tmp_path, config, presenter, worker, judge, planner=None, tiers=None):
    engine = MagicMock()
    engine.agent_engine = MagicMock()
    engine.agent_engine.approval.approval_callback = None
    run = H2LRun(
        engine,
        config,
        presenter,
        cwd=tmp_path,
        session_id="sess",
        tiers=tiers or _tiers(),
    )
    run._worker_factory = lambda *a, **k: worker
    run._judge_factory = lambda *a, **k: judge
    if planner is not None:
        run._planner_factory = lambda *a, **k: planner
    return run


def _plan(tmp_path, *stories):
    return Plan(goal="g", design="d", stories=list(stories))


class TestHappyPath:
    async def test_serial_two_stories_pass(self, tmp_path):
        presenter = RecordingPresenter()
        worker, judge = FakeWorker(), FakeJudge()
        run = _run(tmp_path, _config(), presenter, worker, judge)
        summary = await run.run_plan(
            _plan(tmp_path, _story(tmp_path, "S1"), _story(tmp_path, "S2"))
        )
        assert summary.all_passed
        assert [s.state for s in summary.stories] == [StoryState.PASSED] * 2
        assert [c[0] for c in worker.calls] == ["S1", "S2"]
        assert presenter.events[0] == ("start", "S1", "gateway:a", 1)
        assert presenter.events[-1] == ("done", 2)
        assert summary.stories[0].model == "gateway:a"
        assert summary.stories[0].score == 90
        assert summary.stop_cause == "done"

    async def test_round_robin_assignment(self, tmp_path):
        worker = FakeWorker()
        run = _run(tmp_path, _config(), RecordingPresenter(), worker, FakeJudge())
        await run.run_plan(
            _plan(
                tmp_path,
                _story(tmp_path, "S1"),
                _story(tmp_path, "S2"),
                _story(tmp_path, "S3"),
            )
        )
        assert [c[1] for c in worker.calls] == ["gateway:a", "gateway:b", "gateway:a"]

    async def test_worker_gets_config_tools_and_iterations(self, tmp_path):
        worker = FakeWorker()
        cfg = _config(
            worker_tools=["Edit"], allow_worker_read=False, worker_max_iterations=7
        )
        run = _run(tmp_path, cfg, RecordingPresenter(), worker, FakeJudge())
        await run.run_plan(_plan(tmp_path, _story(tmp_path, "S1")))
        assert worker.calls[0][4] == ["Edit"]

    async def test_usage_split_by_tier(self, tmp_path):
        report = WorkerReport(
            story_id="S1",
            worker=LOW_A,
            diff="+x",
            usage=UsageTotals(input_tokens=100, output_tokens=10),
        )
        verdict = Verdict(
            story_id="S1",
            passed=True,
            score=95,
            judge="model",
            usage=UsageTotals(input_tokens=30, output_tokens=3),
        )
        run = _run(
            tmp_path,
            _config(),
            RecordingPresenter(),
            FakeWorker([report]),
            FakeJudge([verdict]),
        )
        summary = await run.run_plan(_plan(tmp_path, _story(tmp_path, "S1")))
        assert summary.usage_by_tier["low"].input_tokens == 100
        assert summary.usage_by_tier["high"].input_tokens == 30
        assert summary.stories[0].usage.input_tokens == 130


class TestDependencies:
    async def test_dependency_order_respected(self, tmp_path):
        worker = FakeWorker()
        run = _run(tmp_path, _config(), RecordingPresenter(), worker, FakeJudge())
        await run.run_plan(
            _plan(tmp_path, _story(tmp_path, "S2", deps=["S1"]), _story(tmp_path, "S1"))
        )
        assert [c[0] for c in worker.calls] == ["S1", "S2"]

    async def test_run_plan_adds_implicit_shared_file_edge(self, tmp_path):
        shared = str(tmp_path / "shared.py")
        worker = FakeWorker()
        run = _run(
            tmp_path, _config(max_parallel=2), RecordingPresenter(), worker, FakeJudge()
        )
        summary = await run.run_plan(
            _plan(
                tmp_path,
                _story(tmp_path, "S1", files=[shared]),
                _story(tmp_path, "S2", files=[shared]),
            )
        )
        assert run.plan.stories[1].depends_on == ["S1"]
        assert summary.all_passed

    async def test_blocked_dependency_skips_dependents(self, tmp_path):
        fail = Verdict(
            story_id="S1", passed=False, failures=["no"], feedback="fix", judge="model"
        )
        judge = FakeJudge([fail, fail])
        run = _run(
            tmp_path,
            _config(max_retries=1, escalation="block"),
            RecordingPresenter(),
            FakeWorker(),
            judge,
        )
        summary = await run.run_plan(
            _plan(
                tmp_path,
                _story(tmp_path, "S1"),
                _story(tmp_path, "S2", deps=["S1"]),
                _story(tmp_path, "S3"),
            )
        )
        by_id = {s.id: s for s in summary.stories}
        assert by_id["S1"].state == StoryState.BLOCKED
        assert by_id["S1"].reason == "retries_exhausted"
        assert by_id["S2"].state == StoryState.SKIPPED
        assert by_id["S2"].reason == "dependency_blocked:S1"
        assert by_id["S3"].state == StoryState.PASSED
        assert summary.stop_cause == "partial"


class TestRetries:
    async def test_retry_goes_to_next_model_with_feedback(self, tmp_path):
        fail = Verdict(
            story_id="S1",
            passed=False,
            failures=["missing"],
            feedback="add the import",
            judge="model",
        )
        worker = FakeWorker()
        run = _run(
            tmp_path,
            _config(max_retries=2),
            RecordingPresenter(),
            worker,
            FakeJudge([fail]),
        )
        summary = await run.run_plan(_plan(tmp_path, _story(tmp_path, "S1")))
        assert summary.all_passed
        assert worker.calls[0] == (
            "S1",
            "gateway:a",
            None,
            1,
            ["Bash", "Edit", "Write"],
        )
        assert worker.calls[1][:4] == ("S1", "gateway:b", "add the import", 2)
        assert summary.stories[0].attempts == 2
        assert summary.stories[0].model == "gateway:b"
        assert len(summary.stories[0].verdicts) == 2

    async def test_single_model_pool_retries_same_model(self, tmp_path):
        fail = Verdict(
            story_id="S1", passed=False, failures=["x"], feedback="f", judge="model"
        )
        worker = FakeWorker()
        tiers = _tiers()
        tiers.low = tiers.low[:1]
        run = _run(
            tmp_path,
            _config(low=[LOW_A]),
            RecordingPresenter(),
            worker,
            FakeJudge([fail]),
            tiers=tiers,
        )
        await run.run_plan(_plan(tmp_path, _story(tmp_path, "S1")))
        assert [c[1] for c in worker.calls] == ["gateway:a", "gateway:a"]


class TestEscalation:
    def _failing(self, n):
        return [
            Verdict(
                story_id="S1", passed=False, failures=["x"], feedback="f", judge="model"
            )
            for _ in range(n)
        ]

    async def test_block_policy(self, tmp_path):
        run = _run(
            tmp_path,
            _config(max_retries=1, escalation="block"),
            RecordingPresenter(),
            FakeWorker(),
            FakeJudge(self._failing(2)),
        )
        summary = await run.run_plan(_plan(tmp_path, _story(tmp_path, "S1")))
        assert summary.stories[0].state == StoryState.BLOCKED
        assert summary.stories[0].attempts == 2

    async def test_high_policy_runs_high_model_with_full_tools(self, tmp_path):
        worker = FakeWorker()
        judge = FakeJudge(self._failing(2))
        tiers = _tiers()
        run = _run(
            tmp_path,
            _config(max_retries=1, escalation="high"),
            RecordingPresenter(),
            worker,
            judge,
            tiers=tiers,
        )
        summary = await run.run_plan(_plan(tmp_path, _story(tmp_path, "S1")))
        assert summary.stories[0].state == StoryState.PASSED
        assert summary.stories[0].reason == "escalated_to_high"
        assert summary.stories[0].model == "claude:opus"
        last = worker.calls[-1]
        assert last[1] == "claude:opus"
        assert "Glob" in last[4] and "Grep" in last[4]
        # The escalation attempt is judged deterministically only.
        assert len(judge.calls) == 2

    async def test_high_policy_deterministic_failure_blocks(self, tmp_path):
        bad = WorkerReport(
            story_id="S1", worker=HIGH, diff="+x", verify_exit=1, verify_output="FAIL"
        )
        worker = FakeWorker([WorkerReport(story_id="S1", worker=LOW_A, diff="+x"), bad])
        run = _run(
            tmp_path,
            _config(max_retries=0, escalation="high"),
            RecordingPresenter(),
            worker,
            FakeJudge(self._failing(1)),
        )
        summary = await run.run_plan(_plan(tmp_path, _story(tmp_path, "S1")))
        assert summary.stories[0].state == StoryState.BLOCKED
        assert summary.stories[0].reason == "escalation_failed"

    async def test_ask_policy_consults_presenter(self, tmp_path):
        presenter = RecordingPresenter(escalate="block")
        run = _run(
            tmp_path,
            _config(max_retries=0, escalation="ask"),
            presenter,
            FakeWorker(),
            FakeJudge(self._failing(1)),
        )
        summary = await run.run_plan(_plan(tmp_path, _story(tmp_path, "S1")))
        assert ("escalation", "S1") in presenter.events
        assert summary.stories[0].state == StoryState.BLOCKED

    async def test_ask_policy_retry_once_more(self, tmp_path):
        presenter = RecordingPresenter(escalate="retry")
        worker = FakeWorker()
        run = _run(
            tmp_path,
            _config(max_retries=0, escalation="ask"),
            presenter,
            worker,
            FakeJudge(self._failing(1)),
        )
        summary = await run.run_plan(_plan(tmp_path, _story(tmp_path, "S1")))
        assert summary.stories[0].state == StoryState.PASSED
        assert len(worker.calls) == 2


class TestCancellationAndErrors:
    async def test_cancelled_worker_marks_story_and_skips_dependents(self, tmp_path):
        cancelled = WorkerReport(
            story_id="S1", worker=LOW_A, success=False, cancelled=True
        )
        run = _run(
            tmp_path,
            _config(),
            RecordingPresenter(),
            FakeWorker([cancelled]),
            FakeJudge(),
        )
        summary = await run.run_plan(
            _plan(tmp_path, _story(tmp_path, "S1"), _story(tmp_path, "S2", deps=["S1"]))
        )
        by_id = {s.id: s for s in summary.stories}
        assert by_id["S1"].state == StoryState.CANCELLED
        assert by_id["S2"].state == StoryState.SKIPPED
        assert by_id["S2"].reason == "dependency_cancelled:S1"

    async def test_rate_limit_stops_run_and_keeps_state(self, tmp_path):
        limited = WorkerReport(
            story_id="S1", worker=LOW_A, success=False, error="429 rate limit exceeded"
        )
        run = _run(
            tmp_path,
            _config(),
            RecordingPresenter(),
            FakeWorker([limited]),
            FakeJudge(),
        )
        summary = await run.run_plan(
            _plan(tmp_path, _story(tmp_path, "S1"), _story(tmp_path, "S2"))
        )
        assert summary.stop_cause == "rate_limited"
        by_id = {s.id: s for s in summary.stories}
        assert by_id["S1"].state == StoryState.PENDING
        assert by_id["S2"].state == StoryState.PENDING

    async def test_planner_progress_forwarded_to_presenter(self, tmp_path):
        captured = {}

        def planner_factory(*a, **k):
            captured["observer"] = k.get("on_tool_call")
            captured["max_iterations"] = k.get("max_iterations")
            planner = MagicMock()
            planner.plan = AsyncMock(
                return_value=PlanResult(plan=_plan(tmp_path, _story(tmp_path, "S1")))
            )
            return planner

        class ProgressPresenter(RecordingPresenter):
            async def on_planner_tool_call(self, tool_call, result):
                self.events.append(("planner_tool", tool_call.name))

        presenter = ProgressPresenter()
        run = _run(tmp_path, _config(), presenter, FakeWorker(), FakeJudge())
        run._planner_factory = planner_factory
        await run.run("goal")
        observer = captured["observer"]
        assert observer is not None
        assert captured["max_iterations"] is None
        await observer(MagicMock(name="Read"), MagicMock(error=None))
        assert any(e[0] == "planner_tool" for e in presenter.events)

    async def test_wait_heartbeat_forwarded_with_labels(self, tmp_path):
        captured = {}

        def judge_factory(*a, **k):
            captured["judge_wait"] = k.get("on_wait")
            return FakeJudge()

        class WaitingPresenter(RecordingPresenter):
            async def on_wait(self, label, seconds):
                self.events.append(("wait", label, seconds))

        presenter = WaitingPresenter()
        worker = FakeWorker()
        run = _run(tmp_path, _config(), presenter, worker, FakeJudge())
        run._judge_factory = judge_factory
        await run.run_plan(_plan(tmp_path, _story(tmp_path, "S1")))
        await captured["judge_wait"](12.0)
        assert ("wait", "judge claude:opus", 12.0) in presenter.events

    async def test_plan_failure_reported(self, tmp_path):
        planner = MagicMock()
        planner.plan = AsyncMock(
            return_value=PlanResult(error="plan_failed: no submit")
        )
        run = _run(
            tmp_path,
            _config(),
            RecordingPresenter(),
            FakeWorker(),
            FakeJudge(),
            planner=planner,
        )
        summary = await run.run("goal")
        assert summary.stop_cause == "plan_failed"
        assert summary.stories == []

    async def test_plan_declined_by_presenter(self, tmp_path):
        planner = MagicMock()
        planner.plan = AsyncMock(
            return_value=PlanResult(plan=_plan(tmp_path, _story(tmp_path, "S1")))
        )
        presenter = RecordingPresenter(decision=PlanDecision(accepted=False))
        worker = FakeWorker()
        run = _run(tmp_path, _config(), presenter, worker, FakeJudge(), planner=planner)
        summary = await run.run("goal")
        assert summary.stop_cause == "cancelled"
        assert worker.calls == []

    async def test_presenter_edited_plan_is_used(self, tmp_path):
        planner = MagicMock()
        planner.plan = AsyncMock(
            return_value=PlanResult(
                plan=_plan(tmp_path, _story(tmp_path, "S1"), _story(tmp_path, "S2"))
            )
        )
        edited = _plan(tmp_path, _story(tmp_path, "S2"))
        presenter = RecordingPresenter(
            decision=PlanDecision(accepted=True, plan=edited)
        )
        worker = FakeWorker()
        run = _run(tmp_path, _config(), presenter, worker, FakeJudge(), planner=planner)
        summary = await run.run("goal")
        assert [c[0] for c in worker.calls] == ["S2"]
        assert summary.passed == 1

    async def test_plan_approval_can_be_skipped(self, tmp_path):
        planner = MagicMock()
        planner.plan = AsyncMock(
            return_value=PlanResult(plan=_plan(tmp_path, _story(tmp_path, "S1")))
        )
        presenter = RecordingPresenter(decision=PlanDecision(accepted=False))
        run = _run(
            tmp_path,
            _config(plan_approval=False),
            presenter,
            FakeWorker(),
            FakeJudge(),
            planner=planner,
        )
        summary = await run.run("goal")
        assert summary.passed == 1
        assert not any(e[0] == "approve_plan" for e in presenter.events)


class TestConcurrency:
    async def test_independent_stories_overlap_under_max_parallel_two(self, tmp_path):
        worker = FakeWorker(delay=0.05)
        run = _run(
            tmp_path, _config(max_parallel=2), RecordingPresenter(), worker, FakeJudge()
        )
        await run.run_plan(
            _plan(tmp_path, _story(tmp_path, "S1"), _story(tmp_path, "S2"))
        )
        assert worker.max_active == 2

    async def test_serial_under_max_parallel_one(self, tmp_path):
        worker = FakeWorker(delay=0.02)
        run = _run(
            tmp_path, _config(max_parallel=1), RecordingPresenter(), worker, FakeJudge()
        )
        await run.run_plan(
            _plan(tmp_path, _story(tmp_path, "S1"), _story(tmp_path, "S2"))
        )
        assert worker.max_active == 1

    async def test_approval_callback_serialized_and_prefixed(self, tmp_path):
        engine = MagicMock()
        agent_engine = engine.agent_engine
        seen = []
        active = {"n": 0, "max": 0}

        async def original(operation):
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
            await asyncio.sleep(0.02)
            seen.append(operation.description)
            active["n"] -= 1
            return True

        agent_engine.approval.approval_callback = original

        class ApprovingWorker(FakeWorker):
            async def run(self, story, ref, provider, **kw):
                op = Operation(
                    type=OperationType.FILE_WRITE,
                    description=f"Write {story.id}",
                    data={},
                )
                await agent_engine.approval.approval_callback(op)
                return await super().run(story, ref, provider, **kw)

        run = H2LRun(
            engine,
            _config(max_parallel=2),
            RecordingPresenter(),
            cwd=tmp_path,
            session_id="s",
            tiers=_tiers(),
        )
        run._worker_factory = lambda *a, **k: ApprovingWorker()
        run._judge_factory = lambda *a, **k: FakeJudge()
        await run.run_plan(
            _plan(tmp_path, _story(tmp_path, "S1"), _story(tmp_path, "S2"))
        )
        assert active["max"] == 1
        assert any(d.startswith("[H2L S1 · gateway:a] Write S1") for d in seen)
        # Restored afterwards.
        assert agent_engine.approval.approval_callback is original

    async def test_parent_provider_untouched(self, tmp_path):
        engine = MagicMock()
        engine.current_provider.model = "session"
        run = H2LRun(
            engine,
            _config(),
            RecordingPresenter(),
            cwd=tmp_path,
            session_id="s",
            tiers=_tiers(),
        )
        run._worker_factory = lambda *a, **k: FakeWorker()
        run._judge_factory = lambda *a, **k: FakeJudge()
        await run.run_plan(_plan(tmp_path, _story(tmp_path, "S1")))
        assert engine.current_provider.model == "session"
        engine.send_message_with_tools.assert_not_called()


class TestEvents:
    async def test_run_scoped_identity(self, tmp_path):
        ids = []

        async def _spy(event_type, data, *a, **k):
            ids.append(
                (
                    fleet_events.current_agent_id(),
                    fleet_events.current_parent_id(),
                    data,
                )
            )

        class SpyWorker(FakeWorker):
            async def run(self, story, ref, provider, **kw):
                await fleet_events.emit_event(
                    None, {"where": "worker", "run": kw["run_id"]}
                )
                return await super().run(story, ref, provider, **kw)

        run = _run(tmp_path, _config(), RecordingPresenter(), SpyWorker(), FakeJudge())
        from unittest.mock import patch

        with patch.object(fleet_events, "emit_event", _spy):
            await run.run_plan(_plan(tmp_path, _story(tmp_path, "S1")))
        worker_events = [e for e in ids if e[2].get("where") == "worker"]
        assert worker_events and worker_events[0][2]["run"] == "h2l-sess"
