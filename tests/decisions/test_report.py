import pytest

from omnimancer.decisions.report import EvaluationReport, render_report, summarize


def record(**overrides):
    data = dict(
        case_id="case-1",
        split="test",
        category="simple",
        prompt="Replace a label.",
        expected="fast",
        arm="jev",
        repeat=0,
        choice="fast",
        target="fast",
        status="selected",
        confidence=0.9,
        elapsed_ms=100,
        input_tokens=100,
        output_tokens=10,
        probabilities={"fast": 0.95, "deep": 0.05},
    )
    return data | overrides


def test_failed_decisions_remain_in_denominator():
    rows = [
        record(),
        record(
            case_id="case-2",
            status="timeout",
            choice=None,
            target="deep",
            confidence=None,
            probabilities={},
        ),
    ]
    summary = summarize(rows)["jev"]
    assert summary["raw_accuracy"] == 0.5
    assert summary["routed_accuracy"] == 0.5
    assert summary["errors"] == 1
    assert summary["coverage"] == 0.5


def test_render_escapes_untrusted_case_text_and_has_no_active_assets():
    report = EvaluationReport(records=[record(prompt="<script>alert(1)</script>")])
    html = render_report(report)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "Content-Security-Policy" in html
    assert "default-src 'none'" in html
    assert "<script" not in html
    assert "135 passed" not in html


def test_public_schema_rejects_private_fields():
    with pytest.raises(ValueError):
        EvaluationReport(records=[record(private_path="/private/example")])
    with pytest.raises(ValueError):
        EvaluationReport(records=[record()], api_key="synthetic-secret")


def test_development_split_excluded_from_test_metrics():
    rows = [record(), record(split="development", choice="deep", target="deep")]
    assert summarize(rows)["jev"]["raw_accuracy"] == 1
    assert summarize(rows)["jev"]["count"] == 1


def test_invalid_probability_and_duration_rejected():
    with pytest.raises(ValueError):
        EvaluationReport(records=[record(elapsed_ms=float("nan"))])
    with pytest.raises(ValueError):
        EvaluationReport(records=[record(probabilities={"fast": 2})])


def test_unattempted_calls_do_not_lower_latency_percentiles():
    rows = [record(elapsed_ms=300), record(status="timeout", elapsed_ms=2000)]
    rows += [
        record(status="budget_exhausted", elapsed_ms=0, choice=None, target="deep")
        for _ in range(8)
    ]
    summary = summarize(rows)["jev"]
    assert summary["count"] == 10
    assert summary["raw_accuracy"] == 0.2
    assert summary["p50_ms"] == 300
    assert summary["p95_ms"] == 2000
    assert summary["latency_count"] == 2
    assert summary["not_attempted"] == 8


def test_no_attempts_have_no_inference_latency():
    report = EvaluationReport(records=[record(status="offline", elapsed_ms=0)])
    summary = summarize(report.records)["jev"]
    assert summary["p50_ms"] is None
    assert summary["latency_count"] == 0
    assert "Not measured" in render_report(report)


def test_report_write_does_not_follow_predicted_temporary_symlink(tmp_path):
    from omnimancer.decisions.evaluation import save_report

    victim = tmp_path / "must-not-change"
    victim.write_text("preserve me")
    (tmp_path / "report.json.tmp").symlink_to(victim)
    save_report(EvaluationReport(), tmp_path / "report")
    assert victim.read_text() == "preserve me"


def test_missing_task_metadata_is_unknown_instead_of_zero_or_deep():
    from omnimancer.decisions.report import TaskResult

    task = TaskResult(
        case_id="timed-out",
        arm="jev",
        expected="fast",
        target=None,
        success=True,
        elapsed_ms=120000,
        worker_ms=None,
        stop_cause="timeout",
        check="pass",
        routing_status="invalid_output",
    )
    html = render_report(EvaluationReport(tasks=[task]))
    assert "jev / unknown" in html
    assert "120.00 / unknown" in html
    assert "0 known; 1 unknown" in html
    assert "1 / 1" in html
    assert task.model_dump()["turns"] is None
