"""Allowlisted evaluation records and a standalone, script-free HTML report."""

import math
from collections import defaultdict
from html import escape
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Finite = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Slug = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,63}$")]
Label = Literal["fast", "deep"]
ATTEMPTED_STATUSES = frozenset(
    {
        "selected",
        "low_confidence",
        "timeout",
        "http_error",
        "invalid_response",
        "network_error",
    }
)


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CaseResult(PublicModel):
    case_id: Slug
    split: Literal["development", "test"]
    category: Literal["simple", "complex", "ambiguous", "adversarial"]
    prompt: Annotated[str, Field(max_length=4000)]
    expected: Label
    arm: Literal["baseline", "heuristic", "jev"]
    repeat: Annotated[int, Field(ge=0)]
    choice: Label | None = None
    target: Label
    status: Literal[
        "selected",
        "low_confidence",
        "missing_key",
        "invalid_state",
        "timeout",
        "http_error",
        "invalid_response",
        "network_error",
        "budget_exhausted",
        "baseline",
        "heuristic",
        "offline",
    ]
    confidence: Probability | None = None
    probabilities: dict[Label, Probability] = Field(default_factory=dict)
    elapsed_ms: Finite = 0
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    estimated_cost_usd: Finite | None = None
    model: Annotated[str, Field(pattern=r"^[a-zA-Z0-9._-]{1,100}$")] | None = None
    cold_connection: bool = False


class TaskResult(PublicModel):
    case_id: Slug
    arm: Literal["baseline", "jev"]
    expected: Label
    target: Label | None
    success: bool
    elapsed_ms: Finite
    worker_ms: Finite | None
    routing_ms: Finite | None = None
    turns: Annotated[int, Field(ge=0)] | None = None
    tool_calls: Annotated[int, Field(ge=0)] | None = None
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    stop_cause: Literal[
        "done",
        "nudge_exhausted",
        "max_iterations",
        "repeat_abort",
        "timeout",
        "provider_error",
        "invalid_output",
    ]
    check: Literal["pass", "fail", "timeout", "error"]
    routing_status: Annotated[str, Field(pattern=r"^[a-z_]{1,40}$")] = "baseline"


class Manifest(PublicModel):
    date: Annotated[str, Field(pattern=r"^[0-9TZ:.+\-]+$")] = "2026-09-19"
    source_commit: Annotated[str, Field(pattern=r"^[a-f0-9]{7,40}$")] | None = None
    source_dirty: bool | None = None
    implementation_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")] | None = (
        None
    )
    fixture_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")] | None = None
    policy_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")] | None = None
    task_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")] | None = None
    task_policy_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")] | None = None
    seed: int = 19
    repetitions: Annotated[int, Field(ge=1, le=20)] = 3
    planned_jev_decisions: Annotated[int, Field(ge=0)] = 0
    planned_task_runs: Annotated[int, Field(ge=0)] = 0
    live: bool = False
    model: Literal["jev-1.13.0"] = "jev-1.13.0"
    confidence_threshold: Probability = 0.8
    python_version: Annotated[str, Field(pattern=r"^[0-9.]+$")] | None = None
    httpx_version: Annotated[str, Field(pattern=r"^[0-9.]+$")] | None = None
    fast_worker: (
        Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")] | None
    ) = None
    deep_worker: (
        Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")] | None
    ) = None
    targeted_tests_passed: Annotated[int, Field(ge=0)] | None = None
    required_checks: Literal["pending", "passed", "baseline_failures"] = "pending"


class BudgetSummary(PublicModel):
    cap_usd: Finite
    reserved_usd: Finite
    estimated_usage_usd: Finite
    attempts: Annotated[int, Field(ge=0)]
    unsettled_attempts: Annotated[int, Field(ge=0)]


class EvaluationReport(PublicModel):
    manifest: Manifest = Field(default_factory=Manifest)
    budget: BudgetSummary | None = None
    records: list[CaseResult] = Field(default_factory=list)
    tasks: list[TaskResult] = Field(default_factory=list)


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, explicitly retaining timeout observations."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarize(records: list, split: str = "test") -> dict:
    groups = defaultdict(list)
    for record in records:
        row = record.model_dump() if isinstance(record, CaseResult) else record
        if row["split"] == split:
            groups[row["arm"]].append(row)
    output = {}
    for arm, rows in groups.items():
        count = len(rows)
        selected = [r for r in rows if r["status"] == "selected"]
        measured = [
            r for r in rows if arm != "jev" or r["status"] in ATTEMPTED_STATUSES
        ]
        warm = [r["elapsed_ms"] for r in measured if not r.get("cold_connection")]
        output[arm] = {
            "count": count,
            "latency_count": len(measured),
            "not_attempted": count - len(measured),
            "raw_accuracy": sum(r["choice"] == r["expected"] for r in rows) / count,
            "routed_accuracy": sum(r["target"] == r["expected"] for r in rows) / count,
            "coverage": len(selected) / count,
            "errors": sum(
                r["status"]
                not in ("selected", "low_confidence", "baseline", "heuristic")
                for r in rows
            ),
            "under_routes": sum(
                r["target"] == "fast" and r["expected"] == "deep" for r in rows
            ),
            "over_routes": sum(
                r["target"] == "deep" and r["expected"] == "fast" for r in rows
            ),
            "p50_ms": percentile([r["elapsed_ms"] for r in measured], 0.5),
            "p95_ms": percentile([r["elapsed_ms"] for r in measured], 0.95),
            "warm_p50_ms": percentile(warm, 0.5),
            "warm_p95_ms": percentile(warm, 0.95),
        }
    return output


STYLE = """
:root{color-scheme:light;
--paper:#f6f8ff;
--ink:#1b2350;
--muted:#596386;

--fast:#3155cc;
--deep:#6b429a;
--bad:#b2384b;
--good:#126b62;
--line:#dce2f4}
*{box-sizing:border-box}body{margin:0;
background:var(--paper);
color:var(--ink);

font:16px/1.6 ui-sans-serif,system-ui,sans-serif}main{max-width:1180px;
margin:auto;

padding:52px 28px}header{border-left:8px solid var(--fast);
padding:10px 28px 24px;

background:white}h1{font:500 clamp(36px,6vw,64px)/1.1 Georgia,serif;
max-width:850px;

margin:12px 0 20px}h2{font-size:24px;
margin-top:38px}.eyebrow{font:12px monospace;

letter-spacing:.13em;
text-transform:uppercase;
color:var(--muted)}p{max-width:940px}
.lead{font-size:19px}.muted,small{color:var(--muted)}.route{display:flex;
gap:2px;

align-items:stretch;
margin:26px 0;
flex-wrap:wrap}.route>div{padding:18px 22px;

background:#e6ecff;
flex:1;
min-width:160px}.route .fast{border-top:4px solid var(--fast)}
.route .deep{border-top:4px solid var(--deep)}.table-wrap{overflow:auto;
background:white;

border:1px solid var(--line);
border-radius:8px}table{border-collapse:collapse;
width:100%;

font-variant-numeric:tabular-nums}th,td{text-align:left;
vertical-align:top;
padding:12px 15px;

border-bottom:1px solid var(--line)}th{font-size:12px;
text-transform:uppercase;

letter-spacing:.035em;
background:#eef1fa;
white-space:nowrap}td{font-size:14px}
.bad{color:var(--bad);
font-weight:600}.good{color:var(--good);
font-weight:600}
.badge{display:inline-block;
font:12px monospace;
padding:5px 9px;
border:1px solid var(--line);

border-radius:5px;
background:white}.notice{background:#ece9f7;
border-left:4px solid var(--deep);

padding:16px 22px;
margin:22px 0}.case{min-width:320px;
max-width:540px}.num{white-space:nowrap}
details{background:white;
border:1px solid var(--line);
margin:16px 0;
padding:16px;
border-radius:8px}
summary{cursor:pointer;
font-weight:650}code{font:12px ui-monospace,monospace;
overflow-wrap:anywhere}
a{color:var(--fast)}a:focus,summary:focus{outline:3px solid var(--deep);
outline-offset:4px}
footer{margin-top:45px;
color:var(--muted);
font-size:13px} @media(max-width:650px){
main{padding:24px 14px}
header{padding:8px 15px}th,td{padding:10px}.case{min-width:260px}}
@media print{main{padding:0}details{break-inside:avoid}.table-wrap{overflow:visible}}
"""


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    return (
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _known_total(values: list[int | None]) -> str:
    total = sum(value for value in values if value is not None)
    missing = sum(value is None for value in values)
    return f"{total} known; {missing} unknown" if missing else str(total)


def render_report(report: EvaluationReport) -> str:
    """Escape dynamic text; accept only the public schema, never raw logs."""
    report = EvaluationReport.model_validate(report.model_dump())
    m = report.manifest
    summaries = summarize(report.records)
    mode = "Live synthetic pilot" if m.live else "Offline preview · no Jev measurement"
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src
'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Omnimancer · TypeSafe routing
evaluation</title><style>{STYLE}</style></head><body><main>
<header><div class="eyebrow">Omnimancer / routing experiment / {escape(m.date)}</div>
<h1>Choosing the worker</h1><span class="badge">{mode}</span>
<p class="lead">Can a short classification step reduce the time to a correctly
completed task?
This report separates routing-policy agreement from actual task outcomes.</p></header>
<div class="route"><div>Task<br><strong>One typed decision</strong></div>
<div class="fast">Confident, straightforward<br><strong>Fast worker</strong></div>
<div class="deep">Complex, uncertain or failed<br><strong>Configured
baseline</strong></div></div>"""]
    parts.append(
        "<h2>Held-out routing decisions</h2><p>Labels were authored before inference "
        "and encode the stated routing policy. They do not establish the minimum "
        "model capable of each task. Baseline keeps the configured deep worker; "
        "the heuristic is an added keyword comparator, not existing "
        "Omnimancer behavior.</p>"
    )
    metric_rows = []
    for arm in ("baseline", "heuristic", "jev"):
        if arm not in summaries:
            continue
        s = summaries[arm]
        metric_rows.append(
            [
                arm,
                str(s["count"]),
                f'{s["raw_accuracy"]:.1%}' if s["latency_count"] else "Not measured",
                f'{s["routed_accuracy"]:.1%}',
                str(s["under_routes"]),
                str(s["over_routes"]),
                str(s["errors"]),
                (
                    f'{s["p50_ms"]:.1f} / {s["p95_ms"]:.1f} ms'
                    if s["latency_count"]
                    else "Not measured"
                ),
            ]
        )
    parts.append(
        _table(
            [
                "Arm",
                "Predictions",
                "Raw agreement",
                "After fallback",
                "Under-route",
                "Over-route",
                "Errors / skips",
                "p50 / p95",
            ],
            metric_rows,
        )
    )
    jev = summaries.get("jev")
    if jev:
        parts.append(
            "<p>Jev selected a target above the confidence threshold on "
            f'<strong>{jev["coverage"]:.1%}</strong> of held-out predictions. '
            "Errors remain in denominators; low confidence and errors retain "
            "the baseline. Predictions repeat cases, so they are not "
            "independent samples.</p>"
        )
        parts.append(
            f'<p>Latency sample: {jev["latency_count"]} attempted calls; '
            f'{jev["not_attempted"]} skipped decisions are excluded from timing '
            "but retained in agreement denominators.</p>"
        )
    cold = [
        r.elapsed_ms
        for r in report.records
        if r.arm == "jev" and r.cold_connection and r.status in ATTEMPTED_STATUSES
    ]
    warm = [
        r.elapsed_ms
        for r in report.records
        if r.arm == "jev" and not r.cold_connection and r.status in ATTEMPTED_STATUSES
    ]
    if cold:
        parts.append(
            f"<p>First connection: {cold[0]:.1f} ms. Remaining calls: {len(warm)}; "
            f"warm p50 / p95: {percentile(warm, .5) or 0:.1f} / "
            f"{percentile(warm, .95) or 0:.1f} ms. Timing includes transport "
            "and validation, not just server inference.</p>"
        )
    parts.append("<h2>Completed-task comparison</h2>")
    if not report.tasks:
        parts.append(
            '<div class="notice">No executable task results yet. Classification '
            "speed and label agreement cannot establish faster or better "
            "task completion.</div>"
        )
    else:
        task_rows = []
        for arm in ("baseline", "jev"):
            tasks = [t for t in report.tasks if t.arm == arm]
            if tasks:
                times = [t.elapsed_ms for t in tasks]
                task_rows.append(
                    [
                        arm,
                        f"{sum(t.success for t in tasks)} / {len(tasks)}",
                        f"{(percentile(times, .5) or 0)/1000:.2f} s",
                        f"{(percentile(times, .95) or 0)/1000:.2f} s",
                        _known_total([t.turns for t in tasks]),
                        _known_total([t.tool_calls for t in tasks]),
                    ]
                )
        parts.append(
            _table(
                [
                    "Arm",
                    "Acceptance passed",
                    "p50 total",
                    "p95 total",
                    "Model calls",
                    "Tool calls",
                ],
                task_rows,
            )
        )
        parts.append(
            "<p>Paired synthetic repairs use disposable workspaces and the same "
            "hidden acceptance checks. Some task concepts overlap the development "
            "set; this is an illustrative suite, not an independent generalization "
            "test. Full elapsed time includes classifier, "
            "process startup, provider initialization and model loading; worker "
            "time is measured separately. Both workers share one local inference "
            "service, so model-load effects matter. Failed and timed-out tasks "
            "remain in the sample. Missing worker metadata is unknown, not zero; "
            "incomplete call totals explicitly identify missing runs. A passing "
            "final repair can coexist with a timeout stop.</p>"
        )
        parts.append(
            _table(
                ["Task", "Arm / target", "Acceptance", "Total / worker", "Stop"],
                [
                    [
                        escape(t.case_id),
                        f"{t.arm} / {t.target or 'unknown'}",
                        f'<span class="{"good" if t.success else "bad"}">'
                        f"{t.check}</span>",
                        f"{t.elapsed_ms/1000:.2f} / "
                        + (
                            f"{t.worker_ms/1000:.2f} s"
                            if t.worker_ms is not None
                            else "unknown s"
                        ),
                        escape(t.stop_cause),
                    ]
                    for t in report.tasks
                ],
            )
        )
    parts.append("<h2>Disagreements and uncertainty</h2>")
    failures = [
        r
        for r in report.records
        if r.arm == "jev"
        and r.split == "test"
        and (r.choice != r.expected or r.target != r.expected or r.status != "selected")
    ]

    def rows_for(records: list[CaseResult]) -> list[list[str]]:
        return [
            [
                f'<strong>{escape(r.case_id)}</strong><br><span class="muted">'
                f"{r.category} · repeat {r.repeat+1}</span>",
                f'<div class="case">{escape(r.prompt)}</div>',
                r.expected,
                f'{r.choice or "—"} → {r.target}',
                f"{r.confidence:.3f}" if r.confidence is not None else "—",
                escape(r.status),
                f"{r.elapsed_ms:.1f} ms",
            ]
            for r in records
        ]

    heads = [
        "Case",
        "Synthetic task",
        "Expected",
        "Raw → applied",
        "Confidence",
        "Status",
        "Time",
    ]
    parts.append(
        _table(heads, rows_for(failures))
        if failures
        else "<p>No held-out Jev disagreements or uncertainty records in this run.</p>"
    )
    parts.append(
        "<details><summary>Every synthetic classification, including "
        "development cases</summary>"
    )
    parts.append(_table(heads, rows_for([r for r in report.records if r.arm == "jev"])))
    parts.append("</details><h2>Cost and reproducibility</h2>")
    if report.budget:
        b = report.budget
        parts.append(
            "<p>Known TypeSafe usage estimate: "
            f"<strong>${b.estimated_usage_usd:.6f}</strong>. Conservative "
            f"reservations: ${b.reserved_usd:.3f} of ${b.cap_usd:.2f}; "
            f"{b.attempts} attempts, {b.unsettled_attempts} without settled usage. "
            "Reservations are committed before calls and never refunded, "
            "including errors. Local worker API charge: $0; electricity "
            "and subscription development work are excluded.</p>"
        )
    parts.append(
        "<p>Pricing assumption: Jev 1.13.0 at $0.042 per million input tokens; "
        "output free, verified 2026-09-19. These are usage estimates, not an "
        "invoice. Model confidence measures the returned distribution and "
        "does not guarantee correctness.</p>"
    )
    parts.append(
        _table(
            ["Manifest field", "Value"],
            [
                [escape(k), f"<code>{escape(str(v))}</code>"]
                for k, v in m.model_dump().items()
                if v is not None
            ],
        )
    )
    parts.append("""<div class="notice"><strong>Limits.</strong>
Small synthetic dataset, authored
labels, fixed policy and one local hardware setup. No unattended permission decisions
are enabled. A default-disabled experiment can be useful even when it shows no
improvement; this report makes no general speed or safety guarantee.</div>
<footer>Sources: <a
href="https://x.com/sydneyrunkle/status/2100754364545761643">Building a Harness with
Jev</a> ·
<a href="https://docs.typesafe.ai/models">TypeSafe models and pricing</a> ·
<a href="https://docs.typesafe.ai/confidence">Confidence</a> ·
<a href="https://docs.typesafe.ai/model-jaggedness/jev-1.13">Known limitations</a>.
Standalone artifact: no scripts, remote assets, telemetry, raw logs, credentials or
private endpoint names.</footer>
</main></body></html>""")
    return "\n".join(parts)
