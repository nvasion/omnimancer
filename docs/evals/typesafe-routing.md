# Local TypeSafe routing pilot — 2026-09-19

**The tested worker pair did not improve speed or repair acceptance.** Jev
classified the authored policy quickly, but routing some tasks to the worker
named `fast` made this local run substantially slower and introduced failures.
Keep the feature opt-in. A smaller model is not necessarily a faster worker.

Open the [standalone HTML](typesafe-routing.html) in a browser to inspect the
decisions, uncertainties, individual task outcomes, cost, and provenance.
The [public JSON](typesafe-routing.json) contains the allowlisted measurements.
See [usage and reproduction commands](../typesafe-routing.md).

## Executable repairs

Both arms ran the same 12 synthetic Python repairs once, with hidden acceptance
checks, randomized paired order (seed 19), eight model turns maximum, and a
120-second worker deadline. Baseline always used `qwen3.8:27b`; routing could
select `gemma4:12b` as `fast` or retain Qwen as `deep`. Both used the same local
inference service. The names express the hypothesis being tested.

| Measurement | Baseline | TypeSafe routing |
|---|---:|---:|
| Final repair passes acceptance | 11 / 12 | 9 / 12 |
| Passing repair and normal completion | 11 / 12 | 7 / 12 |
| Total elapsed p50 | 6.55 s | 9.69 s |
| Total elapsed p95 | 14.62 s | 120.19 s |
| Sum of elapsed times | 91.20 s | 604.71 s |
| Worker timeouts | 0 | 3 |

Percentiles use nearest rank and include failures and timeouts. Total time
includes startup, routing, initialization/model loading, execution, and
acceptance. Two timed-out routed workers had written passing repairs; these
count toward final-file acceptance, while their timeout stop remains visible.

Both arms failed `recursive-merge-immutable`. Routing additionally failed
`extension-filtering` and `safe-divide`. The two runs with confirmed Gemma
selection took 98.75 s (failed) and 79.84 s (passed), versus baseline 7.45 s
and 9.63 s. The three timed-out runs have unknown applied targets because no
valid final metadata arrived; do not infer their targets or usage from the
classification-only suite. Seven confirmed routed runs retained the deep
worker, five of them through low-confidence fallback.

This is a small illustrative suite, with one observation per arm/task. Some
task concepts overlap development cases. Shared model-loading effects,
stochastic worker behavior, and provider-internal retries can affect time.
The result does not isolate classifier overhead from worker/model behavior
and does not establish performance on real projects.

## Routing-policy agreement

The fixed policy and confidence threshold (0.8) were measured on 20 development
and 60 held-out synthetic cases, balanced between `fast` and `deep`, with three
repetitions. Expected labels describe policy preferences, not independently
verified minimum worker capability. Repetitions are not independent samples.

| Held-out measurement (180 predictions per arm) | Baseline | Keyword heuristic | Jev |
|---|---:|---:|---:|
| Raw agreement | 50.0% | 63.3% | 100.0% |
| Agreement after confidence fallback | 50.0% | 63.3% | 93.3% |
| Under-routes (`deep` label sent to `fast`) | 0 | 66 | 0 |
| Over-routes (`fast` label retained on `deep`) | 90 | 0 | 12 |

Jev's held-out selection coverage was **85.6%**, with no API/validation errors.
Decision p50/p95 was **169.3 / 247.0 ms**, including transport and validation.
The keyword heuristic is an experimental comparator, not prior Omnimancer
behavior. High label agreement did not predict better executable outcomes.

## Cost and provenance

There were 253 reserved TypeSafe attempts: one connection smoke test, 240
classification calls, and 12 task-routing attempts. Known token-based usage
was **$0.004848606**. Three timed-out workers did not report usage, so this is
an incomplete estimate, not the invoice. The shared ledger conservatively
reserved **$0.759 of the $5 cap**, including those attempts. Reservations were
never refunded. Local worker API charges were zero; electricity and development
subscriptions are excluded. The pricing assumption is documented in the guide.

Measurements ran against clean source commit
`f59f6c293a35681ac5a1705c36aa77f4f547218b`. The JSON manifest records its package,
fixture, policy, and task hashes. A subsequent reporting correction makes
missing metadata nullable: for `optional-value-parse`, `safe-divide`, and
`slug-normalize`, unobserved route, timing split, turns, tool calls, and token
counts were normalized to `null` for publication. Measured total times,
acceptance outcomes, stop causes, and classification records were preserved.
The original measurement commit remains the provenance anchor. Current code
emits these nulls directly; an old report cannot be extended with a different
implementation hash. Start a fresh classification report to rerun current code.

Observed worker metadata from the local model service:

| Worker | Parameters | Quantization | Weight digest |
|---|---|---|---|
| `qwen3.8:27b` | 27.3B | Q4_K_M | `22130167c4c20e20c7b71454612966ca8e8171e9b3cc8ab6ce8aa6cbfec79643` |
| `gemma4:12b` | 11.9B | Q4_K_M | `4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c` |

Python was 3.14.7 and httpx 0.28.1. The source was validated with unit and
integration tests, typing, formatting/lint checks, and package builds. The
published report contains synthetic tasks and aggregate metadata, excludes
credentials and private endpoints, and uses script-free HTML with no remote
assets.

The next experiment should measure candidate workers on separate development
repairs first, then freeze the worker pair and policy before a new held-out
comparison. This pilot does not justify enabling the tested pair by default.
