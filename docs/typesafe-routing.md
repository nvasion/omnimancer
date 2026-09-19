# Experimental TypeSafe model routing

Omnimancer can ask TypeSafe's Jev classifier which configured worker should
handle a headless task. This is an opt-in experiment aimed at reducing time to
correct task completion. It makes one decision before the worker starts.

The idea comes from [Building a Harness with Jev](https://x.com/sydneyrunkle/status/2100754364545761643).
The article also proposes tool-risk classification. This implementation covers
model routing only: the classifier cannot approve tools or change permissions.
Vendor speed/cost comparisons are motivation, not measured Omnimancer results.
See the [TypeSafe introduction](https://docs.typesafe.ai/introduction).

## Run a routed task

1. Configure at least two supported worker targets in Omnimancer. This first
   version supports its OpenAI and OpenAI-compatible provider implementations.
   Azure deployments and other provider implementations are excluded.
2. Copy [the example policy](../examples/typesafe/routing.json) and replace its
   provider keys and model names with your configured targets. Models must be
   the provider's current configured model, or present in its initialized
   provider catalog or your custom model catalog. No discovery call is added.
3. Supply `TYPESAFE_API_KEY` through your secret manager or process environment.
   Never put the key in the policy or commit it.
4. Run:

```sh
omn -p "Normalize whitespace in the slug helper" \
  --routing-policy routes.json --output-format json
```

Enabling routing sends the **task text** and selection criteria to TypeSafe.
This includes text piped into the task. It does not send configuration,
provider names, endpoints, conversation history, or tool outputs. A task can
itself contain private data; choose inputs accordingly.

Existing `pre_send_message` hooks also run before the classifier call with
`provider="typesafe"` and the Jev model. A blocking hook veto prevents that
request. Normal worker hooks still run at their existing boundaries.

Omit `--routing-policy` to keep existing behavior. Explicit `--provider`,
`--model`, `--base-url`, and `--resume` bypass routing entirely. The flag is
headless-only. Routing never saves changes to your configuration.

The example uses two labels, `fast` and `deep`, with confidence threshold 0.8
and a two-second classifier deadline. It treats routine localized edits as
fast and algorithms, stateful invariants, security, concurrency, unknown
causes, and ambiguous requirements as deep. These are policy preferences,
not evidence that a particular model can or cannot solve a task.

Set `"mode": "shadow"` to record recommendations while retaining your current
worker. This still calls TypeSafe. In route mode, only a validated decision
above the threshold changes the worker. Missing keys, invalid policy/state,
unavailable targets, low confidence, timeout, HTTP errors, and invalid
responses retain the configured baseline. Fewer than two valid targets means
no API request. Failed model switches restore both providers and chat context.

The classifier uses one fixed HTTPS endpoint, no redirects or automatic
retries, bounded response size, and cancellation propagation. Existing worker
retries and error-driven fallback retain their usual behavior.

## Read the result

JSON and stream-JSON output add `routing` only when the feature is enabled:

```json
{
  "status": "routed",
  "target": "fast",
  "applied_target": "fast",
  "choice": "fast",
  "confidence": 1.0,
  "probabilities": {"fast": 1.0, "deep": 0.0},
  "model": "jev-1.13.0",
  "elapsed_ms": 350.0,
  "input_tokens": 360,
  "output_tokens": 31,
  "estimated_cost_usd": 0.00001512
}
```

This is an illustrative record. `choice` is the raw valid classification;
`target` is the recommendation after confidence filtering; `applied_target`
is populated only after a successful switch. For example, shadow mode can
recommend `fast` without applying it. Failure statuses are categories, never
raw exception text. Text output remains focused on the worker response.

Existing `usage` and `total_cost_usd` count worker usage only. Add
`routing.estimated_cost_usd` separately for a combined estimate. Pricing for
the pinned `jev-1.13.0` model was verified on 2026-09-19: **$0.042 per million
input tokens, output free**. Other classifier models have no cost estimate.
See [models and pricing](https://docs.typesafe.ai/models). Confidence describes
the returned distribution, not a correctness guarantee; see
[confidence](https://docs.typesafe.ai/confidence) and
[known limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

## Reproduce the classification evaluation

No new Python dependency is required. Install the project and run the offline
preview first; it creates JSON and standalone HTML without inference:

```sh
python -m omnimancer.decisions.evaluation \
  --cases examples/typesafe/cases.json \
  --policy examples/typesafe/routing.json \
  --output /tmp/typesafe-routing-preview
```

For real measurements, provide `TYPESAFE_API_KEY` in the environment:

```sh
eval_dir="$(mktemp -d)"  # create once; reuse for this entire experiment
python -m omnimancer.decisions.evaluation \
  --cases examples/typesafe/cases.json \
  --policy examples/typesafe/routing.json \
  --output "$eval_dir/typesafe-routing" \
  --live --budget-usd 5 --ledger "$eval_dir/budget.sqlite" \
  --repetitions 3 --seed 19
```

Use the **same ledger** for every invocation in an experiment, including task
replays. It reserves $0.003 before each Jev attempt, more than the maximum
charge under the verified 64K request bound. Reservations survive crashes
and are never refunded, including timeouts. Reopening cannot raise or reset
the cap. Known token usage is recorded separately; neither number is an
invoice. Removing or replacing the ledger starts a different budget.
The ledger bounds evaluation calls, not ordinary `omn --routing-policy` runs.
Live budget storage requires an owned private directory (0700) on POSIX and
an owned regular ledger file (0600). Symlinks and shared storage are rejected;
the owner's account must remain trusted. `mktemp -d` creates suitable storage.

The frozen fixtures contain 20 development and 60 held-out cases, balanced
between labels. Three repeats yield 240 classifier attempts. Only prompt text
is sent; IDs and expected labels never reach the classifier. The report compares
the configured deep baseline, a deterministic keyword heuristic, and Jev.
It includes raw agreement, agreement after fallback, coverage, under-routing,
over-routing, errors, and end-to-end decision latency. Errors stay in the
denominator. Repeated predictions are not independent samples. The policy and
confidence threshold must be fixed before measuring the held-out set.

## Reproduce the paired task comparison

The separate executable suite contains 12 synthetic Python repairs, with
predetermined acceptance checks. Some concepts overlap the development set;
this is an illustrative task suite, not an independent generalization test.
The replay command uses disposable configurations and workspaces, only keyless
loopback workers, a shared classifier budget, and per-task deadlines. It
requires Linux and `bwrap` to isolate acceptance execution. See `--help`:

```sh
python -m omnimancer.decisions.replay \
  --tasks examples/typesafe/tasks.json \
  --policy examples/typesafe/routing.json \
  --report "$eval_dir/typesafe-routing.json" --output "$eval_dir/typesafe-routing" \
  --endpoint http://127.0.0.1:11434/v1 \
  --live --budget-usd 5 --ledger "$eval_dir/budget.sqlite" \
  --seed 19 --timeout 120 --max-turns 8
```

Replay requires a live classification report with matching frozen-policy and
implementation hashes. Missing hashes and offline reports are rejected before
any call. The implementation hash covers every Python source file in the
loaded Omnimancer package, including response validators, schemas, integration,
CLI, and replay code. Git provenance comes from that package's own checkout,
with dirty state recorded; installed packages outside Git have unknown commit
provenance. Third-party/runtime versions are separate metadata.

Results are saved after each task so an interrupted run retains completed rows.
The default models are the example policy models; use matching worker tags in
the policy for another comparison.

Both arms use the same hidden checks. Baseline always uses the configured deep
worker; the routed arm uses the module above. Full task elapsed time includes
classifier latency, process startup, provider initialization, and model loading.
Worker process time (minus classification) and stop causes are retained separately. Acceptance success means the final file passes its checks, including when a worker times out after writing a passing repair. The stop cause remains visible. Existing provider-internal HTTP retries may occur within the process deadline; reported model turns are not a count of all HTTP attempts. A shared local
inference service can incur model-load delays; local worker API cost is zero,
but electricity and development subscriptions are not included.

## Publication and tests

Reports use an allowlisted schema and reject unknown fields. HTML escapes
dynamic text and has no scripts, remote assets, or telemetry. Raw logs,
exceptions, credentials, local config, and private endpoint names are excluded.
This does not magically anonymize arbitrary input: use the bundled synthetic
fixtures and inspect both the JSON and HTML before publication.

```sh
pytest tests/decisions -q
```

Tests cover transport/response validation, deadline/cancellation, fallback,
actual model selection and rollback, permission preservation, explicit
selection bypass, persistent budget limits, and report injection resistance.
The experiment remains disabled by default regardless of benchmark outcome.
