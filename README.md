# vision-input-check

[![PyPI](https://img.shields.io/pypi/v/vision-input-check)](https://pypi.org/project/vision-input-check/)
[![Python](https://img.shields.io/pypi/pyversions/vision-input-check)](https://pypi.org/project/vision-input-check/)
[![CI](https://github.com/Gjusev/vision-input-check/actions/workflows/test.yml/badge.svg)](https://github.com/Gjusev/vision-input-check/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

Regression tests for **images sent to vision LLM APIs**. Run a labeled image
suite through controlled encoding variants, separate pixel-identical
deliveries (answers must not move) from lossy deliveries (measure by how
much they move), repeat to expose variance, and turn the result into CI
gates.

> **News (2026-09-25).** OpenAI reported an image-encoding bug that degraded
> [GPT-6 Sol/Luna](https://developers.openai.com/api/docs/changelog) and
> recommended re-running image evaluations. This package exists for exactly
> that workflow: re-run a fixed suite across delivery variants and compare
> against known-good answers. Nothing here assumes the bug is still present
> or that other stacks are immune — run the suite and look at your numbers.

## How it works

1. **Variants.** Each fixture image is re-encoded into two classes:
   - *identical* — `original`, `png-reencode`, `exif-strip`: pixels are
     verified unchanged (`sizes + band set + per-pixel diff`) before the
     variant is used; the variant name alone is never trusted as proof.
   - *lossy* — `jpeg-75`, `jpeg-40`, `resize-50`, `grayscale`, `webp`:
     information is destroyed on purpose; answers may legitimately change.
2. **Runs.** Each fixture/variant pair is asked `--runs` times (default 3).
   A provider answers; an exact assertion judges each raw answer. No LLM
   judge.
3. **Metrics** (descriptive; N=3 is not statistical significance):
   - cell accuracy = correct answers / N
   - variant aggregate = mean cell accuracy over eligible fixtures
   - delta = variant aggregate − original aggregate over the same fixtures
   - a fixture whose `original` mixes pass/fail is **flaky** and is
     excluded from strict metrics; denominators are published.
4. **Gates** compare the strict aggregates to your thresholds.

### Metrics the gates watch

| metric | formula | demo value | meaning |
|---|---|---|---|
| `identical_delta_min` | min over `png-reencode`, `exif-strip` of (aggregate delta vs original) | −1/6 | worst aggregate accuracy *change* caused by a pixel-identical delivery |
| `lossy_accuracy_min` | min over lossy variants of aggregate accuracy | 5/6 | worst aggregate accuracy under degraded delivery |
| `flaky_count` | fixtures whose `original` mixes pass/fail | 0 | baseline instability; excluded from strict metrics |

These are aggregate-level minima, not the worst single cell: one noisy cell
does not fail the identical gate by itself.

## Quick start (no credentials)

```bash
pip install vision-input-check
vision-check demo                       # exit 1: seeded failures prove the harness
vision-check demo --max-identical-delta 0.17 --min-lossy-accuracy 0.8   # exit 0
vision-check demo --json                # pure JSON on stdout, errors on stderr
```

The demo renders six synthetic invoices with Pillow and replays canned
answers (marked *synthetic* everywhere). Two failures are seeded on
purpose — `inv-003/png-reencode` and `inv-005/jpeg-40` — so the strict
gates fail and the relaxed gates pass. **The seeded failures demonstrate
the harness; they say nothing about any real provider.**

## Running your own suite

```bash
# offline, against recorded answers
vision-check run suite.jsonl --provider replay --responses responses.json

# live, against any OpenAI-compatible endpoint (extra: pip install 'vision-input-check[live]')
export VISION_API_KEY=...
export VISION_BASE_URL=https://api.openai.com/v1   # /v1 suffix is preserved
export VISION_MODEL=gpt-6.1-sol                    # optional
vision-check run suite.jsonl --provider live --model gpt-6.1-sol \
  --max-identical-delta 0.0 --min-lossy-accuracy 0.95 --max-flaky 0 \
  --output result.json --html report.html
```

Exit codes: `0` gates satisfied · `1` gates failed (artifacts written) ·
`2` configuration, data, or transport error. Transport failures are never
turned into wrong answers.

### CLI flags

| flag | default | meaning |
|---|---|---|
| `--runs N` | 3 | repetitions per fixture/variant; must be ≥ 1 |
| `--max-identical-delta X` | 0.0 | positive *magnitude* of aggregate degradation allowed on identical variants; gate passes when `identical_delta_min >= -X` |
| `--min-lossy-accuracy X` | disabled | gate passes when the worst lossy aggregate ≥ X |
| `--max-flaky N` | disabled | gate passes when `flaky_count <= N`; applies even when flaky fixtures are excluded from other metrics |
| `--json` | off | stdout carries only the result JSON; errors go to stderr |
| `--output FILE` | demo: `visioncheck-demo/result.json` | write result JSON |
| `--html FILE` | demo: `visioncheck-demo/report.html` | write standalone HTML report |

Gate semantics: gates check **answer correctness**, so they detect answer
regressions correlated with an input transformation. They do *not* detect
provider-side preprocessing that is applied to every input equally (it
moves `original` too, and deltas cancel), and they do not establish
causality — repeating runs reduces uncertainty, it does not eliminate
noise.

### Python API

```python
from vision_input_check import (
    ReplayProvider, build_demo_suite, compute_gates, evaluate_suite,
)

fixtures = build_demo_suite("visioncheck-demo")
result = evaluate_suite(fixtures, ReplayProvider(DEMO_RESPONSES), runs=3)
gates = compute_gates(result, max_identical_delta=0.17)
payload = result.to_dict(gates=gates)   # matches the bundled JSON Schema (schema_version "1.0")
print(result.summary())
```

Custom providers implement one method:

```python
class AnswerProvider(Protocol):
    def ask(self, fixture, variant, encoded: bytes, run_index: int) -> str: ...
```

The live adapter is a single OpenAI-compatible chat-completions client
(httpx, `[live]` extra): it sends the question and the image, never the
expected answer; retries are bounded and skip 4xx caller errors; error
messages never contain the API key or request headers. Close it with a
`with` block or `.close()`.

## Suite and replay formats

`suite.jsonl` (one fixture per line; `.json` arrays also work). Image paths
resolve relative to the suite file, so a suite travels with its images.

```json
{"id": "inv-001", "image": "images/inv-001.png",
 "question": "What is the invoice number printed on this invoice?",
 "assert": {"kind": "equals", "expected": "INV-1001"}}
```

Assertion kinds: `equals` (stripped exact match), `contains` (substring),
`regex` (`re.search`), `numeric` (first number in each side compared within
`tolerance`).

`numeric` accepts both separator conventions: `1,234.56` and `1.234,56`
both parse to 1234.56, and the rightmost separator wins in mixed input.
Documented ambiguities: a lone dot is always a decimal point (`1.234` →
1.234, never 1234); a lone comma with exactly three trailing digits is
thousands grouping (`1,234` → 1234) while one or two trailing digits are a
comma decimal (`12,50` → 12.5). If your answers are ambiguous under both
conventions, prefer `equals`/`regex` on a normalized string.

`responses.json` (replay): `{fixture_id: {variant: [answer, answer, ...]}}`.
Lists cycle deterministically by run index, so any `--runs` value works.
Replay never calls a service and never generates answers with a model;
invalid shapes fail before the first run (exit 2).

The `result.json` document is validated by a bundled JSON Schema
(`schema_version "1.0"`); evolution within a schema version is additive.

## Reproduce in Kaggle

`kaggle-kernel/offline/` holds a push-ready kernel that installs
`vision-input-check==0.1.0` from PyPI, runs the offline demo with relaxed
gates, prints a summary, and writes `result.json` + `report.html`. The
relaxed gates exist to let the seeded demo failure through — they are not a
production recommendation. The kernel requires the version to be published
on PyPI first.

```bash
export KAGGLE_API_TOKEN=...   # or use `kaggle` login; never commit the token
kaggle kernels push -p kaggle-kernel/offline
kaggle kernels status gjusev/vision-input-check-demo
kaggle kernels output gjusev/vision-input-check-demo -p ./kaggle-out
```

## How this compares

| tool | what it is best at | where vision-input-check differs |
|---|---|---|
| [promptfoo](https://github.com/promptfoo/promptfoo) | prompt/provider matrices with assertions, including image inputs and custom assertion scripts | evaluates *model behavior* across inputs; vision-input-check systematically varies the *delivery encoding of the same image* and gates on aggregate deltas across repetitions |
| [DeepEval](https://github.com/confident-ai/deepeval) | PyTest-style LLM metrics (G-Eval, multimodal metrics) with rich scoring | measures quality with graded metrics; vision-input-check uses exact assertions to make *identical-input variance* visible and gateable |
| [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | benchmark-style evaluation of vision-language models over large datasets | benchmarks capability once; vision-input-check is a small regression harness for *your* images in CI, with pixel-identity verification of what actually got sent |
| [BackstopJS](https://github.com/garris/BackstopJS) | visual regression of rendered web pages via CSS pixel diffs | compares pixels, no LLM involved; vision-input-check checks the *model's answers* stay stable when bytes change |
| [Argos](https://github.com/argos-ci/argos) | screenshot visual-regression hosting for UIs | same pixel-diff domain; no provider round-trip, no answer assertions |

In short: promptfoo/DeepEval/VLMEvalKit evaluate models; BackstopJS/Argos
diff pixels. vision-input-check sits in the gap: it verifies that
*pixel-identical deliveries behave identically* and quantifies how lossy
deliveries move answers, as a CI gate.

## Limitations

- The client cannot distinguish provider-side preprocessing from model
  behavior; if a provider re-encodes everything, identical-class deltas
  will not see it.
- Repeating runs reduces uncertainty; it does not eliminate noise and does
  not prove causality. Do not read significance into three repetitions.
- Exact assertions fit narrow extraction tasks, not creative or open-ended
  evaluation; there is no LLM judge by design.
- Synthetic replay proves the harness works, never the quality of a model.
- Lossy variants have no obligation to preserve answers: `lossy_accuracy_min`
  is a regression *baseline*, not a correctness contract.

## Development

```bash
make install   # uv sync (pytest + ruff via dependency group)
make test      # offline pytest (sockets blocked), live tests opt-in only
make lint      # ruff (E, F, I, UP, RUF), line length 100
make build     # uv build
make demo      # relaxed-gate demo, exit 0
```

CI (`.github/workflows/test.yml`) runs Python 3.10–3.13, Ruff, the offline
test suite, both demo gate modes, and a clean wheel install with a smoke
test. Publishing (`.github/workflows/publish.yml`) uses Trusted Publishing
on release; no tokens live in the repo.

## License

Apache-2.0 — see [LICENSE](LICENSE).
