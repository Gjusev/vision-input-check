# vision-input-check

[![PyPI](https://img.shields.io/pypi/v/vision-input-check)](https://pypi.org/project/vision-input-check/)
[![Python](https://img.shields.io/pypi/pyversions/vision-input-check)](https://pypi.org/project/vision-input-check/)
[![CI](https://github.com/Gjusev/vision-input-check/actions/workflows/test.yml/badge.svg)](https://github.com/Gjusev/vision-input-check/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

<a href="https://github.com/Gjusev/vision-input-check/releases/download/v0.1.0/brag.mp4">
  <img src="https://github.com/Gjusev/vision-input-check/releases/download/v0.1.0/brag.gif" alt="20 second demo: a seeded png-reencode failure trips the identical gate, then relaxed gates pass" width="640">
</a>

Regression tests for images sent to vision LLM APIs. You hand it a labeled
image suite. It re-encodes every image several ways, asks your model the
same question about each encoding, and checks whether the answers survive.
Pixel-identical encodings should not move an answer. Lossy encodings have
no such obligation, so the tool measures how far they push the model and
turns all of it into CI gates.

> **News (2026-09-25).** OpenAI reported an image-encoding bug that degraded
> [GPT-6 Sol/Luna](https://developers.openai.com/api/docs/changelog) and
> recommended re-running image evaluations. That is this package's whole
> job. Nothing here assumes the bug is still present or that other stacks
> are immune. Run the suite and look at your own numbers.

## How it works

1. **Variants.** Each fixture image is re-encoded into two classes:
   identical (`original`, `png-reencode`, `exif-strip`) and lossy
   (`jpeg-75`, `jpeg-40`, `resize-50`, `grayscale`, `webp`). Before an
   identical variant is used, the tool verifies the pixels really did not
   change: dimensions, band set, per-pixel diff. The variant name alone is
   never trusted as proof.
2. **Runs.** Every fixture/variant pair is asked `--runs` times (default 3).
   A provider answers, an exact assertion judges each raw answer, and there
   is no LLM judge.
3. **Metrics.** Cell accuracy is correct answers over N. The variant
   aggregate is the mean over eligible fixtures. Delta is the variant
   aggregate minus the original aggregate over the same fixtures. A fixture
   whose `original` mixes pass and fail is flaky: it gets excluded from
   strict metrics and the denominators say so.
4. **Gates** compare the strict aggregates against your thresholds.

The demo value of `identical_delta_min` is -1/6 because one seeded failure
in six fixtures moves the aggregate by exactly one fixture. That number is
the tool demonstrating itself.

## Quick start, no credentials

```bash
pip install vision-input-check
vision-check demo                       # exit 1: seeded failures trip strict gates
vision-check demo --max-identical-delta 0.17 --min-lossy-accuracy 0.8   # exit 0
vision-check demo --json                # JSON on stdout, errors on stderr
```

The demo renders six synthetic invoices with Pillow and replays canned
answers, marked as synthetic everywhere. Two failures are seeded on
purpose (`inv-003/png-reencode` and `inv-005/jpeg-40`), so strict gates
fail and relaxed gates pass. The seeded failures prove the harness works.
They say nothing about any real provider.

## Run your own suite

```bash
# offline, against recorded answers
vision-check run suite.jsonl --provider replay --responses responses.json

# live, against any OpenAI-compatible endpoint (extra: pip install 'vision-input-check[live]')
export VISION_API_KEY=...
export VISION_BASE_URL=https://api.openai.com/v1   # the /v1 suffix is kept
export VISION_MODEL=gpt-6.1-sol                    # optional
vision-check run suite.jsonl --provider live --model gpt-6.1-sol \
  --max-identical-delta 0.0 --min-lossy-accuracy 0.95 --max-flaky 0 \
  --output result.json --html report.html
```

Exit codes: 0 gates satisfied, 1 gates failed (artifacts still written),
2 configuration, data, or transport error. A transport failure never
becomes a wrong answer.

### Flags

| flag | default | meaning |
|---|---|---|
| `--runs N` | 3 | repetitions per fixture/variant, must be at least 1 |
| `--max-identical-delta X` | 0.0 | positive magnitude of aggregate degradation allowed on identical variants; the gate passes when `identical_delta_min >= -X` |
| `--min-lossy-accuracy X` | disabled | passes when the worst lossy aggregate is at least X |
| `--max-flaky N` | disabled | passes when `flaky_count <= N`; applies even while flaky fixtures are excluded from other metrics |
| `--json` | off | stdout carries only the result JSON, errors go to stderr |
| `--output FILE` | demo writes `visioncheck-demo/result.json` | where to write the result |
| `--html FILE` | demo writes `visioncheck-demo/report.html` | standalone HTML report |

Gate semantics, stated plainly: the gates watch answer correctness, so
they catch answer regressions that correlate with an input transformation.
They cannot detect provider-side preprocessing applied to every input
equally, because that moves `original` too and deltas cancel. Repeating
runs narrows uncertainty; it does not eliminate noise and it does not
prove causality.

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
(httpx, the `[live]` extra). It sends the question and the image, never
the expected answer. Retries are bounded and skip 4xx caller errors.
Error messages never contain the API key or request headers. Close it
with a `with` block or `.close()`.

## Suite and replay formats

`suite.jsonl`, one fixture per line (a `.json` array also works). Image
paths resolve relative to the suite file, so a suite travels with its
images.

```json
{"id": "inv-001", "image": "images/inv-001.png",
 "question": "What is the invoice number printed on this invoice?",
 "assert": {"kind": "equals", "expected": "INV-1001"}}
```

Assertion kinds: `equals` (stripped exact match), `contains` (substring),
`regex` (`re.search`), `numeric` (first number on each side, compared
within `tolerance`).

`numeric` accepts both separator conventions: `1,234.56` and `1.234,56`
both parse to 1234.56, and in mixed input the rightmost separator wins.
Ambiguities, stated openly: a lone dot is always a decimal point (`1.234`
parses to 1.234, never 1234). A lone comma with exactly three trailing
digits is thousands grouping (`1,234` gives 1234), while one or two
trailing digits make it a comma decimal (`12,50` gives 12.5). If your
answers are ambiguous under both conventions, prefer `equals` or `regex`
on a normalized string.

`responses.json` (replay) maps `fixture_id` to `variant` to a list of
answers. Lists cycle deterministically by run index, so any `--runs`
value works. Replay never calls a service and never generates answers
with a model. Invalid shapes fail before the first run (exit 2).

`result.json` is validated by a bundled JSON Schema with
`schema_version "1.0"`. Evolution within a schema version is additive.

## Reproduce in Kaggle

The kernel already ran end to end on Kaggle: it installs
`vision-input-check==0.1.0` from PyPI, runs the offline demo with relaxed
gates, and leaves `result.json` plus `report.html` in the output.

Kernel: [gjusev/vision-input-check-demo](https://www.kaggle.com/code/gjusev/vision-input-check-demo)

To push your own copy:

```bash
export KAGGLE_API_TOKEN=...   # or a classic API key in ~/.kaggle/kaggle.json
kaggle kernels push -p kaggle-kernel/offline
kaggle kernels status gjusev/vision-input-check-demo
kaggle kernels output gjusev/vision-input-check-demo -p ./kaggle-out
```

The relaxed gates exist to let the seeded demo failure through. They are
tuned to this demo and are not a production recommendation. The kernel
needs the release to exist on PyPI first.

## How this compares

| tool | what it is best at | where vision-input-check differs |
|---|---|---|
| [promptfoo](https://github.com/promptfoo/promptfoo) | prompt and provider matrices with assertions, including image inputs | evaluates model behavior across inputs; this tool varies the delivery encoding of the same image and gates on aggregate deltas across repetitions |
| [DeepEval](https://github.com/confident-ai/deepeval) | pytest-style LLM metrics, including multimodal ones | grades quality with scored metrics; this tool uses exact assertions so identical-input variance becomes visible and gateable |
| [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | benchmark evaluation of vision-language models over large datasets | benchmarks capability once; this tool is a small regression harness for your own images in CI, with pixel identity verified on what actually got sent |
| [BackstopJS](https://github.com/garris/BackstopJS) | visual regression of rendered web pages via pixel diffs | compares pixels with no model involved; this tool checks that model answers stay stable when bytes change |
| [Argos](https://github.com/argos-ci/argos) | screenshot visual regression hosting for UIs | same pixel-diff domain, no provider round-trip, no answer assertions |

promptfoo, DeepEval, and VLMEvalKit evaluate models. BackstopJS and Argos
diff pixels. This package works the gap between them: it verifies that
pixel-identical deliveries behave identically, and it quantifies how lossy
deliveries move answers, as a CI gate.

## Limitations

- The client cannot tell provider-side preprocessing apart from model
  behavior. If a provider re-encodes everything, identical-class deltas
  will not see it.
- Repeating runs reduces uncertainty. It does not eliminate noise and it
  does not prove causality. Do not read significance into three
  repetitions.
- Exact assertions fit narrow extraction tasks, not creative or open-ended
  evaluation. There is no LLM judge, by design.
- Synthetic replay proves the harness works, never the quality of a model.
- Lossy variants have no obligation to preserve answers, so
  `lossy_accuracy_min` is a regression baseline, not a correctness
  contract.

## Development

```bash
make install   # uv sync, pulls pytest and ruff via the dependency group
make test      # offline pytest, sockets blocked, live tests opt-in only
make lint      # ruff (E, F, I, UP, RUF), line length 100
make build     # uv build
make demo      # relaxed-gate demo, exit 0
```

CI (`.github/workflows/test.yml`) runs Python 3.10 through 3.13, Ruff, the
offline suite, both demo gate modes, and a clean wheel install with a
smoke test. Publishing (`.github/workflows/publish.yml`) uses Trusted
Publishing on release; no tokens live in the repo.

## License

Apache-2.0, see [LICENSE](LICENSE).
