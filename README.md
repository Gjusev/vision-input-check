<p align="center">
  <img src="https://raw.githubusercontent.com/Gjusev/vision-input-check/main/docs/assets/logo.png" width="100" height="100" alt="vision-input-check logo">
</p>

<h1 align="center">vision-input-check</h1>

<p align="center"><strong>Same pixels. Different answers? Measure the regression.</strong></p>

[![PyPI](https://img.shields.io/pypi/v/vision-input-check)](https://pypi.org/project/vision-input-check/)
[![Python](https://img.shields.io/pypi/pyversions/vision-input-check)](https://pypi.org/project/vision-input-check/)
[![CI](https://github.com/Gjusev/vision-input-check/actions/workflows/test.yml/badge.svg)](https://github.com/Gjusev/vision-input-check/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

[Quick start](#quick-start-no-credentials) · [Metrics](#what-it-measures) ·
[Your suite](#run-your-own-suite) · [Run on Kaggle](https://www.kaggle.com/code/gjusev/vision-input-check-demo)

Changing image encoding can change what a vision API gets right. This Python
CLI runs a labeled image suite through **eight delivery variants**, repeats each
request, and turns answer correctness into regression metrics and CI gates.

It separates pixel-identical re-encodes from lossy transformations and reports
unstable baselines explicitly. Scoring uses exact assertions, not an LLM judge.

**Offline replay · No API key for the demo · Standalone HTML · Python 3.10+**

## Quick start, no credentials

```bash
pip install vision-input-check
vision-check demo
```

**Exit 1 is expected.** Two synthetic replay failures are seeded on purpose.
Open `visioncheck-demo/report.html` to inspect the fixture/variant matrix and
expand failing cells for expected answers, actual answers, and request settings.

To exercise the passing exit path with the same seeded failures:

```bash
vision-check demo --max-identical-delta 0.17 --min-lossy-accuracy 0.8
```

These relaxed thresholds are for this demo only. After installation, the demo
runs entirely offline. It demonstrates the harness, not a provider regression.

<a href="https://github.com/Gjusev/vision-input-check/releases/download/v0.1.0/brag.mp4">
  <img src="https://github.com/Gjusev/vision-input-check/releases/download/v0.1.0/brag.gif" alt="Demo: a seeded PNG re-encode failure trips the strict gate; relaxed gates pass" width="720">
</a>

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

## What it measures

| Metric | Definition | Interpretation |
|---|---|---|
| Cell accuracy | Correct answers / repeats for one fixture/variant | Reliability for that transformed input |
| Variant accuracy | Mean cell accuracy over eligible fixtures | Accuracy for a delivery transformation |
| Delta vs. original | Variant accuracy − original accuracy over the same fixtures | Change from the baseline |
| `identical_delta_min` | Lowest aggregate delta among non-original identical variants | Worst correctness drop with unchanged pixels |
| `lossy_accuracy_min` | Lowest aggregate accuracy among lossy variants | Worst accuracy under delivery degradation |
| `flaky_count` | Fixtures whose original mixes pass and fail | Baselines excluded from strict aggregates |

No eligible fixtures means strict metrics are `null` and gates fail. Flakiness
means mixed **correctness**, not merely different answer strings.

The default demo evaluates **6 fixtures × 8 variants × 3 repeats = 144 replay answers**:

| Seeded case | Observed result |
|---|---|
| `inv-003 / png-reencode` | Incorrect answers; `identical_delta_min = -1/6` |
| `inv-005 / jpeg-40` | Incorrect answers; `lossy_accuracy_min = 5/6` |
| Original baselines | All correct; `flaky_count = 0` |

Images are normalized to RGB before variants are made. Pixel-identity checks
apply to those generated inputs, not arbitrary original file bytes or metadata.

## Run your own suite

```bash
# offline, against recorded answers
vision-check run suite.jsonl --provider replay --responses responses.json

pip install "vision-input-check[live]"
export VISION_API_KEY="your-api-key"
export VISION_BASE_URL=https://api.openai.com/v1   # the /v1 suffix is kept
export VISION_MODEL="your-vision-model"
vision-check run suite.jsonl --provider live \
  --max-identical-delta 0.0 --min-lossy-accuracy 0.95 --max-flaky 0 \
  --output result.json --html report.html
```

In PowerShell, use `$env:VISION_API_KEY = "..."` and
`$env:VISION_MODEL = "..."` instead of `export`. Choose a vision endpoint/model
compatible with the adapter's chat-completions request settings.
`--model` overrides the environment variable.

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
    DEMO_RESPONSES,
    ReplayProvider,
    build_demo_suite,
    compute_gates,
    evaluate_suite,
)

fixtures = build_demo_suite("visioncheck-demo")
result = evaluate_suite(fixtures, ReplayProvider(DEMO_RESPONSES), runs=3)
gates = compute_gates(result, max_identical_delta=0.17, min_lossy_accuracy=0.8)
payload = result.to_dict(gates=gates)   # matches the bundled JSON Schema (schema_version "1.0")
print(result.summary())
```

Custom providers implement
`ask(fixture, variant, encoded: bytes, run_index: int) -> str`, where `variant`
is the variant name.

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
{"id":"inv-001","image":"images/inv-001.png","question":"What is the invoice number?","assert":{"kind":"equals","expected":"INV-1001"}}
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

The wheel includes a JSON Schema for `result.json` with
`schema_version "1.0"`. Evolution within a schema version is additive.

## Reproduce in Kaggle

The kernel already ran end to end on Kaggle: it installs
`vision-input-check==0.1.0` from PyPI, runs the offline demo with relaxed
gates, and leaves `result.json` plus `report.html` in the output.

Kernel: [gjusev/vision-input-check-demo](https://www.kaggle.com/code/gjusev/vision-input-check-demo)

To push your own copy:

```bash
export KAGGLE_API_TOKEN="your-token"
kaggle kernels push -p kaggle-kernel/offline
kaggle kernels status gjusev/vision-input-check-demo
kaggle kernels output gjusev/vision-input-check-demo -p ./kaggle-out
```

PowerShell: `$env:KAGGLE_API_TOKEN = "your-token"`. Keep tokens out of files.
To publish a fork, update the kernel owner/id in `kernel-metadata.json` first.

The relaxed gates exist to let the seeded demo failure through. They are
tuned to this demo and are not a production recommendation. The kernel
needs the release to exist on PyPI first.

## Why this exists

On **September 25, 2026**, the
[OpenAI API changelog](https://developers.openai.com/api/docs/changelog)
reported an image-encoding fix affecting GPT-6 Sol/Luna and recommended rerunning
image evaluations. This package provides a repeatable input-variation experiment.
It does not assume that bug remains or that its demo reproduces it.

## How this compares

| tool | what it is best at | where vision-input-check differs |
|---|---|---|
| [promptfoo](https://github.com/promptfoo/promptfoo) | prompt and provider matrices with assertions, including image inputs | evaluates model behavior across inputs; this tool varies the delivery encoding of the same image and gates on aggregate deltas across repetitions |
| [DeepEval](https://github.com/confident-ai/deepeval) | LLM evaluation metrics and test workflows | this tool bundles exact assertions, repeat runs, and explicit identical/lossy input groups |
| [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | benchmark evaluation of vision-language models over large datasets | this tool focuses on your own images and delivery variants, with pixel identity checked before requests |
| [BackstopJS](https://github.com/garris/BackstopJS) | visual regression of rendered web pages via pixel diffs | compares pixels with no model involved; this tool checks that model answers stay stable when bytes change |
| [Argos](https://github.com/argos-ci/argos) | screenshot visual regression hosting for UIs | same pixel-diff domain, no provider round-trip, no answer assertions |

Broader frameworks can support related experiments. This package bundles one
specific experiment with offline replay, pixel checks, explicit denominators,
and exit-code gates. It does not claim to replace those frameworks.

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
- Gates aggregate correctness, not answer-string identity. Improvements on some
  fixtures can offset regressions on others; inspect the per-fixture report too.
- Lossy variants have no obligation to preserve answers, so
  `lossy_accuracy_min` is a regression baseline, not a correctness
  contract.

## Development

```bash
make install   # uv sync, pulls pytest and ruff via the dependency group
make test      # offline pytest, sockets blocked, live tests opt-in only
make lint      # Ruff: src, tests, and Kaggle scripts; line length 100
make build     # uv build
make demo      # relaxed-gate demo, exit 0
```

Without Make: `uv sync`, `uv run ruff check src tests kaggle-kernel`,
`uv run pytest -q -m "not live"`, and `uv build`.

CI (`.github/workflows/test.yml`) runs Python 3.10 through 3.13, Ruff, the
offline suite, both demo gate modes, and a clean wheel install with a
smoke test. Publishing (`.github/workflows/publish.yml`) uses Trusted
Publishing on release; no tokens live in the repo.

## License

[Apache-2.0](LICENSE) · [Logos and social preview](docs/assets/README.md)
