"""Evaluation engine: runs, scoring, aggregates, and CI gates.

Definitions (N = runs per fixture/variant, default 3):

  cell accuracy      correct answers / N for one fixture/variant pair
  variant aggregate  mean of cell accuracies over eligible fixtures
  delta (aggregate)  mean of (variant cell accuracy - original cell accuracy)
                     over the same eligible fixtures
  identical_delta_min  minimum aggregate delta across identical variants
                     other than "original"
  lossy_accuracy_min   minimum aggregate accuracy across lossy variants
  flaky              a fixture whose original variant mixes pass/fail

Flaky fixtures are excluded from strict metrics; the denominators section
reports what was excluded. With zero eligible fixtures the strict metrics
are ``None`` and every gate is reported as failed: an unevaluated result is
never presented as passing.

With N around 3, these numbers are descriptive, not statistically
significant; the report says so instead of claiming significance.
"""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

from .types import (
    Fixture,
    FixtureResult,
    SuiteError,
    VariantResult,
    VisionCheckResult,
    check_assertion,
)
from .variants import BUILTIN_VARIANTS, apply_variant, verify_variant_class

MAX_IDENTICAL_DELTA_DEFAULT = 0.0


class AnswerRunner(Protocol):
    def ask(self, fixture: Fixture, variant: str, encoded: bytes, run_index: int) -> str:
        ...


def evaluate_suite(
    fixtures: list[Fixture],
    provider: AnswerRunner,
    runs: int = 3,
    provider_config: dict[str, Any] | None = None,
) -> VisionCheckResult:
    """Run every fixture through every built-in variant ``runs`` times.

    Raises SuiteError for structurally invalid input (empty suite, duplicate
    ids, non-positive runs) and when an identical-class variant loses pixels.
    Provider failures (transport, missing replay answers) propagate as-is so
    the CLI can report a configuration/transport error, never a wrong answer.
    """
    if runs < 1:
        raise SuiteError(f"runs must be >= 1, got {runs}")
    if not fixtures:
        raise SuiteError("suite is empty; nothing to evaluate")
    ids = [f.id for f in fixtures]
    if len(set(ids)) != len(ids):
        raise SuiteError("fixture ids must be unique")

    results: list[list[VariantResult]] = []
    original_passes: list[list[bool]] = []
    for fixture in fixtures:
        with Image.open(fixture.image) as handle:
            source = handle.convert("RGB")
        per_variant: list[VariantResult] = []
        original_variant_passes: list[bool] = []
        for spec in BUILTIN_VARIANTS:
            encoded = apply_variant(source, spec.name)
            if spec.kind == "identical" and not verify_variant_class(spec.name, source, encoded):
                raise SuiteError(
                    f"variant {spec.name!r} is not pixel-identical for fixture {fixture.id!r}; "
                    "identical-class variants must preserve pixels exactly"
                )
            answers: list[str] = []
            passes: list[bool] = []
            for run_index in range(runs):
                answer = provider.ask(fixture, spec.name, encoded, run_index)
                answers.append(str(answer))
                passes.append(check_assertion(str(answer), fixture.assertion))
            accuracy = sum(passes) / runs
            per_variant.append(
                VariantResult(
                    variant=spec.name,
                    kind=spec.kind,
                    answers=answers,
                    passes=passes,
                    accuracy=accuracy,
                    distinct_answers=len(set(answers)),
                )
            )
            if spec.name == "original":
                original_variant_passes = passes
        results.append(per_variant)
        original_passes.append(original_variant_passes)

    fixtures_out: list[FixtureResult] = []
    for fixture, per_variant, orig_passes in zip(fixtures, results, original_passes):
        flaky = 0 < sum(orig_passes) < runs
        reason = "original answers mix pass and fail across runs" if flaky else ""
        baseline = next(v.accuracy for v in per_variant if v.variant == "original")
        for v in per_variant:
            if v.variant != "original":
                v.delta_vs_baseline = v.accuracy - baseline
        fixtures_out.append(
            FixtureResult(
                fixture_id=fixture.id,
                variants=per_variant,
                flaky=flaky,
                flaky_reason=reason,
                question=fixture.question,
                expected=fixture.assertion.expected,
            )
        )

    return _assemble_result(fixtures_out, runs, provider_config)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _assemble_result(
    fixtures_out: list[FixtureResult],
    runs: int,
    provider_config: dict[str, Any] | None,
) -> VisionCheckResult:
    """Compute aggregates, denominators, and the strict metric summary."""
    eligible = [f for f in fixtures_out if not f.flaky]
    excluded = [f.fixture_id for f in fixtures_out if f.flaky]

    aggregates: dict[str, dict[str, Any]] = {}
    for name in [s.name for s in BUILTIN_VARIANTS]:
        cells = [
            next(v for v in f.variants if v.variant == name).accuracy for f in eligible
        ]
        entry: dict[str, Any] = {
            "kind": next(s.kind for s in BUILTIN_VARIANTS if s.name == name),
            "eligible_fixtures": len(cells),
            "accuracy": _mean(cells) if cells else None,
        }
        if name == "original":
            entry["delta_vs_baseline"] = None
        else:
            deltas = [
                next(v for v in f.variants if v.variant == name).delta_vs_baseline
                for f in eligible
            ]
            entry["delta_vs_baseline"] = (
                _mean([d for d in deltas if d is not None]) if cells else None
            )
        aggregates[name] = entry

    identical_deltas = [
        aggregates[n]["delta_vs_baseline"]
        for n, a in aggregates.items()
        if a["kind"] == "identical" and n != "original" and a["delta_vs_baseline"] is not None
    ]
    lossy_accs = [
        aggregates[n]["accuracy"]
        for n, a in aggregates.items()
        if a["kind"] == "lossy" and a["accuracy"] is not None
    ]

    return VisionCheckResult(
        samples=len(eligible) * len(BUILTIN_VARIANTS) * runs,
        runs=runs,
        identical_delta_min=min(identical_deltas) if identical_deltas else None,
        lossy_accuracy_min=min(lossy_accs) if lossy_accs else None,
        flaky_count=len(excluded),
        fixtures=fixtures_out,
        provider=dict(provider_config or {}),
        aggregates=aggregates,
        denominators={
            "eligible_fixtures": len(eligible),
            "total_fixtures": len(fixtures_out),
            "excluded_flaky": len(excluded),
        },
        exclusions=excluded,
    )


def compute_gates(
    result: VisionCheckResult,
    max_identical_delta: float = MAX_IDENTICAL_DELTA_DEFAULT,
    min_lossy_accuracy: float | None = None,
    max_flaky: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate CI gates against the strict aggregates.

    Gate defaults (documented; see README):
      max_identical_delta  0.0 — always active. Any negative aggregate delta
                           on an identical variant fails. The threshold is a
                           positive magnitude of allowed degradation; the
                           gate checks observed >= -threshold.
      min_lossy_accuracy   disabled by default (None). When set, requires the
                           worst aggregate lossy accuracy >= threshold.
      max_flaky            disabled by default (None). When set, requires
                           flaky_count <= threshold.

    If no fixtures remain eligible after flaky exclusion, no gate is
    presented as passed — an empty evaluation is a failure, not a pass.
    """
    no_eligible = result.denominators.get("eligible_fixtures", 0) == 0
    gates: list[dict[str, Any]] = []
    observed = result.identical_delta_min
    ok = observed is not None and observed >= -max_identical_delta
    gates.append(
        {
            "name": "identical_delta_min",
            "operator": ">=",
            "threshold": -max_identical_delta,
            "observed": observed,
            "passed": ok and not no_eligible,
            "description": (
                "worst aggregate accuracy delta across identical variants "
                '(png-reencode, exif-strip); threshold 0.0 means identical inputs '
                "must not move answers at all"
            ),
        }
    )
    if min_lossy_accuracy is not None:
        observed = result.lossy_accuracy_min
        ok = observed is not None and observed >= min_lossy_accuracy
        gates.append(
            {
                "name": "lossy_accuracy_min",
                "operator": ">=",
                "threshold": min_lossy_accuracy,
                "observed": observed,
                "passed": ok and not no_eligible,
                "description": (
                    "worst aggregate accuracy across lossy variants (jpeg-75, "
                    "jpeg-40, resize-50, grayscale, webp); measures how far "
                    "degraded delivery moves answers"
                ),
            }
        )
    if max_flaky is not None:
        observed = result.flaky_count
        ok = observed <= max_flaky
        gates.append(
            {
                "name": "max_flaky",
                "operator": "<=",
                "threshold": max_flaky,
                "observed": observed,
                "passed": ok and not no_eligible,
                "description": (
                    "number of fixtures whose original variant mixes pass/fail "
                    "across runs; flaky fixtures are excluded from strict metrics"
                ),
            }
        )
    return gates
