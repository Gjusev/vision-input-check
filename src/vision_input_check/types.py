"""Typed schemas for suites and results."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

#: Version of the ``result.json`` document shape. Evolution within this
#: version is additive only: existing keys never change meaning or type.
SCHEMA_VERSION = "1.0"


class SuiteError(ValueError):
    """Raised when a suite file, replay file, or fixture cannot be parsed."""


@dataclass
class Assertion:
    """How to judge a raw model answer.

    kind:
      equals   exact match after whitespace strip
      contains expected substring appears in the answer
      regex    re.search(expected, answer)
      numeric  first number in answer compared to expected within tolerance

    ``numeric`` uses :func:`parse_number` on both sides, so both separator
    conventions are accepted; see that function for the documented
    ambiguities.
    """

    kind: str
    expected: str
    tolerance: float = 0.0

    KINDS = ("equals", "contains", "regex", "numeric")

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise SuiteError(f"assertion kind must be one of {self.KINDS}, got {self.kind!r}")


@dataclass
class Fixture:
    id: str
    image: str  # resolved path to the source image
    question: str
    assertion: Assertion
    note: str = ""


@dataclass
class VariantSpec:
    name: str
    kind: str  # "identical" or "lossy"


@dataclass
class VariantResult:
    variant: str
    kind: str
    answers: list[str]
    passes: list[bool]
    accuracy: float
    delta_vs_baseline: float | None = None  # None for the baseline itself
    distinct_answers: int = 0  # number of unique raw answers across runs


@dataclass
class FixtureResult:
    fixture_id: str
    variants: list[VariantResult]
    flaky: bool
    flaky_reason: str = ""
    question: str = ""  # additive in 1.0: keeps reports self-contained
    expected: str = ""  # additive in 1.0: what the assertion wants


@dataclass
class VisionCheckResult:
    samples: int
    runs: int
    identical_delta_min: float | None
    lossy_accuracy_min: float | None
    flaky_count: int
    fixtures: list[FixtureResult]
    schema_version: str = SCHEMA_VERSION
    provider: dict[str, Any] = field(default_factory=dict)
    aggregates: dict[str, Any] = field(default_factory=dict)
    denominators: dict[str, Any] = field(default_factory=dict)
    exclusions: list[str] = field(default_factory=list)

    def to_dict(self, gates: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Serialize the result. Existing keys keep their names and meaning.

        ``gates`` (when given) and the additive keys added in schema 1.0
        (``schema_version``, ``provider``, ``aggregates``, ``denominators``,
        ``exclusions``, per-variant ``distinct_answers``) extend the document
        without changing any original key.
        """
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "samples": self.samples,
            "runs": self.runs,
            "identical_delta_min": self.identical_delta_min,
            "lossy_accuracy_min": self.lossy_accuracy_min,
            "flaky_count": self.flaky_count,
            "aggregates": self.aggregates,
            "denominators": self.denominators,
            "exclusions": self.exclusions,
            "fixtures": [
                {
                    "fixture_id": f.fixture_id,
                    "flaky": f.flaky,
                    "flaky_reason": f.flaky_reason,
                    "question": f.question,
                    "expected": f.expected,
                    "variants": [
                        {
                            "variant": v.variant,
                            "kind": v.kind,
                            "answers": v.answers,
                            "passes": v.passes,
                            "accuracy": v.accuracy,
                            "delta_vs_baseline": v.delta_vs_baseline,
                            "distinct_answers": v.distinct_answers,
                        }
                        for v in f.variants
                    ],
                }
                for f in self.fixtures
            ],
        }
        if gates is not None:
            payload["gates"] = gates
        return payload

    def summary(self) -> str:
        ident = "n/a" if self.identical_delta_min is None else f"{self.identical_delta_min:+.4f}"
        lossy = "n/a" if self.lossy_accuracy_min is None else f"{self.lossy_accuracy_min:.4f}"
        worst_ident = _worst(self, "identical")
        worst_lossy = _worst(self, "lossy")
        eligible = self.denominators.get("eligible_fixtures", "?")
        total = self.denominators.get("total_fixtures", "?")
        return "\n".join(
            [
                f"samples={self.samples} runs={self.runs}",
                f"identical_delta_min={ident}"
                + (f" (worst: {worst_ident[0]}/{worst_ident[1]})" if worst_ident else ""),
                f"lossy_accuracy_min={lossy}"
                + (f" (worst: {worst_lossy[0]}/{worst_lossy[1]})" if worst_lossy else ""),
                f"flaky_fixtures={self.flaky_count}",
                f"eligible_fixtures={eligible}/{total}"
                + (f" excluded: {', '.join(self.exclusions)}" if self.exclusions else ""),
            ]
        )


def _worst(result: VisionCheckResult, kind: str) -> tuple[str, str] | None:
    """Fixture/variant holding the worst observed score of one kind (descriptive)."""
    worst: tuple[float, str, str] | None = None
    for fixture in result.fixtures:
        for variant in fixture.variants:
            if variant.kind != kind or variant.delta_vs_baseline is None:
                continue
            score = variant.delta_vs_baseline if kind == "identical" else variant.accuracy
            if worst is None or score < worst[0]:
                worst = (score, fixture.fixture_id, variant.variant)
    if worst is None:
        return None
    return worst[1], worst[2]


_NUM = re.compile(r"-?\d[\d.,]*")


def parse_number(text: str) -> float | None:
    """First number in *text*, tolerant of separator conventions.

    Accepted conventions:
      - ``1,234.56``  (comma thousands, dot decimal)
      - ``1.234,56``  (dot thousands, comma decimal)
      - ``12.50``     (dot decimal)
      - ``12,50``     (comma decimal)
      - ``1234``      (plain)

    Documented ambiguities (the parser picks one deterministic reading):
      - ``1.234`` is read as a dot decimal (1.234), never as the German
        thousands form (1234); a lone dot is always decimal.
      - ``1,234`` with a lone comma and exactly three trailing digits is read
        as comma thousands (1234); with one or two trailing digits it is read
        as a comma decimal (1.234). ``1,2345`` (four trailing digits) is read
        as thousands (12345).
      - For inputs with both separators, the *last* separator is the decimal
        point and the other one is stripped as thousands grouping.
    """
    match = _NUM.search(text)
    if match is None:
        return None
    raw = match.group(0).rstrip(".,")
    has_comma, has_dot = "," in raw, "." in raw
    if has_comma and has_dot:
        # The rightmost separator is the decimal point by convention.
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif has_comma:
        head, _, tail = raw.rpartition(",")
        if 1 <= len(tail) <= 2:
            raw = head + "." + tail  # comma decimal: 12,50
        else:
            raw = raw.replace(",", "")  # comma thousands: 1,234
    try:
        return float(raw)
    except ValueError:  # pragma: no cover - _NUM only yields valid float text
        return None


def load_suite(path: str) -> list[Fixture]:
    """Load a .jsonl or .json suite of fixtures.

    Image paths are resolved relative to the suite file's directory, so a
    suite can be moved with its images. IDs must be unique and the suite
    non-empty. Raises :class:`SuiteError` on any structural problem.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as error:
        raise SuiteError(f"cannot read suite {path!r}: {error.strerror or error}") from error
    try:
        rows: Any = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if path.endswith(".jsonl")
            else json.loads(text)
        )
    except json.JSONDecodeError as error:
        raise SuiteError(f"suite {path!r} is not valid JSON: {error}") from error
    if not isinstance(rows, list) or not rows:
        raise SuiteError("suite must be a non-empty array of fixtures")

    base_dir = os.path.dirname(os.path.abspath(path))
    fixtures: list[Fixture] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise SuiteError(f"fixture entry is not an object: {row!r}")
        try:
            fixture_id = str(row["id"])
            if fixture_id in seen:
                raise SuiteError(f"duplicate fixture id {fixture_id!r}")
            seen.add(fixture_id)
            image = str(row["image"])
            if image and not os.path.isabs(image):
                image = os.path.normpath(os.path.join(base_dir, image))
            tolerance = float(row["assert"].get("tolerance", 0.0))
            assertion = Assertion(
                kind=str(row["assert"]["kind"]),
                expected=str(row["assert"]["expected"]),
                tolerance=tolerance,
            )
            fixtures.append(
                Fixture(
                    id=fixture_id,
                    image=image,
                    question=str(row["question"]),
                    assertion=assertion,
                    note=str(row.get("note", "")),
                )
            )
        except KeyError as error:
            raise SuiteError(f"missing field {error} in fixture {row.get('id', '?')!r}") from error
        except (SuiteError, TypeError, ValueError) as error:
            if isinstance(error, SuiteError):
                raise
            raise SuiteError(f"invalid fixture {row.get('id', '?')!r}: {error}") from error
    return fixtures


def check_assertion(answer: str, assertion: Assertion) -> bool:
    """Judge one raw answer against one assertion."""
    expected = assertion.expected.strip()
    got = answer.strip()
    if assertion.kind == "equals":
        return got == expected
    if assertion.kind == "contains":
        return expected in got
    if assertion.kind == "regex":
        try:
            return re.search(expected, got) is not None
        except re.error as error:
            raise SuiteError(f"invalid regex {expected!r}: {error}") from error
    got_num = parse_number(got)
    exp_num = parse_number(expected)
    if got_num is None or exp_num is None:
        return False
    return abs(got_num - exp_num) <= assertion.tolerance + 1e-9
