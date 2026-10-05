"""Evaluation math: aggregates, flaky exclusion, gates, error semantics."""

import pytest

from vision_input_check.demo_suite import DEMO_RESPONSES, build_demo_suite
from vision_input_check.evaluate import compute_gates, evaluate_suite
from vision_input_check.providers import ReplayProvider
from vision_input_check.types import Assertion, Fixture, SuiteError


@pytest.fixture()
def demo_fixtures(tmp_path):
    return build_demo_suite(str(tmp_path / "demo"))


class TestDemoNumbers:
    def test_seeded_aggregates(self, demo_fixtures):
        result = evaluate_suite(demo_fixtures, ReplayProvider(DEMO_RESPONSES), runs=3)
        assert result.identical_delta_min == pytest.approx(-1 / 6)
        assert result.lossy_accuracy_min == pytest.approx(5 / 6)
        assert result.flaky_count == 0
        assert result.samples == 6 * 8 * 3
        assert result.denominators == {
            "eligible_fixtures": 6,
            "total_fixtures": 6,
            "excluded_flaky": 0,
        }

    def test_default_gate_fails_relaxed_gate_passes(self, demo_fixtures):
        result = evaluate_suite(demo_fixtures, ReplayProvider(DEMO_RESPONSES), runs=3)
        assert [g["passed"] for g in compute_gates(result)] == [False]
        relaxed = compute_gates(result, max_identical_delta=0.17, min_lossy_accuracy=0.8)
        assert [g["passed"] for g in relaxed] == [True, True]

    def test_distinct_answers_tracked(self, demo_fixtures):
        result = evaluate_suite(demo_fixtures, ReplayProvider(DEMO_RESPONSES), runs=3)
        inv003 = next(f for f in result.fixtures if f.fixture_id == "inv-003")
        png = next(v for v in inv003.variants if v.variant == "png-reencode")
        assert png.answers == ["INV-1008", "INV-1088", "unreadable"]
        assert png.distinct_answers == 3 and png.accuracy == 0.0
        inv006 = next(f for f in result.fixtures if f.fixture_id == "inv-006")
        original = next(v for v in inv006.variants if v.variant == "original")
        assert original.passes == [True, True, True]
        assert original.distinct_answers == 3  # wording varies, verdict does not


class TestFlaky:
    def test_flaky_fixture_excluded_but_max_flaky_gate_applies(self, demo_fixtures):
        # inv-001's original alternates correct/incorrect -> flaky fixture.
        responses = {k: {v: list(a) for v, a in d.items()} for k, d in DEMO_RESPONSES.items()}
        responses["inv-001"]["original"] = ["INV-1001", "INV-9999", "INV-1001"]
        result = evaluate_suite(demo_fixtures, ReplayProvider(responses), runs=3)
        assert result.flaky_count == 1
        assert result.exclusions == ["inv-001"]
        assert result.denominators["eligible_fixtures"] == 5
        assert result.denominators["total_fixtures"] == 6
        # Strict metric now averages over 5 fixtures; png-reencode still -1/5.
        assert result.identical_delta_min == pytest.approx(-1 / 5)
        gates = compute_gates(result, max_flaky=0)
        flaky_gate = next(g for g in gates if g["name"] == "max_flaky")
        assert flaky_gate["observed"] == 1 and flaky_gate["passed"] is False

    def test_all_flaky_means_no_gate_passes(self, demo_fixtures):
        responses = {k: {v: list(a) for v, a in d.items()} for k, d in DEMO_RESPONSES.items()}
        for answers in responses.values():
            correct = answers["original"][0]
            answers["original"] = [correct, "definitely wrong"]
        result = evaluate_suite(demo_fixtures, ReplayProvider(responses), runs=2)
        assert result.flaky_count == 6
        assert result.identical_delta_min is None
        assert result.lossy_accuracy_min is None
        for gate in compute_gates(result, max_identical_delta=1.0, max_flaky=99):
            assert gate["passed"] is False


class TestValidation:
    def test_runs_must_be_positive(self, demo_fixtures):
        with pytest.raises(SuiteError, match="runs"):
            evaluate_suite(demo_fixtures, ReplayProvider(DEMO_RESPONSES), runs=0)

    def test_empty_suite(self):
        with pytest.raises(SuiteError, match="empty"):
            evaluate_suite([], ReplayProvider(DEMO_RESPONSES))

    def test_duplicate_ids(self):
        fixture = Fixture("f1", "x.png", "q", Assertion("equals", "y"))
        with pytest.raises(SuiteError, match="unique"):
            evaluate_suite([fixture, fixture], ReplayProvider(DEMO_RESPONSES))

    def test_transport_failure_is_error_not_wrong_answer(self, demo_fixtures):
        class Broken:
            def ask(self, fixture, variant, encoded, run_index):
                raise RuntimeError("connection reset")

        with pytest.raises(RuntimeError, match="connection reset"):
            evaluate_suite(demo_fixtures, Broken(), runs=1)

    def test_missing_image_is_suite_error(self, tmp_path):
        fixture = Fixture("f1", str(tmp_path / "missing.png"), "q", Assertion("equals", "y"))
        with pytest.raises(OSError):
            evaluate_suite([fixture], ReplayProvider(DEMO_RESPONSES))
