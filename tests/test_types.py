"""parse_number, assertions, and suite loading."""

import json

import pytest

from vision_input_check.types import (
    Assertion,
    SuiteError,
    check_assertion,
    load_suite,
    parse_number,
)


class TestParseNumber:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1,234.56", 1234.56),   # en convention
            ("1.234,56", 1234.56),   # de convention
            ("1234", 1234.0),
            ("-12.5", -12.5),
            ("12.50", 12.5),         # dot decimal
            ("12,50", 12.5),         # comma decimal (1-2 trailing digits)
            ("1,234", 1234.0),       # lone comma + 3 digits -> thousands
            ("1.234", 1.234),        # documented ambiguity: lone dot is decimal
            ("Total: 1.234,50 EUR", 1234.5),
            ("There are 12 items.", 12.0),
            ("no numbers here", None),
            ("", None),
        ],
    )
    def test_conventions(self, text, expected):
        assert parse_number(text) == expected


class TestAssertions:
    def test_equals(self):
        assert check_assertion(" INV-1001 ", Assertion("equals", "INV-1001"))
        assert not check_assertion("INV-1002", Assertion("equals", "INV-1001"))

    def test_contains(self):
        assert check_assertion("Customer: Acme GmbH", Assertion("contains", "Acme"))
        assert not check_assertion("Customer: Beta GmbH", Assertion("contains", "Acme"))

    def test_regex(self):
        assert check_assertion("Date 2026-03-14 ok", Assertion("regex", r"\d{4}-\d{2}-\d{2}"))
        assert not check_assertion("no date", Assertion("regex", r"\d{4}-\d{2}-\d{2}"))

    def test_numeric(self):
        # Both separator conventions compare equal within tolerance.
        assert check_assertion("1.234,50 EUR", Assertion("numeric", "1,234.50", 0.01))
        assert not check_assertion("99 EUR", Assertion("numeric", "100", 0.01))
        assert not check_assertion("nothing", Assertion("numeric", "100", 0.01))

    def test_invalid_regex_is_config_error(self):
        with pytest.raises(SuiteError):
            check_assertion("x", Assertion("regex", "(unclosed"))


class TestLoadSuite:
    def _write(self, tmp_path, rows, name="suite.jsonl"):
        path = tmp_path / name
        if name.endswith(".jsonl"):
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        else:
            path.write_text(json.dumps(rows), encoding="utf-8")
        return str(path)

    @staticmethod
    def _row(image="images/a.png"):
        return {
            "id": "f1",
            "image": image,
            "question": "q?",
            "assert": {"kind": "equals", "expected": "yes"},
        }

    def test_relative_paths_resolve_to_suite_dir(self, tmp_path):
        (tmp_path / "images").mkdir()
        path = self._write(tmp_path, [self._row()])
        fixtures = load_suite(path)
        assert fixtures[0].image == str(tmp_path / "images" / "a.png")

    def test_json_array_suite(self, tmp_path):
        path = self._write(tmp_path, [self._row()], name="suite.json")
        assert len(load_suite(path)) == 1

    def test_missing_field(self, tmp_path):
        bad = {"id": "f1", "question": "q?", "assert": {"kind": "equals", "expected": "y"}}
        with pytest.raises(SuiteError, match="missing field"):
            load_suite(self._write(tmp_path, [bad]))

    def test_duplicate_ids(self, tmp_path):
        with pytest.raises(SuiteError, match="duplicate"):
            load_suite(self._write(tmp_path, [self._row(), self._row()]))

    def test_empty_suite(self, tmp_path):
        with pytest.raises(SuiteError, match="non-empty"):
            load_suite(self._write(tmp_path, []))

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        with pytest.raises(SuiteError, match="not valid JSON"):
            load_suite(str(path))

    def test_unreadable_file(self, tmp_path):
        with pytest.raises(SuiteError, match="cannot read"):
            load_suite(str(tmp_path / "nope.jsonl"))
