"""CLI behavior: exit codes, JSON output, determinism, artifact writing."""

import json
import os

import pytest

from vision_input_check.cli import main
from vision_input_check.demo_suite import DEMO_RESPONSES


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _write_responses(path):
    path.write_text(json.dumps(DEMO_RESPONSES), encoding="utf-8")
    return str(path)


class TestDemo:
    def test_default_gates_fail_with_artifacts(self):
        assert main(["demo"]) == 1
        assert os.path.exists("visioncheck-demo/result.json")
        assert os.path.exists("visioncheck-demo/report.html")
        assert os.path.exists("visioncheck-demo/suite.jsonl")

    def test_relaxed_gates_pass(self):
        assert main(["demo", "--max-identical-delta", "0.17", "--min-lossy-accuracy", "0.8"]) == 0

    def test_json_stdout_is_pure_json(self, capsys):
        assert main(["demo", "--json"]) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["schema_version"] == "1.0"
        assert payload["provider"]["kind"] == "replay"
        assert any(not g["passed"] for g in payload["gates"])

    def test_deterministic_output(self, capsys):
        main(["demo", "--json"])
        first = capsys.readouterr().out
        main(["demo", "--json"])
        assert capsys.readouterr().out == first

    def test_invalid_runs(self):
        assert main(["demo", "--runs", "0"]) == 2

    def test_non_finite_threshold(self):
        assert main(["demo", "--max-identical-delta", "inf"]) == 2

    def test_negative_flaky_threshold(self):
        assert main(["demo", "--max-flaky", "-1"]) == 2


class TestRun:
    def test_replay_requires_responses_file(self, tmp_path):
        suite = "visioncheck-demo/suite.jsonl"
        main(["demo"])  # create the suite
        assert main(["run", suite, "--provider", "replay"]) == 2

    def test_replay_run_passes_relaxed(self, tmp_path):
        main(["demo"])
        responses = _write_responses(tmp_path / "responses.json")
        code = main(
            [
                "run", "visioncheck-demo/suite.jsonl",
                "--provider", "replay", "--responses", responses,
                "--max-identical-delta", "0.17", "--min-lossy-accuracy", "0.8",
                "--output", "out/result.json", "--html", "out/report.html",
            ]
        )
        assert code == 0
        assert os.path.exists("out/result.json") and os.path.exists("out/report.html")

    def test_missing_suite_file(self, tmp_path):
        assert main(["run", "ghost.jsonl", "--provider", "replay",
                     "--responses", _write_responses(tmp_path / "r.json")]) == 2

    def test_replay_missing_pair_is_config_error(self, tmp_path):
        main(["demo"])
        partial = tmp_path / "partial.json"
        partial.write_text(json.dumps({"inv-001": {"original": ["INV-1001"]}}), encoding="utf-8")
        assert main(["run", "visioncheck-demo/suite.jsonl", "--provider", "replay",
                     "--responses", str(partial)]) == 2

    def test_run_deterministic(self, tmp_path, capsys):
        main(["demo"])
        capsys.readouterr()  # discard demo summary
        responses = _write_responses(tmp_path / "r.json")
        args = ["run", "visioncheck-demo/suite.jsonl", "--provider", "replay",
                "--responses", responses, "--json"]
        main(args)
        first = capsys.readouterr().out
        main(args)
        assert capsys.readouterr().out == first


class TestLiveCli:
    def test_live_without_key_is_config_error(self, monkeypatch):
        monkeypatch.delenv("VISION_API_KEY", raising=False)
        main(["demo"])
        code = main(["run", "visioncheck-demo/suite.jsonl", "--provider", "live"])
        assert code == 2
