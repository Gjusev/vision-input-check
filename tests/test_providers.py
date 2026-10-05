"""Replay validation and LiveProvider against a mocked HTTP transport."""

import httpx
import pytest

from vision_input_check.providers import LiveProvider, ReplayProvider
from vision_input_check.types import Assertion, Fixture, SuiteError

FIXTURE = Fixture(
    id="f1", image="x.png", question="q?", assertion=Assertion("equals", "yes")
)


class TestReplay:
    def test_cycles_deterministically_by_run_index(self):
        provider = ReplayProvider({"f1": {"original": ["a", "b"]}})
        seen = [provider.ask(FIXTURE, "original", b"", i) for i in range(3)]
        assert seen == ["a", "b", "a"]

    def test_missing_answers_is_suite_error(self):
        provider = ReplayProvider({"f1": {"original": ["a"]}})
        with pytest.raises(SuiteError, match="no answers"):
            provider.ask(FIXTURE, "jpeg-75", b"", 0)

    def test_empty_store_rejected(self):
        with pytest.raises(SuiteError):
            ReplayProvider({})

    def test_invalid_types_rejected(self):
        bad = [
            {"f1": {"original": "not-a-list"}},
            {"f1": {"original": []}},
            {"f1": {"original": [1, 2]}},
            {"f1": "not-a-dict"},
            ["not", "a", "dict"],
        ]
        for store in bad:
            with pytest.raises(SuiteError):
                ReplayProvider(store)

    def test_from_file_missing(self, tmp_path):
        with pytest.raises(SuiteError, match="cannot read"):
            ReplayProvider.from_file(str(tmp_path / "nope.json"))

    def test_from_file_invalid_json(self, tmp_path):
        path = tmp_path / "responses.json"
        path.write_text("{broken", encoding="utf-8")
        with pytest.raises(SuiteError, match="not valid JSON"):
            ReplayProvider.from_file(str(path))

    def test_from_file_ok(self, tmp_path):
        import json

        path = tmp_path / "responses.json"
        path.write_text(json.dumps({"f1": {"original": ["yes"]}}), encoding="utf-8")
        provider = ReplayProvider.from_file(str(path))
        assert provider.ask(FIXTURE, "original", b"", 0) == "yes"


def _mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=5.0)


class TestLive:
    def _provider(self, handler, **kwargs):
        provider = LiveProvider(api_key="sk-test", **kwargs)
        provider._client = _mock(handler)
        return provider

    def test_answer_extracted_and_v1_preserved(self):
        calls = []

        def handler(request):
            calls.append(str(request.url))
            return httpx.Response(200, json={"choices": [{"message": {"content": " yes "}}]})

        provider = self._provider(
            handler, base_url="https://api.example.com/v1", model="m1"
        )
        with provider:
            assert provider.ask(FIXTURE, "original", b"img", 0) == "yes"
        assert len(calls) == 1
        assert calls[0] == "https://api.example.com/v1/chat/completions"

    def test_payload_carries_image_not_expected_answer(self):
        seen = {}

        def handler(request):
            seen["body"] = request.read()
            return httpx.Response(200, json={"choices": [{"message": {"content": "a"}}]})

        provider = self._provider(handler)
        with provider:
            provider.ask(FIXTURE, "original", b"PNGBYTES", 0)
        body = seen["body"].decode()
        assert "PNGBYTES" not in body  # image is base64, not raw
        assert "yes" not in body  # expected answer never sent
        assert "q?" in body and "aW1n" not in body  # question present

    def test_auth_error_fails_fast_without_retry(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            return httpx.Response(401, json={"error": "bad key"})

        provider = self._provider(handler)
        with pytest.raises(RuntimeError, match="401"):
            with provider:
                provider.ask(FIXTURE, "original", b"", 0)
        assert len(attempts) == 1  # no retry on caller errors

    def test_transient_500_retries_bounded(self):
        attempts = []

        def handler(request):
            attempts.append(1)
            if len(attempts) < 3:
                return httpx.Response(500, json={"error": "boom"})
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        provider = self._provider(handler, max_retries=3)
        with provider:
            assert provider.ask(FIXTURE, "original", b"", 0) == "ok"
        assert len(attempts) == 3

    def test_retries_exhausted_error_has_no_secrets(self):
        def handler(request):
            return httpx.Response(500, json={"error": "boom"})

        provider = self._provider(handler, max_retries=1)
        with pytest.raises(RuntimeError) as excinfo:
            with provider:
                provider.ask(FIXTURE, "original", b"", 0)
        message = str(excinfo.value)
        assert "sk-test" not in message
        assert "Authorization" not in message

    def test_from_env_requires_key(self, monkeypatch):
        monkeypatch.delenv("VISION_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="VISION_API_KEY"):
            LiveProvider.from_env()
