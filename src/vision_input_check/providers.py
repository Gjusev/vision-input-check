"""Answer providers: offline replay and one OpenAI-compatible live adapter."""

from __future__ import annotations

import base64
import json
import os
import random
import time
from typing import Any, Protocol

from .types import Fixture, SuiteError


class AnswerProvider(Protocol):
    def ask(self, fixture: Fixture, variant: str, encoded: bytes, run_index: int) -> str:
        """Return the model's raw answer for one fixture/variant/run."""
        ...


class ReplayProvider:
    """Offline provider: canned answers keyed by fixture id and variant.

    Store shape: ``{"<fixture id>": {"<variant>": ["answer", ...]}}``.
    Answers cycle deterministically by ``run_index`` if a list is shorter
    than the run count, so replay is deterministic for any ``--runs``.
    Replay never calls a service and never generates answers with a model.
    """

    def __init__(self, store: dict[str, dict[str, list[str]]]) -> None:
        self._store = _validated_store(store)

    def ask(self, fixture: Fixture, variant: str, encoded: bytes, run_index: int) -> str:
        try:
            answers = self._store[fixture.id][variant]
        except KeyError as error:
            raise SuiteError(
                f"replay store has no answers for {fixture.id!r} / {variant!r}"
            ) from error
        return answers[run_index % len(answers)]

    @classmethod
    def from_file(cls, path: str) -> ReplayProvider:
        """Load a replay JSON file; raises SuiteError for unreadable or invalid data."""
        try:
            with open(path, encoding="utf-8") as handle:
                store = json.load(handle)
        except OSError as error:
            raise SuiteError(
                f"cannot read replay file {path!r}: {error.strerror or error}"
            ) from error
        except json.JSONDecodeError as error:
            raise SuiteError(f"replay file {path!r} is not valid JSON: {error}") from error
        return cls(store)


def _validated_store(store: object) -> dict[str, dict[str, list[str]]]:
    """Type-check the replay store eagerly so bad data fails before any run."""
    if not isinstance(store, dict) or not store:
        raise SuiteError("replay store must be a non-empty object")
    for fixture_id, variants in store.items():
        if not isinstance(variants, dict) or not variants:
            raise SuiteError(f"replay entry for {fixture_id!r} must be a non-empty object")
        for variant, answers in variants.items():
            if (
                not isinstance(answers, list)
                or not answers
                or not all(isinstance(a, str) for a in answers)
            ):
                raise SuiteError(
                    f"replay answers for {fixture_id!r}/{variant!r} "
                    "must be a non-empty list of strings"
                )
    return store  # type: ignore[return-value]  # shape proven above


# Status codes that are the caller's fault; retrying them cannot help.
_NO_RETRY_STATUS = frozenset({400, 401, 403, 404, 405, 422})


class LiveProvider:
    """Live provider against any OpenAI-compatible chat completions endpoint.

    Configuration comes from the environment so secrets never appear in
    arguments or artifacts:
      VISION_API_KEY   required
      VISION_BASE_URL  default https://api.openai.com/v1  (the /v1 suffix is kept)
      VISION_MODEL     default gpt-6.1-sol

    The request sends the question and the image only — never the expected
    answer, the assertion, or any replay data.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-6.1-sol",
        timeout: float = 60.0,
        max_retries: int = 2,
        transport: object | None = None,
    ) -> None:
        import httpx  # optional [live] extra

        # ``transport`` exists for tests (httpx.MockTransport); production code
        # leaves it as None.

        # The full URL is built per request instead of using httpx.base_url,
        # so a configured "/v1" suffix can never be dropped by URL joining.
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._model = model
        self._max_retries = max_retries

    @classmethod
    def from_env(cls, model: str | None = None) -> LiveProvider:
        api_key = os.environ.get("VISION_API_KEY")
        if not api_key:
            raise RuntimeError("VISION_API_KEY is not set; live mode needs it")
        return cls(
            api_key=api_key,
            base_url=os.environ.get("VISION_BASE_URL", "https://api.openai.com/v1"),
            model=model or os.environ.get("VISION_MODEL", "gpt-6.1-sol"),
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LiveProvider:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def ask(self, fixture: Fixture, variant: str, encoded: bytes, run_index: int) -> str:
        from .variants import media_type_for

        media = media_type_for(variant)
        data_uri = "data:" + media + ";base64," + base64.b64encode(encoded).decode("ascii")
        payload = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": 200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": fixture.question},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        }
        return self._post(payload, fixture.id, variant)

    def _post(self, payload: dict[str, Any], fixture_id: str, variant: str) -> str:
        import httpx  # optional [live] extra

        last_error = "no attempts made"
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post(self._url, json=payload, headers=self._headers)
                if response.status_code in _NO_RETRY_STATUS:
                    # Caller error: fail fast, retrying leaks rate limit or hides misconfig.
                    raise RuntimeError(
                        f"live request rejected for {fixture_id}/{variant}: "
                        f"HTTP {response.status_code} (request not retried)"
                    )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return str(content).strip()
            except httpx.HTTPStatusError as error:
                last_error = f"HTTP {error.response.status_code}"
            except httpx.HTTPError as error:
                last_error = type(error).__name__
            except (KeyError, IndexError, TypeError) as error:
                raise RuntimeError(
                    f"live response for {fixture_id}/{variant} has unexpected shape: "
                    f"{type(error).__name__}"
                ) from error
            if attempt < self._max_retries:
                time.sleep(min(0.5 * (2**attempt), 4.0) + random.uniform(0, 0.25))
        raise RuntimeError(
            f"live request failed for {fixture_id}/{variant} after "
            f"{self._max_retries + 1} attempts: {last_error}"
        )
