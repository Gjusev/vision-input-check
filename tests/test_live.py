"""Real-API smoke test. Opt-in only: VISION_LIVE_TESTS=1 and VISION_API_KEY set.

Excluded from CI with `-m "not live"`; conftest also blocks sockets unless
the opt-in variable is present. Requires network on purpose.
"""

import os

import pytest

from vision_input_check.demo_suite import build_demo_suite
from vision_input_check.providers import LiveProvider

pytestmark = pytest.mark.live

requires_live = pytest.mark.skipif(
    os.environ.get("VISION_LIVE_TESTS") != "1" or not os.environ.get("VISION_API_KEY"),
    reason="set VISION_LIVE_TESTS=1 and VISION_API_KEY to run live smoke",
)


@requires_live
def test_live_answers_one_fixture(tmp_path):
    fixtures = build_demo_suite(str(tmp_path / "demo"))
    with LiveProvider.from_env() as provider:
        answer = provider.ask(fixtures[0], "original", open(fixtures[0].image, "rb").read(), 0)
    assert isinstance(answer, str) and answer
