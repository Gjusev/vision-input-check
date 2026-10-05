"""Offline-by-default test configuration.

Every normal test runs with sockets blocked; a test that touches the network
fails loudly instead of hanging. Real-API tests are opt-in via
VISION_LIVE_TESTS=1 (and are always excluded in CI with `-m "not live"`).
"""

import os
import socket

import pytest


@pytest.fixture(autouse=True, scope="session")
def _block_network():
    if os.environ.get("VISION_LIVE_TESTS") == "1":
        yield
        return

    def _deny(*args, **kwargs):
        raise AssertionError("network access is disabled in offline tests")

    original = socket.socket.connect
    socket.socket.connect = _deny
    yield
    socket.socket.connect = original
