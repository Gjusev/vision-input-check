"""vision-input-check: regression tests for images sent to vision LLM APIs.

Runs a labeled image suite through controlled encoding and delivery
variants, separates pixel-identical changes (answers must not move) from
lossy stress variants (measure by how much they move), and turns the
result into CI gates.
"""

from __future__ import annotations

from .demo_suite import DEMO_RESPONSES, DEMO_SEEDED, build_demo_suite
from .evaluate import compute_gates, evaluate_suite
from .providers import AnswerProvider, LiveProvider, ReplayProvider
from .report import render_html
from .types import (
    SCHEMA_VERSION,
    Assertion,
    Fixture,
    FixtureResult,
    SuiteError,
    VariantResult,
    VariantSpec,
    VisionCheckResult,
    check_assertion,
    load_suite,
    parse_number,
)
from .variants import BUILTIN_VARIANTS, apply_variant

__version__ = "0.1.0"

__all__ = [
    "BUILTIN_VARIANTS",
    "DEMO_RESPONSES",
    "DEMO_SEEDED",
    "SCHEMA_VERSION",
    "AnswerProvider",
    "Assertion",
    "Fixture",
    "FixtureResult",
    "LiveProvider",
    "ReplayProvider",
    "SuiteError",
    "VariantResult",
    "VariantSpec",
    "VisionCheckResult",
    "__version__",
    "apply_variant",
    "build_demo_suite",
    "check_assertion",
    "compute_gates",
    "evaluate_suite",
    "load_suite",
    "parse_number",
    "render_html",
]
