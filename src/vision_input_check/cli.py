"""Command line interface: ``vision-check demo`` and ``vision-check run``.

Exit codes:
  0  all gates satisfied
  1  evaluation ran, artifacts written, at least one gate failed
  2  configuration, data, or transport error (nothing evaluated)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

from .demo_suite import DEMO_DIR, DEMO_RESPONSES, DEMO_SEEDED, build_demo_suite
from .evaluate import (
    MAX_IDENTICAL_DELTA_DEFAULT,
    compute_gates,
    evaluate_suite,
)
from .providers import LiveProvider, ReplayProvider
from .report import render_html
from .types import SuiteError, load_suite


def _add_gate_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs", type=int, default=3, help="runs per fixture/variant (default 3)")
    parser.add_argument(
        "--max-identical-delta",
        type=float,
        default=MAX_IDENTICAL_DELTA_DEFAULT,
        help="positive magnitude of aggregate identical-variant degradation allowed (default 0.0)",
    )
    parser.add_argument(
        "--min-lossy-accuracy",
        type=float,
        default=None,
        help="minimum aggregate accuracy across lossy variants (default: gate disabled)",
    )
    parser.add_argument(
        "--max-flaky",
        type=int,
        default=None,
        help="maximum number of fixtures with unstable original (default: gate disabled)",
    )
    parser.add_argument("--json", action="store_true", help="print only the result JSON on stdout")
    parser.add_argument("--output", default=None, help="path for result.json")
    parser.add_argument("--html", default=None, help="path for report.html")


def _validate(args: argparse.Namespace) -> None:
    """Raise SuiteError for invalid configuration before anything runs."""
    if args.runs < 1:
        raise SuiteError(f"--runs must be >= 1, got {args.runs}")
    for flag, value in (("max-identical-delta", args.max_identical_delta),):
        if not math.isfinite(value):
            raise SuiteError(f"--{flag} must be a finite number, got {value!r}")
    if args.min_lossy_accuracy is not None and not math.isfinite(args.min_lossy_accuracy):
        raise SuiteError("--min-lossy-accuracy must be a finite number")
    if args.max_flaky is not None and args.max_flaky < 0:
        raise SuiteError(f"--max-flaky must be >= 0, got {args.max_flaky}")


def _finish(args: argparse.Namespace, payload: dict, image_paths: dict, label: str,
            synthetic: bool, seeded: list) -> int:
    """Write artifacts, emit output, return the gate exit code."""
    if args.output:
        directory = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(directory, exist_ok=True)
        with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
    if args.html:
        render_html(payload, image_paths, args.html, label, synthetic, seeded)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0 if all(g["passed"] for g in payload.get("gates", [])) else 1
    print(label)
    print("samples=" + str(payload["samples"]) + " runs=" + str(payload["runs"]))
    print("identical_delta_min=" + str(payload["identical_delta_min"]))
    print("lossy_accuracy_min=" + str(payload["lossy_accuracy_min"]))
    print("flaky_fixtures=" + str(payload["flaky_count"]))
    den = payload.get("denominators", {})
    print(
        "eligible_fixtures=" + str(den.get("eligible_fixtures"))
        + "/" + str(den.get("total_fixtures"))
    )
    for gate in payload.get("gates", []):
        state = "PASS" if gate["passed"] else "FAIL"
        print("gate " + gate["name"] + ": " + state)
    written = [p for p in (args.output, args.html) if p]
    if written:
        print("artifacts: " + ", ".join(written))
    return 0 if all(g["passed"] for g in payload.get("gates", [])) else 1


def cmd_demo(args: argparse.Namespace) -> int:
    _validate(args)
    os.makedirs(DEMO_DIR, exist_ok=True)
    fixtures = build_demo_suite(DEMO_DIR)
    result = evaluate_suite(
        fixtures,
        ReplayProvider(DEMO_RESPONSES),
        runs=args.runs,
        provider_config={"kind": "replay"},
    )
    gates = compute_gates(
        result,
        max_identical_delta=args.max_identical_delta,
        min_lossy_accuracy=args.min_lossy_accuracy,
        max_flaky=args.max_flaky,
    )
    payload = result.to_dict(gates=gates)
    output = args.output or os.path.join(DEMO_DIR, "result.json")
    html_path = args.html or os.path.join(DEMO_DIR, "report.html")
    args.output, args.html = output, html_path
    image_paths = {f.id: f.image for f in fixtures}
    label = "vision-input-check demo (synthetic replay)"
    return _finish(args, payload, image_paths, label, synthetic=True, seeded=list(DEMO_SEEDED))


def cmd_run(args: argparse.Namespace) -> int:
    _validate(args)
    fixtures = load_suite(args.suite)
    image_paths = {f.id: f.image for f in fixtures}
    if args.provider == "replay":
        if not args.responses:
            raise SuiteError("--provider replay requires --responses FILE")
        provider = ReplayProvider.from_file(args.responses)
        provider_config = {"kind": "replay"}
        label = "vision-input-check run (replay; answers are canned simulation data)"
        synthetic, seeded = True, []
    else:
        provider = LiveProvider.from_env(args.model)
        provider_config = {
            "kind": "live",
            "model": args.model or os.environ.get("VISION_MODEL", ""),
            "base_url": os.environ.get("VISION_BASE_URL", "https://api.openai.com/v1"),
        }
        label = "vision-input-check run (live provider)"
        synthetic, seeded = False, []
    try:
        result = evaluate_suite(
            fixtures, provider, runs=args.runs, provider_config=provider_config
        )
    finally:
        if args.provider == "live":
            provider.close()
    gates = compute_gates(
        result,
        max_identical_delta=args.max_identical_delta,
        min_lossy_accuracy=args.min_lossy_accuracy,
        max_flaky=args.max_flaky,
    )
    payload = result.to_dict(gates=gates)
    return _finish(args, payload, image_paths, label, synthetic=synthetic, seeded=seeded)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision-check",
        description="Regression tests for images sent to vision LLM APIs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the built-in synthetic demo offline")
    _add_gate_flags(demo)

    run = sub.add_parser("run", help="run a suite file")
    run.add_argument("suite", help="suite .jsonl or .json file")
    run.add_argument(
        "--provider",
        choices=("replay", "live"),
        default="replay",
        help="answer source: canned replay or a live OpenAI-compatible API",
    )
    run.add_argument("--responses", default=None, help="replay answers JSON (required for replay)")
    run.add_argument(
        "--model", default=None, help="model for live provider (overrides VISION_MODEL)"
    )
    _add_gate_flags(run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "demo":
            return cmd_demo(args)
        return cmd_run(args)
    except (SuiteError, RuntimeError, ImportError, KeyError, OSError) as error:
        print("vision-check: error: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
