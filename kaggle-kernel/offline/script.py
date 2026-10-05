"""
Reproduce the vision-input-check demo in a Kaggle notebook (CPU, offline
after the install step).

The script installs vision-input-check==0.1.0 from PyPI, runs the built-in
demo offline, prints a summary, and leaves result.json and report.html in
the working directory (on Kaggle: /kaggle/working).

IMPORTANT: the demo is a synthetic replay whose failures are seeded on
purpose, which is why these RELAXED gates (--max-identical-delta 0.17
--min-lossy-accuracy 0.8) let it pass. They are tuned to this demo, not a
recommendation for production; with default gates the demo exits 1.

The kernel needs the 0.1.0 release to exist on PyPI first. Run it after
publishing, or pin a wheel URL instead.
"""

import json
import subprocess
import sys

PKG = "vision-input-check==0.1.0"

print("installing", PKG)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", PKG], check=True)

# Relaxed gates on purpose: the seeded demo failure (inv-003/png-reencode,
# inv-005/jpeg-40) must stay visible while the harness run itself succeeds.
cmd = [
    "vision-check", "demo",
    "--max-identical-delta", "0.17",
    "--min-lossy-accuracy", "0.8",
    "--output", "result.json",
    "--html", "report.html",
]
completed = subprocess.run(cmd)
if completed.returncode != 0:
    sys.exit("demo failed unexpectedly; the harness or package version changed")

with open("result.json", encoding="utf-8") as handle:
    result = json.load(handle)

print()
print("=== vision-input-check demo summary (synthetic replay) ===")
print("identical_delta_min:", result["identical_delta_min"])
print("lossy_accuracy_min:", result["lossy_accuracy_min"])
print("flaky_count:", result["flaky_count"])
print(
    "eligible_fixtures:", result["denominators"]["eligible_fixtures"],
    "/", result["denominators"]["total_fixtures"],
)
for gate in result["gates"]:
    print("gate", gate["name"], "PASS" if gate["passed"] else "FAIL")
print()
print("seeded failures (expected, demonstrate the harness):")
print("  inv-003/png-reencode -> identical_delta_min ~= -1/6")
print("  inv-005/jpeg-40      -> lossy_accuracy_min ~= 5/6")
print("artifacts written: result.json, report.html, visioncheck-demo/")
