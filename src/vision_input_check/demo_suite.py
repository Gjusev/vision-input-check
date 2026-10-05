"""Self-contained synthetic demo: six Pillow-rendered invoices, fixed questions,
and a fixed replay simulation (DEMO_RESPONSES).

The demo is a harness demonstration, not a model evaluation: every answer is
canned, and two failures are seeded on purpose:

  inv-003 / png-reencode  wrong in every run   -> identical_delta_min = -1/6
  inv-005 / jpeg-40       wrong in every run   -> lossy_accuracy_min = 5/6

Default gates therefore fail (exit 1); relaxed gates pass (exit 0). The
seeded failures prove the harness detects regressions; they say nothing
about any real provider. Reports are labelled as synthetic replay.
"""

from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

from .types import Fixture, load_suite
from .variants import IDENTICAL_VARIANTS, LOSSY_VARIANTS

DEMO_DIR = "visioncheck-demo"

DEMO_FIXTURES = [
    {
        "id": "inv-001",
        "image": "images/inv-001.png",
        "question": "What is the invoice number printed on this invoice?",
        "assert": {"kind": "equals", "expected": "INV-1001"},
        "note": "equals assertion; invoice number in the header",
    },
    {
        "id": "inv-002",
        "image": "images/inv-002.png",
        "question": "What is the total amount in EUR on this invoice?",
        "assert": {"kind": "numeric", "expected": "1,234.50", "tolerance": 0.01},
        "note": "numeric assertion; the image prints the European form 1.234,50",
    },
    {
        "id": "inv-003",
        "image": "images/inv-003.png",
        "question": "What is the invoice number printed on this invoice?",
        "assert": {"kind": "equals", "expected": "INV-1003"},
        "note": "seeded failure target for png-reencode",
    },
]

DEMO_FIXTURES += [
    {
        "id": "inv-004",
        "image": "images/inv-004.png",
        "question": "What date is printed on this invoice?",
        "assert": {"kind": "regex", "expected": r"^\d{4}-\d{2}-\d{2}$"},
        "note": "regex assertion; answer like 2026-03-14",
    },
    {
        "id": "inv-005",
        "image": "images/inv-005.png",
        "question": "Which customer is named on this invoice?",
        "assert": {"kind": "contains", "expected": "Acme"},
        "note": "contains assertion; seeded failure target for jpeg-40",
    },
    {
        "id": "inv-006",
        "image": "images/inv-006.png",
        "question": "How many line items are listed on this invoice?",
        "assert": {"kind": "numeric", "expected": "12", "tolerance": 0.0},
        "note": "numeric assertion; replay answers vary in wording, not correctness",
    },
]

_DEMO_ANSWERS = {
    "inv-001": ["INV-1001"],
    "inv-002": ["The total is 1.234,50 EUR."],
    "inv-003": ["INV-1003"],
    "inv-004": ["2026-03-14"],
    "inv-005": ["Acme Retail Co."],
    "inv-006": ["12", "There are 12 items listed.", "12 items"],
}

#: Explicit replay simulation: fixture id -> variant -> answers per run cycle.
DEMO_RESPONSES = {
    fid: {variant: list(answers) for variant in (*IDENTICAL_VARIANTS, *LOSSY_VARIANTS)}
    for fid, answers in _DEMO_ANSWERS.items()
}

# Seeded failure 1: png-reencode breaks inv-003 in every run.
DEMO_RESPONSES["inv-003"]["png-reencode"] = ["INV-1008", "INV-1088", "unreadable"]
# Seeded failure 2: jpeg-40 breaks inv-005 in every run.
DEMO_RESPONSES["inv-005"]["jpeg-40"] = ["Beta GmbH", "Beta Logistik", "Beta North"]

#: (fixture, variant) pairs that fail on purpose, for report banners.
DEMO_SEEDED = (
    ("inv-003", "png-reencode", "identical"),
    ("inv-005", "jpeg-40", "lossy"),
)

_INVOICE_DATA = [
    ("INV-1001", "2026-01-12", "Acme Logistics GmbH", 4, "840.50 EUR"),
    ("INV-1002", "2026-01-19", "Beta Handel AG", 7, "1.234,50 EUR"),
    ("INV-1003", "2026-01-26", "Acme Retail Co.", 9, "2.499,00 EUR"),
    ("INV-1004", "2026-03-14", "Acme Foods LLC", 5, "96.20 EUR"),
    ("INV-1005", "2026-02-02", "Acme Retail Co.", 12, "1.180,75 EUR"),
    ("INV-1006", "2026-02-09", "Beta Nord GmbH", 12, "340.00 EUR"),
]


def _render_invoice(
    path: str, number: str, date: str, customer: str, items: int, total: str
) -> None:
    image = Image.new("RGB", (640, 400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    lines = [
        "ACME SUPPLIES GmbH - Invoice",
        "Number: " + number,
        "Date: " + date,
        "Customer: " + customer,
        "Items: " + str(items),
        "Total: " + total,
    ]
    y = 60
    for line in lines:
        draw.text((60, y), line, fill="black", font=font)
        y += 44
    draw.rectangle((40, 40, 600, 360), outline="black", width=2)
    image.save(path, format="PNG")


def build_demo_suite(out_dir: str = DEMO_DIR) -> list[Fixture]:
    """Render the demo invoices and write suite.jsonl; return loaded fixtures.

    Creates ``<out_dir>/images/`` with six PNGs and ``<out_dir>/suite.jsonl``
    with image paths relative to the suite file.
    """
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    suite_path = os.path.join(out_dir, "suite.jsonl")
    with open(suite_path, "w", encoding="utf-8", newline="\n") as handle:
        for spec in DEMO_FIXTURES:
            handle.write(json.dumps(spec) + "\n")
    for spec, data in zip(DEMO_FIXTURES, _INVOICE_DATA):
        _render_invoice(os.path.join(out_dir, str(spec["image"])), *data)
    return load_suite(suite_path)
