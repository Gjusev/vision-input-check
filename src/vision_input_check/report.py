"""Standalone HTML report: no server, no external assets, all content escaped."""

from __future__ import annotations

import base64
import html
import io
import os

from PIL import Image

from .variants import media_type_for

_STATUS = {
    True: ("<span class='ok'>PASS</span>", "cell ok"),
    False: ("<span class='bad'>FAIL</span>", "cell bad"),
}


def _thumbnail(path: str, size: int = 120) -> str:
    """Base64 PNG data URI for one fixture thumbnail."""
    with Image.open(path) as handle:
        thumb = handle.convert("RGB")
        thumb.thumbnail((size, size))
        buffer = io.BytesIO()
        thumb.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _format_number(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}" if isinstance(value, float) and value < 0 else f"{value:.4f}"


def render_html(
    result: dict,
    image_paths: dict[str, str],
    out_path: str,
    provider_label: str,
    synthetic: bool,
    seeded: list[tuple[str, str, str]] | None = None,
) -> str:
    """Render result dict (as produced by VisionCheckResult.to_dict) to HTML.

    ``image_paths`` maps fixture id -> source image path for thumbnails.
    ``synthetic`` labels the report as a replay simulation whose failures are
    seeded; ``seeded`` lists (fixture, variant, kind) pairs called out in the
    banner. Returns the written path.
    """
    esc = html.escape
    e = lambda s: esc(str(s), quote=True)  # noqa: E731
    style = """
    body { font-family: system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
    h1 { font-size: 1.3rem; } h2 { font-size: 1.05rem; margin-top: 2rem; }
    table { border-collapse: collapse; margin: 0.5rem 0; }
    th, td { border: 1px solid #ccc; padding: 4px 10px; font-size: 0.85rem; text-align: left; }
    th { background: #f2f2f2; }
    .ok { color: #0a7d24; font-weight: 600; }
    .bad { color: #b3261e; font-weight: 600; }
    .banner { background: #fff3cd; border: 1px solid #e0c568; padding: 0.7rem 1rem;
              border-radius: 6px; margin: 1rem 0; }
    .note { color: #555; font-size: 0.82rem; }
    td.img img { max-width: 110px; }
    details summary { cursor: pointer; color: #b3261e; font-weight: 600; }
    details div { margin: 6px 0; }
    .mono { font-family: ui-monospace, monospace; font-size: 0.8rem; }
    """
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>vision-input-check report</title>",
        "<style>" + style + "</style></head><body>",
        "<h1>vision-input-check &mdash; " + e(provider_label) + "</h1>",
    ]
    if synthetic:
        parts.append(
            "<div class='banner'><strong>Synthetic replay.</strong> Every answer "
            "below is canned simulation data. The failures are seeded on purpose "
            "to demonstrate the harness; they say nothing about any real provider "
            "or model.</div>"
        )
    if seeded:
        items = ", ".join(e(f + " / " + v) for f, v, _k in seeded)
        parts.append(
            "<p class='note'>Seeded failures: " + items + " (expected regressions).</p>"
        )
    # --- gates ---
    gates = result.get("gates", [])
    parts.append("<h2>Gates</h2>")
    parts.append("<table><tr><th>gate</th><th>operator</th><th>threshold</th>"
                 "<th>observed</th><th>result</th></tr>")
    for g in gates:
        cls, _ = _STATUS[bool(g["passed"])]
        obs = "n/a" if g["observed"] is None else f'{g["observed"]:.4f}'
        parts.append(
            f"<tr><td class='mono'>{e(g['name'])}</td><td>{e(g['operator'])}</td>"
            f"<td class='mono'>{e(g['threshold'])}</td><td class='mono'>{obs}</td>"
            f"<td>{cls}</td></tr>"
            f"<tr><td colspan='5' class='note'>{e(g['description'])}</td></tr>"
        )
    parts.append("</table>")

    # --- strict metrics + denominators ---
    den = result.get("denominators", {})
    parts.append("<h2>Strict metrics</h2><table>")
    parts.append("<tr><th>metric</th><th>value</th></tr>")
    for name in ("identical_delta_min", "lossy_accuracy_min", "flaky_count"):
        val = result.get(name)
        val_s = "n/a" if val is None else f"{val:.4f}" if isinstance(val, float) else str(val)
        parts.append(f"<tr><td class='mono'>{e(name)}</td><td class='mono'>{val_s}</td></tr>")
    parts.append(
        f"<tr><td class='mono'>eligible_fixtures</td>"
        f"<td class='mono'>{e(den.get('eligible_fixtures'))} / "
        f"{e(den.get('total_fixtures'))}</td></tr>"
    )
    if result.get("exclusions"):
        parts.append(
            "<tr><td class='mono'>excluded_flaky</td><td class='mono'>"
            + e(", ".join(result["exclusions"])) + "</td></tr>"
        )
    parts.append("</table>")

    # --- aggregates per variant ---
    parts.append("<h2>Aggregate accuracy per variant</h2><table>")
    parts.append("<tr><th>variant</th><th>kind</th><th>eligible</th><th>accuracy</th>"
                 "<th>delta vs original</th></tr>")
    for name, agg in result.get("aggregates", {}).items():
        delta = (
            "baseline" if agg["delta_vs_baseline"] is None else f"{agg['delta_vs_baseline']:+.4f}"
        )
        acc = "n/a" if agg["accuracy"] is None else f"{agg['accuracy']:.4f}"
        parts.append(
            f"<tr><td class='mono'>{e(name)}</td><td>{e(agg['kind'])}</td>"
            f"<td>{e(agg['eligible_fixtures'])}</td><td class='mono'>{acc}</td>"
            f"<td class='mono'>{delta}</td></tr>"
        )
    parts.append("</table>")

    # --- fixtures x variants matrix ---
    variants = list(result.get("aggregates", {}).keys())
    parts.append("<h2>Fixtures &times; variants</h2><table>")
    header = "<tr><th>fixture</th><th>image</th>"
    header += "".join("<th class='mono'>" + e(v) + "</th>" for v in variants)
    parts.append(header + "</tr>")
    for fx in result.get("fixtures", []):
        parts.append(
            "<tr><td class='mono'>" + e(fx["fixture_id"]) + "</td>"
            "<td class='img'><img src='" + e(_thumbnail(image_paths[fx["fixture_id"]]))
            + "' alt=''></td>"
        )
        for v in fx["variants"]:
            good = sum(1 for p in v["passes"] if p)
            total = len(v["passes"])
            summary = str(good) + "/" + str(total)
            if good == total:
                parts.append("<td class='cell ok'>" + summary + "</td>")
                continue
            detail = "<details><summary>" + summary + "</summary>"
            detail += "<div class='note'>question: " + e(fx.get("question", "")) + "</div>"
            detail += "<div class='note'>expected: " + e(fx.get("expected", "")) + "</div>"
            for i, (a, p) in enumerate(zip(v["answers"], v["passes"])):
                badge = _STATUS[p][0]
                detail += "<div class='mono'>run " + str(i) + ": " + e(a) + " " + badge + "</div>"
            detail += "<div class='mono note'>request: provider="
            detail += e(result.get("provider", {}).get("kind", "?"))
            model = result.get("provider", {}).get("model")
            if model:
                detail += " model=" + e(model)
            detail += " media=" + e(media_type_for(v["variant"]))
            detail += "</div></details>"
            parts.append("<td>" + detail + "</td>")
        parts.append("</tr>")
    parts.append("</table>")

    # --- interpretation notes ---
    parts.append("<h2>How to read this report</h2><ul class='note'>")
    parts.append(
        "<li>Identical gates check that pixel-identical deliveries do not move "
        "answers. They do not detect provider-side preprocessing that applies "
        "to all inputs equally, because that affects original too.</li>"
    )
    parts.append(
        "<li>Lossy gates measure how far degraded delivery moves answers; "
        "lossy variants have no obligation to preserve answers.</li>"
    )
    parts.append(
        "<li>Repeating runs reduces uncertainty; it does not eliminate noise or "
        "prove causality. Accuracy over ~3 runs is descriptive, not statistically "
        "significant.</li>"
    )
    parts.append(
        "<li>Exact assertions suit narrow extraction tasks, not open-ended "
        "evaluation; there is no LLM judge.</li>"
    )
    parts.append(
        "<li>A replay/synthetic report proves the harness works, not the quality "
        "of any model.</li>"
    )
    parts.append("</ul>")
    parts.append("</body></html>")

    directory = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("".join(parts))
    return out_path
