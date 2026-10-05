"""The bundled JSON Schema matches what the CLI actually emits."""

import json
from pathlib import Path

from vision_input_check.cli import main
from vision_input_check.types import SCHEMA_VERSION

SCHEMA_PATH = (
    Path(__file__).parents[1] / "src" / "vision_input_check" / "vision_check_result.schema.json"
)


def test_schema_file_is_valid_json():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == SCHEMA_VERSION


def test_demo_result_conforms_to_schema_required_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        main(["demo", "--json"])
    result = json.loads(buffer.getvalue())
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert set(schema["required"]) <= set(result)
    for fixture in result["fixtures"]:
        assert set(schema["properties"]["fixtures"]["items"]["required"]) <= set(fixture)
        for variant in fixture["variants"]:
            assert set(
                schema["properties"]["fixtures"]["items"]["properties"]["variants"]["items"]["required"]
            ) <= set(variant)
    for gate in result["gates"]:
        assert set(schema["properties"]["gates"]["items"]["required"]) <= set(gate)
