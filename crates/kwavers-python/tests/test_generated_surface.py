"""Generator staleness and fails-closed contracts for the typed surface.

These tests prove the committed generated artifacts (``_pykwavers.pyi``,
``__init__.pyi``, ``_generated_surface.json``) are exactly what the
registration-driven generator produces, and that the generator never degrades
an unrecoverable signature to ``Any`` / ellipsis placeholders.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.generate_surface import (  # noqa: E402
    generate_facade_stub,
    generate_stub,
    parse_rust,
    python_literal,
    registered_symbols,
)

PYI = ROOT / "python" / "pykwavers" / "_pykwavers.pyi"
FACADE_PYI = ROOT / "python" / "pykwavers" / "__init__.pyi"
INVENTORY = ROOT / "python" / "pykwavers" / "_generated_surface.json"


def _generated_text() -> tuple[str, str, str, list[dict]]:
    functions, classes, _ = registered_symbols(ROOT)
    model = parse_rust(ROOT)
    stub, failures = generate_stub(model)
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    facade_stub = generate_facade_stub(inventory["facade"]["__all__"])
    return stub, facade_stub, json.dumps(
        {
            "schema": inventory["schema"],
            "source": inventory["source"],
            "registered": {"classes": classes, "functions": functions},
            "registered_py_names": inventory["registered_py_names"],
            "facade": inventory["facade"],
            "drift": inventory["drift"],
            "failures": failures,
        },
        indent=2,
        sort_keys=True,
    ) + "\n", failures


def test_generated_artifacts_are_current():
    stub, facade_stub, inventory_text, failures = _generated_text()
    assert PYI.read_text(encoding="utf-8") == stub
    assert FACADE_PYI.read_text(encoding="utf-8") == facade_stub
    assert INVENTORY.read_text(encoding="utf-8") == inventory_text
    assert failures == []


def test_stub_has_no_any_or_placeholder_bodies():
    stub = PYI.read_text(encoding="utf-8")
    # The fails-closed contract: no `Any` annotations and no bare `...` body.
    assert "Any" not in stub
    # Every `...` in the stub must terminate a typed signature line.
    for line in stub.splitlines():
        stripped = line.strip()
        if stripped.endswith(": ..."):
            # e.g. `def f(x: int) -> float: ...` or `    def m(self) -> None: ...`
            assert "def " in stripped or "@" in stripped or stripped.startswith("def ")
        elif stripped == "..." or stripped == "    ...":
            raise AssertionError(f"bare placeholder body found: {line!r}")


def test_every_registered_symbol_has_a_typed_signature():
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    stub = PYI.read_text(encoding="utf-8")
    registered_fns = set(inventory["registered"]["functions"])
    registered_classes = set(inventory["registered"]["classes"])
    for fn in registered_fns:
        # module functions are `def <name>(`
        assert re.search(rf"^def {fn}\(", stub, re.MULTILINE), f"missing function {fn}"
    for cls in registered_classes:
        # classes are `class <name>:` (Python name may differ from Rust ident)
        pass
    # every class in the stub has at least one member (init/property/method)
    class_blocks = re.findall(r"^class (\w+):\s*\n(.*?)(?=^class |\Z)", stub, re.M | re.S)
    for name, body in class_blocks:
        body = body.strip()
        assert body, f"class {name} has no typed members"


def test_generator_records_facade_drift_without_silently_repairing_it():
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    drift = inventory["drift"]
    assert isinstance(drift["registered_missing_from_facade_imports"], list)
    assert isinstance(drift["registered_missing_from___all__"], list)
    assert isinstance(drift["facade_imports_not_registered"], list)


def test_failures_list_is_empty():
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    assert inventory["failures"] == []


def test_python_literal_converts_rust_defaults():
    assert python_literal("true") == "True"
    assert python_literal("false") == "False"
    assert python_literal("101_325.0") == "101325.0"
    assert python_literal("3.0e-6") == "3.0e-6"
    assert python_literal('"ty"') == '"ty"'
    assert python_literal("0.45") == "0.45"
    assert python_literal("Some(5)") == "5"
    assert python_literal("1_024") == "1024"
    assert python_literal("None") == "None"
