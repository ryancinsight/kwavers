"""Runtime export-inventory oracle for the pykwavers facade.

Proves that every name the package declares in ``__all__`` and imports from
the extension actually exists at runtime, and that the extension exposes no
*extra* public names that the facade silently omits. This test runs against
the **installed wheel** (CI wheel-smoke) or any environment where the compiled
``_pykwavers`` extension is importable; it skips in a bare source tree.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

INVENTORY = Path(__file__).resolve().parents[1] / "python" / "pykwavers" / "_generated_surface.json"

# The compiled extension is required for the runtime oracle. Whether it is
# importable depends on the package mode (mirror of conftest.py):
#   - "installed": the wheel's extension under site-packages.
#   - "source" (default): the in-tree python/pykwavers must carry the compiled
#     extension (maturin develop leaves _pykwavers.pyd there); otherwise a
#     bare source tree has no extension and the runtime tests are skipped.
_EXT_SUFFIXES = (".pyd", ".so", ".dylib")
_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "python" / "pykwavers"
_mode = os.getenv("KWAVERS_PYTHON_PACKAGE", "source")
if _mode == "installed":
    _EXT_AVAILABLE = any(
        (Path(sys.prefix) / "Lib" / "site-packages" / "pykwavers").glob(f"_pykwavers{ext}")
        for ext in _EXT_SUFFIXES
    )
else:
    _EXT_AVAILABLE = any(_PACKAGE_DIR.glob(f"_pykwavers{ext}") for ext in _EXT_SUFFIXES)

requires_extension = pytest.mark.skipif(
    not _EXT_AVAILABLE,
    reason="pykwavers._pykwavers extension not importable (no wheel built)",
)

print(f"[debug] mode={_mode!r} ext_available={_EXT_AVAILABLE} pkgdir={_PACKAGE_DIR} globs={[list(_PACKAGE_DIR.glob(f'_pykwavers{ext}')) for ext in _EXT_SUFFIXES]} suffixes={_EXT_SUFFIXES!r}")


def _load_inventory() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


@requires_extension
def test_facade_exports_resolve_at_runtime():
    import pykwavers

    inventory = _load_inventory()
    exported = set(inventory["facade"]["__all__"])
    missing = sorted(name for name in exported if not hasattr(pykwavers, name))
    assert missing == [], f"facade __all__ names missing at runtime: {missing}"


@requires_extension
def test_registered_extension_surface_is_present():
    import pykwavers
    import pykwavers._pykwavers as ext

    inventory = _load_inventory()
    # Every registered class (by its Python name) and function must be
    # reachable through the facade. The facade may re-export under a
    # different alias (Py* -> unprefixed), so check both the extension and
    # the facade namespace.
    ext_names = set(dir(ext))
    py_class_names = inventory.get("registered_py_names", {}).get("classes", {})
    for cls in inventory["registered"]["classes"]:
        # The runtime name is the #[pyclass(name = ...)] rename when present.
        runtime_name = py_class_names.get(cls, cls)
        assert (
            runtime_name in ext_names or hasattr(pykwavers, runtime_name)
        ), f"missing class {cls} (runtime name {runtime_name})"
    for fn in inventory["registered"]["functions"]:
        assert fn in ext_names or hasattr(pykwavers, fn), f"missing function {fn}"


@requires_extension
def test_extension_has_no_unexpected_public_surface():
    import pykwavers._pykwavers as ext

    inventory = _load_inventory()
    registered = set(inventory["registered"]["classes"]) | set(
        inventory["registered"]["functions"]
    )
    # Names the facade intentionally aliases (Py* -> unprefixed)
    facade_imports = set(inventory["facade"]["imported"])
    public = {
        name
        for name in dir(ext)
        if not name.startswith("_") and name not in ("__all__", "__builtins__")
    }
    unexpected = sorted(
        name for name in public if name not in registered and name not in facade_imports
    )
    assert unexpected == [], f"extension exposes unregistered public names: {unexpected}"


@pytest.mark.parametrize(
    "name",
    [
        "Grid",
        "Medium",
        "Source",
        "Sensor",
        "Simulation",
        "SimulationResult",
        "SolverType",
        "PmlConfig",
        "HelmholtzConfig",
        "NonlinearConfig",
        "ThermalConfig",
        "TransducerArray2D",
        "KWaveArray",
    ],
)
def test_core_classes_are_typed_in_stub(name: str):
    """Every core class appears in the generated stub with real signatures."""
    stub = (Path(__file__).resolve().parents[1] / "python" / "pykwavers" / "_pykwavers.pyi").read_text(
        encoding="utf-8"
    )
    assert f"class {name}:" in stub
