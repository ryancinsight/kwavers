"""Fresh-interpreter oracle for optional `pykwavers` submodules."""

from __future__ import annotations

import importlib
import importlib.abc
import sys

OPTIONAL_SUBMODULES = (
    "comparison",
    "kwave_bridge",
    "kwave_python_bridge",
)


class BlockMatplotlib(importlib.abc.MetaPathFinder):
    """Make the comparison extra unavailable without mutating the environment."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "matplotlib" or fullname.startswith("matplotlib."):
            message = "No module named 'matplotlib'"
            raise ModuleNotFoundError(message, name="matplotlib")
        return None


def verify_base_import() -> None:
    """Verify that importing the base package does not touch optional modules."""
    sys.meta_path.insert(0, BlockMatplotlib())
    import pykwavers

    assert pykwavers.Grid is not None
    for name in OPTIONAL_SUBMODULES:
        assert name not in vars(pykwavers)
        assert f"pykwavers.{name}" not in sys.modules
        assert name not in pykwavers.__all__


def verify_missing_extra() -> None:
    """Verify that explicit submodule imports preserve the dependency error."""
    sys.meta_path.insert(0, BlockMatplotlib())
    import pykwavers

    try:
        importlib.import_module("pykwavers.comparison")
    except ModuleNotFoundError as error:
        assert error.name == "matplotlib"
    else:
        message = "comparison imported without its required extra"
        raise AssertionError(message)

    try:
        from pykwavers import comparison  # noqa: F401
    except ModuleNotFoundError as error:
        assert error.name == "matplotlib"
    else:
        message = "from-import succeeded without its required extra"
        raise AssertionError(message)

    assert "comparison" not in vars(pykwavers)
    assert "pykwavers.comparison" not in sys.modules


def main() -> None:
    """Run the selected contract in a fresh interpreter."""
    if len(sys.argv) != 2:
        message = "usage: optional_submodule_contract.py {base|access}"
        raise SystemExit(message)
    contract = sys.argv[1]
    if contract == "base":
        verify_base_import()
    elif contract == "access":
        verify_missing_extra()
    else:
        message = f"unknown contract: {contract}"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
