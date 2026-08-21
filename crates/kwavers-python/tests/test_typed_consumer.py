"""Strict typed-consumer contract for the generated pykwavers surface.

A representative consumer imports public names from ``pykwavers`` and uses
them under mypy ``--strict``. This proves the installed ``.pyi`` stubs resolve
real names with concrete types (no ``Any`` placeholders). The test skips when
mypy is unavailable locally; the CI typed-consumer job installs it.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
FIXTURE = ROOT / "tests" / "typed_consumer_fixture.py"

FIXTURE_TEXT = '''\
"""Typed consumer that exercises the generated pykwavers surface."""

from typing import List, Optional, Tuple

import numpy as np

import pykwavers


def build_grid() -> pykwavers.Grid:
    grid = pykwavers.Grid(nx=64, ny=64, nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    return grid


def build_medium() -> pykwavers.Medium:
    return pykwavers.Medium.homogeneous(sound_speed=1500.0, density=1000.0)


def run_simulation() -> pykwavers.SimulationResult:
    grid = build_grid()
    medium = build_medium()
    source = pykwavers.Source.plane_wave(
        grid=grid, frequency=1.0e6, amplitude=1.0e5
    )
    sensor = pykwavers.Sensor.point(position=(0.01, 0.01, 0.01))
    sim = pykwavers.Simulation(grid, medium, source, sensor)
    result = sim.run(time_steps=10, dt=1.0e-8)
    n: int = result.num_sensors
    return result


def solver_kind() -> pykwavers.SolverType:
    return pykwavers.SolverType.PSTD


def array_param(x: np.ndarray) -> float:
    return float(x.sum())


# Duck-typed parameters resolved through the audited DUCK_TYPES table.
def duck_typed_usage(p0_2d: np.ndarray, signal_1d: np.ndarray,
                     mask: np.ndarray, alpha_field: np.ndarray) -> None:
    # Source.from_initial_pressure accepts a 2D or 3D f64 ndarray.
    source_a = pykwavers.Source.from_initial_pressure(p0_2d)
    _ = source_a.frequency
    # Source.from_mask accepts a 1D or 2D f64 signal.
    source_b = pykwavers.Source.from_mask(mask, signal_1d, frequency=1.0e6)
    _ = source_b
    # Medium heterogeneous constructor alpha_power accepts float or ndarray.
    medium_scalar = pykwavers.Medium(
        sound_speed=np.full((4, 4, 4), 1500.0),
        density=np.full((4, 4, 4), 1000.0),
        alpha_power=0.8,
    )
    medium_field = pykwavers.Medium(
        sound_speed=np.full((4, 4, 4), 1500.0),
        density=np.full((4, 4, 4), 1000.0),
        alpha_power=alpha_field,
    )
    _ = medium_scalar, medium_field


def duck_typed_simulation_args(source_single: "pykwavers.Source",
                               sensor: "pykwavers.Sensor",
                               grid: "pykwavers.Grid",
                               medium: "pykwavers.Medium") -> None:
    # Simulation.__init__ source accepts Source | TransducerArray2D |
    # list of either; sensor accepts Sensor | TransducerArray2D.
    sim_single = pykwavers.Simulation(grid, medium, source_single, sensor)
    sim_list = pykwavers.Simulation(grid, medium, [source_single], sensor)
    _ = sim_single, sim_list


def typed_usage() -> Tuple[int, Optional[float], str]:
    grid = build_grid()
    dx: float = grid.dx
    nx: int = grid.nx
    solver = solver_kind()
    label: str = str(solver)
    medium = build_medium()
    c: float = medium.sound_speed
    return nx, dx, label
'''


@pytest.mark.skipif(shutil.which("mypy") is None, reason="mypy not installed")
def test_typed_consumer_passes_strict_mypy():
    if not FIXTURE.exists():
        FIXTURE.write_text(FIXTURE_TEXT, encoding="utf-8")
    # Run mypy against the package sources + the fixture using the project's
    # configured [tool.mypy] settings (python_version 3.8, strict).
    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "--strict",
        "--python-version",
        "3.8",
        str(PYTHON_SOURCE / "pykwavers"),
        str(FIXTURE),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert proc.returncode == 0, f"mypy failed:\n{proc.stdout}\n{proc.stderr}"


def test_stub_has_no_unnamed_object_parameters():
    """No parameter in the generated stub may carry a bare ``object`` type.

    Duck-typed PyAny parameters must resolve through the audited DUCK_TYPES
    table; heterogeneous container values (Dict[str, object] values,
    Tuple[object, ...]) are honest and remain permitted.
    """
    import ast

    stub = PYTHON_SOURCE / "pykwavers" / "_pykwavers.pyi"
    tree = ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arg_nodes = list(node.args.args) + list(node.args.kwonlyargs)
            arg_nodes += list(node.args.posonlyargs)
            for arg in arg_nodes:
                if isinstance(arg.annotation, ast.Name) and arg.annotation.id == "object":
                    offenders.append(f"{node.name}:{arg.arg}")
    assert not offenders, f"bare object parameters remain: {offenders}"
