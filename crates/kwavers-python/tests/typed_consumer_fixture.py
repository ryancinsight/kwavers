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


class _Reading:
    """An object carrying the Aequitas quantity protocol.

    Deliberately not a `pyaequitas` import: a dimensioned parameter is typed
    structurally, so a consumer type-checks against it without depending on
    that package at all.
    """

    @property
    def __aequitas_base__(self) -> float:
        return 2.5e-6

    @property
    def __aequitas_dimension__(
        self,
    ) -> Tuple[Tuple[int, int, int, int, int, int, int], str]:
        return ((1, 0, 0, 0, 0, 0, 0), "base")


def dimensioned_arguments() -> Tuple[float, float]:
    # Both forms are accepted: a float in canonical SI base units, which is
    # the contract these signatures always had, and a protocol quantity.
    from_floats = pykwavers.mems_clamped_plate_resonance(
        1.7e11, 2.0e-6, 0.28, 3100.0, 4.0e-5
    )
    length = _Reading()
    from_quantity = pykwavers.mems_clamped_plate_resonance(
        1.7e11, length, 0.28, 3100.0, length
    )
    return from_floats, from_quantity
