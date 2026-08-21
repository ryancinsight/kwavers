"""Typed facade for the registered core ``pykwavers`` classes."""

from types import ModuleType

from ._pykwavers import (
    Grid,
    HelmholtzConfig,
    Medium,
    NonlinearConfig,
    PmlConfig,
    Sensor,
    Simulation,
    SimulationResult,
    SolverType,
    Source,
    ThermalConfig,
)

_pykwavers: ModuleType

__all__ = [
    "Grid",
    "HelmholtzConfig",
    "Medium",
    "NonlinearConfig",
    "PmlConfig",
    "Sensor",
    "Simulation",
    "SimulationResult",
    "SolverType",
    "Source",
    "ThermalConfig",
]
