import pytest

import pykwavers as kw


@pytest.fixture
def grid():
    return kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)


@pytest.fixture
def medium():
    return kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)


@pytest.fixture
def source(grid):
    return kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)


@pytest.fixture
def sensor():
    return kw.Sensor.point(position=(0.0016, 0.0016, 0.0016))
