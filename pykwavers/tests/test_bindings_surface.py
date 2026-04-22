"""
Smoke validation for the canonical pykwavers binding surface.

These tests target the ownership split into `pykwavers/src/bindings/*` and
verify that the public Python API still exposes the expected constructors and
execution path on the CPU-only default feature set.
"""

import numpy as np

import pykwavers as kw


def test_public_symbols_are_exposed():
    assert hasattr(kw, "Grid")
    assert hasattr(kw, "Medium")
    assert hasattr(kw, "Source")
    assert hasattr(kw, "KWaveArray")
    assert hasattr(kw, "TransducerArray2D")
    assert hasattr(kw, "Sensor")
    assert hasattr(kw, "Simulation")
    assert hasattr(kw, "SimulationResult")
    assert hasattr(kw, "SolverType")


def test_heterogeneous_medium_and_mask_sensor_surface():
    c = np.ones((16, 16, 16), dtype=np.float64) * 1500.0
    c[8:, :, :] = 1800.0
    rho = np.ones((16, 16, 16), dtype=np.float64) * 1000.0
    medium = kw.Medium(sound_speed=c, density=rho)

    assert not medium.is_homogeneous
    assert medium.sound_speed == 1800.0
    assert "heterogeneous" in repr(medium).lower()

    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[4, 8, 8] = True
    mask[12, 8, 8] = True
    sensor = kw.Sensor.from_mask(mask)

    assert sensor.sensor_type == "mask"
    assert sensor.num_sensors == 2


def test_kwave_array_and_transducer_surface():
    arr = kw.KWaveArray()
    arr.add_disc_element(position=(1.0e-3, 1.0e-3, 0.0), diameter=0.5e-3)
    arr.add_rect_element(position=(2.0e-3, 1.0e-3, 0.0), dims=(0.5e-3, 0.5e-3, 1.0e-3))
    assert arr.num_elements == 2

    transducer = kw.TransducerArray2D(
        number_elements=8,
        element_width=0.3e-3,
        element_length=2.0e-3,
        element_spacing=0.4e-3,
        sound_speed=1540.0,
        frequency=1.0e6,
    )
    transducer.set_focus_distance(15.0e-3)
    transducer.set_transmit_apodization("Hanning")

    assert transducer.number_elements == 8
    assert transducer.focus_distance == 15.0e-3
    assert transducer.transmit_apodization == "Hanning"


def test_simulation_cpu_surface_runs_end_to_end():
    grid = kw.Grid(nx=16, ny=16, nz=16, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    source = kw.Source.point(position=(0.2e-3, 0.8e-3, 0.8e-3), frequency=1.0e6, amplitude=1.0e5)

    mask = np.zeros((16, 16, 16), dtype=bool)
    mask[8, 8, 8] = True
    mask[10, 8, 8] = True
    sensor = kw.Sensor.from_mask(mask)
    sensor.set_record(["p", "p_max", "p_rms"])

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.FDTD, pml_size=4)
    result = sim.run(time_steps=12, dt=1.0e-8)

    assert result.time_steps == 12
    assert result.sensor_data.shape == (2, 12)
    assert result.sensor_data_shape == (2, 12)
    assert result.num_sensors == 2
    assert result.p_max is not None
    assert result.p_rms is not None
    assert np.all(np.isfinite(result.sensor_data))
    assert np.all(np.isfinite(result.time))


def test_simulation_exposes_kspace_correction_mode():
    grid = kw.Grid(nx=8, ny=8, nz=8, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    source = kw.Source.point(position=(0.2e-3, 0.8e-3, 0.8e-3), frequency=1.0e6, amplitude=1.0e5)
    sensor = kw.Sensor.grid()

    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.FDTD, pml_size=4)
    assert sim.kspace_correction == "none"

    sim.set_kspace_correction("spectral")
    assert sim.kspace_correction == "spectral"


def test_initial_pressure_accepts_2d_and_lifts_to_volume():
    p0 = np.zeros((8, 12), dtype=np.float64)
    p0[3, 5] = 2.5

    source = kw.Source.from_initial_pressure(p0)
    lifted = np.asarray(source.initial_pressure)

    assert source.source_type == "p0"
    assert lifted.shape == (8, 12, 1)
    assert lifted[3, 5, 0] == 2.5
    assert np.max(np.abs(lifted)) == 2.5
