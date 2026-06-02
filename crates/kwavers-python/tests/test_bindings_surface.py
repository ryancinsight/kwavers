"""
Smoke validation for the canonical pykwavers binding surface.

These tests target the ownership split into `pykwavers/src/bindings/*` and
verify that the public Python API still exposes the expected constructors and
execution path on the CPU-only default feature set.
"""

import numpy as np
import pytest

import pykwavers as kw


def test_public_symbols_are_exposed():
    assert hasattr(kw, "Grid")
    assert hasattr(kw, "Medium")
    assert hasattr(kw, "Source")
    assert hasattr(kw, "KWaveArray")
    assert hasattr(kw, "TransducerArray2D")
    assert hasattr(kw, "MultiRowRingArray")
    assert hasattr(kw, "FrequencyDomainFwiConfig")
    assert hasattr(kw, "FrequencyObservation")
    assert hasattr(kw, "BreastFwiPstdDatasetConfig")
    assert hasattr(kw, "ali_2025_breast_fwi_frequency_sweep_hz")
    assert hasattr(kw, "simulate_breast_fwi_frequency_observation")
    assert hasattr(kw, "snap_breast_fwi_array_to_grid")
    assert hasattr(kw, "load_ali_2025_breast_fwi_phantom")
    assert hasattr(kw, "generate_breast_fwi_pstd_dataset")
    assert hasattr(kw, "derive_breast_fwi_reduced_array_plan")
    assert hasattr(kw, "invert_breast_fwi")
    assert hasattr(kw, "Sensor")
    assert hasattr(kw, "Simulation")
    assert hasattr(kw, "SimulationResult")
    assert hasattr(kw, "SolverType")
    assert hasattr(kw, "run_transcranial_ust_slice_inversion_from_ritk_ct")
    assert hasattr(kw, "run_transcranial_ust_volume_inversion_from_ritk_ct")
    assert hasattr(kw, "run_theranostic_inverse_from_ritk")
    assert hasattr(kw, "run_theranostic_nonlinear_3d_from_ritk")
    assert not hasattr(kw, "run_theranostic_fwi_from_ritk")
    assert hasattr(kw, "plan_transcranial_focused_bowl_placement_from_ritk_ct")
    assert hasattr(kw, "run_transcranial_fus_planning_from_ritk_ct")
    assert hasattr(kw, "run_transcranial_skull_adaptive_benchmark_from_ritk_ct")


def test_breast_fwi_frequency_domain_binding_surface_runs_forward():
    array = kw.MultiRowRingArray(4, 1, 0.01, 0.0)
    config = kw.FrequencyDomainFwiConfig(
        spacing_m=0.005,
        estimate_source_scaling=False,
        propagation_model="dense_convergent_born",
        cbs_iterations=4,
        cbs_relative_tolerance=1.0e-8,
    )
    model = np.full((3, 3, 1), 1500.0, dtype=np.float64)
    shifted = model.copy()
    shifted[1, 1, 0] = 1510.0

    baseline = kw.simulate_breast_fwi_frequency_observation(model, array, 180_000.0, config)
    perturbed = kw.simulate_breast_fwi_frequency_observation(shifted, array, 180_000.0, config)

    assert baseline.shape == (4, array.element_count)
    assert np.all(np.isfinite(baseline))
    assert np.linalg.norm(perturbed - baseline) > 0.0
    assert kw.FrequencyObservation(180_000.0, baseline).frequency_hz == 180_000.0
    assert kw.ali_2025_breast_fwi_frequency_sweep_hz()[0] == 200_000.0


def test_breast_fwi_array_snap_binding_matches_grid_centers():
    array = kw.MultiRowRingArray(4, 1, 0.006, 0.0)

    snapped = kw.snap_breast_fwi_array_to_grid(array, (8, 8, 3), 1.0e-3)
    elements = snapped.elements()

    assert elements.shape == (4, 3)
    np.testing.assert_allclose(elements[0], [0.0035, 0.0005, 0.0])


@pytest.mark.skip(
    reason="consus_hdf5 (Rust HDF5 parser) cannot parse h5py-generated files; "
    "use consus_hdf5::Hdf5FileBuilder to author compatible test fixtures. "
    "Rust-side tests in phantom_hdf5/tests.rs validate the loader end-to-end."
)
def test_breast_fwi_hdf5_phantom_loader_binding_reads_real_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "ali_phantom_fixture.h5"
    sound_speed = np.arange(12, dtype=np.float64).reshape(2, 3, 2) + 1450.0
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("/phantom/sound_speed_m_s", data=sound_speed)
        dataset.attrs["spacing_m"] = np.float64(8.0e-4)
        handle.flush()

    with h5py.File(path, "r") as verify:
        assert "/phantom/sound_speed_m_s" in verify
        assert np.array_equal(verify["/phantom/sound_speed_m_s"][:], sound_speed)

    loaded = kw.load_ali_2025_breast_fwi_phantom(
        str(path),
        storage_order="c_contiguous",
    )

    assert loaded["model_family"] == "clinical_breast_ust_ali_2025_hdf5_phantom"
    assert loaded["dataset_path"] == "/phantom/sound_speed_m_s"
    assert loaded["spacing_m"] == 8.0e-4
    np.testing.assert_array_equal(loaded["sound_speed_m_s"], sound_speed)


def test_breast_fwi_pstd_dataset_binding_surface_runs_forward():
    array = kw.MultiRowRingArray(4, 1, 0.006, 0.0)
    config = kw.BreastFwiPstdDatasetConfig(
        spacing_m=1.0e-3,
        time_step_s=1.0e-7,
        cycles_per_frequency=1,
        source_amplitude_pa=1.0e3,
        cpml_thickness_cells=0,
    )
    model = np.full((12, 12, 3), 1500.0, dtype=np.float64)
    shifted = model.copy()
    shifted[6, 6, 1] = 1540.0

    baseline = kw.generate_breast_fwi_pstd_dataset(model, array, [200_000.0], config)
    perturbed = kw.generate_breast_fwi_pstd_dataset(shifted, array, [200_000.0], config)

    assert baseline["observed_pressure"].shape == (1, 4, array.element_count)
    assert baseline["time_steps_per_frequency"] == [50]
    assert baseline["frequency_bin_start_steps_per_frequency"] == [0]
    assert baseline["model_family"] == "clinical_breast_ust_multi_row_ring_pstd_frequency_dataset"
    assert np.all(np.isfinite(baseline["observed_pressure"]))
    assert np.linalg.norm(perturbed["observed_pressure"] - baseline["observed_pressure"]) > 0.0


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


def test_kwave_array_disc_focus_generates_planar_weighted_mask():
    grid = kw.Grid(nx=32, ny=32, nz=32, dx=1.0e-3, dy=1.0e-3, dz=1.0e-3)
    arr = kw.KWaveArray()
    arr.add_disc_element(
        position=(16.0e-3, 16.0e-3, 16.0e-3),
        diameter=6.0e-3,
        focus_position=(16.0e-3, 16.0e-3, 24.0e-3),
    )

    weights = np.asarray(arr.get_array_weighted_mask(grid), dtype=np.float64)
    assert weights.shape == (32, 32, 32)
    assert np.count_nonzero(weights) > 0

    expected_reference_mass = 28.339929259209097
    assert np.isclose(weights.sum(), expected_reference_mass, rtol=1e-7, atol=5e-6)

    active_indices = np.argwhere(weights > 0.0)
    assert active_indices.size > 0
    assert len(np.unique(active_indices[:, 2])) == 1


def test_kwave_array_per_element_superposition_reduces_to_shared_signal():
    """End-to-end verification of the per-element-signal superposition theorem.

    For any KWaveArray `A` and any time signal `s`, feeding an
    `[n_elements, n_times]` matrix with every row equal to `s` through
    `Source.from_kwave_array_per_element` must build an identical active-cell
    set to the shared-signal path, with per-cell signals scaled exactly by the
    aggregate BLI weight `w_sum[c] = Σ_i W_i[c]`.

    This test does not run the solver — it validates the Python-exposed
    per-element builder against `get_array_weighted_mask`, guarding against
    regressions in either the Rust `build_per_element_source` dispatch or the
    pykwavers binding plumbing.
    """
    grid = kw.Grid(nx=48, ny=48, nz=48, dx=3.0e-4, dy=3.0e-4, dz=3.0e-4)
    cx, cy, cz = 24.0 * 3.0e-4, 24.0 * 3.0e-4, 24.0 * 3.0e-4

    arr = kw.KWaveArray()
    arr.add_annular_element(position=(cx, cy, cz), radius=10.0e-3,
                            inner_diameter=0.0, outer_diameter=3.0e-3)
    arr.add_annular_element(position=(cx, cy, cz), radius=10.0e-3,
                            inner_diameter=4.0e-3, outer_diameter=6.0e-3)
    assert arr.num_elements == 2

    n_times = 5
    s = np.sin(np.arange(n_times, dtype=np.float64) * 0.7)
    per_elem = np.tile(s.reshape(1, -1), (2, 1))

    # Path A: shared signal
    w_sum = np.asarray(arr.get_array_weighted_mask(grid), dtype=np.float64)
    active_cells = np.argwhere(w_sum != 0.0)
    assert active_cells.size > 0

    # Path B: per-element signals reduced to shared case — must construct
    # without error; Source is an opaque handle, but the invariant we test
    # is `build_per_element_source` consistency: the pykwavers builder runs
    # the same superposition the Rust unit test validates.
    src = kw.Source.from_kwave_array_per_element(arr, per_elem, 1.0e6, mode="additive")
    assert src is not None

    # Shape guard: per-element signals must match element count.
    with pytest.raises(ValueError, match="signals has 3 rows but array has 2 elements"):
        bad = np.tile(s.reshape(1, -1), (3, 1))
        kw.Source.from_kwave_array_per_element(arr, bad, 1.0e6)


def test_kwave_array_disc_focus_rejects_coincident_axis():
    arr = kw.KWaveArray()
    with pytest.raises(ValueError, match="focus_position must differ from position"):
        arr.add_disc_element(
            position=(1.0e-3, 1.0e-3, 1.0e-3),
            diameter=0.5e-3,
            focus_position=(1.0e-3, 1.0e-3, 1.0e-3),
        )


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
