"""
pykwavers: Python Bindings for kwavers Ultrasound Simulation Library

A Rust-backed Python library for acoustic wave simulation with an API
compatible with k-Wave/k-wave-python for direct comparison and validation.

## Quick Start

```python
import pykwavers as kw
import numpy as np

# Create computational grid (similar to kWaveGrid)
grid = kw.Grid(nx=128, ny=128, nz=128, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)

# Define acoustic medium (similar to k-Wave medium struct)
medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)

# Create acoustic source
source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)

# Create sensor for field recording
sensor = kw.Sensor.point(position=(0.01, 0.01, 0.01))

# Run simulation
sim = kw.Simulation(grid, medium, source, sensor)
result = sim.run(time_steps=1000, dt=1e-8)

# Access results
print(f"Sensor data shape: {result.sensor_data.shape}")
print(f"Final time: {result.final_time:.2e} s")
```

## API Design Philosophy

The API mirrors k-Wave's structure for ease of comparison:
- **Grid**: Computational domain (equivalent to `kWaveGrid`)
- **Medium**: Acoustic properties (equivalent to k-Wave `medium` struct)
- **Source**: Wave excitation (equivalent to k-Wave `source` struct)
- **Sensor**: Field recording (equivalent to k-Wave `sensor` struct)
- **Simulation**: Main orchestrator (equivalent to `kspaceFirstOrder3D`)

## Architecture

Following Clean Architecture principles:
- **Presentation Layer**: Python API (this package)
- **Domain Layer**: Core kwavers library (Rust)
- **Dependency Direction**: Python → Rust (unidirectional)

## Mathematical Foundations

- **Wave Equation**: ∂²p/∂t² = c²∇²p + source terms
- **Discretization**: FDTD (2nd/4th/6th/8th order) or PSTD (spectral)
- **Stability**: CFL condition dt ≤ (dx/c_max)/√3
- **Boundaries**: PML (Perfectly Matched Layers)
- **Absorption**: Power-law α(ω) = α₀|ω|^y

## References

1. Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for simulation
   and reconstruction of photoacoustic wave fields." J. Biomed. Opt., 15(2).
2. kwavers architecture documentation
3. k-wave-python documentation

Author: Ryan Clanton PhD (@ryancinsight)
License: MIT
Repository: https://github.com/ryancinsight/kwavers
"""

import importlib.machinery
import importlib.util
import os
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(_REPO_ROOT))
    for _runtime_dir in (
        Path(sys.base_prefix) / "bin",
        Path(sys.exec_prefix) / "bin",
        Path(sys.executable).parent,
    ):
        if _runtime_dir.exists():
            os.add_dll_directory(str(_runtime_dir))
    _pkg_stable_abi = Path(__file__).with_name("libpython3.dll")
    for _python3_dll in (
        Path(sys.base_prefix) / "libpython3.dll",
        Path(sys.base_prefix) / "bin" / "libpython3.dll",
        Path(sys.exec_prefix) / "libpython3.dll",
        Path(sys.exec_prefix) / "bin" / "libpython3.dll",
    ):
        if _python3_dll.exists():
            if (
                not _pkg_stable_abi.exists()
                or _pkg_stable_abi.stat().st_size != _python3_dll.stat().st_size
            ):
                try:
                    shutil.copy2(_python3_dll, _pkg_stable_abi)
                except PermissionError:
                    pass
            break

# Import Rust extension module
def _newer_local_extension() -> Path | None:
    """Return a newer workspace-built extension when the package copy is stale.

    The in-tree Python package is used during book generation and pytest runs.
    On Windows, `maturin develop` can leave `python/pykwavers/_pykwavers.pyd`
    older than `target/release/pykwavers.dll`; importing the stale package copy
    then fails before tests can set `PYKWAVERS_EXTENSION_PATH`.  The explicit
    environment override still wins.  This fallback only applies inside a
    source checkout where target artifacts exist and are newer than the package
    extension.
    """
    if os.name != "nt":
        return None
    package_extension = Path(__file__).with_name("_pykwavers.pyd")
    repo_root = _REPO_ROOT
    candidates = (
        repo_root / "target" / "release" / "pykwavers.dll",
        repo_root / "target" / "maturin" / "pykwavers.dll",
        repo_root / "target" / "debug" / "pykwavers.dll",
    )
    package_mtime = package_extension.stat().st_mtime if package_extension.exists() else -1.0
    available = [path for path in candidates if path.exists()]
    newer = [path for path in available if path.stat().st_mtime > package_mtime]
    if not newer:
        return None
    return max(newer, key=lambda path: path.stat().st_mtime)


_extension_override = os.getenv("PYKWAVERS_EXTENSION_PATH")
if not _extension_override:
    local_extension = _newer_local_extension()
    if local_extension is not None:
        _extension_override = str(local_extension)
if _extension_override:
    _extension_path = Path(_extension_override).expanduser().resolve()
    if os.name == "nt":
        _extension_dir = _extension_path.parent
        _stable_abi_dll = _extension_dir / "libpython3.dll"
        _python_candidates = (
            Path(sys.base_prefix) / "libpython3.dll",
            Path(sys.base_prefix) / "bin" / "libpython3.dll",
            Path(sys.base_prefix) / "python3.dll",
            Path(sys.base_prefix) / "bin" / "python3.dll",
        )
        for _python3_dll in _python_candidates:
            if _python3_dll.exists():
                if (
                    not _stable_abi_dll.exists()
                    or _stable_abi_dll.stat().st_size != _python3_dll.stat().st_size
                ):
                    try:
                        shutil.copy2(_python3_dll, _stable_abi_dll)
                    except PermissionError:
                        if (
                            not _stable_abi_dll.exists()
                            or _stable_abi_dll.stat().st_size != _python3_dll.stat().st_size
                        ):
                            _runtime_dir = _extension_dir / f"_pykwavers_runtime_{os.getpid()}"
                            _runtime_dir.mkdir(parents=True, exist_ok=True)
                            _runtime_extension_path = _runtime_dir / _extension_path.name
                            shutil.copy2(_extension_path, _runtime_extension_path)
                            shutil.copy2(_python3_dll, _runtime_dir / "libpython3.dll")
                            _extension_path = _runtime_extension_path
                            _extension_dir = _runtime_dir
                            _stable_abi_dll = _runtime_dir / "libpython3.dll"
                break
        os.add_dll_directory(str(_extension_dir))
        # Also add the package directory: GNU-toolchain DLLs (libstdc++, libgcc,
        # libwinpthread) are co-located with _pykwavers.pyd, not with the target dll.
        _pkg_dir_override = str(Path(__file__).parent)
        if _pkg_dir_override != str(_extension_dir):
            os.add_dll_directory(_pkg_dir_override)
    _loader = importlib.machinery.ExtensionFileLoader(
        f"{__name__}._pykwavers",
        str(_extension_path),
    )
    _spec = importlib.util.spec_from_loader(
        f"{__name__}._pykwavers",
        _loader,
        origin=str(_extension_path),
    )
    if _spec is None:
        raise ImportError(f"Failed to create module spec for {_extension_path}")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[f"{__name__}._pykwavers"] = _module
    _loader.exec_module(_module)
elif os.name == "nt":
    # The override path was not taken (pyd is current), but libpython3.dll lives
    # next to the pyd.  Add the package directory so Windows DLL search finds it.
    _pkg_dir = str(Path(__file__).parent)
    os.add_dll_directory(_pkg_dir)

# Import Python submodules for comparison and validation
from . import comparison, kwave_bridge, kwave_python_bridge
from .parity_targets import PARITY_THRESHOLDS, evaluate_parity
from ._pykwavers import (
    # Core classes
    Grid,
    GpuPstdSession,
    Medium,
    Sensor,
    Simulation,
    SimulationResult,
    SolverType,
    Source,
    KWaveArray,
    TransducerArray2D,
    MultiRowRingArray,
    FrequencyDomainFwiConfig,
    FrequencyObservation,
    BreastFwiPstdDatasetConfig,
    ali_2025_breast_fwi_frequency_sweep_hz,
    simulate_breast_fwi_frequency_observation,
    snap_breast_fwi_array_to_grid,
    load_ali_2025_breast_fwi_phantom,
    generate_breast_fwi_pstd_dataset,
    simulate_breast_fwi_pstd_finite_window_born_observation,
    diagnose_breast_fwi_homogeneous_direct_field,
    diagnose_breast_fwi_observation_pair,
    breast_fwi_scaled_observation_residual_metrics,
    breast_fwi_source_channel_residual_diagnostics,
    breast_fwi_source_receiver_mask,
    breast_fwi_passive_receiver_mask,
    breast_fwi_source_excitation_diagnostics,
    breast_fwi_sine_frequency_bin_coefficient,
    breast_fwi_acquisition_identifiability,
    breast_fwi_reconstruction_metrics,
    breast_fwi_table1_parity,
    breast_fwi_operator_equivalence_diagnostics,
    breast_fwi_scattering_increment_diagnostics,
    prepare_breast_fwi_reduced_phantom,
    derive_breast_fwi_reduced_array_geometry,
    derive_breast_fwi_reduced_array_plan,
    invert_breast_fwi,
    # Thermal diffusion / Pennes bioheat (KWaveDiffusion equivalent)
    ThermalSimulation,
    ThermalResult,
    # 3-D residual cavitation-gas (lacuna) void-fraction field
    ResidualGasField,
    # Phase 22: PID, Registration, Bubble Field
    PIDController,
    BubbleField,
    # Field-surrogate cache (cached PSTD focal kernels for planners)
    FocalKernel,
    KernelCube,
    place_kernel_at_focus,
    plan_abdominal_array_placement_from_ritk_ct,
    plan_transcranial_focused_bowl_placement_from_ritk_ct,
    run_transcranial_fus_planning_from_ritk_ct,
    run_transcranial_skull_adaptive_benchmark_from_ritk_ct,
    run_transcranial_ust_slice_inversion_from_ritk_ct,
    run_transcranial_ust_volume_inversion_from_ritk_ct,
    load_ct_nifti,
    run_theranostic_inverse_from_ritk,
    run_theranostic_nonlinear_3d_from_ritk,
    resample_to_target_grid,
    kspace_line_recon,
    time_reversal_reconstruction,
    passive_acoustic_map_das,
    beamform_image_delay_and_sum,
    simulate_receive_rf,
    # Signal generation
    tone_burst,
    create_cw_signals,
    get_win,
    # Geometry
    make_disc,
    make_ball,
    make_sphere,
    make_circle,
    make_line,
    # Unit conversion
    db2neper,
    neper2db,
    freq2wavenumber,
    hounsfield2density,
    hounsfield2soundspeed,
    # Water properties (temperature-dependent)
    water_sound_speed,
    water_density,
    water_absorption,
    water_nonlinearity,
    # Signal processing
    add_noise,
    # Metadata
    __author__,
    __version__,
    # ── FFT / DFT ────────────────────────────────────────────────────────────
    fft1,
    ifft1,
    fft3,
    ifft3,
    # ── Analytical wave physics ───────────────────────────────────────────────
    tone_burst_waveform,
    pulse_train_waveform,
    fubini_waveform,
    shock_vapor_pulse_waveform,
    goldberg_shock_parameter_sweep,
    shock_enhanced_absorption_gain,
    shock_waveform_pressure,
    shock_heat_source_density,
    standing_wave_1d,
    plane_wave_pressure_1d,
    spherical_wave_pressure,
    reflection_pressure_coeff,
    transmission_pressure_coeff,
    power_law_attenuation_np_m,
    absorption_power_law_db_cm,
    stokes_kirchhoff_absorption_np_m,
    fdtd_phase_error_1d,
    pstd_phase_error,
    kspace_correction_error,
    fdtd_cfl_limit,
    fubini_harmonic_amplitude,
    fubini_harmonic_spectrum,
    shock_formation_distance,
    westervelt_harmonic_evolution,
    # ── Analytical transducer physics ────────────────────────────────────────
    circular_piston_directivity,
    linear_array_factor,
    grating_lobe_angles,
    apodization_weights,
    delay_law_focus_2d,
    beam_pattern_2d,
    circular_piston_onaxis,
    focused_bowl_onaxis,
    focused_bowl_element_positions_3d,
    bli_stencil_weights,
    linear_array_positions,
    near_field_distance,
    steering_focus_point,
    delay_law_steer_2d,
    beam_pattern_magnitude,
    beam_pattern_2d_magnitude,
    multi_focus_delay_laws_2d,
    multi_focus_field_magnitude_2d,
    linear_array_aperiodic_positions,
    steered_beam_pattern_1d,
    steering_grating_lobe_ratio_1d,
    safe_steering_halfangle,
    electronic_steering_efficiency,
    # ── Analytical cavitation physics ────────────────────────────────────────
    intrinsic_threshold_cavitation_probability,
    frequency_dependent_intrinsic_threshold_pa,
    cumulative_cavitation_probability,
    prf_efficacy_factor,
    minnaert_resonance_hz,
    blake_threshold_pa,
    rayleigh_collapse_time_s,
    rayleigh_plesset_rk4,
    keller_miksis_rk4,
    bubble_power_spectrum,
    bubble_acoustic_emission_pressure,
    hann_windowed_power_spectrum,
    ensemble_emission_superposition,
    simulate_bubble_emission,
    simulate_coated_bubble_emission,
    epstein_plesset_dissolution_time,
    shelled_dissolution_time,
    wood_sound_speed,
    bubbly_cloud_attenuation,
    bubbly_cloud_phase_velocity,
    cavitation_emission_bands,
    cumulative_cavitation_dose,
    cavitation_controller_pressure,
    integrate_receiver_array_psd,
    emission_energy_in_volume,
    sonication_schedule,
    forward_delivery_fraction,
    received_signal_fraction,
    pressure_transmission_coefficient,
    interface_pressure_enhancement,
    lacuna_cavitation_susceptibility,
    lacuna_void_fraction,
    histotripsy_kill_fraction,
    histotripsy_lethal_dose,
    histotripsy_pulses_for_lesion_radius,
    histotripsy_lesion_radius_m,
    inertial_cavitation_dose,
    clipped_lateral_radius_for_clearance,
    ellipsoid_respects_allowed_mask,
    scale_measured_emission_spectrum,
    delivered_histotripsy_progress,
    # ── Analytical medium / tissue properties ────────────────────────────────
    water_sound_speed_temperature,
    water_density_temperature,
    ba_parameter,
    tissue_absorption_db_cm,
    kramers_kronig_sound_speed,
    tissue_properties,
    histotripsy_tissue_properties,
    tissue_thermal_properties,
    # ── Safety / dosimetry ───────────────────────────────────────────────────
    mechanical_index,
    mechanical_index_field,
    thermal_index_soft_tissue,
    thermal_index_bone,
    cem43_cumulative,
    arrhenius_damage_integral,
    arrhenius_cumulative,
    fda_ispta_limit_mw_cm2,
    fda_isppa_limit_w_cm2,
    # ── Skull / transcranial ─────────────────────────────────────────────────
    skull_insertion_loss_two_way_db,
    skull_phase_screen,
    hu_to_sound_speed_schneider,
    hu_to_density_schneider,
    strehl_ratio,
    skull_surface_temperature_rise,
    skull_transfer_matrix_transmission,
    skull_transmission_spectrum,
    # ── Photoacoustics ───────────────────────────────────────────────────────
    hbo2_molar_absorption,
    hb_molar_absorption,
    gruneisen_parameter_water,
    pa_sphere_pressure_signal,
    pa_axial_resolution,
    spectroscopic_unmixing_lstsq,
    # ── Elastography ─────────────────────────────────────────────────────────
    shear_wave_speed,
    voigt_complex_modulus,
    springpot_complex_modulus,
    # ── Bubble dynamics (Keller–Miksis / CEM43) ──────────────────────────────
    solve_keller_miksis,
    compute_cem43,
    cem43_at_temperatures,
    # ── Thermal: Beer-Lambert depth profiles + 3-D heat-source density ──────────
    adiabatic_temperature_rise_kelvin,
    gaussian_power_deposition_2d,
    bioheat_focal_temperature_rise,
    hifu_focal_pressure_gain,
    acoustic_intensity_from_amplitude,
    acoustic_intensity_depth_profile,
    acoustic_power_deposition_depth_profile,
    acoustic_heat_source_density,
    # ── BBB permeability, closure kinetics, CEUS ─────────────────────────────
    bbb_permeability_hill,
    bbb_closure_kinetics,
    ceus_backscatter_signal,
    # ── Sonogenetics / neuromodulation ───────────────────────────────────────
    compute_acoustic_membrane_tension_py,
    boltzmann_open_probability_py,
    coupled_channel_drive_py,
    gaussian_beam_pressure_field_py,
    simulate_lif_neuron_py,
    # ── RTM / adaptive beamforming ───────────────────────────────────────────
    focused_gaussian_beam_2d,
    backprop_green_function_2d,
    rtm_imaging_condition,
    rtm_multi_frequency_fusion,
    temporal_modulation_frequencies,
    standing_wave_suppression_gain,
    standing_wave_spatial_frequency_cycles_m,
    standing_wave_modulation_period_hz,
    standing_wave_field_1d,
    standing_wave_intensity_statistics,
)

# Config builders: focused simulation configuration (PyO3)
from ._pykwavers import (
    PmlConfig,
    HelmholtzConfig,
    NonlinearConfig,
    ThermalConfig,
)

# Pure-Python k-Wave parity utilities
from .kwave_parity import (
    angular_spectrum_cw,
    backward_angular_spectrum_cw,
    cart2grid,
    extract_amp_phase,
    gaussian,
    gaussian_source_2d,
    grid2cart,
    spect,
)

# Public API
__all__ = [
    # Core classes
    "Grid",
    "GpuPstdSession",
    "Medium",
    "Source",
    "TransducerArray2D",
    "MultiRowRingArray",
    "FrequencyDomainFwiConfig",
    "FrequencyObservation",
    "BreastFwiPstdDatasetConfig",
    # Config builders: focused simulation configuration
    "PmlConfig",
    "HelmholtzConfig",
    "NonlinearConfig",
    "ThermalConfig",
    "Sensor",
    "Simulation",
    "SimulationResult",
    "SolverType",
    "ResidualGasField",
    # Thermal diffusion / Pennes bioheat
    "ThermalSimulation",
    "ThermalResult",
    # Phase 22: PID, Registration, Bubble Field
    "PIDController",
    "BubbleField",
    "plan_abdominal_array_placement_from_ritk_ct",
    "plan_transcranial_focused_bowl_placement_from_ritk_ct",
    "resample_to_target_grid",
    "kspace_line_recon",
    "time_reversal_reconstruction",
    "run_transcranial_ust_slice_inversion_from_ritk_ct",
    "run_transcranial_ust_volume_inversion_from_ritk_ct",
    "load_ct_nifti",
    "run_theranostic_inverse_from_ritk",
    "run_theranostic_nonlinear_3d_from_ritk",
    "run_transcranial_fus_planning_from_ritk_ct",
    "run_transcranial_skull_adaptive_benchmark_from_ritk_ct",
    "ali_2025_breast_fwi_frequency_sweep_hz",
    "simulate_breast_fwi_frequency_observation",
    "snap_breast_fwi_array_to_grid",
    "load_ali_2025_breast_fwi_phantom",
    "generate_breast_fwi_pstd_dataset",
    "simulate_breast_fwi_pstd_finite_window_born_observation",
    "diagnose_breast_fwi_homogeneous_direct_field",
    "diagnose_breast_fwi_observation_pair",
    "breast_fwi_scaled_observation_residual_metrics",
    "breast_fwi_source_channel_residual_diagnostics",
    "breast_fwi_source_receiver_mask",
    "breast_fwi_passive_receiver_mask",
    "breast_fwi_source_excitation_diagnostics",
    "breast_fwi_sine_frequency_bin_coefficient",
    "breast_fwi_acquisition_identifiability",
    "breast_fwi_reconstruction_metrics",
    "breast_fwi_table1_parity",
    "breast_fwi_operator_equivalence_diagnostics",
    "breast_fwi_scattering_increment_diagnostics",
    "prepare_breast_fwi_reduced_phantom",
    "derive_breast_fwi_reduced_array_geometry",
    "derive_breast_fwi_reduced_array_plan",
    "invert_breast_fwi",
    "passive_acoustic_map_das",
    "beamform_image_delay_and_sum",
    "simulate_receive_rf",
    # Submodules
    "comparison",
    "kwave_python_bridge",
    "kwave_bridge",
    # Signal generation
    "tone_burst",
    "create_cw_signals",
    "get_win",
    # Geometry (matching k-Wave toolbox)
    "make_disc",
    "make_ball",
    "make_sphere",
    "make_circle",
    "make_line",
    # Unit conversion
    "db2neper",
    "neper2db",
    "freq2wavenumber",
    "hounsfield2density",
    "hounsfield2soundspeed",
    # Water properties (temperature-dependent, matching k-wave-python)
    "water_sound_speed",
    "water_density",
    "water_absorption",
    "water_nonlinearity",
    # Signal processing
    "add_noise",
    # k-Wave parity utilities (pure Python)
    "gaussian",
    "spect",
    "extract_amp_phase",
    "cart2grid",
    "grid2cart",
    "angular_spectrum_cw",
    "backward_angular_spectrum_cw",
    "gaussian_source_2d",
    # Metadata
    "__version__",
    "__author__",
    # ── FFT / DFT ────────────────────────────────────────────────────────────
    "fft1",
    "ifft1",
    "fft3",
    "ifft3",
    # ── Analytical wave physics ───────────────────────────────────────────────
    "tone_burst_waveform",
    "pulse_train_waveform",
    "fubini_waveform",
    "shock_vapor_pulse_waveform",
    "goldberg_shock_parameter_sweep",
    "shock_enhanced_absorption_gain",
    "shock_waveform_pressure",
    "shock_heat_source_density",
    "standing_wave_1d",
    "plane_wave_pressure_1d",
    "spherical_wave_pressure",
    "reflection_pressure_coeff",
    "transmission_pressure_coeff",
    "power_law_attenuation_np_m",
    "absorption_power_law_db_cm",
    "stokes_kirchhoff_absorption_np_m",
    "fdtd_phase_error_1d",
    "pstd_phase_error",
    "kspace_correction_error",
    "fdtd_cfl_limit",
    "fubini_harmonic_amplitude",
    "fubini_harmonic_spectrum",
    "shock_formation_distance",
    "westervelt_harmonic_evolution",
    # ── Analytical transducer physics ────────────────────────────────────────
    "circular_piston_directivity",
    "linear_array_factor",
    "grating_lobe_angles",
    "apodization_weights",
    "delay_law_focus_2d",
    "beam_pattern_2d",
    "circular_piston_onaxis",
    "focused_bowl_onaxis",
    "focused_bowl_element_positions_3d",
    "bli_stencil_weights",
    "linear_array_positions",
    "near_field_distance",
    "steering_focus_point",
    "delay_law_steer_2d",
    "beam_pattern_magnitude",
    "beam_pattern_2d_magnitude",
    "multi_focus_delay_laws_2d",
    "multi_focus_field_magnitude_2d",
    "linear_array_aperiodic_positions",
    "steered_beam_pattern_1d",
    "steering_grating_lobe_ratio_1d",
    "safe_steering_halfangle",
    "electronic_steering_efficiency",
    # ── Analytical cavitation physics ────────────────────────────────────────
    "intrinsic_threshold_cavitation_probability",
    "frequency_dependent_intrinsic_threshold_pa",
    "cumulative_cavitation_probability",
    "prf_efficacy_factor",
    "minnaert_resonance_hz",
    "blake_threshold_pa",
    "rayleigh_collapse_time_s",
    "rayleigh_plesset_rk4",
    "keller_miksis_rk4",
    "bubble_power_spectrum",
    "bubble_acoustic_emission_pressure",
    "hann_windowed_power_spectrum",
    "ensemble_emission_superposition",
    "simulate_bubble_emission",
    "simulate_coated_bubble_emission",
    "epstein_plesset_dissolution_time",
    "shelled_dissolution_time",
    "wood_sound_speed",
    "bubbly_cloud_attenuation",
    "bubbly_cloud_phase_velocity",
    "cavitation_emission_bands",
    "cumulative_cavitation_dose",
    "cavitation_controller_pressure",
    "integrate_receiver_array_psd",
    "emission_energy_in_volume",
    "sonication_schedule",
    "forward_delivery_fraction",
    "received_signal_fraction",
    "pressure_transmission_coefficient",
    "interface_pressure_enhancement",
    "lacuna_cavitation_susceptibility",
    "lacuna_void_fraction",
    "histotripsy_kill_fraction",
    "histotripsy_lethal_dose",
    "clipped_lateral_radius_for_clearance",
    "ellipsoid_respects_allowed_mask",
    "scale_measured_emission_spectrum",
    "delivered_histotripsy_progress",
    "histotripsy_pulses_for_lesion_radius",
    "histotripsy_lesion_radius_m",
    "inertial_cavitation_dose",
    # ── Analytical medium / tissue properties ────────────────────────────────
    "water_sound_speed_temperature",
    "water_density_temperature",
    "ba_parameter",
    "tissue_absorption_db_cm",
    "kramers_kronig_sound_speed",
    "tissue_properties",
    "histotripsy_tissue_properties",
    "tissue_thermal_properties",
    # ── Safety / dosimetry ───────────────────────────────────────────────────
    "mechanical_index",
    "mechanical_index_field",
    "thermal_index_soft_tissue",
    "thermal_index_bone",
    "cem43_cumulative",
    "arrhenius_damage_integral",
    "arrhenius_cumulative",
    "fda_ispta_limit_mw_cm2",
    "fda_isppa_limit_w_cm2",
    # ── Skull / transcranial ─────────────────────────────────────────────────
    "skull_insertion_loss_two_way_db",
    "skull_phase_screen",
    "hu_to_sound_speed_schneider",
    "hu_to_density_schneider",
    "strehl_ratio",
    "skull_surface_temperature_rise",
    "skull_transfer_matrix_transmission",
    "skull_transmission_spectrum",
    # ── Photoacoustics ───────────────────────────────────────────────────────
    "hbo2_molar_absorption",
    "hb_molar_absorption",
    "gruneisen_parameter_water",
    "pa_sphere_pressure_signal",
    "pa_axial_resolution",
    "spectroscopic_unmixing_lstsq",
    # ── Elastography ─────────────────────────────────────────────────────────
    "shear_wave_speed",
    "voigt_complex_modulus",
    "springpot_complex_modulus",
    # ── Bubble dynamics (Keller–Miksis / CEM43) ──────────────────────────────
    "solve_keller_miksis",
    "compute_cem43",
    "cem43_at_temperatures",
    # ── Thermal: Beer-Lambert depth profiles + 3-D heat-source density ──────────
    "adiabatic_temperature_rise_kelvin",
    "gaussian_power_deposition_2d",
    "bioheat_focal_temperature_rise",
    "hifu_focal_pressure_gain",
    "acoustic_intensity_depth_profile",
    "acoustic_power_deposition_depth_profile",
    "acoustic_intensity_from_amplitude",
    "acoustic_heat_source_density",
    # ── BBB permeability, closure kinetics, CEUS ─────────────────────────────
    "bbb_permeability_hill",
    "bbb_closure_kinetics",
    "ceus_backscatter_signal",
    # ── Sonogenetics / neuromodulation ───────────────────────────────────────
    "compute_acoustic_membrane_tension_py",
    "boltzmann_open_probability_py",
    "coupled_channel_drive_py",
    "gaussian_beam_pressure_field_py",
    "simulate_lif_neuron_py",
    # ── RTM / adaptive beamforming ───────────────────────────────────────────
    "focused_gaussian_beam_2d",
    "backprop_green_function_2d",
    "rtm_imaging_condition",
    "rtm_multi_frequency_fusion",
    "temporal_modulation_frequencies",
    "standing_wave_suppression_gain",
    "standing_wave_spatial_frequency_cycles_m",
    "standing_wave_modulation_period_hz",
    "standing_wave_field_1d",
    "standing_wave_intensity_statistics",
]

# Module-level metadata
__doc_format__ = "numpy"
__license__ = "MIT"
__copyright__ = "Copyright 2026 Ryan Clanton PhD"
