# Parity Matrix: pykwavers vs k-wave-python

> **Generated**: 2026-02-18
> **Test Baseline**: 353 passed, 29 skipped, 0 failed
> **k-wave-python version**: 0.4.1

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Implemented in pykwavers AND tested for k-wave parity |
| 🔶 | Implemented in pykwavers, tested locally, but NO k-wave parity test |
| ❌ | Not implemented in pykwavers (k-wave-python has it) |
| 🔵 | pykwavers-only feature (no k-wave equivalent) |

---

## 1. Grid (kWaveGrid)

| Feature | pykwavers | k-wave-python | Parity Test | Test File | Notes |
|---------|-----------|---------------|-------------|-----------|-------|
| 3D grid creation (Nx,Ny,Nz,dx,dy,dz) | ✅ | ✅ | ✅ | test_grid_parity.py | TestGridCreation |
| Grid dimensions (nx,ny,nz) | ✅ | ✅ | ✅ | test_grid_parity.py | |
| Grid spacing (dx,dy,dz) | ✅ | ✅ | ✅ | test_grid_parity.py | |
| Total points | ✅ | ✅ | ✅ | test_grid_parity.py | |
| Domain size | ✅ | ✅ | 🔶 | test_grid_parity.py | TestDomainSize — local only |
| CFL time step | ✅ | ✅ | ✅ | test_grid_parity.py | TestGridParityWithKWave |
| Wavenumber arrays (kx,ky,kz) | ✅ | ✅ | ✅ | test_grid_parity.py | TestKSpaceGridParityWithKWave |
| makeTime() equivalent | ❌ | ✅ | — | — | kgrid.makeTime() not exposed |
| dt/Nt properties | ❌ | ✅ | — | — | kgrid.dt, kgrid.Nt not exposed |
| Nyquist frequency | ❌ | ✅ | — | — | Not tested |
| 2D grid support | 🔶 | ✅ | — | — | Grid(nx,ny,1,...) works quasi-2D |
| PPW (points per wavelength) | ✅ | ✅ | 🔶 | test_grid_parity.py | TestGridUtilities |

## 2. Medium (kWaveMedium)

| Feature | pykwavers | k-wave-python | Parity Test | Test File | Notes |
|---------|-----------|---------------|-------------|-----------|-------|
| Homogeneous (c, rho) | ✅ | ✅ | ✅ | test_medium_parity.py | TestMediumParityWithKWave |
| Heterogeneous (c_array, rho_array) | ✅ | ✅ | ✅ | test_heterogeneous_parity.py | TestHeterogeneousMediumParity |
| Absorption coefficient (alpha_coeff) | ✅ | ✅ | ✅ | test_examples_parity.py | TestExampleAbsorbingMedium |
| Absorption power (alpha_power) | ✅ | ✅ | 🔶 | test_medium_parity.py | TestMediumAbsorption |
| Nonlinearity (BonA) | 🔶 | ✅ | — | — | Exposed but no k-wave parity test |
| is_homogeneous flag | ✅ | ✅ | 🔶 | test_utilities.py | TestFeatureGaps |
| Impedance derived property | 🔶 | ✅ | — | test_medium_parity.py | TestDerivedAcousticProperties |
| Bulk modulus derived property | 🔶 | ✅ | — | test_medium_parity.py | TestDerivedAcousticProperties |
| Reflection/transmission coeffs | 🔶 | ❌ | — | test_medium_parity.py | pykwavers-only |
| Spatially varying absorption | ✅ | ✅ | 🔶 | test_heterogeneous_parity.py | In inclusion test |
| alpha_mode | ❌ | ✅ | — | — | Not implemented |
| alpha_sign | ❌ | ✅ | — | — | Not implemented |

## 3. Source (kSource)

| Feature | pykwavers | k-wave-python | Parity Test | Test File | Notes |
|---------|-----------|---------------|-------------|-----------|-------|
| Plane wave source | ✅ | ✅ | ✅ | test_solver_parity.py | TestPlaneWaveParity |
| Point source | ✅ | ✅ | ✅ | test_solver_parity.py | TestPointSourceParity |
| Mask source (p_mask + p) | ✅ | ✅ | ✅ | test_solver_parity.py | _run_pykwavers |
| Initial pressure (p0) | ✅ | ✅ | 🔶 | test_utilities.py | TestFeatureGaps |
| Velocity source (ux,uy,uz) | ✅ | ✅ | 🔶 | test_source_parity.py | TestVelocitySource |
| Plane wave direction | ✅ | ✅ | 🔶 | test_utilities.py | TestFeatureGaps |
| Source mode (additive/dirichlet) | ✅ | ✅ | 🔶 | test_source_parity.py | TestSourceMode |
| Multi-source superposition | ✅ | ✅ | 🔶 | test_phase5_features.py | No k-wave parity |
| Stress sources (u_mask) | ❌ | ✅ | — | — | Not implemented |
| Transducer source | 🔶 | ✅ | — | — | TransducerArray2D partially done |
| source.p_frequency_ref | ❌ | ✅ | — | — | Not implemented |

## 4. Sensor (kSensor)

| Feature | pykwavers | k-wave-python | Parity Test | Test File | Notes |
|---------|-----------|---------------|-------------|-----------|-------|
| Point sensor | ✅ | ✅ | ✅ | test_sensor_parity.py | TestSensorCreation |
| Mask sensor (boolean) | ✅ | ✅ | ✅ | test_sensor_parity.py | TestSensorParityWithKWave |
| Grid sensor (full field) | ✅ | ✅ | 🔶 | test_sensor_parity.py | TestSensorRecording |
| Multi-sensor data (2D output) | ✅ | ✅ | ✅ | test_examples_parity.py | TestExampleAtArrayAsSensor |
| sensor.record = ['p'] | ✅ | ✅ | ✅ | test_solver_parity.py | All k-wave tests use it |
| sensor.record = ['p_max'] | ❌ | ✅ | — | — | Not implemented |
| sensor.record = ['u'] | ❌ | ✅ | — | — | Not implemented |
| sensor.record = ['I'] | ❌ | ✅ | — | — | Not implemented |
| sensor.directivity | ❌ | ✅ | — | — | kSensorDirectivity missing |
| reorder_sensor_data | ❌ | ✅ | — | — | Not implemented |
| Beamforming (delay-and-sum) | 🔶 | ❌ | — | test_sensor_parity.py | pykwavers-only |

## 5. Solver (kspaceFirstOrder3D)

| Feature | pykwavers | k-wave-python | Parity Test | Test File | Notes |
|---------|-----------|---------------|-------------|-----------|-------|
| FDTD solver | ✅ | N/A | ✅ | test_solver_parity.py | TestPlaneWaveParity (FDTD vs k-wave) |
| PSTD solver | ✅ | ✅ | ✅ | test_solver_parity.py | TestPlaneWaveParity (PSTD xfail) |
| Hybrid solver | 🔵 | ❌ | — | test_pstd_hybrid_solvers.py | pykwavers-only |
| PML (perfectly matched layers) | ✅ | ✅ | ✅ | test_solver_parity.py | pml_size parameter |
| Configurable PML size | ✅ | ✅ | 🔶 | test_solver_parity.py | TestPMLConfigurations |
| PML alpha | ❌ | ✅ | — | — | Not configurable |
| pml_inside option | ❌ | ✅ | — | — | Always pml_inside=True |
| data_cast option | ❌ | ✅ | — | — | Not implemented |
| save_to_disk option | ❌ | ✅ | — | — | Not implemented |
| GPU support | ❌ | ✅ | — | — | Not implemented |
| Spatial convergence | 🔶 | ✅ | — | test_solver_parity.py | TestSolverConvergence |
| Temporal convergence | 🔶 | ✅ | — | test_solver_parity.py | TestSolverConvergence |
| kspaceFirstOrder2D | ❌ | ✅ | — | — | quasi-2D via nz=1/2 only |
| kspaceFirstOrder1D | ❌ | ✅ | — | — | Not implemented |
| Solver cross-validation | 🔵 | ❌ | — | test_pstd_hybrid_solvers.py | FDTD vs PSTD vs Hybrid |

## 6. Signal Generation Utilities

| Feature | pykwavers | k-wave-python | Parity Test | Test File | Notes |
|---------|-----------|---------------|-------------|-----------|-------|
| tone_burst | ✅ | ✅ | 🔶 | test_utilities_parity.py | Window difference (Hanning vs Gaussian) |
| create_cw_signals | ✅ | ❌ | — | test_utilities_parity.py | pykwavers-only |
| get_win | ✅ | ✅ | 🔶 | test_utilities_parity.py | TestSignalUtilities |
| add_noise | ✅ | ✅ | 🔶 | test_new_utils_parity.py | TestAddNoise |
| gaussian | ❌ | ✅ | — | — | Not implemented |
| reorder_binary_sensor_data | ❌ | ✅ | — | — | Not implemented |
| spect | ❌ | ✅ | — | — | Not implemented |
| extract_amp_phase | ❌ | ✅ | — | — | Not implemented |

## 7. Geometry / Map Generation Utilities

| Feature | pykwavers | k-wave-python | Parity Test | Test File | Notes |
|---------|-----------|---------------|-------------|-----------|-------|
| make_disc | ✅ | ✅ | 🔶 | test_utilities_parity.py | TestGeometryUtilities |
| make_ball | ✅ | ✅ | 🔶 | test_utilities_parity.py | TestGeometryUtilities |
| make_sphere (alias) | ✅ | ✅ | 🔶 | test_new_utils_parity.py | TestMakeSphere |
| make_circle | ✅ | ✅ | 🔶 | test_new_utils_parity.py | TestMakeCircle |
| make_line | ✅ | ❌ | 🔶 | test_utilities_parity.py | pykwavers-only |
| make_arc | ❌ | ✅ | — | — | Not implemented |
| make_multi_arc | ❌ | ✅ | — | — | Not implemented |
| make_multi_bowl | ❌ | ✅ | — | — | Not implemented |
| make_spherical_section | ❌ | ✅ | — | — | Not implemented |
| make_cart_circle | ❌ | ✅ | — | — | Not implemented |
| make_cart_sphere | ❌ | ✅ | — | — | Not implemented |
| make_cart_rect | ❌ | ✅ | — | — | Not implemented |
| make_cart_arc | ❌ | ✅ | — | — | Not implemented |
| make_cart_bowl | ❌ | ✅ | — | — | Not implemented |
| make_cart_multi_arc | ❌ | ✅ | — | — | Not implemented |

## 8. Unit Conversion Utilities

| Feature | pykwavers | k-wave-python | Parity Test | Test File | Notes |
|---------|-----------|---------------|-------------|-----------|-------|
| db2neper | ✅ | ✅ | 🔶 | test_new_utils_parity.py | Different convention |
| neper2db | ✅ | ✅ | 🔶 | test_utilities_parity.py | Different convention |
| freq2wavenumber | ✅ | ❌ | 🔶 | test_new_utils_parity.py | pykwavers-only |
| hounsfield2density | ✅ | ✅ | 🔶 | test_new_utils_parity.py | TestHounsfieldConversions |
| hounsfield2soundspeed | ✅ | ✅ | 🔶 | test_new_utils_parity.py | TestHounsfieldConversions |
| water_sound_speed | ✅ | ✅ | 🔶 | test_new_utils_parity.py | TestWaterProperties |
| water_density | ✅ | ✅ | 🔶 | test_new_utils_parity.py | TestWaterProperties |
| water_absorption | ✅ | ✅ | 🔶 | test_new_utils_parity.py | TestWaterProperties |
| water_nonlinearity | ✅ | ✅ | 🔶 | test_new_utils_parity.py | TestWaterProperties |
| fit_power_law_params | ❌ | ✅ | — | — | Not implemented |
| cart2grid | ❌ | ✅ | — | — | Not implemented |
| grid2cart | ❌ | ✅ | — | — | Not implemented |
| smooth | ❌ | ✅ | — | — | Not implemented |
| gaussian_filter | ❌ | ✅ | — | — | Not implemented |
| filter_time_series | ❌ | ✅ | — | — | Not implemented |

## 9. TransducerArray (kWaveTransducerSimple / NotATransducer)

| Feature | pykwavers | k-wave-python | Parity Test | Test File | Notes |
|---------|-----------|---------------|-------------|-----------|-------|
| Array creation | ✅ | ✅ | 🔶 | test_transducer_array.py | TestTransducerArray2D |
| Element count/spacing | ✅ | ✅ | 🔶 | test_transducer_array.py | |
| Focus distance | ✅ | ✅ | 🔶 | test_transducer_array.py | |
| Steering angle | ✅ | ✅ | 🔶 | test_transducer_array.py | |
| Apodization windows | ✅ | ✅ | 🔶 | test_transducer_array.py | |
| Active element masking | ✅ | ✅ | 🔶 | test_transducer_array.py | |
| Input signal setting | ✅ | ✅ | 🔶 | test_transducer_array.py | |
| Simulation integration | ❌ | ✅ | — | — | Not usable as Source yet |
| combine_sensor_data | ❌ | ✅ | — | — | Not implemented |
| NotATransducer (passive) | ❌ | ✅ | — | — | Not implemented |

---

## Coverage Summary

### By Component

| Component | Total Features | ✅ K-Wave Parity | 🔶 Local Only | ❌ Missing | 🔵 Extra |
|-----------|---------------|-------------------|---------------|------------|----------|
| Grid | 11 | 5 | 3 | 3 | 0 |
| Medium | 12 | 3 | 5 | 2 | 2 |
| Source | 11 | 3 | 5 | 2 | 1 |
| Sensor | 11 | 4 | 2 | 4 | 1 |
| Solver | 14 | 3 | 3 | 6 | 2 |
| Signal Gen | 8 | 0 | 4 | 4 | 0 |
| Geometry | 14 | 0 | 5 | 9 | 0 |
| Conversions | 15 | 0 | 9 | 6 | 0 |
| Transducer | 10 | 0 | 7 | 3 | 0 |
| **Total** | **106** | **18** | **43** | **39** | **6** |

### Parity Test Gap Analysis

**Features with implementation but NO k-wave parity test (🔶 → ✅ candidates):**

1. **Grid**: Domain size, PPW computation, 2D grid
2. **Medium**: Absorption power, nonlinearity (BonA), impedance, bulk modulus, heterogeneous absorption
3. **Source**: Initial pressure (p0), velocity source, plane wave direction, source mode, multi-source
4. **Sensor**: Grid sensor, beamforming
5. **Solver**: PML configurations, spatial/temporal convergence
6. **Signal**: tone_burst, get_win, add_noise, create_cw_signals
7. **Geometry**: make_disc, make_ball, make_sphere, make_circle, make_line
8. **Conversions**: All 9 functions (db2neper, neper2db, freq2wavenumber, hounsfield*, water_*)
9. **Transducer**: All 7 features

### Priority Actions

**P0 — Immediate (enhance existing tests with k-wave parity):**
- Add k-wave parity to `tone_burst` (test_utilities_parity.py)
- Add k-wave parity to `make_disc` / `make_ball` (test_utilities_parity.py)
- Add k-wave parity to `get_win` (test_utilities_parity.py)
- Add k-wave parity to `water_*` properties (test_new_utils_parity.py)

**P1 — High (fill key simulation parity gaps):**
- Initial pressure (p0) source parity with k-wave
- Velocity source parity with k-wave
- Multi-sensor mask parity with k-wave (2D output shape)
- Heterogeneous medium absorption parity

**P2 — Medium (expose more k-wave features):**
- `make_arc` geometry
- `smooth` filter
- `filter_time_series`
- `sensor.record` options beyond `['p']`
- TransducerArray2D as simulation source

**P3 — Low (completeness):**
- Cartesian geometry variants (make_cart_*)
- `make_multi_arc`, `make_multi_bowl`
- kspaceFirstOrder2D / 1D
- GPU support
- alpha_mode / alpha_sign

---

## Test File Index

| File | Tests | Focus | K-Wave? |
|------|-------|-------|---------|
| test_basic.py | 5 | Smoke test / binding validation | No |
| test_grid_parity.py | 25 | Grid dimensions, k-space arrays | Some |
| test_source_parity.py | 42 | Source types, modes, signals | Some |
| test_sensor_parity.py | 35 | Sensor types, recording, arrays | Some |
| test_medium_parity.py | 40 | Medium properties, absorption | Some |
| test_solver_parity.py | ~30 | FDTD/PSTD vs k-wave, heterogeneous | Yes (slow) |
| test_examples_parity.py | ~25 | End-to-end scenarios vs k-wave | Yes (slow) |
| test_kwave_comparison.py | 3 | SimulationConfig framework vs k-wave | Yes (slow) |
| test_kwave_validation.py | 1 | Plane wave 2D validation | Yes (slow) |
| test_kwave_difference_diagnostics.py | 4 | Timing/phase/amplitude diagnostics | Yes (slow) |
| test_heterogeneous_parity.py | 4 | Layered/gradient/inclusion media | Yes (slow) |
| test_phase5_features.py | 16 | Multi-source, solver enum, superposition | No |
| test_plane_wave_timing.py | 9 | Arrival time validation | No |
| test_pstd_hybrid_solvers.py | 14 | PSTD/Hybrid wiring, dispersion, energy | No |
| test_source_injection.py | 4 | Source injection correctness | No |
| test_transducer_array.py | 10 | TransducerArray2D API | No |
| test_utilities.py | ~40 | Error metrics, CFL, integration, gaps | No |
| test_utilities_parity.py | ~15 | Signal/geometry/conversion parity | Some |
| test_new_utils_parity.py | 39 | New utility functions parity | Some |
