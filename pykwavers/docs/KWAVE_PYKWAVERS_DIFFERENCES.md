# k-wave-python vs pykwavers: Current Difference Map

This note tracks *measured* differences from focused parity diagnostics and is updated as solver behavior converges.

## Current snapshot (2026-02-14)

Diagnostics from `tests/test_kwave_difference_diagnostics.py` on cached k-wave reference runs:

- **Plane wave, FDTD**
  - Raw: `L2=0.814`, `Linf=1.325`, `corr=0.637`
  - Lag-aligned: `L2=0.716`, `Linf=0.991`, `corr=0.725`
  - Best lag: `+2` samples
  - RMS ratio (py/kw): `0.879`

- **Plane wave, PSTD**
  - Raw: `L2=0.993`, `Linf=1.118`, `corr=0.287`
  - Lag-aligned: `L2=0.927`, `Linf=0.975`, `corr=0.445`
  - Best lag: `-3` samples
  - RMS ratio (py/kw): `0.458`

## Interpretation

- **FDTD** is qualitatively aligned with k-wave-python, with modest timing lag and moderate amplitude/shape error.
- **PSTD** remains the dominant gap:
  - lower correlation,
  - significant amplitude attenuation (`~0.46x` RMS vs k-wave in plane-wave probe),
  - point-source case shows opposite amplitude bias (`~2.48x` RMS),
  - timing offset alone does not explain total mismatch.

## Resolved Issues

### Temporal Sinc Normalization Convention (RESOLVED 2026-02-12)

The primary PSTD parity regression was identified and fixed:

- **Root Cause**: Treeby temporal correction was using normalized sinc (`sin(πx)/(πx)`) instead of unnormalized sinc (`sin(x)/x`).
- **Fix**: Default changed to unnormalized sinc in `kwavers/src/solver/forward/pstd/utils.rs`.
- **Impact**: Point PSTD L2 improved from 5.526 to 2.016, correlation improved from -0.122 to 0.620.

## Remaining PSTD Parity Issues

### 1. Stepping Order Differences

**kwavers current order:**
1. update_velocity(dt)
2. apply_dynamic_velocity_sources(dt)
3. update_density(dt)
4. apply_pressure_sources(time_index, dt)
5. update_pressure()
6. apply_boundary(time_index)
7. record sensor

**k-wave order (inferred from C++ binary structure):**
1. Apply ALL sources first (velocity + pressure as mass source)
2. Compute velocity updates
3. Compute density updates  
4. Compute pressure updates
5. Apply boundary
6. Record sensor

**Status**: Partially addressed with environment variable experiments. Full reordering may be needed.

### 2. PML Application Differences

- kwavers applies PML separately to velocity, density (each split component), and pressure
- k-wave applies PML more holistically

This can cause different absorption characteristics.

### 3. Source Scaling Verification

Both implementations use equivalent formulas:
- Mass source scale: `2 * dt / (N * c0 * dx)`
- k-space correction: `kappa = sinc(c_ref * k * dt / 2)`
- Source kappa: `cos(c_ref * k * dt / 2)`

## Environment Variable Diagnostics

Available for experimentation (no default behavior changes):

| Variable | Purpose |
|----------|---------|
| `KWAVERS_PSTD_SOURCE_TIME_SHIFT` | Shift source time by N samples |
| `KWAVERS_PSTD_SOURCE_GAIN` | Apply gain to source terms |
| `KWAVERS_PSTD_DISABLE_BOUNDARY` | Skip boundary application |
| `KWAVERS_PSTD_RECORD_PRE_BOUNDARY` | Record before boundary |
| `KWAVERS_PSTD_DIAGNOSTICS` | Enable diagnostics output |
| `KWAVERS_PSTD_TEMPORAL_SINC` | Override sinc convention |

## Active Test Policy

- Keep strict PSTD target thresholds in parity policy.
- Keep current PSTD parity assertions as explicit tracking (`xfail`) where strict thresholds are not yet met.
- Do not silently relax PSTD thresholds.

## Next Technical Actions

1. **Consider stepping order reordering** in `stepper.rs` to apply pressure sources BEFORE velocity/density updates
2. Re-run diagnostics to measure improvement after any changes
3. If still failing, investigate PML coupling differences more deeply
4. Consider implementing staggered grid shift operators to match k-wave's exact PSTD formulation

## Parity Test Coverage

The following test modules provide comprehensive parity validation:

| Module | Coverage |
|--------|----------|
| `test_grid_parity.py` | Grid creation, k-space grids, CFL stability |
| `test_medium_parity.py` | Homogeneous/heterogeneous media, absorption |
| `test_source_parity.py` | Point, plane, mask sources, signal generation |
| `test_sensor_parity.py` | Point, mask, array sensors, beamforming |
| `test_solver_parity.py` | FDTD/PSTD vs k-wave, convergence, stability |
| `test_examples_parity.py` | Full simulation scenarios, transducer arrays |

