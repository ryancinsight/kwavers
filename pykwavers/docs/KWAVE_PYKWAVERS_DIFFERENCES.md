# k-wave-python vs pykwavers: Current Difference Map

This note tracks *measured* differences from focused parity diagnostics and is updated as solver behavior converges.

## Current snapshot (2026-02-12)

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

## Internal PSTD telemetry (2026-02-12)

With `KWAVERS_PSTD_DIAGNOSTICS=1`, per-step PSTD internals show:

- Plane-wave PSTD: source-term RMS (`dpx_rms`) is stable (`~1e-3`), but pressure RMS stays below k-wave parity target.
- Point-source PSTD: source-term RMS is much smaller (`~1e-5`) while pressure peaks are intermittent and much larger, with large lag (`+48` samples).

This pattern suggests the dominant issue is not a simple global source amplitude scale. It is more likely in PSTD propagation/boundary coupling and/or update ordering for localized sources.

### Isolation experiments (env-gated, no default behavior change)

- `KWAVERS_PSTD_RECORD_PRE_BOUNDARY=1`: record-before-boundary vs record-after-boundary produced **no meaningful metric change**.
- `KWAVERS_PSTD_DISABLE_BOUNDARY=1`: skipping final boundary application in stepper produced **no meaningful metric change** for current diagnostics.

These results reduce confidence that sensor-record checkpoint timing or the final boundary-call placement is the primary source of the PSTD parity gap.

- `KWAVERS_PSTD_SOURCE_TIME_SHIFT={-2,-1,0,+1,+2}`: shifting source sample index changes lag/correlation slightly but does **not** resolve PSTD mismatch.
  - Point-source PSTD raw at `shift=-2`: `L2=5.436`, `corr=-0.038`, RMS ratio `5.305` (vs baseline `L2=5.526`, `corr=-0.122`, RMS ratio `5.314`).
  - Timing contributes, but is not the dominant driver.

- `KWAVERS_PSTD_SOURCE_GAIN={0.2,0.5,1.0}`: source gain strongly controls PSTD amplitude error.
  - Point-source PSTD raw at `gain=0.2`: `L2=1.546`, RMS ratio `1.063`.
  - Point-source PSTD raw at `gain=0.5`: `L2=2.951`, RMS ratio `2.657`.
  - Point-source PSTD raw at `gain=1.0`: `L2=5.526`, RMS ratio `5.314`.
  - Correlation and lag remain nearly unchanged across gain sweep, indicating residual **phase/order** mismatch after amplitude correction.

- Combined probe (`KWAVERS_PSTD_SOURCE_GAIN=0.2`, `KWAVERS_PSTD_SOURCE_TIME_SHIFT=-2`) further improves point-source raw metric to `L2=1.485`, RMS ratio `1.061`, but still leaves lag/shape mismatch (`corr=-0.038`, best lag `-34`).

- `KWAVERS_PSTD_TEMPORAL_SINC` A/B probe identified a **true implementation regression** in PSTD temporal correction convention:
  - `normalized` (current regressed path): plane PSTD `L2=0.918`, point PSTD `L2=5.526`, point corr `-0.122`, RMS ratio `5.314`.
  - `unnormalized` (`sin(x)/x`): plane PSTD `L2=0.993`, point PSTD `L2=2.016`, point corr `0.620`, RMS ratio `2.478`.
  - This exactly matches the earlier recovered baseline and confirms normalization convention, not source-gain tuning, was the primary regression driver.

Resolution implemented:

- Treeby temporal correction now defaults to **unnormalized** sinc (`sin(x)/x`) with env override for diagnostics:
  - default behavior: unnormalized
  - optional override: `KWAVERS_PSTD_TEMPORAL_SINC=normalized`

## Likely root-cause buckets

1. **Source scaling / injection consistency**
   - Differences in pressure source application between k-space PSTD and pykwavers PSTD path.
2. **k-space correction and update ordering**
   - Small ordering/scaling discrepancies can produce phase and amplitude drift.
3. **Boundary and PML behavior under PSTD**
   - PML coupling in PSTD can disproportionately affect amplitude.

## Active test policy

- Keep strict PSTD target thresholds in parity policy.
- Keep current PSTD parity assertions as explicit tracking (`xfail`) where strict thresholds are not yet met.
- Do not silently relax PSTD thresholds.

## Root Cause Analysis from Source Comparison

After comparing source code between kwavers and k-wave-python, the following discrepancies have been identified:

### Key Code Files

| File | Role |
|------|------|
| `kwavers/src/solver/forward/pstd/implementation/core/stepper.rs` | Main stepping loop |
| `kwavers/src/solver/forward/pstd/utils.rs` | k-space corrections (kappa, source_kappa) |
| `kwavers/src/solver/forward/fdtd/source_handler.rs` | Source scaling calculations |
| `pykwavers/.venv/Lib/site-packages/kwave/kspaceFirstOrder3D.py` | k-wave-python reference |
| `pykwavers/.venv/Lib/site-packages/kwave/kWaveSimulation_helper/scale_source_terms_func.py` | k-wave source scaling |

### 1. k-space operator (kappa) computation

**k-wave-python (kspaceFirstOrder3D.py):**
```python
# Kappa (k-space correction for derivatives)
k_sim.kappa = np.fft.ifftshift(np.sinc(c_ref * k * dt / 2))
# Source kappa (k-space correction for source injection)
k_sim.source_kappa = np.fft.ifftshift(np.cos(c_ref * k * dt / 2))
```

**kwavers (utils.rs):**
```rust
// Treeby2010 correction - uses normalized sinc
let temporal_correction = sinc_normalized(c_ref * dt * k_mag / 2.0);
// source_kappa computed as:
let source_kappa = k_mag.mapv(|k| (0.5 * c_ref * config.dt * k).cos());
```

**Status:** EQUIVALENT - Both use the same formulas. The normalized sinc matches `np.sinc(x/π)`.

### 2. Mass source scaling

**k-wave-python (scale_source_terms_func.py):**
```python
# Uniform grid additive pressure source:
source_p = source_p * (2 * dt / (N * c0 * dx))
```

**kwavers (stepper.rs):**
```rust
let mass_source_scale = 2.0 * dt / (n_dim * self.c_ref * dx_min)
```

**Status:** EQUIVALENT - Both use the same scaling formula.

### 3. Stepping order (PRIMARY SUSPECT)

**kwavers current order:**
1. update_velocity(dt)
2. apply_dynamic_velocity_sources(dt)
3. update_density(dt)
4. apply_pressure_sources(time_index, dt) ← mass source injection HERE
5. update_pressure()
6. apply_boundary(time_index)
7. record sensor

**k-wave order (inferred from k-wave structure):**
1. Apply ALL sources first (velocity + pressure as mass source)
2. Compute velocity updates
3. Compute density updates  
4. Compute pressure updates
5. Apply boundary
6. Record sensor

**Issue:** kwavers injects pressure sources AFTER velocity/density updates but BEFORE pressure update. k-wave may inject sources BEFORE any field updates.

### 4. PML application differences

- kwavers applies PML separately to velocity, density (each split component), and pressure
- k-wave applies PML more holistically

This can cause different absorption characteristics.

## Proposed Fixes

### Fix 1: Adjust stepping order to match k-wave

Modify `stepper.rs` to apply sources before field updates:

```rust
// Proposed new order:
// 1. Apply pressure sources first (as mass sources)
// 2. Apply velocity sources
// 3. Update velocity
// 4. Update density  
// 5. Update pressure
// 6. Apply boundary
// 7. Record sensor
```

### Fix 2: Add environment variables for experimentation

Already implemented:
- `KWAVERS_PSTD_SOURCE_TIME_SHIFT` - Shift source time by N samples
- `KWAVERS_PSTD_SOURCE_GAIN` - Apply gain to source terms
- `KWAVERS_PSTD_DISABLE_BOUNDARY` - Skip boundary
- `KWAVERS_PSTD_RECORD_PRE_BOUNDARY` - Record before boundary
- `KWAVERS_PSTD_DIAGNOSTICS` - Enable diagnostics output

## Active test policy

- Keep strict PSTD target thresholds in parity policy.
- Keep current PSTD parity assertions as explicit tracking (`xfail`) where strict thresholds are not yet met.
- Do not silently relax PSTD thresholds.

## Next technical actions

1. **PRIORITY**: Reorder stepping in `stepper.rs` to apply pressure sources BEFORE velocity/density updates
2. Verify kappa computation uses exact k-wave formula (verified - already matches)
3. Re-run diagnostics to measure improvement after stepping order fix
4. If still failing, investigate PML coupling differences more deeply
5. Consider implementing staggered grid shift operators to match k-wave's exact PSTD formulation

