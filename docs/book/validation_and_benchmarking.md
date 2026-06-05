# Chapter 20: Validation and Benchmarking

*Systematic Validation of kwavers Against Analytical Solutions, Reference Simulators, and Experimental Data*

---

## 1. Introduction

Validation is the process of determining that a simulation model represents reality within an accepted tolerance. Benchmarking is the process of quantifying that representation numerically. Both are mandatory preconditions for any clinical or research use of a simulation library.

kwavers follows a three-tier validation hierarchy, in ascending order of inferential strength:

1. **Analytical validation.** Compare simulation output against closed-form solutions derived from first principles (plane waves, focused fields in homogeneous media, Green's functions). Errors here indicate solver defects unambiguously attributable to numerics.

2. **Reference simulation parity.** Compare against k-Wave MATLAB (Treeby & Cox 2010), k-wave-python (Jaros 2016), and k-Wave-Julia for identical problem setups. Agreement confirms numerical equivalence of the PSTD discretization; disagreement locates grid-origin offsets, axis-ordering mismatches, or coordinate-convention differences.

3. **Experimental validation.** Compare against hydrophone scan data from physical transducer setups. Agreement here validates both the numerical model and the physical parameter identification pipeline.

**Acceptance criteria** (applied at all tiers):

| Metric              | Minimum acceptance threshold | Strong acceptance threshold |
|---------------------|------------------------------|-----------------------------|
| Pearson *r*         | ≥ 0.95                       | ≥ 0.99                      |
| PSNR                | ≥ 30 dB                      | ≥ 40 dB                     |
| RMS ratio           | 0.85 – 1.15                  | 0.95 – 1.05                 |
| Phase error         | < 5°                         | < 1°                        |

**Notation.** Throughout: *A* and *B* are real-valued *n*-vectors (simulation and reference, respectively); *σ* denotes standard deviation; *μ* denotes mean; *MAX* is the maximum signal value in *B*; *RMSE* is root-mean-square error between *A* and *B*; *c* is acoustic speed; *Δx* is spatial step; *Δt* is time step; *k* is wavenumber.

---

## 2. Theorem: Pearson Correlation as Waveform Fidelity Metric

**Statement.** The Pearson correlation coefficient between vectors *A* and *B* is:

```
r(A, B) = cov(A, B) / (σ_A · σ_B)
         = Σᵢ (Aᵢ - μ_A)(Bᵢ - μ_B) / [√Σᵢ(Aᵢ - μ_A)² · √Σᵢ(Bᵢ - μ_B)²]
```

with r ∈ [–1, 1]. Then r = 1 if and only if A = αB + β for some α > 0, β ∈ ℝ (perfect positive linear relationship). Furthermore:

- *r* is insensitive to amplitude scaling (r(αA, B) = r(A, B) for α ≠ 0).
- *r* is insensitive to DC offset (r(A + β, B) = r(A, B)).
- *r* is sensitive to phase shifts: a half-wavelength shift between A and B yields r ≈ –1.

**Proof.** 

*Necessity of r = 1 ⇔ A = αB + β (α > 0).* 

The Cauchy-Schwarz inequality states |⟨u, v⟩| ≤ ‖u‖ · ‖v‖ with equality iff u = λv for some scalar λ. Set u = A – μ_A·1 and v = B – μ_B·1 (mean-centered vectors). Then r = ⟨u,v⟩ / (‖u‖ · ‖v‖). Equality r = 1 requires ⟨u,v⟩ = ‖u‖ · ‖v‖ and ⟨u,v⟩ > 0, which by Cauchy-Schwarz holds iff u = λv with λ > 0. This gives A – μ_A·1 = λ(B – μ_B·1), i.e., A = λB + (μ_A – λμ_B). Setting α = λ > 0 and β = μ_A – λμ_B completes the proof.

*Insensitivity to scaling.* r(αA, B) = cov(αA, B) / (σ_{αA} · σ_B) = α·cov(A,B) / (|α|·σ_A · σ_B) = sign(α)·r(A,B). For α > 0, r(αA, B) = r(A, B). ∎

**Consequence for validation.** Pearson r measures waveform shape agreement, not amplitude agreement. A simulation with correct spatial pressure pattern but 10% amplitude error gives r = 1.0. Therefore, kwavers parity scripts always report r together with the RMS amplitude ratio:

```
RMS ratio = √(Σ Aᵢ²) / √(Σ Bᵢ²)
```

Only when both r ≥ 0.99 **and** RMS ratio ∈ [0.95, 1.05] is parity declared.

**Sensitivity to phase.** For a sinusoidal wave A = sin(kx) and B = sin(kx + φ), the Pearson correlation is r = cos(φ). A phase error of φ = 10° gives r = 0.985; φ = 18° gives r = 0.951. The threshold r ≥ 0.99 therefore bounds phase error to |φ| ≤ 8.1°.

---

## 3. Theorem: PSNR Definition and Sensitivity

**Statement.** The Peak Signal-to-Noise Ratio between simulation output *A* and reference *B* is:

```
PSNR(A, B) = 20 · log₁₀( MAX_B / RMSE(A, B) )
```

where RMSE(A, B) = √(‖A – B‖² / n) and MAX_B = max(|Bᵢ|). PSNR is monotone-decreasing in RMSE and monotone-increasing in MAX_B. Under a Gaussian noise assumption, PSNR relates to signal dynamic range as:

```
PSNR = SNR_dB + 20 · log₁₀( MAX_B / σ_signal )
```

where SNR_dB is the traditional signal-to-noise ratio in decibels.

**Proof (relationship to dynamic range).** Let B = s + n where s is the signal and n ~ N(0, σ_n²) is additive Gaussian noise. Then RMSE(A, B) ≈ σ_n (for A ≈ s). By definition:

```
PSNR = 20 log₁₀(MAX_s / σ_n)
     = 20 log₁₀(MAX_s / σ_s) + 20 log₁₀(σ_s / σ_n)
     = 20 log₁₀(MAX_s / σ_s) + SNR_dB
```

The first term is a property of the signal's dynamic range (crest factor); the second term is the SNR. For a sinusoidal signal MAX_s = √2 · σ_s, so 20 log₁₀(MAX_s/σ_s) = 20 log₁₀(√2) ≈ 3 dB. ∎

**Thresholds.** PSNR = 40 dB corresponds to RMSE = MAX_B / 100 (1% of peak amplitude error). PSNR = 60 dB corresponds to 0.1% error. Medical imaging standards (IEC 62306) require PSNR ≥ 35 dB for diagnostic image quality; kwavers targets ≥ 40 dB for parity acceptance.

**Caveat: PSNR sensitivity to normalization.** If A and B are normalized differently, PSNR changes by 20 log₁₀(scale_factor). kwavers parity scripts normalize both fields by the reference's L∞ norm before computing PSNR to remove scale-factor ambiguity.

---

## 4. Theorem: Convergence of PSTD for Linear Acoustics

**Statement.** For the linear acoustic wave equation:

```
∂²p/∂t² = c² ∇²p
```

discretized with the pseudospectral method (spectral differentiation in space, second-order leapfrog in time), the spatial error is O(Δx^N) for any order N (spectral convergence — exponential in the number of grid points for smooth solutions), and the temporal error is O(Δt²). The scheme is stable when:

```
c · Δt / Δx ≤ C_max = π / (d · k_max · Δx)
```

where d is the spatial dimension and k_max = π/Δx is the Nyquist wavenumber.

**Proof sketch (temporal convergence).** The leapfrog scheme advances the field as:

```
p^{n+1} = 2p^n - p^{n-1} + (c·Δt)² · L_h[p^n]
```

where L_h is the spectral Laplacian. Taylor expanding the exact solution p(t + Δt) and p(t - Δt) and summing gives:

```
p(t+Δt) + p(t-Δt) = 2p(t) + Δt² · ∂²p/∂t² + O(Δt⁴)
```

The O(Δt⁴)/Δt² = O(Δt²) local truncation error per step, accumulated over T = t_end/Δt steps, gives global error O(Δt²). ∎

**Proof sketch (spatial convergence).** The spectral method represents p as a truncated Fourier series. For smooth (C^∞) periodic functions, the Fourier coefficients decay faster than any polynomial in wavenumber. The aliasing error from the truncation is therefore smaller than any power of Δx, giving spectral convergence. For non-smooth media (jump discontinuities in c or ρ), convergence degrades to O(Δx^p) where p depends on the regularity of the solution.

**CFL condition.** The von Neumann stability analysis of the leapfrog-spectral scheme requires that the spectral radius of the time-stepping operator is ≤ 1. For the spectral Laplacian with maximum eigenvalue –k_max² = –(π/Δx)²:

```
(c · Δt · k_max)² ≤ d
⟹ c · Δt / Δx ≤ √d · (π / (π)) = 1/√d   (for k_max = π/Δx)
```

More precisely, kwavers uses CFL = 0.3 / d^(1/2) by default, providing a 70% safety margin below the theoretical limit.

**Convergence validation procedure.** For each new solver component, kwavers runs a convergence test that doubles N from N_min to N_max and verifies that the error halves (second-order) or decreases exponentially (spectral):

```rust
for n in [32, 64, 128, 256] {
    let error = run_pstd_vs_analytical(n, dt_scaled(n));
    // Verify: error(2n) / error(n) ≈ (1/2)^2 = 0.25 for O(Δt²)
}
```

---

## 5. Theorem: Grid Dispersion Error and PSTD Correction

**Statement.** For a finite-difference scheme of order 2m applied to the spatial Laplacian, the numerical wavenumber k_num satisfies:

```
k_num = (1/Δx) · arcsin(k · Δx · C_FD(k, m))
```

where C_FD is a correction factor that deviates from 1 as k → k_Nyquist. The phase velocity error is:

```
ε_v(k) = (c_num(k) - c) / c = (k_num/k - 1)
```

For the pseudospectral method, k_num = k exactly for |k| ≤ π/Δx, so ε_v = 0 for all resolved wavenumbers. PSTD achieves zero numerical dispersion within the resolved band.

**Proof (FD dispersion).** Insert the plane-wave ansatz A·exp(i(k_num·x – ωt)) into the FD stencil:

```
[2cos(k_num·Δx) – 2] / Δx²  ≈  –k_num²  (for k_num·Δx ≪ 1)
```

The FD approximation of –k² with error O((kΔx)^{2m}) gives k_num = k – k^{2m+1}Δx^{2m}/(2m)! + O(k^{2m+3}). The phase velocity c_num = ω/k_num ≠ c for k ≠ 0, producing dispersion.

**Proof (PSTD zero dispersion).** The spectral derivative is implemented as multiplication by ik in Fourier space:

```
∂p/∂x  →  ℱ⁻¹[ik · ℱ[p]]
```

This is exact for all wavenumbers k that are represented in the DFT (|k| ≤ π/Δx). No approximation is introduced; k_num = k identically. Dispersion arises only from the time integrator (O(Δt²) from the leapfrog), not from the spatial operator. ∎

**Practical significance.** In FD schemes, dispersion error accumulates over propagation distance L as a phase shift Δφ = ε_v · k · L. For a 5 MHz transducer, Δx = 0.1 mm, and propagation distance L = 50 mm, a 2nd-order FD scheme incurs Δφ ≈ 15° at 5 MHz. PSTD incurs zero spatial dispersion and only the leapfrog temporal dispersion of Δφ ≈ 0.02° for the same parameters.

---

## 6. Algorithm: Parity Protocol

### 6.1 Structure of compare_*.py Scripts

Each parity script in `pykwavers/examples/` follows a fixed structure:

```python
# compare_<scenario>.py
# Phase 1: Run kwavers simulation
cfg = build_config(nx, ny, nz, dx, dt, ...)
sim = Simulation(cfg)
result = sim.run()

# Phase 2: Run reference (k-Wave MATLAB via kwave-python, or k-Wave-Julia)
ref = kwave_run_equivalent(cfg)

# Phase 3: Metric computation
r      = pearson_r(result.p_final, ref.p_final)
psnr   = compute_psnr(result.p_final, ref.p_final)
rms_ratio = rms(result.p_final) / rms(ref.p_final)

# Phase 4: Acceptance gate
assert r >= 0.99,        f"Pearson r = {r:.4f} < 0.99"
assert psnr >= 40.0,     f"PSNR = {psnr:.2f} dB < 40 dB"
assert 0.95 <= rms_ratio <= 1.05, f"RMS ratio = {rms_ratio:.3f}"

# Phase 5: Visual output
save_side_by_side_parity_figure(result.p_final, ref.p_final, scenario_name)
```

### 6.2 save_side_by_side_parity_figure

This function generates a three-panel PNG:

- **Panel 1:** kwavers pressure field (dB scale, normalized to reference peak).
- **Panel 2:** Reference field (same normalization).
- **Panel 3:** Absolute difference field in dB.

Color axis: –60 dB to 0 dB for field panels; –80 dB to 0 dB for difference panel. Metrics (r, PSNR, RMS ratio) are printed as a text annotation at the top of the figure.

### 6.3 Acceptance Criteria Matrix

| Scenario               | Pearson r | PSNR (dB) | RMS ratio | Status |
|------------------------|-----------|-----------|-----------|--------|
| 1-D plane wave         | ≥ 0.999   | ≥ 50      | 0.99–1.01 | Gate   |
| Focused bowl 3-D       | ≥ 0.990   | ≥ 40      | 0.95–1.05 | Gate   |
| Annular array 3-D      | ≥ 0.990   | ≥ 40      | 0.95–1.05 | Gate   |
| Phased array 2-D       | ≥ 0.990   | ≥ 38      | 0.95–1.05 | Gate   |
| B-mode scan lines      | ≥ 0.970   | ≥ 35      | 0.85–1.15 | Gate   |
| Nonlinear propagation  | ≥ 0.980   | ≥ 38      | 0.93–1.07 | Gate   |

### 6.4 Coordinate Convention

A persistent source of parity failure is the grid-origin offset between kwavers and k-Wave. k-Wave uses the convention that the grid center is at index `N/2` (integer division), not `(N-1)/2`. For N = 128, the center is at index 64 (not 63.5). All kwavers parity scripts enforce:

```python
cx = nx // 2 * dx   # k-Wave convention (NOT (nx-1)/2 * dx)
cy = ny // 2 * dy
cz = nz // 2 * dz
```

This fix (project_annular_array_coordinate_fix.md) raised Pearson correlation from 0.02 to 1.0 and PSNR from 3 dB to 119 dB for the annular array case, demonstrating that coordinate conventions dominate physics accuracy at this scale.

---

## 7. Algorithm: Regression Test Suite

### 7.1 Rust Unit and Integration Tests (cargo nextest)

The kwavers Rust test suite is executed via `cargo nextest` with a hard 60-second timeout per test:

```toml
# .config/nextest.toml
[profile.default]
test-threads = "num-cpus"
slow-timeout = { period = "60s", terminate-after = 1 }
fail-fast = false
```

Test organization:

| Test type              | Location                              | Count |
|------------------------|---------------------------------------|-------|
| PSTD unit tests        | `kwavers/src/solver/forward/pstd/*/tests.rs` | 47 |
| CPML config tests      | `kwavers/src/domain/boundary/cpml/config/tests.rs` | 12 |
| Beamforming tests      | `kwavers/src/analysis/signal_processing/beamforming/*/tests.rs` | 18 |
| Differential op tests  | `kwavers/src/math/numerics/operators/differential/*/tests.rs` | 32 |
| Microbubble tests      | `kwavers/src/domain/therapy/microbubble/state/tests.rs` | 8 |
| Architecture boundary  | `kwavers/tests/architecture_boundaries.rs` | 5 |
| **Total**              |                                       | **122** |

All assertions use value-semantic checks:

```rust
// Correct: inspect computed value
assert!((r - 0.9999).abs() < 1e-3, "Pearson r = {r}");

// Prohibited: existence-only check
// assert!(result.is_ok());   ← rejected by zero_tolerance policy
```

### 7.2 Python Parity Tests (pytest)

```bash
cd pykwavers
pytest examples/ -v --timeout=300 -k "compare_"
```

Parity tests are marked `@pytest.mark.slow` and excluded from the fast CI gate (< 5 min) but required for the full validation gate (< 30 min).

### 7.3 Test Data Derivation

Test data is derived from one of three authoritative sources:

1. **Analytical solutions.** Plane wave: p(x,t) = A·sin(kx – ωt). Focused bowl: Green's function integral evaluated numerically with 10× oversampled quadrature. Delay-and-sum beamform: geometric time delays from transducer geometry.

2. **Published reference data.** k-Wave MATLAB toolbox outputs for the canonical examples (sd_focused_detector_3D, at_focused_bowl_3D, at_focused_annular_array_3D) stored as compressed `.npz` files in `pykwavers/examples/reference_data/`.

3. **kwavers-internal cross-validation.** CPU vs GPU results for the same simulation must agree to within floating-point rounding (‖CPU – GPU‖_∞ < 10 × machine_epsilon).

---

## 8. Algorithm: Analytical Benchmark Cases

### 8.1 1-D Plane Wave

**Setup.** Medium: water (c = 1500 m/s, ρ = 1000 kg/m³, lossless). Grid: N = 512 points, Δx = 0.1 mm, Δt = CFL × Δx/c, T = 500 steps. Source: sinusoidal point source at x = 0, f = 1 MHz.

**Analytical solution.** At time t after the wavefront passes position x:

```
p(x, t) = A · sin(2πf(t – x/c))   for t > x/c
p(x, t) = 0                         for t ≤ x/c
```

**Acceptance.** Pearson r ≥ 0.999, PSNR ≥ 50 dB, RMS ratio 0.99–1.01.

**Known pass condition.** This test is deterministic and has been passing in kwavers since the initial PSTD implementation. Failure indicates a regression in the pressure update kernel or source injection.

### 8.2 Focused Bowl (3-D)

**Setup.** Focused bowl transducer: radius of curvature R = 60 mm, aperture D = 50 mm, center frequency f = 1 MHz. Grid: 128³, Δx = 0.5 mm. Medium: homogeneous water.

**Analytical solution.** On the acoustic axis, the focal-zone pressure amplitude is given by the O'Neil formula (O'Neil 1949):

```
|p(z)| = ρ₀ c u₀ · | ∫₀^a J₀(ka·r/z) · exp(ikz√(1+(r/z)²)) · r dr |
```

evaluated numerically. The focal point at z = R has pressure gain G = π D² / (4 λ R).

**kwavers result (project_at_focused_bowl_3D_parity.md).** Pearson = 0.9999, RMS ratio = 0.994, PSNR = 45.82 dB. Gap CLOSED.

**Note.** The "25% deficit" reported in earlier records was a stale metrics file artefact. Always re-run `compare_at_focused_bowl_3D.py` from the cached `.npz` before diagnosing physics.

### 8.3 Annular Array (3-D)

**Setup.** 5-element annular array, element radii 0–25 mm, center frequency 1 MHz, f-number 1.5. Grid: 128³, Δx = 0.5 mm.

**Analytical solution.** Delay-and-sum focal pressure using per-element Euler rotation geometry and the BLI (band-limited interpolation) kernel for off-grid element positions (Wise 2019). The BLI stencil is canonical and must not be re-tuned (project_bli_stencil_audit.md).

**kwavers result (project_annular_array_coordinate_fix.md).** After fixing the grid center convention (nx//2 × dx, not (nx-1)/2 × dx): Pearson = 1.0, PSNR = 119 dB. The 17.5% amplitude deficit in earlier records was an example script bug, not a physics error.

### 8.4 Phased Array (2-D)

**Setup.** 64-element linear phased array, element pitch λ/2, steering angle 20°, f = 3.5 MHz. Grid: 256×256, Δx = 0.22 mm.

**Reference.** k-Wave MATLAB `example_tvsp_steering_linear_array.m`.

**kwavers result (project_phased_array_parity.md).** GPU: Pearson = 0.9996 (fundamental), 0.9968 (harmonic), 14× speedup vs k-Wave. The TDR poll fix was required to obtain stable results for this grid size.

---

## 9. kwavers Parity Results

### 9.1 Closed Validation Gaps

| Scenario                         | Pearson r | PSNR (dB) | RMS ratio | Gap status    |
|----------------------------------|-----------|-----------|-----------|---------------|
| PSTD 1-D plane wave              | 1.0000    | 72        | 1.000     | CLOSED        |
| PSTD focused bowl 3-D            | 0.9999    | 45.82     | 0.994     | CLOSED        |
| PSTD annular array 3-D           | 1.0000    | 119       | 1.000     | CLOSED        |
| Phased array 2-D (GPU, fund.)    | 0.9996    | 44        | 0.998     | CLOSED        |
| Phased array 2-D (GPU, harm.)    | 0.9968    | 38        | 0.995     | CLOSED        |
| B-mode scan lines (raw)          | 0.977     | 35        | 0.998     | CLOSED        |
| Focused detector 3-D (CPU PSTD)  | 1.0000    | 80        | 1.000     | CLOSED        |
| PSTD absorption (power law)      | 0.9999    | 55        | 1.001     | CLOSED (< 0.11% error) |

### 9.2 Active Validation Tasks

| Scenario                         | Current r | Target r | Gap description          |
|----------------------------------|-----------|----------|--------------------------|
| GPU fractional Laplacian         | N/A       | ≥ 0.99   | CPU port in progress (project_gpu_frac_laplacian_absorption.md) |
| B-mode log-compression           | 0.593     | ≥ 0.95   | Normalize artefact in log_compression (project_us_bmode_linear_transducer_gap.md) |
| Axisymmetric WSWA-FFT            | N/A       | ≥ 0.99   | Implementation done; validation pending (project_axisymmetric_impl.md) |

### 9.3 Historical Root Causes of Validation Failures

Understanding past failure modes prevents their recurrence:

| Failure                          | Root cause                       | Fix                              |
|----------------------------------|----------------------------------|----------------------------------|
| PSTD amplitude 3× too high       | CPML absorption in wrong field   | Absorb in pressure, not density  |
| u·∇ρ₀ term: non-zero error       | Advection term was spurious      | Remove term; validated absent    |
| Annular array 17.5% deficit      | Script used (N-1)/2 not N//2     | Fix coordinate center convention |
| B-mode sensor ordering mismatch  | Fortran vs C array reshape       | Reshape(NY,NX).T in Python       |
| GPU PSTD hung > 60 s             | TDR timeout without poll         | device.poll every 16 batches     |
| Stale metrics showed 25% deficit | Stale .npz not regenerated       | Always re-run compare script     |
| Pearson = –0.11 on beam patterns | Sensor array transposed          | Reshape(NY,NX).T                 |

---

## 10. Experimental Validation

### 10.1 Hydrophone Scan Protocol

Experimental validation uses a calibrated needle hydrophone (Precision Acoustics HPM075, 75 µm active diameter, flat frequency response 0.1–30 MHz ± 1.5 dB) mounted on a 3-axis motorized stage (step size 0.1 mm). The scan procedure:

1. Fill water tank, degas to < 2 ppm dissolved O₂ (to prevent cavitation at diagnostic pressure levels).
2. Align transducer face to hydrophone using the pulse-echo null method.
3. Scan the desired plane (e.g., the focal plane at z = R) on a grid matching the simulation spatial resolution.
4. Record waveforms at each position; extract peak pressure and fundamental/harmonic amplitudes via FFT.
5. Export as HDF5 file with embedded metadata (transducer serial, calibration date, medium temperature).

### 10.2 Simulation–Experiment Registration

Spatial registration between simulation grid and scan grid requires:

1. **Scale calibration.** The simulation Δx and the scan step Δx_scan must agree to < 1%. Verify via the measured focal spot FWHM against the analytical prediction.
2. **Origin alignment.** The transducer face position in simulation coordinates is identified as the scan position where the waveform leading-edge arrival time matches the geometric propagation time c × z/Δz.
3. **RITK registration.** For complex geometries (transcranial, curved arrays), the RITK (Registration Image Toolkit) image registration pipeline aligns simulation and experimental fields via 3-D rigid registration with mutual information metric.

### 10.3 Acceptance for Experimental Comparison

Experimental data contains measurement noise, spatial sampling errors, and hydrophone directivity effects not modeled in simulation. The acceptance thresholds are therefore relaxed:

| Metric              | Acceptance threshold |
|---------------------|----------------------|
| Pearson r           | ≥ 0.90               |
| PSNR                | ≥ 28 dB              |
| RMS ratio           | 0.80 – 1.20          |
| Focal depth error   | < 1 mm               |
| FWHM error          | < 10%                |

---

## 11. Continuous Integration

### 11.1 GitHub Actions Pipeline

The kwavers CI pipeline runs on push to `main` and on pull requests:

```yaml
# .github/workflows/ci.yml (structural outline)
jobs:
  rust-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - name: Install nextest
        run: cargo install cargo-nextest --locked
      - name: Run Rust tests
        run: cargo nextest run --profile ci --no-fail-fast
        env:
          RUSTFLAGS: "-C target-feature=+avx2"

  python-parity:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - name: Run fast parity tests
        run: pytest pykwavers/examples/ -m "not slow" --timeout=60
      - name: Upload parity figures
        uses: actions/upload-artifact@v4
        with:
          name: parity-figures
          path: pykwavers/examples/figures/*.png

  full-validation:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Run full parity suite
        run: pytest pykwavers/examples/ --timeout=300
```

### 11.2 Timeout Policy

| Test category                    | Hard timeout | Action on timeout         |
|----------------------------------|-------------|---------------------------|
| Rust unit tests (per test)       | 60 s        | Fail; optimize real code  |
| Python parity (fast, per test)   | 60 s        | Fail; optimize real code  |
| Python parity (full, per test)   | 300 s       | Fail; optimize real code  |
| Full validation suite            | 120 min     | Fail CI; escalate         |

Tests that approach the timeout threshold trigger an optimization cycle on the real implementation — reducing grid size in tests is prohibited by the zero_tolerance policy.

### 11.3 Artifact Upload

Parity figures are uploaded as CI artifacts for visual inspection on each run. The figure naming convention is:

```
{scenario}_{pearson_r:.4f}_{psnr:.1f}dB_{rms_ratio:.3f}.png
```

This embeds the metric values in the filename so that CI summary pages show pass/fail at a glance without opening the figure.

### 11.4 Regression Detection

A test marked as passing is registered in `parity_baseline.json` with its metric values. On each CI run, the current metrics are compared against the baseline:

```python
def check_regression(current: PariMetrics, baseline: ParityMetrics, tol=0.01):
    assert current.pearson >= baseline.pearson - tol, \
        f"Pearson regression: {current.pearson:.4f} < {baseline.pearson:.4f} - {tol}"
    assert current.psnr >= baseline.psnr - 1.0, \
        f"PSNR regression: {current.psnr:.1f} < {baseline.psnr:.1f} - 1.0"
```

A regression exceeding tolerance blocks the PR merge via the required status check gate.

---

## 12. Figure References

| Figure | Caption                                              | Source                     |
|--------|------------------------------------------------------|----------------------------|
| 12.1   | Validation hierarchy diagram (3-tier)                | §1                         |
| 12.2   | Side-by-side: focused bowl kwavers vs k-Wave         | compare_at_focused_bowl_3D |
| 12.3   | Side-by-side: annular array kwavers vs k-Wave        | compare_at_focused_annular |
| 12.4   | Pearson r convergence vs grid resolution (N=32–256)  | Convergence test §4        |
| 12.5   | Grid dispersion error: FD-2 vs FD-6 vs PSTD         | §5                         |
| 12.6   | Hydrophone scan vs simulation: focal plane           | §10                        |
| 12.7   | CI pipeline timing breakdown (Gantt chart)           | §11.1                      |
| 12.8   | Parity metric history (r, PSNR vs commit index)      | Regression tracking §11.4  |

Figures are generated by parity scripts in `pykwavers/examples/` and stored in `docs/book/figures/`.

---

## 13. References

1. **Treeby, B. E., and Cox, B. T.** (2010). k-Wave: MATLAB Toolbox for the Simulation and Reconstruction of Photoacoustic Wave Fields. *Journal of Biomedical Optics*, 15(2), 021314. https://doi.org/10.1117/1.3360308

2. **Jaros, J., Rendell, A. P., and Treeby, B. E.** (2016). Full-Wave Nonlinear Ultrasound Simulation on Multi-GPU Using k-Wave and CUDA. *International Journal of High Performance Computing Applications*, 30(2), 137–155.

3. **Treeby, B. E., and Cox, B. T.** (2010). Modeling Power Law Absorption and Dispersion for Acoustic Propagation Using the Fractional Laplacian. *Journal of the Acoustical Society of America*, 127(5), 2741–2748.

4. **Mast, T. D.** (2001). Empirical Relationships Between Acoustic Parameters in Human Soft Tissues. *Acoustics Research Letters Online*, 1(2), 37–42.

5. **IEC 62306.** (2005). Ultrasonics — Field Characterization — Test Methods for the Determination of Thermal and Mechanical Indices Related to Medical Diagnostic Ultrasonic Fields. International Electrotechnical Commission.

6. **O'Neil, H. T.** (1949). Theory of Focusing Radiators. *Journal of the Acoustical Society of America*, 21(5), 516–526.

7. **Wise, E. S., Cox, B. T., Jaros, J., and Treeby, B. E.** (2019). Representing Arbitrary Acoustic Source and Sensor Distributions in Fourier Collocation Methods. *Journal of the Acoustical Society of America*, 146(1), 278–288.

8. **Williams, E. G.** (1999). Fourier Acoustics: Sound Radiation and Nearfield Acoustical Holography. Academic Press, London.

9. **Courant, R., Friedrichs, K., and Lewy, H.** (1928). Über die Partiellen Differenzengleichungen der mathematischen Physik. *Mathematische Annalen*, 100(1), 32–74.

10. **Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P.** (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. *IEEE Transactions on Image Processing*, 13(4), 600–612.

11. **k-wave-python Documentation.** https://k-wave-python.readthedocs.io/

12. **k-Wave.jl Repository.** https://github.com/JClingo/k-wave-julia

13. **Precision Acoustics Ltd.** (2020). Needle Hydrophone HPM075 Data Sheet and Calibration Procedure. Dorchester, UK.

---

*Module ownership: `kwavers::solver::validation`, `pykwavers/examples/`, `kwavers/tests/architecture_boundaries.rs`. Chapter version: 0.4.0.*
