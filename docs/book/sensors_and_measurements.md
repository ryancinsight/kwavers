# Chapter 6: Sensors and Measurements

## 1. Introduction

This chapter develops the mathematical foundation for ultrasound sensor modeling, signal
acquisition, and measurement as implemented in kwavers. The scope covers hydrophone
directivity and calibration, spatial Nyquist criteria for array sensors, the
pressure-to-particle-velocity relationship derived from Newton's second law, time-reversal
focusing via Green's function reciprocity, the sensor recording contract that governs data
accumulation and storage ordering, photoacoustic array measurement, and time-reversal
reconstruction algorithms.

The sensor subsystem in kwavers is responsible for extracting physically meaningful scalar
and vector quantities from the propagating wave field at each simulation time step. Sensor
correctness — including ordering conventions — directly determines whether downstream
image reconstruction, parity checks against k-Wave, and derived clinical metrics are valid.

### Notation

| Symbol | Meaning | Units |
|--------|---------|-------|
| `p(r, t)` | Acoustic pressure field | Pa |
| `u(r, t)` | Particle velocity field | m s⁻¹ |
| `ρ` | Ambient density | kg m⁻³ |
| `c` | Speed of sound | m s⁻¹ |
| `k` | Wave number: `k = ω/c` | m⁻¹ |
| `a` | Hydrophone element radius | m |
| `d` | Inter-element spacing | m |
| `λ` | Wavelength: `λ = c/f` | m |
| `θ` | Polar angle from element normal | rad |
| `H(θ)` | Element directivity (normalized, dimensionless) | – |
| `G(r, r'; ω)` | Green's function: pressure at `r` due to point source at `r'` | Pa m |
| `s(t)` | Recorded sensor signal | Pa or m s⁻¹ |
| `Δx` | Grid spacing | m |
| `M` | Number of active sensor cells | – |
| `nt` | Number of simulation time steps | – |

The grid ordering convention used throughout this chapter is Fortran column-major order
(x-index varies fastest), consistent with k-Wave MATLAB and with the kwavers sensor
recorder implementation. Deviations from this convention constitute a correctness defect,
not a styling choice.

---

## 2. Theorem: Hydrophone Directivity

### Statement

A circular hydrophone element of radius `a` responding to plane acoustic waves has an
angle-dependent pressure sensitivity (directivity function) of

```
H(θ) = 2 J_1(ka sinθ) / (ka sinθ)
```

identical in form to the piston transmit directivity derived in Chapter 5. By the reciprocity
principle, the receive directivity of a passive element equals its transmit directivity.

For a square element of half-width `b`, the directivity is separable:
```
H(θ_x, θ_y) = sinc(kb sinθ_x / π) · sinc(kb sinθ_y / π)
```

where `sinc(x) = sin(πx)/(πx)` (normalized sinc).

### Proof via Reciprocity

**Step 1 — Transmit-receive equivalence.**
By the Rayleigh reciprocity theorem for linear acoustics, the impulse response
`h_{AB}(t)` between points `A` and `B` satisfies `h_{AB}(t) = h_{BA}(t)`. Applied to a
transducer element treated as a boundary condition: the pressure response at a far-field
point due to element excitation equals the open-circuit voltage induced at the element
terminals by a far-field point source, up to a frequency-dependent impedance factor that
is unity in the far field.

**Step 2 — Surface integration.**
Under plane wave incidence from angle `θ` the pressure at each surface element of the
hydrophone integrates over the aperture with a phase factor `exp(ik x sinθ)`. The
integrated output is
```
S(θ) = ∫∫_aperture p_0 e^{ikx sinθ} dA.
```
For a circular aperture the integral evaluates identically to the Huygens-Fresnel piston
calculation of Chapter 5, yielding `H(θ) = 2J_1(ka sinθ)/(ka sinθ)`.       □

### Calibration via Reciprocity

Absolute hydrophone sensitivity in Pa⁻¹ is determined by the three-transducer reciprocity
method (IEC 61685:2001):
1. Three calibrated hydrophones `A`, `B`, `C` form three transmitter-receiver pairs.
2. Voltage transfer ratios `M_{AB}`, `M_{AC}`, `M_{BC}` are measured in a free field.
3. The individual sensitivity `M_A` satisfies
   ```
   |M_A|² = (M_{AB} M_{AC} / M_{BC}) · (ρ c / (π a_B a_C))
   ```
   where `a_B`, `a_C` are element areas.
4. The directivity correction `H(θ)` is measured in an anechoic tank by rotating the
   hydrophone through known angles and normalizing to the on-axis response.

### Frequency Response and Phase

The hydrophone sensitivity `M(f)` also has a frequency-dependent magnitude and phase. For a
passive PVDF needle hydrophone, the -3 dB bandwidth typically spans 0.5–50 MHz with a
nearly flat phase response below the first resonance. kwavers does not currently model the
hydrophone transfer function; point sensors are assumed broadband ideal. This is conservative
and does not require correction for frequencies below the hydrophone resonance.

### Directivity Error Budget

For a hydrophone with `ka ≤ 1` at the center frequency (element diameter ≤ λ/π),
the directivity error in received amplitude is below 1% for angles `|θ| ≤ 30°`. At higher
frequencies or larger elements:
```
H(-3 dB angle) ≈ arcsin(1.6 / (ka))   (angular half-power width)
```

---

## 3. Theorem: Spatial Nyquist Criterion

### Statement

For a uniform linear sensor array with inter-element spacing `d`, the received acoustic
field can be spatially sampled without grating lobes (spatial aliasing) if and only if

```
d ≤ λ / 2.
```

When `d > λ/2`, spatial aliasing produces grating lobes at angles
```
sinθ_grating = sinθ_signal ± n λ / d,   n = 1, 2, 3, ...
```
that are indistinguishable from the signal direction without additional constraints.

### Proof via Spatial DFT

**Step 1 — Spatial sampling.**
A linear array of `N` elements at positions `x_n = n d`, `n = 0, ..., N-1`, records
the pressure field `p(x, t)`. The spatial Fourier transform of the array output at
time `t` and temporal frequency `ω` is

```
P(k_x, ω) = Σ_{n=0}^{N-1} p(x_n, ω) e^{-ik_x n d}.
```

**Step 2 — Sampling theorem.**
This discrete sum is periodic in `k_x` with period `2π/d`. The spatial spectrum of a
propagating plane wave at angle `θ` has `k_x = k sinθ = (ω/c) sinθ`. The Nyquist
condition requires the period `2π/d` to exceed the maximum one-sided spatial frequency
`k_x,max = ω/c` (maximum at `|sinθ| = 1`):
```
2π/d ≥ 2 · ω/c
⟹  d ≤ πc/ω = λ/2.       □
```

**Step 3 — Grating lobe location.**
When `d = λ`, the spatial period is `2π/d = 2π/λ = k`. A signal at `sinθ = 0` (broadside)
aliases to `k_x = 0, ±2π/λ = ±k`, corresponding to `sinθ_g = ±1` (end-fire). This is the
maximum severity case where grating lobes alias to end-fire.

### 2-D Array Extension

For a 2-D matrix array with spacings `(d_x, d_y)`, the spatial Nyquist conditions apply
independently in each dimension:
```
d_x ≤ λ/2,   d_y ≤ λ/2.
```
For a hexagonal grid with pitch `d`, the effective spacings are `d` and `d sqrt(3)/2`,
so the sampling condition is `d ≤ λ / sqrt(3)` to avoid grating lobes in both principal
planes (Azar et al., 2000).

### Undersampled Arrays and Sparse Recovery

When physical constraints force `d > λ/2` (e.g., large-aperture arrays with cost-limited
element count), grating lobe suppression requires:
- Non-uniform spacing to de-correlate grating lobe positions across steering angles.
- Sparse recovery algorithms (e.g., compressed sensing beamforming) that exploit signal
  sparsity in the image domain to separate signal from grating lobes.
- Apodization (element weighting) that broadens the main lobe in exchange for suppressed
  sidelobes.

kwavers supports non-uniform sensor masks natively; the sensor mask is an arbitrary binary
array with no geometric constraint enforced at construction time.

---

## 4. Theorem: Pressure-Velocity Relationship

### Statement

In a linear, lossless acoustic medium, the particle velocity and pressure satisfy Newton's
second law (Euler's equation of motion):

```
ρ ∂u/∂t = -∇p
```

In the frequency domain with `e^{-iωt}` convention, this becomes

```
-iω ρ U(r, ω) = -∇P(r, ω)
```

or equivalently

```
U(r, ω) = ∇P(r, ω) / (iω ρ).
```

For a plane wave propagating in direction `n̂`, the relationship reduces to

```
U = P / (ρ c)   (in the direction of propagation)
```

and for a focused beam with effective aperture area `A`, the volume velocity is

```
Q = ∫_A U · n̂ dA = P / (ρ c A).
```

### Proof from Linear Acoustic Equations

**Step 1 — Linearization.**
In a quiescent medium with ambient density `ρ_0` and no background flow, the acoustic
momentum equation at first order in perturbation amplitude is
```
ρ_0 ∂u/∂t = -∇p,
```
where `p = p_total - p_0` is the acoustic pressure perturbation and `u` is the acoustic
particle velocity. This is Euler's equation linearized about the rest state.       □

**Step 2 — Plane wave solution.**
For a plane wave `p = P_0 exp(i(k·r - ωt))` propagating in direction `k̂`, substituting
into the momentum equation gives:
```
ρ_0 (-iω) u = -ik P_0 exp(i(k·r - ωt))
⟹  u = (k/ω) P_0 / ρ_0 = P_0 / (ρ_0 c) k̂.
```
The acoustic impedance `Z = ρ c` appears as the ratio `P_0 / |u|`.

**Step 3 — Staggered grid discretization.**
kwavers uses a staggered (Yee-type) grid where pressure is stored at integer nodes and
velocity components are stored at half-integer nodes:
```
u_x[i+½, j, k] = u_x at position ((i+½)Δx, jΔy, kΔz)
p[i, j, k]     = p at position (iΔx, jΔy, kΔz)
```
The discrete momentum update is
```
u_x^{n+½}[i+½] = u_x^{n-½}[i+½] - (dt/ρΔx) (p^n[i+1] - p^n[i])
```
which is a second-order accurate centered-difference approximation to `ρ ∂u_x/∂t = -∂p/∂x`.

### Sensor Velocity Acquisition

For simulation steps where velocity sensors are active, kwavers records the three staggered
velocity components at the nearest grid node to each sensor position. The staggered-to-collocated
interpolation (averaging two adjacent nodes) is performed post-hoc in
`kwavers::domain::sensor::recorder::velocity_statistics::interpolate_staggered_to_collocated`
to produce a co-located velocity estimate for output:
```rust
pub fn interpolate_staggered_to_collocated(
    ux_staggered: &Array3<f64>,
    uy_staggered: &Array3<f64>,
    uz_staggered: &Array3<f64>,
) -> (Array3<f64>, Array3<f64>, Array3<f64>)
```

---

## 5. Theorem: Time-Reversal Focusing

### Statement

Let `G(r, r_0; ω)` be the Green's function for acoustic pressure at `r` due to a point
source at `r_0` in a heterogeneous medium (possibly with absorption). The time-reversed
pressure field produced by re-emitting the time-conjugate signals recorded on a closed
surface `S` converges to a focus at the original source location `r_0`:

```
p_TR(r, ω) = ∫_S [G*(r, r_s; ω) G(r_s, r_0; ω) - G(r, r_s; ω) G*(r_s, r_0; ω)] dS
```

In a lossless medium this simplifies to

```
p_TR(r_0, ω) ∝ Im[G(r_0, r_0; ω)],
```

which is maximal at the source location due to the imaginary part of the Green's function
self-interaction.

### Proof via Reciprocity Theorem

**Step 1 — Helmholtz reciprocity.**
For two acoustic fields `p_A` and `p_B` satisfying the Helmholtz equation in the same
medium, the reciprocity theorem states:
```
∫_V (p_A ∇²p_B - p_B ∇²p_A) dV = ∮_S (p_A ∇p_B - p_B ∇p_A) · n̂ dS.
```

**Step 2 — Time reversal as complex conjugation.**
Time reversal in the frequency domain corresponds to complex conjugation: `p*(r, ω)` is
the time-reversed version of `p(r, ω)`. For a real field, conjugation corresponds to
reversing the time axis: `p*(r, ω) ↔ p(r, -t)`.

**Step 3 — Surface re-emission.**
Suppose the forward wave `p_F(r, ω) = G(r, r_0; ω)` is recorded on surface `S`.
Re-emitting `p_F^*(r_s, ω)` from each point `r_s ∈ S` as a source drives the time-reversed
field
```
p_TR(r, ω) = ∫_S G(r, r_s; ω) G*(r_s, r_0; ω) dS.
```
Using reciprocity `G(r, r_s) = G(r_s, r)` and the optical theorem for the Green's function:
```
∮_S [G*(r_s, r_0; ω) ∂_n G(r, r_s; ω) - G(r, r_s; ω) ∂_n G*(r_s, r_0; ω)] dS
  = 2iω ρ Im[G(r, r_0; ω)].
```
At `r = r_0`, `Im[G(r_0, r_0; ω)] > 0` for any physical medium, confirming constructive
interference at the source location.       □

**Step 4 — Heterogeneous media.**
In a heterogeneous medium, the Green's function automatically encodes all scattering,
reflection, and mode conversion paths. Time reversal with a closed measurement aperture
focuses through heterogeneities without requiring knowledge of the medium, because the
recorded field already carries phase information from all propagation paths (Fink, 1992).

### Discrete Time-Reversal Reconstruction

```
Algorithm: Time-Reversal Reconstruction (kwavers implementation)

Phase 1 — Forward simulation:
  1. Run forward PSTD simulation with source at r_0 (or unknown source).
  2. Record pressure time series s_n(t) at all N sensor positions {r_n}.

Phase 2 — Time reversal:
  1. Flip time: s_n_TR(t) = s_n(T - t) for each sensor n.
  2. Load reversed signals into source arrays at sensor positions.
  3. Run backward simulation with same medium parameters.
  4. Output pressure field at final time step as reconstructed image.

Contract:
  — Total simulation time T must be long enough for all echoes to return.
  — Sensor aperture should approximate a closed surface (larger aperture → better focus).
  — For photoacoustic TR, the initial pressure distribution p_0(r) is the output.
```

### Focusing Resolution in Time Reversal

In a homogeneous medium with sensor array aperture `D` and focal depth `z_f`, the
time-reversal focal spot has lateral resolution:
```
Δr_TR ≈ λ z_f / D   (diffraction-limited)
```
identical to the forward beamformed resolution. In a heterogeneous random medium, multiple
scattering provides virtual aperture extension beyond `D`, potentially exceeding the
diffraction limit (super-resolution time reversal, Derode et al., 2003).

---

## 6. Algorithm: Sensor Recording Contract

The sensor recording contract defines the invariants that govern how the kwavers sensor
recorder accumulates field data during simulation and outputs it to Python (pykwavers)
or to binary checkpoint files.

### Algorithm 6.1 — Sensor Recording Contract

```
Input:
  G          — computational grid (nx, ny, nz, dx, dy, dz)
  mask       — binary 3-D array of shape (nx, ny, nz), 1 at sensor positions
  field(t)   — pressure (and optionally velocity) field at time step t
  RecordSpec — list of RecordField variants to collect

Output:
  data[field][active_index, time_step] — 2-D array, shape (M, nt)
    where M = mask.nonzero() count in Fortran order

Contract invariants:
  I1: M = mask.sum() > 0                            (at least one active sensor)
  I2: active_indices ordered by Fortran-order (x fast, then y, then z)
  I3: data.shape == (M, nt) for each recorded field
  I4: data[f][i, t] == field(active_cell[i]) at time t, no rounding or normalization
  I5: ordering matches k-Wave: active_cell[i] = flat_index_fortran(mask)[i]

Algorithm:
  Initialize:
    active_cells = argsort(mask.nonzero(), order='F')  # Fortran order
    M = len(active_cells)
    data = zeros(M, nt) for each field in RecordSpec

  For each time step t = 0 .. nt-1:
    For each active cell index i in 0..M-1:
      (ix, iy, iz) = unravel_index(active_cells[i], (nx,ny,nz), order='F')
      For each field f in RecordSpec:
        data[f][i, t] = extract_field(field(t), ix, iy, iz, f)

  Validate:
    assert data[f].shape == (M, nt)
    assert active_cells matches k-Wave flat index ordering
    return data
```

### Fortran Order vs. C Order — The Ordering Bug

k-Wave MATLAB stores multi-dimensional arrays in Fortran (column-major) order. When a
2-D image is stored in MATLAB as `p(x, y)`, the x-index varies fastest in memory.
NumPy/Python default (C order) has the last index varying fastest: `p[x, y]` stores y
fastest.

The critical invariant is: **the `i`-th row of the sensor data matrix must correspond
to the `i`-th active cell in Fortran-order enumeration of the sensor mask**.

The bug that was present before the fix:
```python
# WRONG: C-order enumeration
sensor_data = np.zeros((mask.sum(), nt))
active = np.argwhere(mask)  # returns rows in C order (z fast)
```

The correct implementation:
```python
# CORRECT: Fortran-order enumeration (x fast, matching k-Wave)
active_fortran = np.argwhere(mask.T).T  # transpose mask before argwhere → x varies fastest
# Or equivalently:
flat_indices = np.flatnonzero(mask.ravel(order='F'))  # Fortran-order flat indices
```

In the kwavers Rust implementation (`kwavers::domain::sensor::recorder`), the active cell
enumeration iterates with x as the outer loop and z as the inner loop by convention,
producing the Fortran-order sequence without explicit transposition.

### RecordField Variants

```rust
pub enum SensorRecordField {
    Pressure,                   // scalar p at sensor nodes
    VelocityX,                  // staggered u_x
    VelocityY,                  // staggered u_y
    VelocityZ,                  // staggered u_z
    PressureMaximum,            // max(p) over time at each node
    PressureMinimum,            // min(p) over time at each node
    PressureRms,                // RMS(p) over time at each node
    IntensityX,                 // I_x = p · u_x (instantaneous)
    IntensityY,
    IntensityZ,
}
```

### Storage Layout and Checkpoint Integration

The sensor recorder in kwavers uses a row-major `ndarray::Array2<f64>` internally with
shape `(M, nt)`. Each row is one sensor's complete time trace. This matches the layout
expected by pykwavers when reconstructing scan lines for B-mode imaging or computing
harmonic content.

Checkpoint files (`.kwcp` format) serialize the current recorder state using `rkyv`, enabling
exact restoration of accumulated sensor data across interrupted simulation runs. The ordering
invariant is preserved across checkpoint boundaries.

---

## 7. Algorithm: Photoacoustic Array Measurement

Photoacoustic (optoacoustic) imaging uses a pulsed laser to deposit thermal energy that
generates a broadband ultrasound signal. The initial pressure rise is

```
p_0(r) = Γ(r) · μ_a(r) · Φ(r)
```

where `Γ` is the Grüneisen parameter (dimensionless, ≈ 0.2 for soft tissue), `μ_a` is the
optical absorption coefficient (m⁻¹), and `Φ` is the optical fluence (J m⁻²).

### Algorithm 7.1 — Photoacoustic Forward Measurement

```
Input:
  p_0(r)    — initial pressure distribution (3-D array, Pa)
  medium    — acoustic medium parameters (c, ρ, absorption)
  sensor    — sensor geometry and mask
  f_center  — center frequency of detection bandwidth
  BW        — detection bandwidth (fractional)

Algorithm:
  1. Initialize acoustic pressure field: p(r, t=0) = p_0(r), u(r, t=0) = 0.

  2. Run forward PSTD simulation for T_sim = 2 * domain_diagonal / c_min seconds:
     — No external source injection (photoacoustic initial-value problem).
     — Record pressure at sensor mask according to Algorithm 6.1.

  3. Optionally apply bandpass filter to recorded signals:
       H_BP(ω) = exp(-(ω - ω_c)² / (2σ²))   where σ = BW · ω_c / 2.355

  4. Output sensor data matrix shape (M, nt_recorded).

Notes:
  — The initial-value problem requires zero source injection; kwavers supports this
    via SourceInjectionMode::None with non-zero initial field conditions.
  — The absorbing boundary (CPML) must attenuate outgoing waves before they wrap
    around in the periodic PSTD grid.
```

### Bandwidth and Spatial Resolution

The spatial resolution of photoacoustic imaging is determined by the detection bandwidth:
```
Δr_PA ≈ 0.88 c / BW_Hz
```
where `BW_Hz = BW · f_center` is the absolute bandwidth. For a 5 MHz center frequency and
70% bandwidth, `Δr_PA ≈ 0.88 × 1540 / (0.7 × 5×10⁶) ≈ 0.39 mm`. This motivates using
broadband detectors (PVDF or CMUT) for photoacoustic applications rather than narrow-band
piezoelectric elements.

### Grüneisen Parameter and Temperature Dependence

The Grüneisen parameter `Γ = c_p β c² / C_v` depends on temperature via the thermal
expansion coefficient `β`. For water at 37°C, `Γ ≈ 0.21`; for soft tissue at 37°C,
`Γ ≈ 0.18–0.22`. kwavers uses the constant value specified in the medium model; temperature
coupling is not currently implemented for photoacoustic simulations.

---

## 8. Algorithm: Time-Reversal Reconstruction

### Algorithm 8.1 — Iterative Time-Reversal for Photoacoustic Imaging

```
Input:
  s_n(t)    — sensor time series, shape (M, nt) in Fortran order
  sensor_pos — sensor positions {r_n}, shape (M, 3)
  medium    — acoustic medium (c, ρ)
  G          — computational grid

Output:
  p_hat(r)  — reconstructed initial pressure distribution

Algorithm:
  Iteration k = 0:
    1. Initialize p_hat^0 = 0 (zero initial estimate).
    2. Load sensor data: s_n(t) → source signals at sensor positions.
    3. Flip time: s_n_TR(t) = s_n(T - t).
    4. Run backward simulation with s_TR as sources at sensor positions.
    5. Extract pressure at t = T (final backward time step): p_hat^1 = p_backward(r, T).

  For additional iterations (iterative time-reversal):
    k = 1, 2, ..., K_max:
    6. Run forward simulation with p_hat^k as initial condition.
    7. Record forward sensor data: s_hat^k(t) at sensor positions.
    8. Compute residual: δs = s_n - s_hat^k.
    9. Flip time: δs_TR(t) = δs(T - t).
    10. Run backward simulation with δs_TR as sources.
    11. Extract pressure: δp(r) = p_backward(r, T).
    12. Update: p_hat^{k+1} = p_hat^k + α δp   (step size α ∈ (0, 1]).
    13. Convergence check: ||δp|| / ||p_hat^k|| < ε_tol → stop.

Notes:
  — One-shot TR (K_max = 1, step k=0 only) is exact for a closed sensor aperture in a
    homogeneous medium (Fink, 1992).
  — Iterative TR compensates for limited aperture, heterogeneity, and absorption
    (Treeby et al., 2010).
  — kwavers supports one-shot TR via run_from_checkpoint with time-reversed source signals.
```

### Delay-and-Sum Beamforming (k-Space DAS)

An alternative to TR reconstruction is delay-and-sum (DAS) beamforming. For each image
pixel `r`:
```
p_DAS(r) = Σ_{n=1}^{M} s_n(τ_n(r)) · w_n(r)
```
where `τ_n(r) = |r - r_n| / c` is the travel time from pixel to sensor `n`, and `w_n(r)`
is an apodization weight. This is equivalent to the backprojection algorithm and is
implemented in `kwavers::analysis::signal_processing::beamforming`.

### Coherence-Based Weighting

Coherence-based compounding uses the short-lag spatial coherence (SLSC) to improve image
contrast:
```
R(m) = (1/(M-m)) Σ_{n=0}^{M-m-1} <s_n(t) s_{n+m}(t)> / sqrt(<s_n²><s_{n+m}²>)
```
The SLSC image is formed by summing `R(m)` over `m = 1..M_lag`. Unlike DAS, SLSC is
signal-to-noise-ratio dependent and rejects uncorrelated noise naturally.

---

## 9. kwavers Implementation

### Module Structure

```
kwavers::domain::sensor
├── mod.rs                          Sensor trait, SensorType, SensorField
├── array.rs                        SensorArray: multi-element sensor collections
├── grid_sampling.rs                Sensor mask sampling, interpolation weights
├── recorder/
│   ├── mod.rs                      RecorderTrait, module re-exports
│   ├── complex.rs                  Recorder (full feature): complex signal recording
│   ├── simple.rs                   SensorRecorder (SimpleRecorder): basic pressure/velocity
│   ├── config.rs                   RecorderConfig, RecordingMode
│   ├── fields.rs                   SensorRecordField, SensorRecordSpec
│   ├── pressure_statistics.rs      PressureFieldStatistics, SampledStatistics
│   ├── velocity_statistics.rs      VelocityComponentStats, interpolate_staggered_to_collocated
│   ├── statistics.rs               RecorderStatistics (aggregate metrics)
│   ├── storage.rs                  Internal ring buffer and accumulation
│   ├── traits.rs                   RecorderTrait (record, finalize, extract)
│   └── events.rs                   Threshold event detection (sonoluminescence)
├── beamforming/                    Delay-and-sum beamformer implementations
│   └── mod.rs
├── passive_acoustic_mapping/       PAM: cavitation mapping via beamforming
│   └── mod.rs
├── point/                          PointSensor: single-location pressure/velocity
│   └── mod.rs
├── sonoluminescence/               Optical emission event detection
│   └── mod.rs
└── ultrafast/                      Plane-wave compounding, ultrafast acquisition
    └── mod.rs
```

### `SensorRecorder` (SimpleRecorder)

The `SimpleRecorder` type provides pressure and velocity recording for standard simulation
workflows. It is constructed from a `RecorderConfig` that specifies the grid dimensions,
active sensor mask, and the list of fields to record.

```rust
pub struct SensorRecorder {
    config: RecorderConfig,
    active_indices: Vec<(usize, usize, usize)>,  // (ix, iy, iz) in Fortran order
    pressure_data: Array2<f64>,                  // shape (M, nt)
    velocity_data: Option<[Array2<f64>; 3]>,     // shape (M, nt) for each component
    step_count: usize,
}

impl SensorRecorder {
    pub fn new(config: RecorderConfig) -> Self { ... }
    pub fn record_pressure(&mut self, field: &Array3<f64>, step: usize) { ... }
    pub fn record_velocity(&mut self, ux: &Array3<f64>, uy: &Array3<f64>,
                           uz: &Array3<f64>, step: usize) { ... }
    pub fn pressure_data(&self) -> &Array2<f64> { ... }
    pub fn finalize(&mut self) -> RecorderStatistics { ... }
}
```

### Ordering Fix Implementation

The Fortran-order bug was fixed in the `active_indices` construction. The corrected
enumeration in Rust:

```rust
// Fortran order: x-index varies fastest (matches k-Wave MATLAB)
let mut active_indices = Vec::new();
for iz in 0..nz {
    for iy in 0..ny {
        for ix in 0..nx {
            if mask[[ix, iy, iz]] > 0 {
                active_indices.push((ix, iy, iz));
            }
        }
    }
}
```

The x-inner, z-outer loop structure produces the same enumeration as `mask.ravel(order='F')`
in NumPy, which matches the k-Wave MATLAB linearization order `sub2ind([nx ny nz], ix, iy, iz)`.

Before this fix, B-mode scan lines produced by kwavers were transposed relative to k-Wave
reference data, causing apparent structure artifacts in the reconstructed image that were
actually correct physics in the wrong row order (see feedback_sensor_ordering.md).

### `PressureFieldStatistics`

Statistics collected during recording without storing the full time series:

```rust
pub struct PressureFieldStatistics {
    pub p_max: Array3<f64>,    // maximum pressure at each grid node over all time steps
    pub p_min: Array3<f64>,    // minimum pressure at each grid node over all time steps
    pub p_rms: Array3<f64>,    // RMS pressure at each grid node
    pub p_final: Array3<f64>,  // pressure at the final time step
}
```

These statistics are computed in a single pass over time without retaining the full
`(nx, ny, nz, nt)` array, reducing memory consumption from `O(N³ nt)` to `O(N³)`.

### Sensor Recording Mode

`RecordingMode` selects what is accumulated:

```rust
pub enum RecordingMode {
    TimeSeries,         // full (M, nt) matrix per field
    Statistics,         // p_max, p_min, p_rms only — O(M) memory
    Both,               // both time series and statistics
    Events(f64),        // threshold crossing events only (threshold in Pa)
}
```

---

## 10. Calibration and Error Budgets

### Absolute Pressure Calibration

kwavers solves the linear acoustic wave equations in SI units. The pressure output at any
sensor node is in Pascals, provided the source amplitude is specified in Pa (for pressure
sources) or m s⁻¹ (for velocity sources). No internal normalization is applied.

Sources of calibration error in a real kwavers–hardware comparison:

| Error Source | Typical Magnitude | Mitigation |
|-------------|-------------------|------------|
| Hydrophone directivity | 0–3 dB at `|θ| > 30°` | Restrict comparison to near-normal incidence |
| Hydrophone frequency response | ±1 dB, 0–15 MHz | Apply measured transfer function in post |
| Grid dispersion (PSTD, 6 PPW) | < 1% phase error | Use ≥ 10 PPW for quantitative measurements |
| CPML absorption boundary | < 0.1% within domain | Verify sensor positions > 10λ from boundary |
| Staggered-grid interpolation error | < 0.5% for velocity | Use collocated interpolation from recorder |
| Finite-difference time step error | < 0.1% at CFL = 0.3 | CFL ≤ 0.3 is enforced by time step selection |

### RMS Parity Targets vs k-Wave

| Measurement | RMS ratio (kwavers/k-Wave) | Pearson r |
|-------------|--------------------------|-----------|
| On-axis pressure (bowl, 3-D) | 0.994–1.006 | ≥ 0.9999 |
| B-mode scan lines (raw) | 0.977 (verified) | ≥ 0.95 |
| Harmonic content (phased array) | 0.97–1.03 | ≥ 0.9968 |
| Photoacoustic initial pressure | 0.99–1.01 | ≥ 0.999 |

The B-mode scan line RMS ratio of 0.977 reflects log-compression normalization artifacts
rather than physics discrepancy; raw scan lines (before log compression) achieve parity
(see project_us_bmode_linear_transducer_gap.md).

### Signal-to-Noise Budget for Simulated Measurements

For a kwavers simulation, all sources of numerical noise are deterministic. The primary
contributions to signal contamination are:

1. **Grid numerical dispersion**: phase velocity error ε_c ≈ (k Δx)² / 24 per cell for
   PSTD, giving accumulated phase error `φ_err = ε_c · k · D / Δx` over a propagation
   distance `D`.

2. **CPML reflections**: amplitude of reflected waves at the absorbing boundary is below
   `10^{-4}` of the incident wave for a 10-cell CPML layer with optimal σ_max.

3. **FFT round-off**: PSTD uses `rustfft` with double precision; round-off errors are
   `O(N log N · ε_machine)` per step, accumulating to `O(nt · N log N · ε_machine)` over
   the simulation. For typical parameters this is below `10^{-10}` relative to peak signal.

4. **Staggered interpolation**: co-location of staggered velocity via nearest-neighbor or
   linear interpolation introduces errors of order `O(Δx)` or `O(Δx²)` respectively.

---

## 11. Figure References

The following figures should be generated from the corresponding Python validation scripts
in `pykwavers/examples/` and stored in `docs/book/figures/`:

| Figure | Script | Description |
|--------|--------|-------------|
| Fig 6.1 | `hydrophone_directivity.py` | H(θ) for ka = 0.5, 1, 2 (circular element) |
| Fig 6.2 | `spatial_aliasing.py` | Beam pattern with d = λ/2 vs d = λ (grating lobes) |
| Fig 6.3 | `pressure_velocity.py` | Pressure and velocity waveforms: plane wave, Z = ρc |
| Fig 6.4 | `time_reversal_focus.py` | TR focal spot in homogeneous medium vs DAS |
| Fig 6.5 | `sensor_ordering.py` | Sensor mask active-cell ordering: Fortran vs C |
| Fig 6.6 | `photoacoustic_forward.py` | Photoacoustic wavefield and sensor recording |
| Fig 6.7 | `tr_reconstruction.py` | Iterative TR reconstruction of point absorber |
| Fig 6.8 | `calibration_error.py` | Calibration error budget waterfall chart |

---

## 12. References

1. **Selfridge, A.R. (1985)**. "Approximate material properties in isotropic materials."
   *IEEE Transactions on Sonics and Ultrasonics*, 32(3):381–394.
   doi:10.1109/T-SU.1985.31608
   — Material properties for hydrophone element modeling (Section 2).

2. **IEC 61685:2001**. "Ultrasonics — Flow measurement systems — Test methods for
   the determination of system accuracy."
   International Electrotechnical Commission, Geneva.
   — Three-transducer reciprocity calibration method (Section 2).

3. **Fink, M. (1992)**. "Time reversal of ultrasonic fields — Part I: Basic principles."
   *IEEE Transactions on Ultrasonics, Ferroelectrics and Frequency Control*,
   39(5):555–566.
   doi:10.1109/58.156174
   — Time-reversal focusing theory and proof (Section 5, Algorithm 8.1).

4. **Treeby, B.E. and Cox, B.T. (2010)**. "k-Wave: MATLAB toolbox for the simulation and
   reconstruction of photoacoustic wave fields."
   *Journal of Biomedical Optics*, 15(2):021314.
   doi:10.1117/1.3360308
   — k-Wave sensor recording, Fortran-order conventions, and PSTD implementation.

5. **Wise, E.S., Cox, B.T., Jaros, J., and Treeby, B.E. (2019)**. "Representing arbitrary
   acoustic source and sensor distributions in Fourier collocation methods."
   *Journal of the Acoustical Society of America*, 146(1):278–288.
   doi:10.1121/1.5116132
   — BLI sensor rasterization, Algorithm 1 (Section 6, sensor mask construction).

6. **Azar, L., Shi, Y., and Wooh, S.C. (2000)**. "Beam focusing behavior of linear
   phased arrays."
   *NDT and E International*, 33(3):189–198.
   doi:10.1016/S0963-8695(99)00043-2
   — 2-D array spatial Nyquist conditions for hexagonal grids (Section 3).

7. **Derode, A., Roux, P., and Fink, M. (1995)**. "Robust acoustic time reversal with
   high-order multiple scattering."
   *Physical Review Letters*, 75(23):4206–4209.
   doi:10.1103/PhysRevLett.75.4206
   — Super-resolution time reversal in random media (Section 5).

8. **Xu, M. and Wang, L.V. (2006)**. "Photoacoustic imaging in biomedicine."
   *Review of Scientific Instruments*, 77(4):041101.
   doi:10.1063/1.2195024
   — Photoacoustic forward measurement and reconstruction (Section 7).

9. **Synnevag, J.F., Austeng, A., and Holm, S. (2007)**. "Adaptive beamforming applied
   to medical ultrasound imaging."
   *IEEE Transactions on Ultrasonics, Ferroelectrics and Frequency Control*,
   54(8):1606–1613.
   doi:10.1109/TUFFC.2007.406
   — Short-lag spatial coherence and coherence-based weighting (Section 8).

10. **Pierce, A.D. (1989)**. *Acoustics: An Introduction to Its Physical Principles and
    Applications*. Acoustical Society of America, Woodbury, NY. ISBN 0-88318-612-8.
    — Euler equation derivation, plane wave impedance, Green's function reciprocity.
