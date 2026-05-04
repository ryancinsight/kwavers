# Chapter 4: Transducer Arrays and Beamforming

**Scope.** This chapter derives the physics of ultrasound transducer arrays from first
principles: element directivity, array factor, grating lobe conditions, delay-and-sum (DAS)
beamforming, apodization, and Fourier-domain beamforming. Theorems establish spatial
resolution and grating-lobe positions. All derivations connect to
`kwavers::domain::source::transducers` and the BLI rasterizer in
`kwavers::domain::source::kwave_array`.

---

## 4.1 Single-Element Radiation

### 4.1.1 Baffled Circular Piston

A circular piston of radius a vibrating in an infinite rigid baffle is the canonical
transducer element model. The on-axis pressure at distance z from the piston face,
driven at angular frequency ω, is (O'Neil 1949):

```
p(z, ω) = 2ρ₀c₀u₀ sin[(k/2)(√(z²+a²) − z)]                             (4.1)
```

where k = ω/c₀, u₀ is the piston face velocity, and the source pressure P₀ = ρ₀c₀u₀.

**Theorem 4.1 (Far-Field Directivity of a Baffled Piston).** In the far field (kr ≫ 1),
the pressure radiated at angle θ from the piston axis at distance r is

```
p(r, θ, ω) = (ikaP₀/r) exp(ikr) D(θ)                                    (4.2)
```

where the element directivity factor is

```
D(θ) = 2J₁(ka sin θ) / (ka sin θ)                                        (4.3)
```

and J₁ is the Bessel function of the first kind of order 1.

*Proof.* The pressure field of a baffled piston is the Rayleigh integral:

```
p(r, θ, ω) = (ikρ₀c₀u₀/2π) ∫∫_S exp(ik|r−r'|)/|r−r'| dS'             (4.4)
```

In the far field, |r−r'| ≈ r − r'·r̂ (paraxial approximation):

```
p ≈ (ikρ₀c₀u₀ exp(ikr))/(2πr) ∫∫_S exp(−ikr'·r̂) dS'
```

Converting to polar coordinates on the piston face (r', φ') with r̂ = (sin θ, 0, cos θ):

```
∫∫_S exp(−ikr' sin θ cos φ') r'dr'dφ'
  = ∫₀^a r' J₀(kr' sin θ) · 2π dr'   [Jacobi-Anger expansion]
  = 2π a J₁(ka sin θ)/(k sin θ)
```

Substituting and simplifying gives (4.2)–(4.3). □

**Definition 4.1 (−6 dB Half-Angle).** The −6 dB beam half-angle θ₆dB is defined by the
first maximum of D(θ) below 0.5. The half-power point occurs at ka sin θ ≈ 1.616,
giving the approximate half-angle:

```
sin θ₋₆dB ≈ 0.257 λ / a                                                   (4.5)
```

The first null (zero of J₁) is at ka sin θ = 3.832, giving sin θ_null = 0.61 λ/a
(Rayleigh criterion).

### 4.1.2 Rectangular Element Directivity

For a rectangular element of width w (in the steering plane) and height h:

```
D(θ, φ) = sinc(kw sin θ / 2) · sinc(kh sin φ / 2)                       (4.6)
```

where sinc(x) = sin(x)/x. This separable form is used in the `kwave_array` element
models for linear and matrix arrays in `kwavers::domain::source::transducers::rectangular`.

---

## 4.2 Array Factor and the Grating Lobe Theorem

### 4.2.1 Linear Array of N Elements

Consider a linear array of N identical elements with equal pitch d along the x-axis.
Element n is at position x_n = nd (n = 0, …, N−1). For electronic steering to angle θ_s,
element n receives a time delay τ_n = nd sin θ_s / c₀, equivalent to a phase shift
φ_n = knd sin θ_s.

**Definition 4.2 (Array Factor).** The array factor for a uniformly weighted linear array
steered to angle θ_s is

```
AF(θ) = Σ_{n=0}^{N-1} w_n exp(iknd(sin θ − sin θ_s))                    (4.7)
```

where w_n are the element weights (apodization).

**Theorem 4.2 (Grating Lobe Positions).** For a uniform array (w_n = 1) with pitch d,
the array factor (4.7) has magnitude maxima at angles θ satisfying

```
d(sin θ − sin θ_s) = mλ,    m ∈ ℤ                                        (4.8)
```

The m = 0 peak is the main lobe. The m ≠ 0 peaks are **grating lobes**.

*Proof.* AF(θ) is a geometric series:

```
AF(θ) = (1 − exp(iNkd(sin θ − sin θ_s))) / (1 − exp(ikd(sin θ − sin θ_s)))
```

|AF(θ)| is maximized when the denominator vanishes, i.e., when kd(sin θ − sin θ_s) = 2πm,
which is equivalent to (4.8). □

**Corollary 4.1 (Grating-Lobe-Free Condition).** No grating lobe exists within the visible
region |sin θ| ≤ 1 if and only if

```
d ≤ λ / (1 + |sin θ_s|)                                                   (4.9)
```

*Proof.* The nearest grating lobe is at sin θ_GL = sin θ_s ± λ/d. For the lobe to remain
outside the visible region: |sin θ_s + λ/d| > 1, giving d < λ/(1−|sin θ_s|). Similarly
for the negative side; the tighter constraint is (4.9). □

**Corollary 4.2 (λ/2 Rule).** For θ_s = 0 (no steering), the grating-lobe-free condition
is d ≤ λ. For maximum steering θ_s,max → 90°, the condition approaches d ≤ λ/2. Hence the
canonical design choice d = λ/2 is grating-lobe-free for all steering angles in the
half-space.

### 4.2.2 Beam Pattern (Combined Directivity × Array Factor)

The full radiation pattern of the phased array is

```
P(θ) = D(θ) · AF(θ)                                                       (4.10)
```

where D(θ) is the single-element directivity (4.3) or (4.6). D(θ) modulates the envelope
of AF(θ), attenuating grating lobes that fall in the D(θ) sidelobe structure.

---

## 4.3 Focusing

### 4.3.1 Time-Delay Focusing Law

**Theorem 4.3 (Delay Law for a Focused Aperture).** Let element n be at position r_n and
the geometric focus be at position r_f. To achieve constructive interference at r_f, element
n is fired with time delay

```
τ_n = (|r_n − r_f| − min_m |r_m − r_f|) / c₀                            (4.11)
```

*Proof.* The travel time from element n to focus r_f is |r_n − r_f|/c₀. For all elements
to arrive simultaneously, element n must be fired earlier by the excess path
(|r_n − r_f| − min_m|r_m − r_f|)/c₀. Subtracting the minimum makes all delays
non-negative. □

This is the canonical delay law implemented in
`kwavers::domain::source::kwave_array::accessors::beamforming`.

### 4.3.2 Lateral Resolution at Focus

**Theorem 4.4 (Lateral Resolution — −6 dB Beamwidth).** For a rectangular aperture of
width L focused at depth z_f in the far field (z_f ≫ L), the −6 dB lateral resolution is

```
Δx₋₆dB ≈ 0.886 λ z_f / L = 0.886 λ F#                                   (4.12)
```

where F# = z_f/L is the f-number of the transducer.

*Proof.* The pressure at the focal plane is the Fourier transform of the aperture function.
For a uniform aperture of width L, the focal field is a sinc function with first zero at
Δx = λz_f/L. The −6 dB full width (half-amplitude) of |sinc(πx/(λz_f/L))|
corresponds to x/[λz_f/L] = ±0.443, giving FWHM = 2 × 0.443 λ z_f/L = 0.886 λ F#. □

### 4.3.3 Axial Resolution

**Theorem 4.5 (Axial Resolution).** The −6 dB axial resolution for a pulse of N cycles
at center frequency f₀ is

```
Δz₋₆dB ≈ N c₀ / (2 f₀)                                                   (4.13)
```

*Proof.* The envelope of the pulse in range has duration τ_pulse = N/f₀ (N cycles at f₀).
Two reflectors separated by Δz are resolvable when the round-trip delay difference
2Δz/c₀ > τ_pulse/2, giving Δz = Nc₀/(4f₀). For a Hann-windowed pulse the −6 dB
criterion yields the factor of 1/2 in the denominator. □

| Quantity | Formula | Example (5 MHz, F# 2, N=2 cycles) |
|----------|---------|----------------------------------|
| Lateral −6 dB FWHM | 0.886 λ F# | 0.886 × 0.31 mm × 2 = 0.55 mm |
| Axial −6 dB | N c₀/(2f₀) | 2 × 1540/(2×5×10⁶) = 0.31 mm |

---

## 4.4 Apodization

### 4.4.1 Window Functions

**Definition 4.3 (Apodization).** Apodization refers to the application of a window
function w_n to element weights before beamforming, suppressing sidelobes at the cost of
a broadened main lobe.

Common apodization windows and their sidelobe performance:

| Window | −6 dB width (rel. uniform) | Peak sidelobe (dB) | Code key |
|--------|---------------------------|-------------------|----------|
| Uniform (rectangular) | 1.00 | −13.2 | `Rectangular` |
| Hann | 1.63 | −31.5 | `Hann` |
| Hamming | 1.47 | −41.8 | `Hamming` |
| Blackman | 1.97 | −57.3 | `Blackman` |
| Tukey (cosine fraction) | 1.0 – 1.63 | −13.2 to −31.5 | `Tukey(r)` |

**Theorem 4.6 (Sidelobe–Resolution Trade-off).** For any apodization window with spectrum
W(k_x), the product of the −6 dB main-lobe width BW₋₆dB and the peak sidelobe level PSL
is bounded below by a constant depending on the window family. No window simultaneously
achieves BW → 0 and PSL → −∞.

*Proof sketch.* By the uncertainty principle (Parseval + Cauchy-Schwarz on window and
its Fourier transform), the spatial-frequency bandwidth product σ_x × σ_{kx} ≥ 1/2.
The direct analog holds for aperture apodization and its radiation pattern. □

**Corollary 4.3.** In tissue harmonic imaging (Chapter 5), transmit apodization is
typically Hann to limit clutter, while receive apodization is also Hann or Blackman for
further sidelobe reduction, at the cost of ≈ 1.6× lateral resolution degradation relative
to a uniform transmit-uniform receive combination.

### 4.4.2 kwavers Apodization Implementation

Apodization is implemented in `kwavers::domain::source::transducers::apodization`.
The `PhasedArrayConfig` stores weight arrays computed at construction time; the beamforming
module in `kwavers::domain::source::transducers::phased_array::beamforming` applies them
when computing transmit delays and signals.

---

## 4.5 Receive Beamforming: Delay-and-Sum

### 4.5.1 DAS Algorithm

**Algorithm 4.1 (Delay-and-Sum Beamforming).**

```
Input:  RF data s_n(t) from N receive elements; sound speed c₀; focus geometry
Output: Beamformed RF signal b(t) at each image point x_p

For each image point x_p:
    For each receive element n = 0..N-1:
        τ_n = 2|r_n − x_p| / c₀         (round-trip delay, Eq. 4.11)
        Compute interpolated sample: ŝ_n = s_n(t − τ_n)
        Apply receive weight w_n (apodization)
    b(t) = Σ_n w_n ŝ_n                  (coherent sum)
```

**Theorem 4.7 (DAS Coherent Signal Enhancement).** If all elements receive independent
zero-mean noise with variance σ² and coherent signal amplitude A, the post-DAS SNR is

```
SNR_DAS = N A² / σ²                                                       (4.14)
```

which scales linearly in array size N.

*Proof.* Signal sums coherently: |Σ_n w_n A|² = N²A² for uniform weights. Noise
variance sums incoherently: Var(Σ_n w_n ε_n) = Nσ². SNR = N²A²/(Nσ²) = NA²/σ². □

### 4.5.2 Fourier-Domain Beamforming (f-k Migration)

For a flat reflector model, the received signal spectrum S(k_x, ω) (spatial frequency
k_x = ω sin θ_r / c₀) is related to the reflectivity function R(k_x, k_z) via a
Stolt migration mapping (Stolt 1978):

```
k_z = √((2ω/c₀)² − k_x²) / 2                                             (4.15)
```

This Fourier-domain migration is exact for homogeneous media and is the basis of
synthetic aperture imaging and coherent plane-wave compounding (Chapter 5).

### 4.5.3 MVDR Adaptive Beamformer

**Definition 4.4 (MVDR Beamformer).** The Minimum Variance Distortionless Response
beamformer selects weights w to minimize output power while preserving a unit response
in the steering direction a(θ_s):

```
min_w  w^H R_xx w    subject to  w^H a(θ_s) = 1                          (4.16)
```

with solution w_MVDR = R_xx^{-1} a / (a^H R_xx^{-1} a), where R_xx is the sample
covariance matrix of the channel data.

MVDR achieves substantially lower sidelobes than DAS by adapting to interference, at
the cost of requiring a stable covariance matrix estimate (typically ≥ 2N snapshots).
In clinical applications R_xx is regularized: R_xx → R_xx + ε·I with ε ~ 1/30 × λ_max.

---

## 4.6 Bandlimited Interpolation (BLI) Source Rasterization

### 4.6.1 Mathematical Basis

Continuous-domain transducer surfaces must be projected onto the discrete computational
grid. Nearest-neighbor rasterization introduces error proportional to the grid spacing;
BLI rasterization, following Wise et al. (2019), achieves spectral accuracy.

**Theorem 4.8 (BLI Rasterization Accuracy — Wise 2019).** For a continuous source
distribution p_s(r) with highest spatial frequency k_max, the BLI representation on a
grid with spacing Δx satisfies

```
‖p_s − p_BLI‖₂ / ‖p_s‖₂ ≤ C (k_max Δx)^(2N_sub)                       (4.17)
```

where N_sub = ⌈1/(π ε_BLI)⌉ is the stencil half-width and ε_BLI is the BLI tolerance.

*Proof sketch.* The BLI kernel is the ideal (sinc-based) interpolation kernel truncated
to a window of width 2N_sub. The truncation error in the Fourier domain is bounded by the
energy in the spectral tail beyond k_max, which decays faster than any polynomial in Δx
for a band-limited function. □

**Definition 4.5 (BLI Stencil Weight).** For a surface sample at continuous coordinate x,
the weight at grid index i is

```
w_i = sinc(π(x_i − x) / Δx)                                             (4.18)
```

where sinc(u) = sin(u)/u. The 3-D weight is the tensor product w_x · w_y · w_z.

In kwavers, `KWaveArray::map_surface_sample()` in
`kwavers::domain::source::kwave_array::bli_kernel` implements (4.18) with a window of
N_sub = ⌈1/(π × 0.05)⌉ = 7 grid cells (default `DISC_BLI_TOLERANCE = 0.05`).

### 4.6.2 Disc Packing for Curved Surfaces

For a curved transducer element (focused bowl, arc, annular ring), the surface is
discretized into a set of point samples. kwavers uses a hexagonal close-packing of discs
on the element surface with packing number `DISC_PACKING_NUMBER = 3`, ensuring sub-grid
coverage without oversampling. Each disc contributes one BLI-stencil call. The element
area is preserved: the number of discs × disc area ≈ element area.

---

## 4.7 Multi-Element Array Configurations

### 4.7.1 Linear Array

A 1-D linear array of N elements, each of width w and pitch d (d ≥ w), provides:
- Electronic beam steering in the azimuth plane (along x)
- Fixed elevation focus determined by element height h
- Lateral resolution: 0.886 λ F# in azimuth; fixed in elevation

**Elevation resolution.** The elevation (y) beam is not dynamically focused in a standard
linear array. The elevation −6 dB beamwidth is set by the element height h:

```
Δy ≈ 0.886 λ z_f / h    (fixed-focus elevation)                          (4.19)
```

At depths away from the fixed-focus elevation lens, the elevation beam broadens,
reducing slice-thickness resolution.

### 4.7.2 1.5-D and Matrix Arrays

A 1.5-D array adds a small number of elevation rows (typically 3–7) to provide limited
dynamic elevation focusing. A full matrix array (N_x × N_y elements) provides fully
dynamic 2-D steering and focusing for volumetric imaging.

| Array type | Azimuth steering | Elevation steering | Elements | Volume imaging |
|------------|-----------------|-------------------|----------|----------------|
| Linear 1-D | ✓ dynamic | ✗ (fixed lens) | 64–512 | ✗ |
| 1.5-D | ✓ dynamic | ✓ limited | 3–7 rows | ✗ |
| Matrix 2-D | ✓ dynamic | ✓ dynamic | N² (thousands) | ✓ |
| Row-column (cross) | ✓ dynamic | ✓ dynamic | 2N | ✓ (limited) |

### 4.7.3 Annular Array and Axisymmetric Focusing

An annular array consists of N concentric rings centered on the acoustic axis. Each ring
generates a spherically symmetric beam at its own focal depth, enabling dynamic depth
focusing with electronic switching. The kwavers annular array implementation in
`kwavers::domain::source::kwave_array` places rings at half-cell-offset grid origins
with centering convention matching k-Wave (integer N/2 centering).

---

## 4.8 Focused Transducer (Bowl / Arc / Hemispherical)

### 4.8.1 Geometric Focus

**Definition 4.6 (Focal Length and F-Number).** For a spherically focused transducer with
aperture radius a and radius of curvature R (focal length), the f-number is F# = R/(2a).

**Theorem 4.9 (On-Axis Pressure Enhancement at Focus).** For a continuous spherically
focused aperture of aperture radius a and focal radius R_f in a lossless medium, the
on-axis pressure gain at the geometric focus relative to the piston face pressure P₀ is

```
G = k a² / (2 R_f)  =  π a² / (λ R_f)                                   (4.20)
```

*Proof.* Every surface element contributes amplitude P₀ dS/(4πR_f) and travels path R_f
to the focus. Integrating over the aperture area πa² and multiplying by the coherence
factor k (from the i·k·exp(ikR_f)/R_f prefactor in the Rayleigh integral) gives G in (4.20).
For HIFU with a = 30 mm, R_f = 60 mm, f = 1 MHz: G = π(0.03)²/(1.5×10⁻³ × 0.06) ≈ 31. □

### 4.8.2 HIFU Focused Bowl

For HIFU applications, a single-element focused bowl of aperture diameter 2a and focal
length R_f produces a focal pressure:

```
p_focus = G × P₀ = (πa²/λR_f) × ρ₀c₀u₀                                 (4.21)
```

Focal gains of 10–50 are typical for therapy transducers. The HIFU bowl in kwavers is
defined in `kwavers::domain::source::transducers::focused::bowl`.

---

## 4.9 Grating Lobe Validation Protocol

**Algorithm 4.2 (Grating Lobe Acceptance Test).**

```
Input:  N-element array, pitch d, center frequency f₀, steering angle θ_s
Output: pass / fail

1. Compute theoretical grating lobe angles from (4.8): sin θ_GL = sin θ_s + m λ/d
2. Run kwavers phasedArray simulation with the same parameters.
3. Extract pressure field at the focal depth over the aperture half-angle.
4. For each theoretical grating lobe angle θ_GL:
   a. Compute simulated pressure at (z_f sin θ_GL, 0, z_f).
   b. Compare with AF(θ_GL) × D(θ_GL) prediction (Eq. 4.10).
   c. Require |P_sim − P_theory| / P_main < 0.05.
5. Verify grating lobe is absent when d = λ/2 (Corollary 4.2).
```

---

## 4.10 Code Mapping

| Concept | kwavers module | Key struct/fn |
|---------|---------------|---------------|
| Phased array | `transducers::phased_array` | `PhasedArrayTransducer` |
| Beamforming modes | `phased_array::beamforming` | `BeamformingMode::{DAS, Fourier, Hadamard}` |
| Apodization | `transducers::apodization` | Window enum |
| Single-element directivity | `transducers::physics::directivity` | `directivity_factor()` |
| Focused bowl | `transducers::focused::bowl` | `FocusedBowl` |
| Arc / ring | `transducers::focused::arc` | `FocusedArc` |
| Annular array | `kwave_array` | `KWaveArray` (annular geometry) |
| BLI rasterization | `kwave_array::bli_kernel` | `map_surface_sample()` |
| Delay law (Eq. 4.11) | `kwave_array::accessors::beamforming` | `compute_delays()` |
| Linear array | `basic::linear_array` | `LinearArray` |
| Matrix array | `basic::matrix_array` | `MatrixArray` |

---

## 4.11 Worked Example: Linear Array Steering and Grating Lobe

**Setup.** 64-element linear array, f₀ = 5 MHz, λ = 0.308 mm, element pitch d = 0.3 mm
(≈ 0.97 λ), uniform apodization, steering angle θ_s = 20°.

**Grating lobe positions (Eq. 4.8, m = ±1):**

```
sin θ_GL = sin 20° + 1 × (λ/d) = 0.342 + (0.308/0.300) = 0.342 + 1.027 → outside visible
sin θ_GL = sin 20° − 1 × (λ/d) = 0.342 − 1.027 = −0.685 → θ_GL = −43.2°
```

At d ≈ λ, the grating lobe enters the visible region for θ_s > 0. Reducing pitch to
d = λ/2 = 0.154 mm eliminates the grating lobe (Corollary 4.2).

**Lateral resolution at 30 mm depth, L = 64 × 0.3 mm = 19.2 mm, F# = 30/19.2 = 1.56:**

```
Δx₋₆dB = 0.886 × 0.308 mm × 1.56 = 0.425 mm
```

With Hann apodization the lateral beamwidth widens by 1.63× to 0.69 mm but the grating
lobe level drops by 18.3 dB (Theorem 4.6).

---

## Appendix 4A: Original Scope (Advanced Beamforming)

This chapter's original scope note included MVDR, subspace methods, plane-wave
compounding, SAFT, 3-D beamforming, neural beamforming, clutter filtering, and
localization. These topics are addressed in Chapter 5 (Ultrasound Imaging) and their
deeper treatment in `kwavers::analysis::signal_processing::beamforming`,
`kwavers::clinical::imaging`, and `kwavers::analysis::ml`.

- ULTRA-SR ULM benchmark: https://doi.org/10.1109/TMI.2024.3388048
- Row-column 3-D super-resolution ultrasound: https://doi.org/10.1016/j.ultrasmedbio.2024.03.020

---

## References

1. O'Neil, H. T. (1949). Theory of focusing radiators. *J. Acoust. Soc. Am.*,
   **21**(5), 516–526. https://doi.org/10.1121/1.1906542

2. Szabo, T. L. (2014). *Diagnostic Ultrasound Imaging: Inside Out* (2nd ed.).
   Academic Press. Chapters 7–9.

3. Jensen, J. A. (1996). Field: A program for simulating ultrasound systems.
   *10th Nordic-Baltic Conf. Biomed. Imag.*, Vol. 34, Suppl. 1, 351–353.

4. Stolt, R. H. (1978). Migration by Fourier transform. *Geophysics*, **43**(1), 23–48.
   https://doi.org/10.1190/1.1440826

5. Wise, E. S., Cox, B. T., Jaros, J., & Treeby, B. E. (2019). Representing arbitrary
   acoustic source and sensor distributions in Fourier collocation methods.
   *J. Acoust. Soc. Am.*, **146**(1), 278–288. https://doi.org/10.1121/1.5116132

6. Montaldo, G., Tanter, M., Bercoff, J., Benech, N., & Fink, M. (2009). Coherent
   plane-wave compounding for very high frame rate ultrasonography and transient
   elastography. *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*, **56**(3), 489–506.
   https://doi.org/10.1109/TUFFC.2009.1067

7. Capon, J. (1969). High-resolution frequency-wavenumber spectrum analysis.
   *Proc. IEEE*, **57**(8), 1408–1418. https://doi.org/10.1109/PROC.1969.7278
