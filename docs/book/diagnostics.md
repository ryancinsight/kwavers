# Chapter 5: Ultrasound Imaging

**Scope.** This chapter covers the full diagnostic ultrasound imaging pipeline: B-mode
pulse-echo, Doppler velocity estimation, contrast-enhanced ultrasound (CEUS), photoacoustic
imaging, shear-wave elastography, and ultrasound localization microscopy (ULM). Each
modality is derived from first principles with formal theorems. Code references map to
`kwavers::clinical::imaging` and `kwavers::analysis::signal_processing`.

---

## 5.1 B-Mode Pulse-Echo Imaging

### 5.1.1 Signal Model

In a pulse-echo configuration the transmitted pulse p_tx(t) propagates into tissue,
scatters from heterogeneities, and returns to the receive aperture. Ignoring multiple
scattering, the received signal from a point scatterer at depth z is

```
s(t) = p_tx(t − 2z/c₀) · r(z) · α_geo(z)                                (5.1)
```

where r(z) is the reflection coefficient at depth z, and α_geo(z) ∝ 1/z is the
geometric spreading factor.

**Definition 5.1 (Point Spread Function).** The PSF of a pulse-echo system is the
response to a single point scatterer:

```
h(x, z) = h_lat(x, z) · h_ax(z)                                           (5.2)
```

where h_lat(x,z) is the lateral beam profile (Chapter 4, Theorem 4.4) and h_ax(z) is
the axial pulse envelope.

### 5.1.2 Time-Gain Compensation

Tissue absorption attenuates the signal as exp(−α₀ f z) (α₀ in dB/cm/MHz, z in cm). The
time-gain compensation (TGC) amplification is

```
TGC(t) = exp(+α₀ f c₀ t / 2)                                             (5.3)
```

compensating round-trip attenuation. The factor of 2 accounts for the round-trip path.

### 5.1.3 Envelope Detection

The B-mode image is the envelope of the beamformed RF signal. Denoting the analytic
signal z(t) = s(t) + i·H{s(t)} (where H is the Hilbert transform):

```
env(t) = |z(t)| = √(s²(t) + H{s}²(t))                                   (5.4)
```

**Theorem 5.1 (Hilbert Transform Envelope).** For a narrowband signal s(t) = A(t)cos(ω₀t + φ),
the analytic signal envelope is |z(t)| = A(t).

*Proof.* The Hilbert transform of cos(ω₀t + φ) is sin(ω₀t + φ). Therefore
z(t) = A(t)(cos + i·sin)(ω₀t + φ) = A(t)exp(i(ω₀t + φ)), and |z| = A(t). □

The log-compressed display dynamic range is typically 40–60 dB:

```
B(t) = 20 log₁₀(env(t) / env_max)    [dB]                               (5.5)
```

### 5.1.4 Lateral Resolution and Contrast

**Definition 5.2 (Contrast-to-Noise Ratio).** For a lesion with mean intensity μ_l
and background mean μ_b and standard deviation σ_b:

```
CNR = |μ_l − μ_b| / σ_b                                                   (5.6)
```

**Definition 5.3 (Contrast Ratio).** CR = 20 log₁₀(μ_l/μ_b) [dB].

---

## 5.2 Plane-Wave Coherent Compounding

### 5.2.1 Plane-Wave Transmit

**Definition 5.4 (Plane-Wave Transmit).** A plane-wave transmit at steering angle θ_i
fires all array elements simultaneously with linear phase ramp. The received channel data
s_i,n(t) are collected for each transmit angle θ_i and receive element n.

**Theorem 5.2 (Plane-Wave Compounding SNR).** Let s_i(r) be the complex beamformed image
for transmit angle θ_i (i = 1,…,N_c). Under coherent compounding:

```
S(r) = Σ_{i=1}^{N_c} s_i(r)                                              (5.7)
```

If speckle noise is zero-mean and independent across angles, the compounded SNR scales as

```
SNR_comp = √N_c × SNR_single                                               (5.8)
```

*Proof.* Signal sums coherently: E[S] = N_c μ. Noise variance sums incoherently:
Var(Σ_i n_i) = N_c σ². SNR = N_c μ / (√(N_c)σ) = √N_c × μ/σ. □

**Corollary 5.1.** N_c = 16 plane-wave angles yield √16 = 4× SNR improvement over a
single plane wave, recovering focused-transmit image quality at a higher frame rate
(frame rate = PRF/N_c versus PRF/(N_scanlines) for focused mode).

### 5.2.2 Fourier-Domain Reconstruction (f-k)

The Stolt mapping (Chapter 4, Eq. 4.15) extends to each plane-wave transmit angle:

```
k_z(ω, k_x) = √((ω/c₀)² − k_x²) + √((ω/c₀)² − (k_x − k_x,tx)²)       (5.9)
```

where k_x,tx = (ω/c₀)sinθ_i is the transmit plane-wave spatial frequency. This
F-K migration is implemented in `kwavers::clinical::imaging::workflows::plane_wave_compounding`.

---

## 5.3 Doppler Ultrasound

### 5.3.1 Doppler Frequency Shift

**Theorem 5.3 (Doppler Shift).** A scatterer moving at velocity v along the beam axis
(angle α to beam) returns a signal shifted by

```
f_D = 2 f₀ v cos α / c₀                                                   (5.10)
```

*Proof.* In the scatterer frame, the incident frequency is f₀(1 + v·cos α/c₀) (Doppler
effect for moving receiver). The scattered wave is further Doppler-shifted by the moving
source: f_r = f₀(1 + v·cos α/c₀)² ≈ f₀(1 + 2v·cos α/c₀) for v ≪ c₀. Subtracting f₀:
f_D = 2f₀ v·cos α/c₀. □

**Definition 5.5 (Maximum Unambiguous Velocity).** For a pulsed wave system with pulse
repetition frequency PRF, the maximum velocity without aliasing (Nyquist limit) is:

```
v_max = c₀ PRF / (4 f₀ cos α)                                             (5.11)
```

The maximum range (depth) is z_max = c₀/(2 PRF). These constraints form the
range-velocity uncertainty of pulsed Doppler.

### 5.3.2 Autocorrelation Velocity Estimator

**Algorithm 5.1 (Kasai Autocorrelation Estimator — Kasai et al. 1985).**

```
Input:  I+Q channel data x_m(t) for ensemble m = 1..M (slow-time packets)
Output: Mean velocity estimate v̂

1. Form complex signal z_m(t) = I_m(t) + i·Q_m(t)
2. Compute lag-1 autocorrelation: R(1) = Σ_{m=1}^{M-1} z_m+1(t) · z_m*(t)
3. Phase estimate: φ̂ = arg(R(1))
4. Velocity estimate: v̂ = c₀ φ̂ / (4π f₀ cos α T_prf)
```

**Theorem 5.4 (Autocorrelation Estimator Variance).** For M ensemble members and
signal-to-clutter ratio SCR, the variance of the velocity estimate is

```
Var(v̂) ≈ v_max² (1 − |R(1)|²) / (π² M |R(1)|²)                         (5.12)
```

*Proof.* The Doppler phase of the lag-1 autocorrelation `R(1) = |R(1)|exp(iΦ)` encodes
velocity as `Φ = 2π f_d T_PRF = π v / v_max`, where `v_max = c₀/(4 f₀ T_PRF)` is the
Nyquist velocity.  For `M` independent ensemble samples at high SCR, `Φ` has Gaussian
distribution with variance `σ²_Φ = (1 − |R(1)|²) / (M |R(1)|²)` by the Cramér-Rao
bound for circular phase estimation (Kay 1993, §3.7).  The velocity is a linear function
of phase: `v = (v_max/π) Φ`, so the velocity variance is

```
Var(v̂) = (v_max/π)² · σ²_Φ = v_max²(1 − |R(1)|²) / (π² M |R(1)|²).
```

The Jacobian `dv/dΦ = v_max/π` is the phase-to-velocity scale factor from the Doppler
relation above. □

Implemented in `kwavers::clinical::imaging::doppler::autocorrelation`.

### 5.3.3 Color Flow Mapping

Color flow imaging applies the Kasai estimator to every pixel in a 2-D frame. The
wall filter (`kwavers::clinical::imaging::doppler::wall_filter`) suppresses slow-moving
tissue clutter (typically by high-pass FIR filter with cutoff ≈ 100–500 Hz) before the
autocorrelation step.

---

## 5.4 Contrast-Enhanced Ultrasound (CEUS)

### 5.4.1 Microbubble Scattering

**Theorem 5.5 (Rayleigh–Plesset Bubble Dynamics — linearized).** For a microbubble of
equilibrium radius R₀, shell stiffness χ, shell viscosity κ_s, and internal gas pressure
p_gas, driven by an incident pressure p_inc(t), the linearized radius perturbation
x = R − R₀ satisfies

```
ρ_l R₀ ẍ + (4κ_s/R₀² + 4μ_l/R₀) ẋ + (3κ_p0/(R₀²) + 2χ/R₀) x = −p_inc(t)  (5.13)
```

where ρ_l is liquid density, μ_l is liquid viscosity, p_0 is ambient pressure, and
κ = 3γ (polytropic index × initial gas pressure).

The natural frequency of the bubble is

```
f_0^{bubble} = (1/2πR₀) √(3κp_0/ρ_l + 2χ/(ρ_l R₀))                     (5.14)
```

For SonoVue-type microbubbles (R₀ ≈ 1–3 μm, χ ≈ 0.5 N/m): f₀ ≈ 2–5 MHz, matching
diagnostic frequencies.

*Proof sketch.* The full Rayleigh-Plesset equation linearized about R₀ gives a damped
harmonic oscillator (5.13). Natural frequency follows from the restoring coefficient. □

**Theorem 5.6 (Scattered Pressure from a Single Bubble).** In the far field
(`kr ≫ 1`, equivalently `r ≫ λ/(2π) ≈ R₀/kR₀`; for `R₀ = 2 μm` and `f = 2 MHz`,
`λ/(2π) ≈ 119 μm`, so the approximation holds for `r ≳ 1 mm` — well within
clinical imaging depth), the scattered pressure is

```
p_s(r) = ρ_l R₀ R̈ / r · exp(−ikr)                                        (5.15)
```

proportional to the bubble wall acceleration R̈. *Proof sketch:* The full multipole
expansion of the radiated pressure retains only the monopole term at `kr ≪ 1` bubble
size (source compactness), giving `p_s = ρ_l Ṡ_bubble / (4πr)` where `Ṡ = d²(4πR³/3)/dt²
= 4π R₀(2Ṙ² + RR̈) ≈ 4π R₀² R̈` (linearized about `R₀` with `Ṙ ≪ 1`). □

Below the resonance frequency this scales as `ω² R₀³` (Rayleigh scattering).
At resonance the scattering cross-section far exceeds the geometric cross-section
(`σ_s ≫ πR₀²`).

**Scope note:** Eq. (5.13) assumes shell stiffness χ is a constant (linear shell
model).  At mechanical index MI > 0.3, lipid and polymer shells exhibit
strain-dependent stiffness and rupture; the linear model loses validity exactly where
clinical contrast imaging is performed.  The kwavers implementation uses the constant-χ
linearized model for resonance frequency estimation and `BubbleDynamics` uses the
full nonlinear RP ODE for CEUS/cavitation simulation.

### 5.4.2 Nonlinear Bubble Scattering for Contrast Imaging

At higher driving pressures (MI > 0.1), the bubble response becomes nonlinear and
generates sub-harmonics (f₀/2), ultra-harmonics (3f₀/2), and super-harmonics (2f₀, 3f₀).
Clinical CEUS receives the fundamental or second harmonic with tissue suppression:

```
CTR = 20 log₁₀(p_bubble_2f / p_tissue_2f)    [dB]                        (5.16)
```

Tissue harmonic ratio p_tissue_2f is set by B/A ≈ 6 (Eq. 3.25).
Bubble harmonic ratio p_bubble_2f is enhanced by resonance by a factor of Q = f₀/(Δf).
Typical CTR at second harmonic: 15–25 dB.

### 5.4.3 Ultrasound Localization Microscopy (ULM)

**Definition 5.6 (ULM Resolution Limit).** At ultra-low microbubble concentration (one
bubble per resolution cell), the center of each bubble PSF can be localized to:

```
σ_loc ≈ FWHM / (2.35 √SNR)                                                (5.17)
```

where FWHM is the diffraction-limited PSF width. For SNR = 25 dB (316), FWHM = 200 μm:
σ_loc ≈ 200/(2.35 × 17.8) ≈ 5 μm. This is 40× sub-diffraction resolution.

**Algorithm 5.2 (ULM Processing Pipeline).**

```
Input:  Ultrafast plane-wave sequence s_frame(t, n, θ) at frame rate ≥ 500 Hz
Output: Super-resolved vascular map

1. CLUTTER FILTER: SVD decomposition of spatio-temporal data matrix;
   retain singular vectors beyond the tissue subspace (typically sv ≥ 10).
2. LOCALIZE: fit Gaussian PSF model to isolated bright spots.
   Report: centroid (x,z), amplitude, width.
3. FILTER: discard localizations with |width − FWHM_expected| > 30%.
4. TRACK: assign localizations across frames using Hungarian algorithm.
   Motion model: constant velocity Kalman filter.
5. ACCUMULATE: bin localizations to 10× oversampled grid.
   Velocity map: mean trajectory velocity per bin.
6. VALIDATE: σ_loc < FWHM/10, track length ≥ 3 frames, flow continuity ≥ 0.9.
```

Implemented in `kwavers::clinical::imaging::functional_ultrasound::ulm`.

---

## 5.5 Photoacoustic Imaging

### 5.5.1 Photoacoustic Wave Generation

**Theorem 5.7 (Photoacoustic Wave Equation).** Under thermal confinement (τ_pulse ≪ τ_th)
and stress confinement (τ_pulse ≪ τ_s), optical absorption generates an initial pressure:

```
p₀(r) = Γ μ_a(r) Φ(r)                                                     (5.18)
```

where Γ = β c₀²/C_p is the Grüneisen parameter (dimensionless), μ_a(r) is the optical
absorption coefficient [m⁻¹], and Φ(r) is the local fluence [J m⁻²].

*Proof.* Under stress confinement, the thermoelastic stress is p₀ = −K β ΔT where K is
the bulk modulus, β is the thermal expansion coefficient, and ΔT = μ_a Φ / (ρ C_p).
Substituting K = ρ c₀² and Γ = β K/(ρ C_p) = β c₀²/C_p gives (5.18). □

**Definition 5.7 (Grüneisen Parameter).** Γ = β c₀²/C_p, with β [K⁻¹] the thermal
expansion coefficient, c₀ [m/s] the sound speed, C_p [J/(kg·K)] the specific heat capacity.

| Tissue | Γ | Notes |
|--------|---|-------|
| Water (37°C) | 0.12 | Baseline |
| Whole blood | 0.18 | Hemoglobin-dominant |
| Fat | 0.70 | Elevated β |
| Breast tissue | 0.15 | Mixed |

### 5.5.2 Reconstruction

The received photoacoustic signal at sensor position r_s is:

```
p(r_s, t) = ∂/∂t [t/(4πc₀²) ∫_{|r−r_s|=c₀t} p₀(r) dΩ]                (5.19)
```

(spherical Radon transform). Time-reversal reconstruction recovers p₀(r) by running the
acoustic propagation backward in time, implemented in
`kwavers::clinical::imaging::reconstruction::acoustic_projection`.

### 5.5.3 Spectroscopic PA Imaging

The absorption spectrum of hemoglobin (HbO₂ / Hb) enables blood oxygen saturation
measurement. At two wavelengths λ₁, λ₂:

```
[μ_a(λ₁)]   [ε_HbO₂(λ₁)  ε_Hb(λ₁)] [c_HbO₂]
[μ_a(λ₂)] = [ε_HbO₂(λ₂)  ε_Hb(λ₂)] [c_Hb  ]                           (5.20)
```

Solving this 2×2 system via Tikhonov regularization gives c_HbO₂ and c_Hb, hence
sO₂ = c_HbO₂/(c_HbO₂ + c_Hb). Implemented in
`kwavers::clinical::imaging::spectroscopy::solvers::tikhonov`.

---

## 5.6 Shear-Wave Elastography

### 5.6.1 Shear Wave Generation and Speed

**Theorem 5.8 (Shear Wave Speed and Stiffness).** In a linear elastic incompressible medium
the shear wave phase velocity c_s and Young's modulus E are related by:

```
c_s = √(μ/ρ)    E = 3μ    G = μ                                           (5.21)
```

where μ is the shear modulus, G = μ the shear modulus, and E = 3G for incompressible tissue.

*Proof.* In an incompressible Kelvin-Voigt viscoelastic solid, the shear wave dispersion
relation is k² = ρω²/(μ + iωη) where η is the shear viscosity. In the purely elastic
limit (η = 0): c_s = ω/k = √(μ/ρ). □

| Tissue type | Shear modulus μ (kPa) | c_s (m/s) | Stiffness class |
|-------------|----------------------|-----------|-----------------|
| Normal liver | 0.8–2 | 0.9–1.4 | Soft |
| Early fibrosis | 2–5 | 1.4–2.2 | Moderate |
| Advanced cirrhosis | 5–15 | 2.2–3.9 | Stiff |
| Breast fat | 0.5–1.5 | 0.7–1.2 | Very soft |
| Breast cancer | 20–200 | 4.5–14 | Hard |

### 5.6.2 Acoustic Radiation Force Impulse (ARFI)

High-intensity focused ultrasound creates an acoustic radiation force F = 2α I/c₀ [N/m³]
that displaces tissue axially. The subsequent shear wave propagation is tracked with
ultrafast imaging. The peak displacement at the focus is

```
u_peak = F Δt / (2ρ c_s)   ≈ α I₀ Δt / (ρ c₀ c_s)                      (5.22)
```

where Δt is the push pulse duration (~1 ms) and I₀ is the focal intensity.

---

## 5.7 Ultrasonic Speed-of-Sound Shift Imaging

Speed-of-sound shift imaging estimates a perturbation map δc(r) = c(r) − c₀ from
differential ultrasonic travel-time shifts measured across a transmit/receive aperture.
It is distinct from B-mode reflectivity: the data are phase or time shifts relative to a
reference sound speed, and the image is a quantitative sound-speed contrast.

For a straight transmit/receive ray Γᵣ with observed-minus-reference time shift δtᵣ:

```
δtᵣ = ∫_{Γᵣ} [1/(c₀ + δc(r)) − 1/c₀] ds
    ≈ −1/c₀² ∫_{Γᵣ} δc(r) ds                                      (5.23)
```

After voxelization with exact ray/pixel intersection lengths ℓᵣᵥ:

```
Σ_v ℓᵣᵥ δc_v = −c₀² δtᵣ                                           (5.24)
```

**Dense regime.** Dense shift imaging uses all measured aperture rows and solves:

```
min_δc  1/2 ||Aδc − b||₂² + λ/2 ||δc||₂² + γ/2 δcᵀLδc              (5.25)
```

where L is the active-pixel graph Laplacian. This produces a spatially dense quantitative
sound-speed shift field for diffuse thermal, fat, fibrosis, or skull-aberration contexts.

**Sparse regime.** Sparse shift imaging uses a deterministic row subset and an L1 prior:

```
min_δc  1/2 ||A_Sδc − b_S||₂² + λ/2 ||δc||₂² + γ/2 δcᵀLδc
        + μ ||δc||₁                                                 (5.26)
```

This targets compact lesions, focal ablation changes, or localized acoustic-property
changes when a full aperture sweep is not available. The implementation keeps dense and
sparse behavior in one API: `ShiftSampling` selects measurement rows and `ShiftPrior`
selects the image prior, while the forward operator remains the same straight-ray
speed-shift model. Ray assembly traverses only the crossed grid cells and stores
nonzero segment lengths in flat row-offset, column, and length arrays, so
construction and operator storage scale with ray path length rather than the full
active-pixel count. Iterative inversion uses caller-reusable PCG/ISTA scratch
buffers through `SoundSpeedShiftWorkspace`, so repeated dense or sparse
reconstructions over compatible grids retain row, Laplacian, prediction,
residual, diagonal, and power-iteration allocations instead of rebuilding them
for each solve.

**Fixed-acquisition planning.** When the aperture, active mask, propagation
model, sensitivity model, and sampling policy are fixed across a time series,
`SoundSpeedShiftPlan` caches the generated acquisition rows and the assembled
operator. A frame then supplies only the measured shift vector in original
acquisition order:

```
b_S^(n)[r] = -c0^2 Delta t_i(r)^(n)
```

where `i(r)` is the original acquisition-row index retained by the sampled
operator row. This preserves the inverse model in (5.25)-(5.26) while
amortizing curved-array row generation, curved-ray subsegment traversal, and
finite-frequency weight assembly across repeated frames.
For frame batches, the same plan drives all frames through one solver workspace
and one reusable sampled-row RHS buffer. `SoundSpeedShiftBatchConfig` defaults
to compact objective retention:

```
summary_n = (J_n(0), J_n(final), iteration_count_n).                         (5.27)
```

Full per-frame objective histories are retained only when explicitly requested,
so long time series do not store every solver objective value by default.

**Curved-array acquisition.** A 2-D curved array is represented as a circular
arc:

```
p_i = c + r[cos(theta_0 + i Delta theta), sin(theta_0 + i Delta theta)]      (5.28)
```

`CurvedArrayShiftScan` emits same-aperture pitch-catch rows in deterministic
transmitter-major order with receiver index `(i + offset) mod N`. These rows
are ordinary `SoundSpeedShiftSample` values, so curved-array acquisition does
not create a separate inverse model; it supplies curved-array transmitter and
receiver coordinates to the same straight-ray operator in (5.24)-(5.26).

**Curved-ray and finite-frequency sensitivity.** The same reconstruction API
also supports a circular-arc propagation row:

```
Γ_r = union_k [q_k, q_{k+1}],   q_k on the transmitter-receiver circular arc.  (5.29)
```

Each subsegment uses the same exact segment/pixel traversal as the straight-ray
case, so the curved row is a sum of exact voxel intersections over the declared
piecewise-linear circular-arc discretization. For finite-frequency sensitivity,
each subsegment uses a compact Fresnel tube around the propagation path:

```
K_vk = exp(-d(v, Γ_k)^2 / (2σ_k²)),   σ_k² = λ s_k(L - s_k) / L.              (5.30)
```

The implementation normalizes the active voxel weights on each subsegment so a
uniform sound-speed shift integrates to the geometric path length. This keeps the
linearized shift equation in (5.24) dimensionally identical while broadening
sensitivity to off-axis pixels.

Implemented in `kwavers::clinical::imaging::reconstruction::sound_speed_shift`.

---

## 5.8 Image Quality Metrics

### 5.8.1 Spatial Resolution Metrics

**Definition 5.8 (FWHM).** Full-width at half-maximum of the PSF cross-section. Measured
from a point target or wire phantom.

**Definition 5.9 (Lateral Resolution, Axial Resolution).** Measured FWHM in lateral and
axial directions for the combined transmit-receive PSF h(x,z).

### 5.8.2 Contrast Metrics

**Definition 5.10 (CR, CNR, gCNR).** For lesion intensity distribution p_l and background
p_b:
- CR = 20 log₁₀(μ_l/μ_b) [dB]
- CNR = |μ_l − μ_b| / √(σ_l² + σ_b²)
- gCNR = 1 − OVL(p_l, p_b), where OVL is the overlap integral of the two distributions

**Theorem 5.9 (gCNR Bound).** For any monotone increasing transformation of the image
data (log compression, TGC), gCNR is invariant. CNR is not invariant under nonlinear
transformations.

*Proof.* The overlap integral OVL = ∫ min(p_l(x), p_b(x)) dx is invariant under monotone
reparameterization by the change-of-variables theorem. □

### 5.8.3 Validation Pass Criteria

| Metric | Formula | Pass criterion |
|--------|---------|----------------|
| Lateral FWHM | PSF cross-section | ≤ 0.886 λ F# × 1.1 |
| Axial FWHM | Pulse envelope | ≤ N_cyc c₀/(2f₀) × 1.1 |
| CNR (anechoic cyst) | Eq. 5.6 | ≥ 3 |
| Doppler velocity error | |v̂ − v| / v | < 2% |
| PA sO₂ error | |sO₂ − sO₂_ref| | < 3% |
| SWE stiffness error | |Ê − E_ref| / E_ref | < 10% |

---

## 5.9 Code Mapping

| Modality | kwavers module | Key struct/fn |
|----------|---------------|---------------|
| B-mode beamforming | `clinical::imaging::workflows` | `PlaneWaveCompounding`, `DAS` |
| Envelope detection | `analysis::signal_processing` | `hilbert_envelope()` |
| Plane-wave F-K | `workflows::plane_wave_compounding` | `FKMigration::reconstruct()` |
| Doppler autocorr. | `clinical::imaging::doppler::autocorrelation` | `KasaiEstimator` |
| Wall filter | `clinical::imaging::doppler::wall_filter` | `WallFilter` |
| Color flow | `clinical::imaging::doppler::color_flow` | `ColorFlowMapper` |
| PA reconstruction | `clinical::imaging::reconstruction` | `AcousticProjection` |
| PA spectroscopy | `clinical::imaging::spectroscopy::solvers::tikhonov` | `SpectralUnmixer` |
| Hemoglobin spectra | `clinical::imaging::chromophores::hemoglobin` | `HemoglobinSpectrum` |
| ULM localization | `functional_ultrasound::ulm::microbubble_detection` | `MicrobubbleDetector` |
| ULM tracking | `functional_ultrasound::ulm::tracking` | `BubbleTracker` |
| ULM super-resolution | `functional_ultrasound::ulm::super_resolution` | `SuperResolutionReconstructor` |
| SWE velocity mapping | `functional_ultrasound::ulm::velocity_mapping` | `VelocityMapper` |
| Vasculature analysis | `functional_ultrasound::vasculature` | `FrangiFilter` |
| Speed-of-sound shift | `clinical::imaging::reconstruction::sound_speed_shift` | `reconstruct_sound_speed_shift()` |

---

## 5.10 Worked Example: B-Mode SNR after Coherent Compounding

**Setup.** 16 plane-wave angles, single plane-wave SNR = 20 dB (√100 = 10 amplitude ratio).

**Compounded SNR (Theorem 5.2):**

```
SNR_comp = √16 × SNR_single = 4 × 20 dB ... (linear: 4 × 10 = 40)
           → 20 log₁₀(40) = 32.0 dB
```

The 12 dB improvement recovers image quality comparable to focused transmit scanning
(typically SNR 30–35 dB) while maintaining the high frame rate of plane-wave imaging:

```
Frame rate = PRF / N_c = 10,000 Hz / 16 = 625 frames/s
```

versus ~30 frames/s for focused scanning with 128 lines.

---

## References

1. Szabo, T. L. (2014). *Diagnostic Ultrasound Imaging: Inside Out* (2nd ed.).
   Academic Press. Chapters 10–13.

2. Kasai, C., Namekawa, K., Koyano, A., & Omoto, R. (1985). Real-time two-dimensional
   blood flow imaging using an autocorrelation technique. *IEEE Trans. Sonics Ultrason.*,
   **32**(3), 458–464. https://doi.org/10.1109/T-SU.1985.31615

3. Montaldo, G., Tanter, M., Bercoff, J., Benech, N., & Fink, M. (2009). Coherent
   plane-wave compounding for very high frame rate ultrasonography and transient
   elastography. *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*, **56**(3), 489–506.
   https://doi.org/10.1109/TUFFC.2009.1067

4. Errico, C., Pierre, J., Pezet, S., et al. (2015). Ultrafast ultrasound localization
   microscopy for deep super-resolution vascular imaging. *Nature*, **527**, 499–502.
   https://doi.org/10.1038/nature16066

5. Dencks, S., & Schmitz, G. (2023). Ultrasound localization microscopy: a review.
   *Z. Med. Phys.*, **33**(4), 394–410. https://doi.org/10.1016/j.zemedi.2023.02.004

6. Wang, L. V. (2009). Multiscale photoacoustic microscopy and computed tomography.
   *Nature Photonics*, **3**, 503–509. https://doi.org/10.1038/nphoton.2009.157

7. Treeby, B. E., & Cox, B. T. (2010). k-Wave: MATLAB toolbox for the simulation and
   reconstruction of photoacoustic wave fields. *J. Biomed. Opt.*, **15**(2), 021314.
   https://doi.org/10.1117/1.3360308

8. Bercoff, J., Tanter, M., & Fink, M. (2004). Supersonic shear imaging: a new technique
   for soft tissue elasticity mapping. *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*,
   **51**(4), 396–409. https://doi.org/10.1109/TUFFC.2004.1295425

9. Rodriguez-Molares, A., Rindal, O. M. H., D'hooge, J., et al. (2020). The generalized
   contrast-to-noise ratio: a unified framework for assessing ultrasound image quality.
   *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*, **67**(4), 745–759.
   https://doi.org/10.1109/TUFFC.2019.2956855

10. Dines, K. A., & Kak, A. C. (1979). Ultrasonic attenuation tomography of soft
   tissues. *Ultrasonic Imaging*, **1**(1), 16–33.

11. ULTRA-SR benchmark: https://doi.org/10.1109/TMI.2024.3388048
