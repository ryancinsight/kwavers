# Chapter 13: Theranostics — Combined Imaging and Therapy

**Scope.** This chapter covers the physics and engineering of closed-loop ultrasound
theranostics: systems where diagnostic imaging estimates biological state and acoustic
therapy updates exposure in response. Topics include bubble dynamics (Rayleigh-Plesset),
passive cavitation detection, ARFI-based treatment monitoring, MR thermometry integration,
and formal closed-loop stability theorems. Code ownership spans
`kwavers_diagnostics` (imaging workflows), `kwavers_therapy` (therapy integration,
microbubble dynamics), and the shared `kwavers_physics::acoustics::bubble_dynamics`
cavitation modules.

---

## 13.1 Bubble Dynamics (recap)

Theranostic cavitation control rests on the single-bubble physics derived in full in the
**Cavitation and Bubble Dynamics** chapter; the results reused in this chapter are:

- **Rayleigh–Plesset** wall dynamics
  `ρ_l(R R̈ + 3/2 Ṙ²) = p_g(R) − p_∞(t) − p_0 − 4μ_l Ṙ/R − 2σ/R`
  with polytropic gas law `p_g = (p_0 + 2σ/R_0)(R_0/R)^{3κ}` — *Cavitation §5.2*, and the
  Keller–Miksis compressible extension for high-amplitude drive — *Cavitation §5.3*.
- **Minnaert resonance** `f_M = (1/2πR_0)√(3κp_0/ρ_l)` (1 μm → 3.3 MHz, 5 μm → 0.65 MHz)
  — *Cavitation §5.5*.
- **Blake threshold** for the onset of inertial collapse — *Cavitation §5.4*.

These ODEs are integrated in kwavers by
`kwavers_physics::acoustics::bubble_dynamics::rayleigh_plesset` (`RayleighPlesset`), with
adaptive time-stepping and the Keller–Miksis model in the same `bubble_dynamics` tree.

---

## 13.2 Passive Cavitation Detection (PCD)

### 13.2.1 Cavitation signatures (recap)

The acoustic-emission discriminants used by the controller below are derived in
*Cavitation §5.6 (Acoustic Emission and Passive Cavitation Detection)*:

- **Stable cavitation (SC):** sub-harmonic (½f₀) and ultra-harmonic (3/2 f₀) spectral
  peaks with low broadband noise (period-2 bubble oscillation; the Floquet multiplier of
  the linearized RP equation crosses −1).
- **Inertial cavitation (IC):** broadband noise from violent aperiodic collapse,
  quantified by the **inertial cavitation dose**
  `ICD = ∫ S(f) df  [Pa² s]` integrated over a band that excludes ±Δf around nf₀
  (n = 1, 2, 3) and ½f₀.

kwavers computes both discriminants from the received spectrum in
`kwavers_physics::acoustics::bubble_dynamics::cavitation_control::detection`
(`broadband`, `spectral`, `subharmonic` sub-modules), which drive the closed-loop
controller of §13.2.2.

### 13.2.2 Cavitation Control for BBB Opening

A PCD-based controller adjusts the therapeutic ultrasound pulse amplitude to maintain
stable cavitation while avoiding inertial cavitation:

```
Algorithm 13.1 (PCD Feedback Control):
Input:  S(f): PCD spectrum; P_n: current pressure amplitude; thresholds SC_min, IC_min
Output: P_{n+1}

1. Compute SC_n = peak power at ½f₀ (sub-harmonic)
2. Compute IC_n = broadband ICD (§13.2.1)
3. If IC_n > IC_thresh: reduce P_{n+1} = γ_down × P_n   (γ_down < 1)
4. Elif SC_n < SC_target: increase P_{n+1} = γ_up × P_n  (γ_up > 1)
5. Else: maintain P_{n+1} = P_n
6. Project P_{n+1} to safe range [P_min, P_max]
```

---

## 13.3 MR Thermometry and Closed-Loop HIFU

### 13.3.1 Proton Resonance Frequency Shift

**Theorem 13.5 (MR Thermometry — PRFS Method).** The proton resonance frequency
(PRF) in water-containing tissues shifts linearly with temperature:

```
f_MR(T) = f_0(1 − α_PRFS T)    α_PRFS ≈ −0.0102 ppm/°C                 (13.7)
```

*Proof.* Hydrogen bonding in liquid water modulates the electron shielding constant σ_c.
The temperature coefficient of σ_c is α_PRFS = dσ_c/dT, which is empirically −0.0102 ppm/°C
for aqueous tissue (De Poorter 1995). The frequency shift Δf = f_0 α_PRFS ΔT. □

Phase difference between reference and post-heating MR images gives ΔT:

```
ΔT(r) = Δφ(r) / (2π f_0 α_PRFS TE)                                      (13.8)
```

where TE is echo time [s] and Δφ is the voxel phase change [rad].
MR thermometry precision: ~1–2 °C at 3 T with TE = 15 ms.

### 13.3.2 Closed-Loop HIFU Controller

**Theorem 13.6 (Closed-Loop Thermal Dose Monotonicity).** Let D_k be the cumulative
CEM43 dose at step k and u_k ≥ 0 the acoustic power.  Define the CEM43 dose rate:

```
φ(T) = R(T)^(T − 43)   where R(T) = 0.25 (T > 43 °C), R(T) = 0.50 (T ≤ 43 °C)
```

The dose update

```
D_{k+1} = D_k + φ(T_k) Δt                                                (13.9)
```

is monotone non-decreasing: D_{k+1} ≥ D_k for all k.

*Proof.* Since `R(T) ∈ {0.25, 0.50} ⊂ (0, 1]`, every real power `R^x = exp(x ln R)`
is strictly positive.  Therefore `φ(T) = R^(T−43) > 0` for all finite `T`.  With
`Δt > 0`, `D_{k+1} − D_k = φ(T_k) Δt > 0`.  By induction the sequence {D_k} is
strictly increasing. □

**Corollary 13.1 (Irreversibility and Safety Constraint).** Because φ > 0, dose is
strictly monotone — it cannot be reduced by decreasing power. Consequently a bounded
safety constraint `D_k ≤ D_max` cannot be enforced by reducing `u_k` once the bound is
approached, since `D_k` is non-decreasing in `k` regardless of `u_k`. Safety must
therefore be guaranteed by pre-treatment planning (Chapter 12), which predicts the dose
evolution `{D_k}` and selects an exposure schedule that keeps the trajectory below
`D_max` for the entire treatment.

---

## 13.4 Theranostic Feedback Architecture

### 13.4.1 State-Estimator Loop

**Algorithm 13.2 (Image-Guided Therapy Loop).**

```
Initialize: acoustic field model; tissue state estimate x̂_0; dose D_0 = 0

Loop (k = 0, 1, 2, ...):
  1. ACQUIRE: diagnostic image y_k (B-mode, MR thermometry, PCD spectrum)
  2. REGISTER: align y_k to therapy frame using deformable registration (RITK)
  3. ESTIMATE: x̂_k = KalmanFilter(x̂_{k-1}, y_k, model)
                 State vector: x = [T (°C), D (CEM43), ρ_b (mm⁻³), c_s (m/s)]ᵀ ∈ ℝ⁴
                 Process noise Q = diag(σ²_T, σ²_D, σ²_b, σ²_c) from acoustic model uncertainty
                 Obs. noise  R = diag(σ²_thermo, σ²_dose, σ²_PCD, σ²_RTT) per modality
                 H maps state → observed: MR phase → T, CEM43 integrator → D, PCD RMS → ρ_b
  4. PREDICT: D̂_{k+1} = D_k + φ(u_k, x̂_k) Δt
  5. PLAN: select u_{k+1} such that D̂_{k+1} ≤ D_target and MI(u_{k+1}) ≤ MI_safe
  6. DELIVER: apply u_{k+1} for Δt seconds
  7. UPDATE: D_{k+1} = D_k + φ(u_{k+1}, x_k) Δt
  8. Terminate when D_{k+1} ≥ D_target or safety limit reached.
```

### 13.4.2 State Uncertainty

State-estimator uncertainty must be propagated explicitly:

```
σ_D² = σ_x² (∂φ/∂x)² Δt²    (linearized uncertainty propagation)          (13.10)
```

Acceptance criterion: D_target − 2σ_D ≥ 0 (dose coverage at 95% confidence).

---

## 13.5 Microbubble-Mediated Drug Delivery

### 13.5.1 Physical Mechanism

Oscillating microbubbles increase local permeability via:

1. **Microstreaming.** Oscillatory bubble motion drives fluid jets that shear endothelial
   cell membranes, increasing pore size transiently.
2. **Sonoporation.** Individual cell membrane permeabilization by bubble contact (§6.5.1).
3. **Tight junction disruption.** BBB-specific: ZO-1, occludin proteins are displaced
   from tight junctions under stable cavitation stress.

**Theorem 13.7 (Drug Uptake Enhancement).** For stable cavitation at MI ≈ 0.3, the
fractional drug uptake enhancement ε relative to passive diffusion scales as

```
ε ∝ R₀² f₀ p_A / (μ_l c₀)                                               (13.11)
```

where p_A is the driving pressure amplitude and f₀ the frequency.

*Proof sketch (dilute-bubble approximation, one bubble per voxel).* Microstreaming
velocity near a single oscillating bubble in unbounded fluid scales as
`u_s ∝ R₀ f₀ p_A/(ρ_l c₀)` (Longuet-Higgins 1998; streaming is proportional to
the oscillation velocity `Ṙ_max ≈ p_A/(ρ_l c₀)` and to the bubble size `R₀`).
Membrane shear stress `τ ∝ μ_l u_s / δ` (Stokes boundary layer, `δ = pore size`).
Drug uptake per-bubble ∝ permeability ∝ τ ∝ `R₀ f₀ p_A / (ρ_l c₀ δ)`. The ratio
to passive diffusion (τ = 0, i.e. p_A → 0) gives (13.11).

**Scope limitation.** The scaling (13.11) holds for a single isolated bubble in the
dilute limit (bubble volume fraction < 1%).  For clinical BBB opening
(bubble concentration ≈ 10⁴–10⁷ mm⁻³), bubble–bubble hydrodynamic coupling,
secondary Bjerknes forces, and collective microstreaming modify the effective shear
field.  Eq. (13.11) is a single-bubble lower bound; collective enhancement depends on
concentration and spacing (Garbin et al. 2009). □

### 13.5.2 Dose–Response Relationship

Clinical BBB opening protocol (Hynynen et al.):

| Parameter | Value |
|-----------|-------|
| Frequency | 0.5 MHz |
| Duty cycle | 10% |
| PRF | 1 Hz |
| Duration | 120 s |
| MI (in situ) | 0.3–0.5 |
| Bubble concentration | ~10⁵–10⁶ /mL |

Typical gadolinium enhancement: 2–5× normal BBB permeability at target; reversible
within 4–6 hours.

---

## 13.6 Histotripsy (cross-reference)

In the theranostic loop, histotripsy is the high-amplitude mechanical-ablation mode
(P_neg > 15–30 MPa, MI > 5–10): dense bubble clouds liquefy tissue without coagulative
necrosis, initiated above the intrinsic cavitation threshold
`P_neg,intrinsic = √(16πσ³/(3 k_B T)) ≈ 26–30 MPa` (water, 37 °C, classical nucleation
theory). The full mechanism, derivation, and the classical-vs-millisecond-pulse regimes
are in the **Histotripsy** chapter and *Cavitation §5.9*. kwavers models the bubble-cloud
dynamics in `kwavers_therapy::therapy::lithotripsy::cavitation_cloud` (`CavitationCloud`).

---

## 13.7 Code Mapping

| Concept | kwavers module | Key struct |
|---------|---------------|------------|
| Bubble ODE (Rayleigh–Plesset, Keller–Miksis) | `kwavers_physics::acoustics::bubble_dynamics::rayleigh_plesset` | `RayleighPlesset` |
| Passive cavitation detection (SC/IC) | `kwavers_physics::acoustics::bubble_dynamics::cavitation_control::detection` | `broadband` / `spectral` / `subharmonic` |
| Microbubble dynamics | `kwavers_therapy::therapy::microbubble_dynamics::service` | `MicrobubbleDynamicsService` |
| Cavitation cloud (histotripsy) | `kwavers_therapy::therapy::lithotripsy::cavitation_cloud` | `CavitationCloud` |
| Therapy orchestrator | `kwavers_therapy::therapy::therapy_integration::orchestrator` | `TherapyIntegrationOrchestrator` |
| Safety controller | `kwavers_therapy::therapy::therapy_integration::safety_controller` | `SafetyController` |
| ULM microbubble detection | `kwavers_analysis::signal_processing::ulm::microbubble_detection` | `UlmDetector` |
| Plane-wave compounding | `kwavers_diagnostics::workflows::plane_wave_compounding` | `PlaneWaveCompounding` |
| Registration | via RITK crate | `DeformableRegistration` |

---

## 13.8 Worked Example: PCD-Controlled BBB Opening

**Setup.** 0.5 MHz transducer, 64-element array, focus at 60 mm depth (brain), SonoVue
microbubbles at 0.1 mL/kg IV. Target: stable cavitation at f₀/2 = 250 kHz, IC avoided.

**Stable cavitation threshold (SC):** MI ≈ 0.2 → P_neg = MI × √f₀ = 0.2 × √0.5 ≈ 0.14 MPa.

**Inertial cavitation threshold (IC):** MI ≈ 0.6 → P_neg = 0.6 × √0.5 ≈ 0.42 MPa.

**Control window:** P_neg ∈ [0.14, 0.42] MPa, corresponding to focal intensities
I ∈ [1300, 11800] W/m². PCD monitors broadband ICD and sub-harmonic power every pulse.

If ICD exceeds threshold (IC onset): power reduced by γ_down = 0.8.
If sub-harmonic < target (SC not established): power increased by γ_up = 1.05.
Convergence to stable cavitation: typically 5–20 pulse iterations.

---

## References

1. Rayleigh, J. W. S. (1917). On the pressure developed in a liquid during the collapse
   of a spherical cavity. *Philos. Mag.*, **34**(200), 94–98.
   https://doi.org/10.1080/14786440808635681

2. Plesset, M. S. (1949). The dynamics of cavitation bubbles. *J. Appl. Mech.*,
   **16**(3), 277–282.

3. Minnaert, M. (1933). On musical air-bubbles and the sounds of running water.
   *Philos. Mag.*, **16**(104), 235–248.
   https://doi.org/10.1080/14786443309462277

4. Hynynen, K., McDannold, N., Vykhodtseva, N., & Jolesz, F. A. (2001). Noninvasive MR
   imaging-guided focal opening of the blood-brain barrier in rabbits. *Radiology*,
   **220**(3), 640–646. https://doi.org/10.1148/radiol.2202001804

5. Xu, Z., Ludomirsky, A., Eun, L. Y., et al. (2004). Controlled ultrasound tissue
   erosion. *IEEE Trans. Ultrason. Ferroelectr. Freq. Control*, **51**(6), 726–736.
   https://doi.org/10.1109/TUFFC.2004.1308731

6. De Poorter, J. (1995). Noninvasive MRI thermometry with the proton resonance
   frequency method: Study of susceptibility effects. *Magn. Reson. Med.*, **34**(3),
   359–367. https://doi.org/10.1002/mrm.1910340313

7. Dencks, S., & Schmitz, G. (2023). Ultrasound localization microscopy.
   *Z. Med. Phys.*, **33**(4), 394–410. https://doi.org/10.1016/j.zemedi.2023.02.004

8. Glioma theranostics: https://doi.org/10.3390/biomedicines12061230
9. Microbubble drug/gene delivery: https://doi.org/10.1016/j.jddst.2023.105312
10. tFUS neuromodulation: https://doi.org/10.1186/s12984-025-01753-2
