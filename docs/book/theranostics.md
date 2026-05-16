# Chapter 7: Theranostics — Combined Imaging and Therapy

**Scope.** This chapter covers the physics and engineering of closed-loop ultrasound
theranostics: systems where diagnostic imaging estimates biological state and acoustic
therapy updates exposure in response. Topics include bubble dynamics (Rayleigh-Plesset),
passive cavitation detection, ARFI-based treatment monitoring, MR thermometry integration,
and formal closed-loop stability theorems. Code ownership spans
`kwavers::clinical::imaging`, `kwavers::clinical::therapy`, and the shared cavitation
and bubble dynamics physics modules.

---

## 7.1 Rayleigh-Plesset Bubble Dynamics

### 7.1.1 Full Nonlinear ODE

**Theorem 7.1 (Rayleigh-Plesset Equation).** For a spherical gas bubble of radius R(t)
in an incompressible Newtonian liquid of density ρ_l and viscosity μ_l, with internal
gas pressure p_g(R) and external acoustic pressure p_∞(t), the bubble wall dynamics satisfy

```
ρ_l (R R̈ + 3/2 Ṙ²) = p_g(R) − p_∞(t) − p_0 − 4μ_l Ṙ/R − 2σ/R         (7.1)
```

where p_0 is the ambient hydrostatic pressure, σ is the surface tension [N m⁻¹], and
the gas pressure follows the polytropic law:

```
p_g(R) = (p_0 + 2σ/R_0)(R_0/R)^{3κ}                                     (7.2)
```

with κ = 1 (isothermal) or κ = γ = C_p/C_v (adiabatic).

*Proof.* Applying Newton's second law to a spherical shell of liquid at radius r > R,
using conservation of mass (∂_t(4πr²ρ) = 0), and integrating the momentum equation from
R to ∞ yields (7.1). □

### 7.1.2 Minnaert Resonance

**Theorem 7.2 (Minnaert Resonance Frequency).** Linearizing (7.1) about equilibrium R₀:

```
f_Minnaert = (1/2πR₀) √(3κ p_0 / ρ_l)                                   (7.3)
```

neglecting surface tension (valid for R₀ > 1 μm).

*Proof.* Set R = R₀(1 + x), |x| ≪ 1. Expand (7.1) to first order:
the gas restoring force coefficient is 3κ p_0/R₀², the inertial coefficient is ρ_l R₀.
Natural frequency: ω₀ = √(3κ p_0/(ρ_l R₀²)), yielding (7.3). □

| R₀ (μm) | f_Minnaert (MHz, water, κ=1.4) |
|---------|-------------------------------|
| 1 | 3.26 |
| 2 | 1.63 |
| 3 | 1.09 |
| 5 | 0.65 |
| 10 | 0.33 |

### 7.1.3 Blake Threshold

**Theorem 7.3 (Blake Threshold Pressure).** The minimum acoustic pressure required to
drive an unbounded bubble collapse (inertial cavitation) is

```
p_Blake = p_0 + 2σ/R_0 · (R_0/R_Blake)³                                  (7.4)
```

with R_Blake = R_0 (4/3 + 4σ/(3R_0 p_0))^{1/3} (critical radius at which the net
restoring force vanishes). For water at p_0 = 101 kPa:

```
P_Blake ≈ p_0 − 0.77 (2σ/(R_0))^3 / p_0²                               (7.5)
```

Numerically: R₀ = 1 μm → P_Blake ≈ 79 kPa (0.079 MPa); R₀ = 5 μm → 5 kPa (0.005 MPa).

---

## 7.2 Passive Cavitation Detection (PCD)

### 7.2.1 Broadband Noise Signature

**Definition 7.1 (Inertial Cavitation Dose, ICD).** The ICD is the integrated broadband
noise power in the received PCD signal over a frequency band excluding harmonics and
sub-harmonics:

```
ICD = ∫_{f_low}^{f_high} S(f) df    [Pa² s]                              (7.6)
```

where S(f) is the power spectral density of the received passband signal and the
integration band excludes ±Δf around nf₀ for n = 1, 2, 3 and ½f₀.

**Theorem 7.4 (Stable vs Inertial Cavitation Signatures).**
- Stable cavitation (SC): generates sub-harmonic (f₀/2) and ultra-harmonic (3f₀/2)
  peaks in the spectrum; broadband noise is low.
- Inertial cavitation (IC): generates broadband noise across all frequencies;
  characteristically observed when R(t) collapses violently.

*Proof.* SC corresponds to period-2 bubble oscillation (parametric instability at f₀/2);
the Floquet multiplier for the linearized RP equation crosses −1. IC corresponds to
aperiodic collapse; the resulting pressure pulse from Gilmore/modified RP has a
broad Fourier transform. □

### 7.2.2 Cavitation Control for BBB Opening

A PCD-based controller adjusts the therapeutic ultrasound pulse amplitude to maintain
stable cavitation while avoiding inertial cavitation:

```
Algorithm 7.1 (PCD Feedback Control):
Input:  S(f): PCD spectrum; P_n: current pressure amplitude; thresholds SC_min, IC_min
Output: P_{n+1}

1. Compute SC_n = peak power at ½f₀ (sub-harmonic)
2. Compute IC_n = broadband ICD from (7.6)
3. If IC_n > IC_thresh: reduce P_{n+1} = γ_down × P_n   (γ_down < 1)
4. Elif SC_n < SC_target: increase P_{n+1} = γ_up × P_n  (γ_up > 1)
5. Else: maintain P_{n+1} = P_n
6. Project P_{n+1} to safe range [P_min, P_max]
```

---

## 7.3 MR Thermometry and Closed-Loop HIFU

### 7.3.1 Proton Resonance Frequency Shift

**Theorem 7.5 (MR Thermometry — PRFS Method).** The proton resonance frequency
(PRF) in water-containing tissues shifts linearly with temperature:

```
f_MR(T) = f_0(1 − α_PRFS T)    α_PRFS ≈ −0.0102 ppm/°C                 (7.7)
```

*Proof.* Hydrogen bonding in liquid water modulates the electron shielding constant σ_c.
The temperature coefficient of σ_c is α_PRFS = dσ_c/dT, which is empirically −0.0102 ppm/°C
for aqueous tissue (De Poorter 1995). The frequency shift Δf = f_0 α_PRFS ΔT. □

Phase difference between reference and post-heating MR images gives ΔT:

```
ΔT(r) = Δφ(r) / (2π f_0 α_PRFS TE)                                      (7.8)
```

where TE is echo time [s] and Δφ is the voxel phase change [rad].
MR thermometry precision: ~1–2 °C at 3 T with TE = 15 ms.

### 7.3.2 Closed-Loop HIFU Controller

**Theorem 7.6 (Closed-Loop Thermal Dose Monotonicity).** Let D_k be the cumulative
CEM43 dose at step k and u_k ≥ 0 the acoustic power.  Define the CEM43 dose rate:

```
φ(T) = R(T)^(T − 43)   where R(T) = 0.25 (T > 43 °C), R(T) = 0.50 (T ≤ 43 °C)
```

The dose update

```
D_{k+1} = D_k + φ(T_k) Δt                                                (7.9)
```

is monotone non-decreasing: D_{k+1} ≥ D_k for all k.

*Proof.* Since `R(T) ∈ {0.25, 0.50} ⊂ (0, 1]`, every real power `R^x = exp(x ln R)`
is strictly positive.  Therefore `φ(T) = R^(T−43) > 0` for all finite `T`.  With
`Δt > 0`, `D_{k+1} − D_k = φ(T_k) Δt > 0`.  By induction the sequence {D_k} is
strictly increasing. □

**Corollary 7.1 (Irreversibility).** Because φ > 0, dose is strictly monotone — it
cannot be reduced by decreasing power.  Safety constraints `D_k ≤ D_max` must be
enforced by pre-treatment planning (Chapter 6) rather than post-onset power reduction.

**Corollary 7.1 (Safety Constraint).** A bounded safety constraint D_k ≤ D_max cannot be
enforced by reducing u_k alone once violated, since dose is non-decreasing. This motivates
the need for pre-treatment planning (Chapter 6) to predict the dose evolution.

---

## 7.4 Theranostic Feedback Architecture

### 7.4.1 State-Estimator Loop

**Algorithm 7.2 (Image-Guided Therapy Loop).**

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

### 7.4.2 State Uncertainty

State-estimator uncertainty must be propagated explicitly:

```
σ_D² = σ_x² (∂φ/∂x)² Δt²    (linearized uncertainty propagation)          (7.10)
```

Acceptance criterion: D_target − 2σ_D ≥ 0 (dose coverage at 95% confidence).

---

## 7.5 Microbubble-Mediated Drug Delivery

### 7.5.1 Physical Mechanism

Oscillating microbubbles increase local permeability via:

1. **Microstreaming.** Oscillatory bubble motion drives fluid jets that shear endothelial
   cell membranes, increasing pore size transiently.
2. **Sonoporation.** Individual cell membrane permeabilization by bubble contact (§6.5.1).
3. **Tight junction disruption.** BBB-specific: ZO-1, occludin proteins are displaced
   from tight junctions under stable cavitation stress.

**Theorem 7.7 (Drug Uptake Enhancement).** For stable cavitation at MI ≈ 0.3, the
fractional drug uptake enhancement ε relative to passive diffusion scales as

```
ε ∝ R₀² f₀ p_A / (μ_l c₀)                                               (7.11)
```

where p_A is the driving pressure amplitude and f₀ the frequency.

*Proof sketch (dilute-bubble approximation, one bubble per voxel).* Microstreaming
velocity near a single oscillating bubble in unbounded fluid scales as
`u_s ∝ R₀ f₀ p_A/(ρ_l c₀)` (Longuet-Higgins 1998; streaming is proportional to
the oscillation velocity `Ṙ_max ≈ p_A/(ρ_l c₀)` and to the bubble size `R₀`).
Membrane shear stress `τ ∝ μ_l u_s / δ` (Stokes boundary layer, `δ = pore size`).
Drug uptake per-bubble ∝ permeability ∝ τ ∝ `R₀ f₀ p_A / (ρ_l c₀ δ)`. The ratio
to passive diffusion (τ = 0, i.e. p_A → 0) gives (7.11).

**Scope limitation.** The scaling (7.11) holds for a single isolated bubble in the
dilute limit (bubble volume fraction < 1%).  For clinical BBB opening
(bubble concentration ≈ 10⁴–10⁷ mm⁻³), bubble–bubble hydrodynamic coupling,
secondary Bjerknes forces, and collective microstreaming modify the effective shear
field.  Eq. (7.11) is a single-bubble lower bound; collective enhancement depends on
concentration and spacing (Garbin et al. 2009). □

### 7.5.2 Dose–Response Relationship

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

## 7.6 Histotripsy

**Definition 7.2 (Histotripsy).** Histotripsy is mechanical tissue liquefaction driven
by dense bubble clouds generated at high negative pressure (P_neg > 15–30 MPa, MI > 5–10).
Unlike HIFU thermal ablation, histotripsy is a non-thermal modality that creates
homogeneous liquefied zones (acellular debris) without coagulative necrosis.

**Theorem 7.8 (Intrinsic Threshold).** Histotripsy initiates when the peak negative
pressure exceeds the intrinsic threshold for cavitation in the absence of nuclei:

```
P_neg,intrinsic = √(16πσ³/(3k_B T))    ≈ 26–30 MPa (water, 37°C)        (7.12)
```

*Derivation.* Classical nucleation theory: the free energy barrier for nucleation of a
vapour nucleus of critical radius r_c = 2σ/P_neg is ΔG = 16πσ³/(3P_neg²). Setting
ΔG = k_B T (thermal nucleation condition) and solving for P_neg gives (7.12). □

Implemented in `kwavers::clinical::therapy::lithotripsy::cavitation_cloud`.

---

## 7.7 Code Mapping

| Concept | kwavers module | Key struct |
|---------|---------------|------------|
| Bubble ODE (RP) | `physics::acoustics::bubble_dynamics` | `RayleighPlesset` |
| Microbubble service | `clinical::therapy::microbubble_dynamics` | `MicrobubbleService` |
| Cavitation cloud | `clinical::therapy::lithotripsy::cavitation_cloud` | `CavitationCloud` |
| Therapy orchestrator | `therapy_integration::orchestrator` | `TherapyOrchestrator` |
| Safety controller | `therapy_integration::safety_controller` | `SafetyController` |
| ULM (diagnostic) | `clinical::imaging::functional_ultrasound::ulm` | `MicrobubbleDetector` |
| Plane-wave compounding | `clinical::imaging::workflows::plane_wave_compounding` | `PlaneWaveCompounding` |
| Registration | via RITK crate | `DeformableRegistration` |

---

## 7.8 Worked Example: PCD-Controlled BBB Opening

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
