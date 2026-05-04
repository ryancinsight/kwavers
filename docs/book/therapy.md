# Chapter 6: Therapeutic Ultrasound

**Scope.** This chapter derives the physical mechanisms of ultrasound therapy: HIFU-induced
heating, the Pennes bioheat equation, thermal dose (CEM43), acoustic radiation force,
sonoporation, lithotripsy, and neuromodulation. Every mechanism is derived from
first principles with formal theorems. Code references map to `kwavers::clinical::therapy`
and the acoustic propagation solvers (Chapters 2–3).

---

## 6.1 Acoustic Intensity and Energy Deposition

### 6.1.1 Time-Averaged Intensity

**Theorem 6.1 (Acoustic Power Deposition).** For a plane wave with pressure amplitude P
propagating in a medium with absorption coefficient α [Np m⁻¹], the volumetric acoustic
power deposition (heat source density) is

```
Q_ac(r) = 2α I(r) = α P²(r) / (ρ₀ c₀)    [W m⁻³]                       (6.1)
```

where I = P²/(2ρ₀c₀) is the time-averaged acoustic intensity.

*Proof.* The intensity of a plane wave decays as I(z) = I₀ exp(−2αz). Conservation of
acoustic energy (Theorem 1.5) relates the divergence of the Poynting vector to
dissipation: Q_ac = −∇·(pu) = 2αI. Substituting I = P²/(2ρ₀c₀) gives (6.1). □

**Remark 6.1.** For non-planar fields (focused beams), Eq. (6.1) holds locally when α
is small (αλ ≪ 1) and the beam is quasi-planar within a resolution cell. For strongly
focused beams the full vector form Q = −∇·⟨p u⟩ must be used.

### 6.1.2 Focal Intensity for a Focused Bowl

For a HIFU focused bowl of aperture 2a, focal length R_f, face pressure P₀, surface
intensity I_face = P₀²/(2ρ₀c₀):

```
I_focal = G² I_face    G = k a²/(2R_f)    (Theorem 4.9)                   (6.2)
```

For a = 30 mm, f = 1 MHz, R_f = 60 mm: G ≈ 31, I_focal/I_face ≈ 961.

---

## 6.2 Pennes Bioheat Equation

### 6.2.1 Derivation

**Theorem 6.2 (Pennes Bioheat Equation).** The temperature T(r, t) in perfused tissue
satisfies

```
ρ_t c_p ∂T/∂t = ∇·(κ ∇T) + Q_ac − ω_b ρ_b c_b (T − T_b) + Q_met         (6.3)
```

where:
- ρ_t, c_p: tissue density [kg m⁻³] and specific heat capacity [J kg⁻¹ K⁻¹]
- κ: thermal conductivity [W m⁻¹ K⁻¹]
- Q_ac = 2αI: acoustic heat source [W m⁻³] (Eq. 6.1)
- ω_b: blood perfusion rate [kg m⁻³ s⁻¹]
- ρ_b, c_b: blood density and specific heat
- T_b: blood temperature (37 °C)
- Q_met: metabolic heat generation (typically ≪ Q_ac during HIFU)

*Proof.* The bioheat equation is the heat equation with three source/sink terms:
(1) conduction ∇·(κ∇T) by Fourier's law; (2) acoustic deposition Q_ac; (3) perfusion
cooling by convective heat exchange with blood flowing at ω_b kg m⁻³ s⁻¹.
Pennes (1948) derived the blood term by modeling perfusion as a spatially distributed
heat exchanger at temperature T_b. □

### 6.2.2 Tissue Thermal Properties

| Tissue | ρ (kg/m³) | c_p (J/kg·K) | κ (W/m·K) | ω_b (kg/m³/s) |
|--------|-----------|--------------|-----------|---------------|
| Liver | 1060 | 3600 | 0.51 | 6.4 × 10⁻³ |
| Kidney | 1050 | 3900 | 0.54 | 24.0 × 10⁻³ |
| Muscle | 1080 | 3640 | 0.50 | 0.5 × 10⁻³ |
| Fat | 940 | 2350 | 0.21 | 0.5 × 10⁻³ |
| Bone (cortical) | 1850 | 1300 | 0.38 | 0 |

### 6.2.3 Simplified Homogeneous Solution

Neglecting perfusion and conduction (short exposures, τ < 1 s), Eq. (6.3) reduces to:

```
∂T/∂t ≈ Q_ac / (ρ_t c_p)  →  ΔT = 2αI τ / (ρ_t c_p)                    (6.4)
```

At HIFU focal intensities (I = 5000 W/cm² = 5×10⁷ W/m², α = 5 Np/m, τ = 1 s):

```
ΔT = 2 × 5 × 5×10⁷ × 1 / (1060 × 3600) ≈ 131 °C / s
```

This confirms the thermal ablation mechanism: tissue reaches 60 °C (protein denaturation)
within ≈ 0.17 s at typical HIFU intensities.

---

## 6.3 Thermal Dose: CEM43

### 6.3.1 Definition

**Definition 6.1 (Cumulative Equivalent Minutes at 43 °C, CEM43).** The thermal dose
accumulated over a treatment at spatially varying temperature T(t) is

```
CEM43 = ∫₀^{t_total} R^{43−T(t)} dt                                       (6.5)
```

where R = 0.5 for T ≥ 43 °C and R = 0.25 for T < 43 °C (Sapareto & Dewey 1984).

**Theorem 6.3 (CEM43 Ablation Threshold).** Irreversible tissue damage (coagulative
necrosis) occurs when

```
CEM43 ≥ 240 min    (muscle, liver, most soft tissue)                       (6.6)
```

*Derivation.* The Arrhenius cell survival model S = exp(−Ω), with damage integral
Ω = A ∫ exp(−E_a/(RT)) dt, is empirically equivalent to (6.6) at 240 min CEM43
for tissues with activation energy E_a ≈ 680 kJ/mol (Dewey 2009). □

| Tissue | CEM43 threshold | Notes |
|--------|----------------|-------|
| Liver | 25 min | Sensitive to thermal ablation |
| Muscle | 240 min | Standard reference |
| Skin | 600 min | Higher threshold |
| Nerve | 5 min | Sensitive |
| Brain (gray matter) | 17 min | — |

### 6.3.2 Discrete CEM43 Accumulation

For a numerical simulation with time step Δt and temperature T^n at step n:

```
CEM43^{N} = Σ_{n=0}^{N-1} R^{43−T^n} · Δt                               (6.7)
```

Implemented in `kwavers::clinical::therapy::metrics` with the discrete summation (6.7)
applied element-wise over the 3-D temperature field.

---

## 6.4 Acoustic Radiation Force

### 6.4.1 Definition and Theorem

**Theorem 6.4 (Acoustic Radiation Force).** The time-averaged body force per unit volume
exerted by an acoustic field on an absorbing medium is

```
F_rad = 2α I / c₀    [N m⁻³]                                              (6.8)
```

in the direction of wave propagation.

*Proof.* The momentum density of the acoustic field is g = I/c₀². The rate of momentum
deposited per unit volume due to absorption is dg/dt = 2α I/c₀ (momentum transfer
proportional to energy deposition × 1/c₀). □

### 6.4.2 ARFI and Shear-Wave Generation

For a push pulse of duration τ_push [s] at focal intensity I_focus [W m⁻²]:

```
F_push = 2α I_focus / c₀ × τ_push    [N m⁻³ · s = Pa]                   (6.9)
```

This creates a tissue displacement u_peak ≈ F_push τ_push / (ρ c_s) and launches shear
waves at c_s (see Chapter 5, Eq. 5.21). The kwavers therapy module tracks radiation force
in `kwavers::clinical::therapy::therapy_integration::acoustic`.

---

## 6.5 Sonoporation and Drug Delivery

### 6.5.1 Bubble Oscillation and Membrane Permeabilization

**Definition 6.2 (Sonoporation).** Sonoporation is the transient increase in cell membrane
permeability caused by oscillating microbubbles in an acoustic field, enabling intracellular
delivery of otherwise membrane-impermeant molecules.

**Theorem 6.5 (Permeabilization Threshold).** Inertial cavitation (IC) onset requires

```
MI ≡ P_neg / √f₀ ≥ MI_IC ≈ 1.0    [kPa / √MHz = MPa^0.5]               (6.10)
```

Stable cavitation (SC, non-inertial), sufficient for gentle sonoporation, occurs at

```
MI_SC ≈ 0.1 – 0.5    (bubble-type and size dependent)                     (6.11)
```

*Derivation.* The inertial cavitation threshold is set by the condition that bubble
collapse time τ_collapse ≈ 0.915 R₀ √(ρ/p_∞) is shorter than the acoustic period 1/f₀.
Solving gives P_neg,IC ∝ √f₀, hence MI = P_neg/√f₀ = const at threshold. □

### 6.5.2 Blood-Brain Barrier Opening

Focused ultrasound combined with intravenous microbubbles opens the blood-brain barrier
(BBB) transiently at MI 0.2–0.6 (SC regime). The mechanism involves endothelial tight
junction disruption by oscillating bubble microstreaming. Key parameters:

| Parameter | Typical range | Clinical standard |
|-----------|--------------|-------------------|
| f₀ | 0.2–1.5 MHz | 0.5–1 MHz |
| Duty cycle | 1–20% | 10% |
| PRF | 1–10 Hz | 1 Hz |
| Duration | 30–120 s | 120 s |
| MI (in situ) | 0.2–0.6 | < 0.8 |

---

## 6.6 Lithotripsy

### 6.6.1 Shock Wave Lithotripsy (SWL)

In extracorporeal shock wave lithotripsy (ESWL), a focused shock wave with P_peak ~
50–100 MPa (positive) and P_neg ~ −5 to −15 MPa fractures kidney stones. The physical
mechanisms are:

1. **Spallation.** Tensile stress wave (reflected shock) at stone-fluid interface
   exceeds stone tensile strength (~10 MPa for calcium oxalate).
2. **Cavitation.** P_neg > 0.5–1 MPa drives inertial cavitation; bubble collapse
   produces microjet velocities ~100 m/s directed at the stone surface.
3. **Fatigue.** Repeated cycles (~ 2000 shocks) accumulate fatigue damage.

**Theorem 6.6 (Stone Tensile Stress from Reflected Shock).** A compressive shock of
peak pressure P_s transmitted into a stone of impedance Z_s ≫ Z_fluid generates a
reflected tensile wave at the distal stone–fluid interface of amplitude

```
p_tensile = −(Z_s − Z_f)/(Z_s + Z_f) × P_s × T_12                       (6.12)
```

where T_12 = 2Z_s/(Z_s+Z_f) is the transmission coefficient at incidence, and the
reflected wave at the stone–fluid boundary has reflection coefficient (Z_f−Z_s)/(Z_s+Z_f) < 0.

*Proof.* Continuity of pressure and normal particle velocity at the boundary requires
applying the standard Fresnel coefficients (Chapter 1, Theorem 1.4) twice (entry and exit).
The minus sign on the reflected wave at the stone–fluid interface (Z_f < Z_s) generates
a tensile phase. □

The stone fracture model in kwavers is in
`kwavers::clinical::therapy::lithotripsy::stone_fracture`.

---

## 6.7 Transcranial Focused Ultrasound Neuromodulation

### 6.7.1 Skull Transmission

**Theorem 6.7 (Skull Insertion Loss).** For a plane wave at normal incidence through a
skull layer of thickness d, density ρ_s, speed c_s (longitudinal), the transmission
coefficient in pressure is

```
T_skull = 4Z_s Z_f exp(ikd) / [(Z_s + Z_f)² − (Z_s − Z_f)² exp(2iαd)]   (6.13)
```

where Z_s = ρ_s c_s is the skull impedance and k = ω/c_s + iα_s is the complex wave
number (α_s skull absorption).

For human temporal bone at 0.5 MHz: typical |T_skull|² ≈ 20–40% intensity.

*Proof.* Layer transfer-matrix method (TMM): apply the 2×2 boundary condition matrix
at the brain-skull and skull-transducer interfaces and solve for the transmitted field.
Result (6.13) follows from the standard TMM for a single layer. □

### 6.7.2 Safety: MI, TI, and Regulatory Limits

**Definition 6.3 (Mechanical Index).** MI = P_neg [MPa] / √f₀ [MHz]. FDA limit: MI ≤ 1.9.

**Definition 6.4 (Thermal Index).** TI = W/(W_deg), the power required for 1 °C
temperature rise. TI_soft tissue (TIS), TI_bone (TIB), TI_cranium (TIC) are mode-specific.
FDA limit: TI ≤ 6 for transient, TI ≤ 2 for prolonged exposures.

| Application | f₀ | MI limit | TI limit | Typical exposure |
|-------------|-----|----------|----------|-----------------|
| Diagnostic B-mode | 5–15 MHz | 1.9 | 6 | Single pulse |
| CEUS | 2–5 MHz | 0.4 | 2 | CW ≤ 5 min |
| tFUS neuromod. | 0.25–1 MHz | 0.5–1.0 | < 2 | Pulsed, 30–120 s |
| HIFU ablation | 1–3 MHz | ≫ 1 (therapeutic) | — | ≤ 10 s/sonication |

---

## 6.8 Therapy Validation Protocol

**Algorithm 6.1 (Therapy Validation Loop).**

```
Input:  transducer geometry, medium properties, exposure parameters
Output: thermal dose map CEM43(r), peak pressure field, MI/TI

1. ACOUSTIC FIELD: run FDTD or PSTD solver with heterogeneous c₀, ρ₀, α.
2. INTENSITY: I(r) = ⟨p(r,t) u_n(r,t)⟩ or I = P_rms²/(2ρ₀c₀) (plane-wave approx.)
3. HEAT SOURCE: Q = 2α I (Theorem 6.1)
4. BIOHEAT: integrate Eq. (6.3) over exposure duration with Crank-Nicolson scheme.
5. THERMAL DOSE: accumulate CEM43 via Eq. (6.7).
6. SAFETY: compute MI = P_neg/√f₀; TI = W/W_deg; compare to FDA limits.
7. VALIDATE:
   a. Homogeneous medium: ΔT against Eq. (6.4) within 5%.
   b. Focal pressure gain against Theorem 4.9 within 10%.
   c. CEM43 ablation zone volume against k-Wave reference within 15%.
```

---

## 6.9 Code Mapping

| Concept | kwavers module | Key struct/fn |
|---------|---------------|---------------|
| HIFU planning | `clinical::therapy::hifu_planning` | `HifuPlanner` |
| Bioheat solver | `clinical::therapy::therapy_integration::tissue` | `BioheatSolver` |
| Thermal dose | `clinical::therapy::metrics` | `CEM43Accumulator` |
| Intensity tracking | `therapy_integration::intensity_tracker` | `IntensityTracker` |
| Lithotripsy | `clinical::therapy::lithotripsy` | `ShockWaveGenerator` |
| Stone fracture | `lithotripsy::stone_fracture` | `StoneFractureModel` |
| Cavitation cloud | `lithotripsy::cavitation_cloud` | `CavitationCloud` |
| Microbubble dynamics | `clinical::therapy::microbubble_dynamics` | `MicrobubbleService` |
| Safety controller | `therapy_integration::safety_controller` | `SafetyController` |
| Therapy orchestrator | `therapy_integration::orchestrator` | `TherapyOrchestrator` |

---

## 6.10 Worked Example: HIFU Ablation Dose

**Setup.** Liver tumor, 1 MHz HIFU, a = 35 mm, R_f = 80 mm, face pressure P₀ = 300 kPa.
- Surface intensity: I_face = P₀²/(2ρ₀c₀) = (3×10⁵)²/(2×1060×1540) ≈ 27.5 W/cm²
- Focal gain: G = kπa²/(2πR_f) = (2π×10⁶/1540)×(0.035)²/(2×0.08) ≈ 30.8
- Focal intensity: I_focal = G² × I_face ≈ 949 × 27.5 ≈ 26,100 W/cm² = 2.61 × 10⁸ W/m²
- Heat source at focus: Q = 2α I = 2 × 7 × 2.61×10⁸ ≈ 3.66 × 10⁹ W/m³
- Temperature rise (no perfusion, τ = 0.5 s): ΔT = Q τ/(ρ c_p) = 3.66×10⁹×0.5/3816×10³ ≈ 479 °C — dominated by thermal diffusion, so in practice ΔT ≈ 60–80 °C
- CEM43 from 60 °C isothermal hold of 1 s: 0.5^(43−60) × 1 = 0.5^{−17} × 1 ≈ 131,072 min >> 240 min threshold

Ablation is achieved in < 1 s per sonication at these parameters, consistent with
HIFU clinical outcomes (Jolesz 2014).

---

## References

1. Pennes, H. H. (1948). Analysis of tissue and arterial blood temperatures in the resting
   human forearm. *J. Appl. Physiol.*, **1**(2), 93–122.
   https://doi.org/10.1152/jappl.1948.1.2.93

2. Sapareto, S. A., & Dewey, W. C. (1984). Thermal dose determination in cancer therapy.
   *Int. J. Radiat. Oncol. Biol. Phys.*, **10**(6), 787–800.
   https://doi.org/10.1016/0360-3016(84)90379-1

3. Dewey, W. C. (2009). Arrhenius relationships from the molecule and cell to the clinic.
   *Int. J. Hyperthermia*, **25**(1), 3–20. https://doi.org/10.1080/02656730902747919

4. Jolesz, F. A. (2014). MRI-guided focused ultrasound surgery. *Annu. Rev. Med.*,
   **65**, 329–348. https://doi.org/10.1146/annurev-med-050913-013754

5. Bailey, M. R., Khokhlova, V. A., Sapozhnikov, O. A., Kargl, S. G., & Crum, L. A.
   (2003). Physical mechanisms of the therapeutic effect of ultrasound.
   *Acoust. Phys.*, **49**(4), 369–388. https://doi.org/10.1134/1.1591291

6. Haar, G. T., & Coussios, C. (2007). High intensity focused ultrasound: Physical
   principles and devices. *Int. J. Hyperthermia*, **23**(2), 89–104.
   https://doi.org/10.1080/02656730601186138

7. Hynynen, K., McDannold, N., Vykhodtseva, N., & Jolesz, F. A. (2001). Noninvasive MR
   imaging-guided focal opening of the blood-brain barrier in rabbits. *Radiology*,
   **220**(3), 640–646. https://doi.org/10.1148/radiol.2202001804

8. Lingeman, J. E. (2007). Lithotripsy systems. *Endotext* (updated 2020).

9. Legon, W., Sato, T. F., Opitz, A., et al. (2014). Transcranial focused ultrasound
   modulates the activity of primary somatosensory cortex in humans. *Nat. Neurosci.*,
   **17**, 322–329. https://doi.org/10.1038/nn.3620

10. FDA. (2019). Guidance for industry and FDA staff: Information for manufacturers
    seeking marketing clearance of diagnostic ultrasound systems and transducers.
    U.S. Food and Drug Administration.

11. BBB opening review: https://doi.org/10.1016/j.jconrel.2024.07.006
12. tFUS neuromodulation: https://doi.org/10.1186/s12984-025-01753-2
