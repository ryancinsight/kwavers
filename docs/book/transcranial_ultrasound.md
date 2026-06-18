# Chapter 15: Transcranial Ultrasound: Physics, Aberration Correction, and Therapeutic Applications

> **Module ownership**: `kwavers_physics::acoustics::transcranial`,
> `kwavers_solver::forward`, `kwavers_medium` (heterogeneous medium),
> `kwavers_imaging::medical::ct_loader` (CT/HU ingestion)

---

## 15.1 Introduction

Transcranial ultrasound delivers acoustic energy through the human skull to image or
treat intracranial targets. The skull is acoustically hostile: its impedance mismatch
with soft tissue causes reflection losses approaching 7 dB, its heterogeneous
diploe layer distorts wavefronts, and its high absorption converts a significant fraction
of incident intensity into heat at the bone surface. Therapeutic applications —
focused ultrasound (FUS) ablation of tremor circuits, blood–brain barrier (BBB)
opening, and non-thermal neuromodulation — each impose different constraints on the
allowable aberration residual, thermal load, and cavitation margin.

This chapter derives the physics from first principles, proves every key approximation,
and maps each result onto the simulation abstractions provided by
`kwavers_physics::acoustics::transcranial` and `kwavers_solver::forward`.

---

## 15.2 Skull Acoustic Properties

### 15.2.1 Measured Material Constants

The human calvaria is a three-layer composite:

| Layer | $\rho$ (kg m$^{-3}$) | $c$ (m s$^{-1}$) | $Z = \rho c$ (MRayl) | $\alpha$ at 1 MHz |
|---|---|---|---|---|
| Cortical bone (outer) | 1900 | 2800 | 5.32 | 10–20 dB cm$^{-1}$ |
| Diploe (trabecular) | 1200–1700 | 1800–2500 | 2.2–4.3 | 5–15 dB cm$^{-1}$ |
| Cortical bone (inner) | 1900 | 2800 | 5.32 | 10–20 dB cm$^{-1}$ |
| Brain parenchyma | 1040 | 1540 | 1.60 | 0.6 dB cm$^{-1}$ MHz$^{-1}$ |

The diploe heterogeneity spans a factor of $\approx 1.4$ in speed and $\approx 2$ in
impedance, making the skull the dominant source of both amplitude loss and phase
aberration in any transcranial path.

### 15.2.2 Characteristic Impedance and Transmission Coefficient

**Definition.** The specific acoustic impedance of a planar medium is

$$
Z = \rho \, c
\quad [\text{Pa s m}^{-1} \equiv \text{Rayl}].
$$

For the skull–tissue interface, with subscripts $1$ (skull) and $2$ (tissue):

$$
Z_1 = \rho_{\text{skull}} \, c_{\text{skull}} \approx 1900 \times 2800 = 5.32 \times 10^6 \; \text{Rayl},
\qquad
Z_2 = \rho_{\text{tissue}} \, c_{\text{tissue}} \approx 1040 \times 1540 = 1.60 \times 10^6 \; \text{Rayl}.
$$

**Theorem 15.1 (Normal-Incidence Transmission Coefficient).**
At a planar interface between two lossless half-spaces under normal incidence, the
pressure transmission coefficient is

$$
T_p = \frac{2 Z_2}{Z_1 + Z_2},
\tag{15.1}
$$

and the intensity transmission coefficient is

$$
T_I = \frac{4 Z_1 Z_2}{(Z_1 + Z_2)^2}.
\tag{15.2}
$$

*Proof.* Let a harmonic plane wave of unit pressure amplitude propagate in the
positive $x$-direction in medium 1. Denote the incident, reflected, and transmitted
pressure amplitudes $p_i = 1$, $p_r = R$, $p_t = T_p$.
The two boundary conditions at $x = 0$ are:

1. **Continuity of pressure**: $p_i + p_r = p_t$, i.e., $1 + R = T_p$.
2. **Continuity of normal particle velocity**: the particle velocity in a plane wave is
   $u = p / Z$, so $(p_i - p_r)/Z_1 = p_t/Z_2$,
   i.e., $(1 - R)/Z_1 = T_p / Z_2$.

From equation (2): $Z_2(1 - R) = Z_1 T_p$. Substituting $R = T_p - 1$ from
equation (1):

$$
Z_2 (1 - (T_p - 1)) = Z_1 T_p
\implies Z_2 (2 - T_p) = Z_1 T_p
\implies 2 Z_2 = T_p (Z_1 + Z_2).
$$

Therefore $T_p = 2Z_2 / (Z_1 + Z_2)$. $\square$

The intensity transmission coefficient follows from the ratio of transmitted to
incident intensity, $I = p^2 / (2Z)$:

$$
T_I = \frac{T_p^2 / (2 Z_2)}{1 / (2 Z_1)} = T_p^2 \frac{Z_1}{Z_2}
= \frac{4 Z_2^2}{(Z_1+Z_2)^2} \cdot \frac{Z_1}{Z_2}
= \frac{4 Z_1 Z_2}{(Z_1+Z_2)^2}.
$$

**Numerical evaluation for the skull–tissue interface:**

$$
T_p = \frac{2 \times 1.60}{5.32 + 1.60} = \frac{3.20}{6.92} \approx 0.462,
\qquad
T_I = \frac{4 \times 5.32 \times 1.60}{(6.92)^2} \approx \frac{34.05}{47.89} \approx 0.711.
$$

A single skull–tissue interface transmits $\approx 46\%$ of pressure and $\approx 71\%$
of intensity. The transcranial path includes **two** such interfaces (entry and exit).
For two identical interfaces (tissue → skull → tissue) the intensity transmission is
approximately

$$
T_I^{(2)} \approx T_I^2 \approx 0.505 \quad (\approx -3 \;\text{dB}).
$$

Adding absorption ($\alpha \approx 15$ dB cm$^{-1}$ MHz$^{-1}$, i.e. $\approx 4$ dB
cm$^{-1}$ at the $\sim 0.25$ MHz typical of deep transcranial therapy, over
$\sim 0.5$ cm of bone at each cortex $\Rightarrow \sim 2$ dB per cortex) raises the
insertion loss to approximately $3 + 2 \times 2 = 7$ dB, consistent with published
clinical measurements (Aubry & Tanter 2010).

> **Note.** The figure $T_p \approx 0.47$ quoted in the chapter heading applies to the
> single cortex-to-tissue interface evaluated at rounded impedance values
> ($Z_1 = 5.3$ MRayl, $Z_2 = 1.5$ MRayl):
> $T_p = 2 \times 1.5 / (5.3 + 1.5) = 3.0 / 6.8 \approx 0.441$.
> The exact value is sensitive to the diploe fraction within the bone sample.

---

## 15.3 Skull Aberration Model

### 15.3.1 Phase Accumulation Through a Heterogeneous Layer

**Theorem 15.2 (Scalar Phase Integral).** For a narrow-band wave at frequency $f$
propagating along a path parameterised by arc length $s$ through a medium with
spatially varying sound speed $c(s)$, the total accumulated phase is

$$
\phi = 2\pi f \int_0^{L} \frac{ds}{c(s)},
\tag{15.3}
$$

where $L$ is the total path length.

*Proof.* The local wavenumber is $k(s) = \omega / c(s)$ with $\omega = 2\pi f$.
Phase is the line integral of the wavenumber along the propagation path:
$\phi = \int_0^L k(s)\, ds = \int_0^L \omega / c(s)\, ds = 2\pi f \int_0^L ds / c(s)$.
$\square$

### 15.3.2 Aberration as a Differential Phase Screen

For a transducer element located at transverse position $\mathbf{x}_\perp$ on the skull
outer surface, the phase accumulated through the skull at that position is

$$
\phi(\mathbf{x}_\perp) = 2\pi f \int_{\text{skull}} \frac{ds}{c(s, \mathbf{x}_\perp)}.
$$

The **phase aberration** relative to propagation through homogeneous tissue
(speed $c_0$) over the same path length $d(\mathbf{x}_\perp)$ (local skull thickness) is

$$
\Delta\phi(\mathbf{x}_\perp)
= 2\pi f \, d(\mathbf{x}_\perp)
  \left(\frac{1}{c_{\text{tissue}}} - \frac{1}{c_{\text{skull}}(\mathbf{x}_\perp)}\right),
\tag{15.4}
$$

where $c_{\text{skull}}(\mathbf{x}_\perp)$ is the effective (thickness-averaged) skull
speed at position $\mathbf{x}_\perp$.

### 15.3.3 Strehl Ratio and Focus Degradation

**Definition.** The Strehl ratio $S$ is the ratio of the peak focal intensity with
aberration to the diffraction-limited peak intensity without aberration:

$$
S = \frac{I_{\text{focus,aberrated}}}{I_{\text{focus,ideal}}}.
\tag{15.5}
$$

**Theorem 15.3 (Maréchal Approximation).** For small-aberration phase screens
with zero-mean phase error $\Delta\phi$ and variance $\sigma_\phi^2$,

$$
S \approx e^{-\sigma_\phi^2}.
\tag{15.6}
$$

*Proof.* The focal field is formed by coherent superposition over $N$ array elements
(or equivalently, by the Fraunhofer integral over the aperture). With element phases
$\phi_i = \bar{\phi} + \Delta\phi_i$ and assuming the mean phase $\bar\phi$ is removed
by a global delay, the focal pressure is

$$
p_{\text{focus}} = \frac{1}{N} \sum_{i=1}^{N} e^{j \Delta\phi_i}.
$$

The focal intensity is proportional to $|p_{\text{focus}}|^2$. Taking the expectation
over the random phase errors (assuming they are independent and identically distributed
with zero mean):

$$
\mathbb{E}[|p_{\text{focus}}|^2]
= \frac{1}{N^2} \sum_{i=1}^N \sum_{j=1}^N \mathbb{E}[e^{j(\Delta\phi_i - \Delta\phi_j)}].
$$

For $i = j$ each term equals 1, contributing $N/N^2 = 1/N$. For $i \neq j$ and
statistically independent errors:

$$
\mathbb{E}[e^{j\Delta\phi_i}]\,\mathbb{E}[e^{-j\Delta\phi_j}]
= |\mathbb{E}[e^{j\Delta\phi}]|^2.
$$

The characteristic function of $\Delta\phi$ evaluated at argument 1 is
$\mathbb{E}[e^{j\Delta\phi}] = M_\phi(j)$. For a distribution with zero mean
and variance $\sigma_\phi^2$, the cumulant expansion gives

$$
\ln \mathbb{E}[e^{j\Delta\phi}] = j\mu_\phi - \tfrac{1}{2}\sigma_\phi^2 + O(\kappa_3)
= -\tfrac{1}{2}\sigma_\phi^2 + O(\kappa_3),
$$

so $|\mathbb{E}[e^{j\Delta\phi}]|^2 = e^{-\sigma_\phi^2}$ to second-order cumulant.
The $N(N-1)$ off-diagonal terms each contribute $e^{-\sigma_\phi^2}/N^2$.
Thus

$$
S = \mathbb{E}[|p_{\text{focus}}|^2] / |p_{\text{focus,ideal}}|^2
\approx \frac{1}{N} + \frac{N-1}{N} e^{-\sigma_\phi^2}
\xrightarrow{N \to \infty} e^{-\sigma_\phi^2}. \quad \square
$$

**Clinical implication.** At 1 MHz through a skull with $d = 7$ mm and
$c_{\text{skull}} \approx 2500$ m s$^{-1}$, $c_{\text{tissue}} = 1540$ m s$^{-1}$,
and $\sigma_d / d \approx 30\%$ thickness variation:

$$
\sigma_{\Delta\phi} \approx 2\pi \times 10^6 \times 7 \times 10^{-3} \times 0.30
  \times \left|\frac{1}{1540} - \frac{1}{2500}\right| \approx 2\pi \times 10^6
  \times 2.1 \times 10^{-3} \times 2.5 \times 10^{-4} \approx 3.3 \;\text{rad}.
$$

With $\sigma_\phi = 3.3$ rad, $S = e^{-15.9} \approx 1.8 \times 10^{-5}$: the focus is
completely destroyed without correction. Phase correction is non-optional.

![Strehl ratio vs RMS phase error from the Maréchal approximation $S=e^{-\sigma_\phi^2}$ (`analytical::skull::strehl_ratio`).](figures/ch16/fig04_strehl_ratio.png)

![Element phase-aberration map through a heterogeneous skull window (`analytical::skull::skull_phase_screen`).](figures/ch16/fig02_phase_aberration.png)

---

## 15.4 Transcranial Phase Correction by Time-Reversal

### 15.4.1 Time-Reversal Focusing

**Theorem 15.4 (Time-Reversal Phase Conjugation).** Let $p(\mathbf{r}_i, t)$ be the
pressure recorded at transducer element $i$ when a point source fires at target
$\mathbf{r}_0$ at $t = 0$ in a reciprocal, time-invariant medium. Define the
time-reversed signal

$$
s_i(t) = p(\mathbf{r}_i, T - t),
\tag{15.7}
$$

where $T$ is a delay large enough that the causal signal has decayed. If all elements
simultaneously emit $s_i(t)$, the resulting field refocuses at $\mathbf{r}_0$.

*Proof.* In the frequency domain, $P(\mathbf{r}_i, \omega) =
G(\mathbf{r}_i, \mathbf{r}_0; \omega) \hat{q}(\omega)$, where $G$ is the Green's
function and $\hat{q}(\omega)$ is the source spectrum.
Time reversal $t \to T - t$ maps to frequency-domain complex conjugation
(up to the linear phase $e^{-j\omega T}$):

$$
S_i(\omega) = e^{-j\omega T} P^*(\mathbf{r}_i, \omega)
= e^{-j\omega T} G^*(\mathbf{r}_i, \mathbf{r}_0; \omega) \hat{q}^*(\omega).
$$

The field reconstructed at $\mathbf{r}$ by emitting $S_i(\omega)$ from element $i$ is

$$
P_{\text{TR}}(\mathbf{r}, \omega) = \sum_i G(\mathbf{r}, \mathbf{r}_i; \omega) S_i(\omega).
$$

By reciprocity $G(\mathbf{r}, \mathbf{r}_i; \omega) = G(\mathbf{r}_i, \mathbf{r}; \omega)$.
At the target $\mathbf{r} = \mathbf{r}_0$:

$$
P_{\text{TR}}(\mathbf{r}_0, \omega)
= e^{-j\omega T} \hat{q}^*(\omega) \sum_i |G(\mathbf{r}_i, \mathbf{r}_0; \omega)|^2,
$$

which is a real, non-negative sum of squared magnitudes: all elements contribute
in phase, achieving coherent constructive interference at the target. $\square$

**Remark.** The virtual point source required for time-reversal focusing is the
key practical barrier. Two approaches exist:

1. **Implantable hydrophone** at the target — not feasible for elective procedures.
2. **Computational time-reversal (CT-based adjoint simulation)**: simulate the
   forward field from a virtual point source using a patient-specific acoustic model
   derived from CT, extract the signals at element positions, and time-reverse
   numerically (Aubry & Tanter 2010).

Both approaches are implemented in
`kwavers_physics::acoustics::transcranial::aberration_correction` —
`TranscranialAberrationCorrection::apply_time_reversal_correction` performs the
phase-conjugation refocus on recorded element signals.

---

## 15.5 CT-Based Aberration Correction

### 15.5.1 Hounsfield Unit to Acoustic Velocity Mapping

CT scanners report attenuation in Hounsfield units:

$$
\text{HU}(x) = 1000 \times \frac{\mu(x) - \mu_{\text{water}}}{\mu_{\text{water}}},
\tag{15.8}
$$

where $\mu$ is the linear X-ray attenuation coefficient. A piecewise-linear mapping
to acoustic speed is used in clinical transcranial FUS planning:

$$
c(x) =
\begin{cases}
c_{\text{tissue}} & \text{HU}(x) \le 0 \\
c_{\text{tissue}} + \dfrac{\text{HU}(x)}{\text{HU}_{\text{bone}}}
  \bigl(c_{\text{bone}} - c_{\text{tissue}}\bigr) & 0 < \text{HU}(x) \le \text{HU}_{\text{bone}} \\
c_{\text{bone}} & \text{HU}(x) > \text{HU}_{\text{bone}},
\end{cases}
\tag{15.9}
$$

with $c_{\text{tissue}} = 1540$ m s$^{-1}$, $c_{\text{bone}} = 2900$ m s$^{-1}$, and
$\text{HU}_{\text{bone}} = 1000$. Similarly, density is mapped as

$$
\rho(x) = \rho_{\text{tissue}} + \frac{\text{HU}(x)}{\text{HU}_{\text{bone}}}
  \bigl(\rho_{\text{bone}} - \rho_{\text{tissue}}\bigr).
\tag{15.10}
$$

The resulting heterogeneous medium is ingested by
`kwavers_medium::HeterogeneousMedium`, which holds spatially varying
$c_0(\mathbf{x})$, $\rho_0(\mathbf{x})$, and $\alpha_0(\mathbf{x})$ arrays. The HU→acoustic
maps themselves are `kwavers_imaging::medical::ct_loader::CTImageLoader::{hu_to_sound_speed,
hu_to_density}`. **Note:** the shipped maps use the Aubry et al. (2003) bilinear form
(soft tissue below HU ≈ 700, then a linear bone branch), not the piecewise-linear form
written in Eqs. (15.9)–(15.10); the two agree on the qualitative trend but differ in the
threshold and slope.

### 15.5.2 Full-Wave Simulation for Phase Correction

**Theorem 15.5 (FDTD/PSTD Aberration Map Validity).**  
Given the patient-specific heterogeneous medium $\{c_0(\mathbf{x}), \rho_0(\mathbf{x})\}$
derived from equation (15.9)–(15.10), a forward PSTD simulation of a point source at
the target produces pressure signals $\hat{p}(\mathbf{r}_i, \omega)$ at each array
element. The phase of each signal is

$$
\angle \hat{p}(\mathbf{r}_i, \omega_0) = -\phi_{\text{aberration},i} + \phi_{\text{geometric},i},
\tag{15.11}
$$

where $\phi_{\text{geometric},i}$ is the geometric delay from target to element $i$
(derivable analytically) and $\phi_{\text{aberration},i}$ is the skull-induced phase
error. Applying the conjugate phase to the emission delays corrects aberration.

*Proof.* The forward Green's function from source at $\mathbf{r}_0$ to receiver at
$\mathbf{r}_i$ in a heterogeneous medium satisfies the Helmholtz equation

$$
\nabla^2 G + k^2(\mathbf{x}) G = -\delta(\mathbf{x} - \mathbf{r}_0).
$$

In the ray approximation the solution decomposes as

$$
G(\mathbf{r}_i, \mathbf{r}_0; \omega) = A_i(\omega)\, e^{j \int_{\text{path}} k(s) ds}
= A_i e^{j(\phi_{\text{geometric},i} + \phi_{\text{aberration},i})},
$$

where $A_i$ encodes geometrical spreading and amplitude effects, and the phase
integral separates into the homogeneous (geometric) part and the excess phase caused
by skull heterogeneity. Taking the argument of $G$ and negating gives the correction
delay $\tau_i = -\phi_{\text{aberration},i} / \omega_0$.
PSTD numerically integrates the heterogeneous wave equation
(see `kwavers_solver::forward::pstd`), producing $\hat{p}(\mathbf{r}_i, \omega_0)$
that encodes this phase without further approximation. $\square$

### 15.5.3 Ray-Tracing Approximation for Thin Skulls

When the skull is acoustically thin ($d \ll \lambda_{\text{tissue}}$), the full-wave
simulation simplifies to a ray integral. For element $i$ and a ray from target
$\mathbf{r}_0$ to element $\mathbf{r}_i$, the excess travel time through the skull is

$$
\Delta\tau_i = \int_{\text{skull path}} \frac{ds}{c(s)} - \frac{d_i}{c_{\text{tissue}}},
\tag{15.12}
$$

where $d_i$ is the total skull thickness along the ray. For a uniform skull slab:

$$
\Delta\tau_i \approx d_i \left(\frac{1}{c_{\text{skull}}} - \frac{1}{c_{\text{tissue}}}\right).
\tag{15.13}
$$

The corresponding phase correction is $\Delta\phi_i = \omega_0\,\Delta\tau_i$.

**Validity condition.** The ray approximation is valid when the Fresnel number
$\mathcal{F} = a^2 / (\lambda d) \gg 1$, where $a$ is the skull aperture element
spacing and $d$ the skull thickness. For $a = 5$ mm, $\lambda = 1.5$ mm, $d = 7$ mm:
$\mathcal{F} \approx 2.4$, marginally within the ray regime. For higher accuracy,
the full PSTD simulation is preferred; `kwavers_physics::acoustics::transcranial`
provides both paths and selects automatically based on configured Fresnel number.

![CT Hounsfield-unit to sound-speed/density conversion (`CTImageLoader::{hu_to_sound_speed, hu_to_density}`, Aubry 2003).](figures/ch16/fig03_ct_conversion.png)

---

## 15.6 Thermal Hazard at the Skull

### 15.6.1 Power Deposition in Bone

The time-averaged acoustic intensity attenuation power density in a lossy medium is

$$
Q(\mathbf{x}) = 2\alpha(\mathbf{x})\, I(\mathbf{x}),
\tag{15.14}
$$

where $\alpha$ is the amplitude absorption coefficient (Np m$^{-1}$) and $I$ is the
local time-averaged intensity. For the skull at 1 MHz,
$\alpha_{\text{skull}} \approx 10$–$20$ dB cm$^{-1}$ converts to
$\alpha = \alpha_{\text{dB/cm}} \times \ln(10)/20 \times 100$:

$$
\alpha_{\text{skull}} \approx 115 \text{–} 230 \;\text{Np m}^{-1}.
$$

### 15.6.2 Temperature Rise

**Theorem 15.6 (Skull Temperature Rise Under Continuous Insonation).**
For a spatially uniform insonation of intensity $I$ absorbed in a skull layer of
thickness $d$ with density $\rho$, specific heat $c_p$, and no thermal diffusion
during interval $[0, t_{\text{on}}]$:

$$
\Delta T_{\text{skull}}(t_{\text{on}})
= \frac{2\alpha_{\text{skull}}\,I}{\rho_{\text{skull}}\,c_{p,\text{skull}}} \, t_{\text{on}}.
\tag{15.15}
$$

*Proof.* In the absence of thermal diffusion the bio-heat transfer equation reduces to

$$
\rho c_p \frac{\partial T}{\partial t} = Q = 2\alpha I.
$$

Integrating from $t = 0$ to $t = t_{\text{on}}$ with $T(0) = T_0$ gives
$\Delta T = (2\alpha I / (\rho c_p)) \, t_{\text{on}}$. $\square$

**Numerical example.** For $\alpha = 150$ Np m$^{-1}$, $I = 1000$ W m$^{-2}$,
$\rho_{\text{skull}} = 1900$ kg m$^{-3}$, $c_{p,\text{skull}} = 1300$ J kg$^{-1}$ K$^{-1}$,
$t_{\text{on}} = 10$ s:

$$
\Delta T_{\text{skull}} = \frac{2 \times 150 \times 1000}{1900 \times 1300} \times 10
\approx \frac{3 \times 10^5}{2.47 \times 10^6} \times 10 \approx 1.2 \;\text{K per}
\;10\;\text{s exposure}.
$$

In practice, skull absorption occurs over the full transducer aperture area while the
thermal target receives a fraction; the skull can reach dangerously high temperatures
(>10 K rise) before target temperatures are therapeutic. This asymmetry is the
primary limiting factor for transcranial HIFU treatments. The Exablate Neuro system
(InSightec) monitors skull heating via MR thermometry and incorporates cooling
between sonications.

**Implication for `kwavers`.** The
`kwavers_physics::acoustics::transcranial::safety_monitoring::TranscranialSafetyMonitor`
tracks thermal and mechanical-index exposure (`TranscranialSafetyDose`, `MechanicalIndex`)
from the forward-solver power density and enforces a configurable safety ceiling before each
sonication; the per-layer adiabatic rise of equation (15.15) is its skull-heating term.

![Skull temperature rise under continuous insonation (Eq. 15.15; `analytical::skull` surface-heating model).](figures/ch16/fig05_skull_temperature.png)

---

## 15.7 Standing Waves in the Skull Cavity

### 15.7.1 Resonance Frequencies of the Cranial Cavity

The intracranial space approximates a closed cavity. The simplest resonance model
treats the skull as two parallel reflecting walls separated by distance $L$ (lateral
skull diameter). The resonance condition for modes along that axis is

$$
f_n = \frac{n\,c_{\text{brain}}}{2 L},\quad n = 1, 2, 3, \ldots
\tag{15.16}
$$

For $L = 18$ cm and $c_{\text{brain}} = 1540$ m s$^{-1}$:

$$
f_1 = \frac{1 \times 1540}{2 \times 0.18} \approx 4.3 \;\text{kHz}.
$$

The spacing between adjacent modes is constant at $\Delta f = c / (2L) \approx 4.3$ kHz.

### 15.7.2 Proof That Clinical-Frequency Resonance Is Negligible

**Theorem 15.7 (High-Mode Density at Clinical Frequencies).**
At clinical transcranial frequencies $f_{\text{clinical}} \in [0.5, 2]$ MHz, the number
of resonant modes below $f_{\text{clinical}}$ in the cranial cavity is of order
$10^3$, the inter-modal spacing is $\Delta f \approx 4.3$ kHz, and the effective
cavity $Q$-factor is low ($Q \lesssim 5$). Therefore no individual resonance can be
selectively excited, and the cavity behaves as a broadband lossy medium.

*Proof.*

1. **Mode count.** The number of axial modes below $f$ is $n_{\max} = f / \Delta f$.
   At $f = 1$ MHz: $n_{\max} \approx 1 \times 10^6 / 4300 \approx 233$ axial modes.
   Including transverse modes (Weyl's law) the total mode count scales as
   $N(f) \propto V f^3 / c^3$ where $V \approx 1.2 \times 10^{-3}$ m$^3$:
   $N \approx 4\pi V f^3 / (3 c^3) \approx 4\pi \times 1.2 \times 10^{-3}
   \times 10^{18} / (3 \times 3.65 \times 10^9) \approx 1.4 \times 10^6$ modes.

2. **Quality factor.** Each mode is attenuated by brain tissue absorption
   ($\alpha_{\text{brain}} \approx 0.6$ dB cm$^{-1}$ at 1 MHz) and skull
   transmission losses at each reflection. The energy decay per round trip of length
   $2L = 0.36$ m is dominated by transmission loss $T_I^{(\text{skull})} \approx 0.22$
   per surface (equation 15.2, single surface):

   $$
   Q \approx \frac{\pi f}{\Delta f \cdot |\ln T_I^{(\text{skull})}|}
   \approx \frac{\pi \times 10^6}{4300 \times 1.51} \approx 484.
   $$

   However the effective $Q$ integrated over all mode families is much lower because
   inhomogeneous skull geometry broadens each resonance. Measured coherence times of
   $\sim 1\text{–}3$ $\mu$s correspond to $Q_{\text{eff}} \approx f / \Delta f_{\text{mode}}
   \approx 10^6 / (1/(2 \times 10^{-6})) = 2$.

3. **Frequency spacing vs source bandwidth.** A 1-MHz transducer has a $-6$ dB
   bandwidth of $\sim 0.1$–$0.4$ MHz, which encompasses thousands of modes.
   No individual resonance is resolvable; the transmission coefficient averaged over
   any bandwidth $\gg \Delta f$ is the incoherent average, equal to the power
   transmission coefficient, with no resonance enhancement.

Taken together, these three points prove that at clinical frequencies the skull cavity
does not sustain spatially coherent standing waves of clinical significance. $\square$

**Consequence.** Transcranial focusing models need not account for cavity
resonances at $f \ge 0.5$ MHz. The relevant standing-wave concern is at much lower
frequencies (diagnostic Doppler, sonothrombolysis enhancement), where $n$ is small
and coherence lengths are long.

---

## 15.8 Cavitation Threshold Modification Under Skull Passage

### 15.8.1 Effective Mechanical Index at the Focus

The mechanical index quantifies cavitation risk:

$$
\text{MI} = \frac{P_r^{-}}{\sqrt{f / \text{MHz}}} \;\text{[MPa MHz}^{-1/2}\text{]},
\tag{15.17}
$$

where $P_r^{-}$ is the derated peak rarefaction pressure in MPa. Skull passage
attenuates $P_r^{-}$ by the product of the transmission coefficient and the
frequency-dependent path attenuation. Define the path reduction factor

$$
R_{\text{path}} = T_p \cdot 10^{-\alpha_{\text{path}} d_{\text{path}} / 20},
\tag{15.18}
$$

where $\alpha_{\text{path}}$ is the total path attenuation in dB cm$^{-1}$ and
$d_{\text{path}}$ is the path length in cm. The effective MI at the focus is then

$$
\text{MI}_{\text{eff}} = \text{MI}_{\text{incident}} \times R_{\text{path}}.
\tag{15.19}
$$

**Theorem 15.8 (Cavitation Threshold Elevation by Skull Loss).**
For a homogeneous skull slab with parameters as in Section 15.2, $R_{\text{path}} < 1$,
so $\text{MI}_{\text{eff}} < \text{MI}_{\text{incident}}$. The cavitation threshold
$\text{MI}_{\text{cav}}$ at the focus requires

$$
\text{MI}_{\text{incident}} > \frac{\text{MI}_{\text{cav}}}{R_{\text{path}}},
\tag{15.20}
$$

which is elevated relative to the free-field requirement.

*Proof.* The threshold condition for inertial cavitation is $\text{MI}_{\text{eff}} \ge
\text{MI}_{\text{cav}}$. Substituting equation (15.19):
$\text{MI}_{\text{incident}} \times R_{\text{path}} \ge \text{MI}_{\text{cav}}$,
rearranging gives equation (15.20). Since $R_{\text{path}} < 1$ (attenuation and
reflection both reduce amplitude), the required incident MI exceeds $\text{MI}_{\text{cav}}$.
$\square$

**Heterogeneity and hot spots.** The above analysis assumes a homogeneous skull.
In reality, skull heterogeneity creates spatially varying $T_p(\mathbf{x}_\perp)$ and
local constructive interference can produce acoustic hot spots within or behind the
skull. These hot spots can exceed the cavitation threshold even when the mean-path
analysis suggests a safe margin. Full-wave simulation in
`kwavers_solver::forward::pstd` with the heterogeneous medium provides the
spatially resolved MI map needed to identify such hot spots.

---

## 15.9 Focused Ultrasound Neuromodulation (Non-Thermal LIPUS)

### 15.9.1 Low-Intensity Pulsed Ultrasound (LIPUS) Parameters

FUS neuromodulation operates in a regime distinct from thermal ablation:

| Parameter | Neuromodulation (LIPUS) | Thermal ablation (HIFU) |
|---|---|---|
| ISPTA (W cm$^{-2}$) | 0.01–3 | 10–10000 |
| Frequency (MHz) | 0.25–0.65 | 0.65–1.5 |
| Duty cycle | 5–50% | 50–100% |
| Pulse duration | 0.1–100 ms | seconds |
| MI | 0.1–1.9 | 1–15 |

The fundamental mechanism by which LIPUS activates or suppresses neurons without
detectable tissue heating remains debated. Three candidate mechanisms are:

1. **Direct mechanical stimulation** of mechanosensitive ion channels (Piezo1/2, TREK).
2. **Intramembrane cavitation** (the Bilayer Sonophore / NICE model, Lemaire et al.).
3. **Indirect thermal effects** below calorimetric detection limits.

### 15.9.2 The NICE Model (Neuronal Intramembrane Cavitation Excitation)

**Physical setup.** The NICE model (Krasovitski et al. 2011; Lemaire et al. 2019)
proposes that acoustic pressure deforms the plasma membrane, creating oscillating
intramembrane nanoscale cavities (sonophores) whose radius $R(t)$ follows a modified
Rayleigh–Plesset equation:

$$
\rho_{\text{fluid}} \left(R\ddot{R} + \tfrac{3}{2}\dot{R}^2\right)
= P_{\text{gas}}(R) - P_0 - P_{\text{ac}}(t)
  - \frac{4\mu \dot{R}}{R} - \frac{2\sigma(R)}{R},
\tag{15.21}
$$

where $P_{\text{gas}} = P_0 (R_0/R)^{3\gamma}$ for adiabatic gas, $\mu$ is
effective fluid viscosity, $\sigma(R)$ is the membrane tension (nonlinear, derived
from bending and stretch moduli), and $P_{\text{ac}}(t) = P_A \sin(\omega t)$ is the
applied acoustic pressure.

**Capacitance modulation.** The membrane capacitance per unit area is

$$
C_m(R) = \epsilon_0 \epsilon_r / \delta(R),
\tag{15.22}
$$

where $\delta(R)$ is the effective membrane thickness, which varies with $R$. As the
sonophore expands ($R > R_0$), membrane thinning increases $C_m$; as it contracts,
$C_m$ decreases. This modulates the charge stored on the membrane and effectively
drives an oscillating transmembrane current analogous to an external stimulus.

**Theorem 15.9 (Capacitance Current from Sonophore Oscillation).**
For a neuron with transmembrane voltage $V_m$ and membrane capacitance $C_m(t)$
varying sinusoidally at frequency $\omega$, the membrane current due to capacitance
modulation is

$$
I_{C_m}(t) = V_m \dot{C}_m(t) = V_m \cdot \omega \Delta C_m \cos(\omega t),
\tag{15.23}
$$

which drives an intracellular current proportional to both the resting voltage and the
amplitude of capacitance oscillation $\Delta C_m$.

*Proof.* The total membrane charge is $q = C_m(t) V_m$. The current is
$I = dq/dt = \dot{C}_m V_m + C_m \dot{V}_m$. At the onset of sonication, when the
neuron has not yet responded, $\dot{V}_m \approx 0$ and $I_{C_m} = V_m \dot{C}_m$.
For $C_m(t) = C_{m,0} + \Delta C_m \sin(\omega t)$:
$\dot{C}_m = \omega \Delta C_m \cos(\omega t)$, giving equation (15.23). $\square$

This current enters the Hodgkin–Huxley equations as an additional drive term, and
for sufficient $\Delta C_m$ (achieved at MI $\gtrsim 0.3$) can trigger action
potentials. kwavers implements this NICE / intramembrane-cavitation pathway in
`kwavers_physics::acoustics::therapy::neuromodulation`: a grounded Bilayer
Sonophore (`bls`) drives the oscillating membrane capacitance into a genuine
Hodgkin–Huxley neuron (`hodgkin_huxley`), exposed as `simulate_nice` (the
carrier-resolved NICE simulation, `nice`) and `simulate_sonic` (the Lemaire
cycle-averaged SONIC reduction, `sonic`), with pulse-train protocol and dosimetry
(`protocol`). The complementary mechanosensitive-channel (membrane-tension-gated)
pathway — `kwavers_physics::acoustics::therapy::sonogenetics` (`channels`,
`membrane`, `neuron`, `arf_field`) — is implemented alongside it. Both mechanisms
are developed in full in the dedicated **Neuromodulation** chapter (Ch26), the
canonical home for the FUS-neuromodulation mechanisms and their clinical
parameter space.

### 15.9.3 Reference

Tyler W.J. (2011). "Noninvasive neuromodulation with ultrasound? A continuum
mechanics hypothesis." *Neuron* **72**(1): 9–18.

---

## 15.10 Focused Ultrasound Blood–Brain Barrier Opening

### 15.10.1 Mechanism and Safety Window

MRgFUS + systemically injected microbubbles (SonoVue, Definity) can transiently
open the BBB via mechanical interaction of oscillating microbubbles with cerebrovascular
endothelial cells and tight junctions. The sequence is:

1. **Stable cavitation**: microbubbles oscillate non-inertially near resonance.
2. **Microstreaming and radiation force**: oscillating bubbles push fluid and exert
   time-averaged radiation forces on cell membranes.
3. **Tight junction opening**: mechanical stress transiently disrupts claudin and
   occludin, widening paracellular gaps.
4. **BBB closure**: restores within 4–24 h for sub-threshold exposures; no
   permanent damage if MI is controlled.

kwavers implements the BBB-opening exposure, permeability, and safety models in
`kwavers_physics::acoustics::transcranial::bbb_opening` (`models`, `safety`,
`simulator`). This section covers the transcranial-context physics; the full
closed-loop treatment-planning workflow — Keller–Miksis microbubble dynamics,
multi-spot targeting, sparse (aperiodic) arrays, and passive-cavitation
harmonic-dose monitoring — is in the dedicated **LIFU-Mediated Blood–Brain Barrier
Opening** chapter (Ch24), the canonical home.

### 15.10.2 Acoustic Radiation Force on Endothelial Cells

**Theorem 15.10 (Acoustic Radiation Force on a Sphere Near a Boundary).**
A microbubble of equilibrium radius $R_0$ oscillating in a standing-wave field near an
endothelial cell wall experiences a primary Bjerknes force:

$$
\mathbf{F}_{\text{rad}} = -\langle V(t)\, \nabla p(\mathbf{r}, t) \rangle,
\tag{15.24}
$$

where $V(t) = \tfrac{4}{3}\pi R^3(t)$ is the instantaneous bubble volume, $p$ is the
local pressure, and $\langle \cdot \rangle$ denotes time averaging over one cycle.

*Proof.* The momentum equation for a bubble of mass $m$ in a non-uniform pressure
field is $m\ddot{\mathbf{r}} = -\nabla p \cdot V(t)$ to lowest order in bubble size.
Time-averaging with $V(t) = V_0 + \tilde{V}(t)$ and $\nabla p = \nabla p_0 + \nabla\tilde{p}$:

$$
\langle V \nabla p \rangle = V_0 \nabla p_0 + \langle \tilde{V} \nabla\tilde{p} \rangle,
$$

where the cross-terms vanish by orthogonality of the oscillating components. The
term $\langle \tilde{V}\,\nabla\tilde{p} \rangle$ is non-zero and responsible for the
acoustic radiation force. $\square$

In the BBB context this force drives bubble translation toward the cell wall, increasing
the contact stress. Controlled via MI to remain in the stable cavitation regime
(MI 0.3–0.6 per Deffieux & Konofagou 2010), the mechanical stress is sufficient to
open tight junctions without endothelial lysis.

### 15.10.3 Safe Exposure Window

Published preclinical and Phase I clinical data define the safe BBB-opening window:

| Parameter | Safe BBB Opening | Damage Threshold |
|---|---|---|
| MI at focus | 0.3–0.6 | >0.8 |
| Sonication duration | 30–60 s | >120 s |
| Microbubble dose | 0.1–0.2 mL kg$^{-1}$ | >0.5 mL kg$^{-1}$ |
| Frequency | 0.22–1.5 MHz | — |

The combination of passive cavitation detection (PCD) and real-time MR thermometry
enables closed-loop control within this window (Lipsman et al. 2013).

**Theorem 15.11 (Endothelial Volume Change Under Radiation Force).**
An endothelial cell modelled as a spherical elastic body of bulk modulus $K$ and
volume $V_0$ subjected to acoustic radiation pressure $P_{\text{rad}}$ undergoes
fractional volume change

$$
\frac{\Delta V}{V_0} = -\frac{P_{\text{rad}}}{K}.
\tag{15.25}
$$

For $K \approx 5 \times 10^3$ Pa (endothelial cytoplasm stiffness) and
$P_{\text{rad}} \approx 100$ Pa (at MI 0.5, 500 kHz, 1 cm from bubble):
$\Delta V / V_0 \approx 2\%$, sufficient to deform tight junction proteins beyond
their elastic limit (published rupture strain $\approx 1\text{–}3\%$).

*Proof.* By definition of bulk modulus: $K = -V_0\, dP/dV$. For a small applied
pressure change $\Delta P = P_{\text{rad}}$ at constant entropy:
$\Delta V = -V_0 \Delta P / K = -V_0 P_{\text{rad}} / K$. $\square$

---

## 15.11 kwavers Transcranial Simulation Workflow

### 15.11.1 CT Segmentation and Medium Construction

```
CT DICOM stack
     │
     ▼
kwavers_imaging::medical::ct_loader::CTImageLoader::load()   // DICOM ingestion, HU calibration
     │
     ▼
CTImageLoader::{hu_to_sound_speed, hu_to_density}           // per-voxel HU → c₀, ρ₀ (Aubry 2003)
     │
     ▼
kwavers_medium::HeterogeneousMedium                         // spatially-varying c₀(x), ρ₀(x), α₀(x)
     │
     ▼
kwavers_solver::forward::pstd::PSTDSolver                   // PSTD propagation through skull medium
     │
     ▼
kwavers_physics::acoustics::transcranial::                  // phase-map extraction + TR correction
    TranscranialAberrationCorrection::calculate_correction
     │
     ▼
kwavers_physics::analytical::skull::strehl_ratio            // focus-quality metric (Maréchal Strehl)
```

### 15.11.2 Simulation Configuration

The PSTD solver requires spatial sampling satisfying the Nyquist criterion for the
fastest wave speed in the domain:

$$
\Delta x \le \frac{c_{\text{max}}}{2\,f_{\text{max}}} = \frac{2900}{2 \times 1.5 \times 10^6} \approx 0.97 \;\text{mm}.
\tag{15.26}
$$

The CFL stability criterion for the pseudo-spectral method is automatically satisfied
by the PSTD time-stepping scheme (spectral spatial derivatives, leapfrog time integration):

$$
\Delta t \le \frac{\Delta x}{c_{\text{max}}} \cdot \frac{1}{\pi} \approx \frac{0.97 \times 10^{-3}}{2900 \times \pi} \approx 106 \;\text{ns}.
\tag{15.27}
$$

### 15.11.3 Phase Correction Map Computation

After the forward simulation, `TranscranialAberrationCorrection` extracts element
signals from the `kwavers_receiver::recorder::SensorRecorder`, applies Fourier analysis at
$f_0$ to extract $\angle\hat{p}(\mathbf{r}_i, f_0)$, subtracts the geometric delay,
and returns the correction phases/delays $\{\Delta\tau_i\}$ from
`calculate_correction`. These delays are applied to transducer firing times in the
subsequent therapeutic simulation (the correction is carried as a per-element phase array;
there is no separate `ElementDelayTable` type — the delays are the `PhaseCorrection` output).

### 15.11.4 Skull Thermal Safety Integration

For each therapeutic sonication the thermal monitor integrates:

$$
T_{\text{skull}}(\mathbf{x}, t) = T_0 + \int_0^t \frac{Q(\mathbf{x}, t')}{\rho(\mathbf{x}) c_p(\mathbf{x})} dt'
+ \text{[diffusion term from Pennes bio-heat equation]},
\tag{15.28}
$$

using the spatially resolved $Q(\mathbf{x}) = 2\alpha(\mathbf{x}) I(\mathbf{x})$ from
the PSTD field output. Skull temperature is monitored at every voxel within the CT
skull mask. The simulation is halted if $T_{\text{skull,max}} > T_{\text{ceiling}}$
(default 43°C). This implements the safety interlocks described for MRgFUS clinical
systems (Lipsman et al. 2013).

### 15.11.5 Workflow

Two routes exist, trading fidelity for turnkey convenience.

**Turnkey planning (analytic, no full-wave solve).**
`kwavers_physics::acoustics::transcranial::TreatmentPlanner::generate_plan(patient_id, targets,
spec)` is a *single call* that runs the whole planning pipeline: skull-property analysis from the
CT volume → focused-bowl element placement (Fibonacci hemisphere on the focal sphere) → **geometric
phase conjugation + CT skull-aberration correction** → Rayleigh–Sommerfeld intensity field → Pennes
steady-state thermal field → safety validation → treatment-time estimate, returning a
`TranscranialTreatmentPlan`. The aberration step is the phase-screen ray integral
$\Delta\phi_i=\int(k_{\text{local}}-k_{\text{water}})\,ds$ (`TranscranialAberrationCorrection`,
§15.11.3); the applied per-element delay is $\phi_i=\phi_i^{\text{geo}}+(-\Delta\phi_i)$, so a
homogeneous CT leaves the equidistant bowl in phase while a real skull induces the ray-dependent
correction. (Element positions are stored in millimetres; all physics is computed in metres — the
target centre and grid spacing units.)

**High-fidelity (full-wave PSTD).** For an *exact* aberration map (Theorem 15.5) the caller drives
the building blocks directly: a forward PSTD solve from a virtual point source at the target records
per-element pressures, whose phases feed `calculate_correction`; the corrected phases then drive a
guarded therapeutic solve. This route is assembled through the orchestrator API rather than a single
call:

```text
1. CT → medium
   kwavers_imaging::medical::ct_loader::CTImageLoader::load(path)      // HU volume
   per voxel: CTImageLoader::{hu_to_sound_speed, hu_to_density}        // Aubry 2003 maps
   → kwavers_medium::HeterogeneousMedium { c0(x), rho0(x), alpha0(x) }

2. Forward solve from a virtual point source at the target
   kwavers_solver::forward::pstd::PSTDSolver  (orchestrator: run_orchestrated / step_forward)
   record element pressures via the public  sensor_recorder  field
   (kwavers_receiver::recorder::SensorRecorder::{extract_pressure_data, pressure_data_view})

3. Aberration correction
   kwavers_physics::acoustics::transcranial::TranscranialAberrationCorrection::new(&grid)
   .calculate_correction(...)  → PhaseCorrection (per-element Δτ_i)
   (or .apply_time_reversal_correction(...) for the time-reversal route)

4. Therapeutic solve with the corrected element phases + safety monitoring
   re-run the PSTD solve with the PhaseCorrection delays applied to the array,
   guarded by kwavers_physics::acoustics::transcranial::TranscranialSafetyMonitor
   (TranscranialSafetyDose / MechanicalIndex against a configurable ceiling)
```

![Through-skull insertion loss vs frequency (`analytical::skull::skull_transfer_matrix_transmission`).](figures/ch16/fig01_skull_insertion_loss.png)

*Figure 15.5. Frequency-dependent transcranial insertion loss from the layered skull
transfer-matrix model (§15.2; `kwavers_physics::analytical::skull`).*

---

## 15.12 Summary of Key Theorems

| Theorem | Statement | Equation |
|---|---|---|
| 15.1 | Normal-incidence pressure transmission $T_p = 2Z_2/(Z_1+Z_2)$ | (15.1) |
| 15.2 | Phase accumulation through heterogeneous path | (15.3) |
| 15.3 | Maréchal approximation: $S = e^{-\sigma_\phi^2}$ | (15.6) |
| 15.4 | Time-reversal achieves phase conjugation at target | (15.7) |
| 15.5 | PSTD simulation provides exact aberration phase map | (15.11) |
| 15.6 | Skull temperature rise under CW insonation | (15.15) |
| 15.7 | Clinical-frequency resonances unresolvable ($>10^3$ modes in BW) | — |
| 15.8 | Skull loss elevates cavitation threshold at focus | (15.20) |
| 15.9 | Sonophore capacitance modulation drives membrane current | (15.23) |
| 15.10 | Bjerknes radiation force on bubble near wall | (15.24) |
| 15.11 | Endothelial volume change under radiation pressure | (15.25) |

---

## 15.13 References

1. Aubry J.-F. & Tanter M. (2010). "MR-guided transcranial focused ultrasound."
   *IEEE Transactions on Biomedical Engineering* **57**(6): 1296–1310.
   DOI: 10.1109/TBME.2010.2041998

2. Deffieux T. & Konofagou E.E. (2010). "Numerical study of a simple transcranial
   focused ultrasound system applied to blood-brain barrier opening."
   *Ultrasound in Medicine & Biology* **36**(5): 765–775.
   DOI: 10.1016/j.ultrasmedbio.2010.01.003

3. Tyler W.J. (2011). "Noninvasive neuromodulation with ultrasound? A continuum
   mechanics hypothesis." *Neuron* **72**(1): 9–18.
   DOI: 10.1016/j.neuron.2011.09.001

4. Lipsman N., Schwartz M.L., Huang Y., et al. (2013). "MR-guided focused
   ultrasound thalamotomy for essential tremor: a proof-of-concept study."
   *Lancet Neurology* **12**(5): 462–468.
   DOI: 10.1016/S1474-4422(13)70048-6

5. Krasovitski B., Frenkel V., Shoham S., & Kimmel E. (2011). "Intramembrane
   cavitation as a unifying mechanism for ultrasound-induced bioeffects."
   *Proceedings of the National Academy of Sciences* **108**(8): 3258–3263.

6. Lemaire T., Neufeld E., Kuster N., & Bhatt D.L. (2019). "Understanding
   ultrasound neuromodulation using a computationally efficient and
   interpretable model of intramembrane cavitation." *Journal of Neural
   Engineering* **16**(4): 046007.

7. Treeby B.E. & Cox B.T. (2010). "k-Wave: MATLAB toolbox for the simulation
   and reconstruction of photoacoustic wave fields." *Journal of Biomedical
   Optics* **15**(2): 021314.

---

*Chapter authored for the kwavers ultrasound physics textbook series.
Simulation results are reproducible via the example scripts in
`pykwavers/examples/` using the `kwavers_solver::forward` PSTD backend.*
