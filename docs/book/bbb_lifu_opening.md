# Chapter 24 — LIFU-Mediated Blood–Brain Barrier Opening

> **Prerequisite:** Chapter 9 (Cavitation and Bubble Dynamics), Chapter 15
> (Safety and Dosimetry), Chapter 23 (Passive Acoustic Mapping).
> Familiarity with Pennes bioheat transfer and Keller–Miksis bubble dynamics
> is assumed.

---

## 24.1 Scope

Low-Intensity Focused Ultrasound (LIFU) combined with systemically injected
microbubbles (MBs) transiently opens the Blood–Brain Barrier (BBB) via
cavitation-mediated mechanotransduction.  Unlike thermal HIFU, LIFU operates
in the stable-cavitation regime (MI ≈ 0.3–0.5) to achieve reversible
permeabilisation without tissue damage, providing a clinically actionable drug
delivery window of 6–24 hours.

This chapter covers: the Keller–Miksis microbubble dynamics under LIFU
driving, the MI parameter space separating stable from inertial cavitation,
the Hill-function permeability enhancement model, thermal safety analysis via
Pennes bioheat + CEM43, CEUS contrast enhancement, and the BBB closure
kinetics.  A kwavers `PhysicsCatalog` simulation workflow is given in §24.7.

---

## 24.2 Microbubble dynamics — Keller–Miksis equation

A coated MB of equilibrium radius $R_0$ in blood driven by acoustic pressure
$p_a \sin(\omega t)$ obeys the Keller–Miksis equation (Keller & Miksis 1980;
Prosperetti 1988):

$$
\left(1 - \frac{\dot R}{c}\right) R\ddot R
+ \frac{3}{2}\left(1 - \frac{\dot R}{3c}\right)\dot R^2
= \frac{1}{\rho_L}\left(1 + \frac{\dot R}{c}\right)(p_L - p_\infty)
+ \frac{R}{\rho_L c}\dot p_L
$$

with liquid wall pressure:

$$
p_L(R) = \left(P_0 + \frac{2\sigma}{R_0}\right)\left(\frac{R_0}{R}\right)^{3\kappa}
        - \frac{2\sigma}{R} - \frac{4\mu\dot R}{R} - \frac{4\xi\dot R}{R^2}
$$

The shell viscosity term $4\xi\dot R/R^2$ (Doinikov–Dayton neo-Hookean model)
is critical for SonoVue-type phospholipid shells where $\xi \approx 1.5\;\text{nm·Pa·s}$.

> **Theorem 24.1 (Linear resonance frequency).**
> *For small oscillations $R = R_0 + x$, $|x| \ll R_0$, the linearised
> Keller–Miksis equation reduces to a damped harmonic oscillator with natural
> frequency (Minnaert 1933; Prosperetti 1977):*
> $$f_{res} = \frac{1}{2\pi R_0}\sqrt{\frac{1}{\rho_L}\left(3\kappa P_0 + (3\kappa-1)\frac{2\sigma}{R_0}\right)}$$

**Proof sketch.**  Substitute $R = R_0(1+\epsilon)$, expand all terms to first
order in $\epsilon$, collect the $\ddot\epsilon$ and $\epsilon$ coefficients.
The zero-radiation-damping ($c \to \infty$) limit gives the Minnaert formula;
the finite-$c$ correction adds the $O(R_0/\lambda)$ radiation term.  $\square$

For SonoVue at $R_0 = 1.5\;\mu\text{m}$ in blood this gives $f_{res} \approx 2.0\;\text{MHz}$,
supporting the clinical choice of 0.5–1.5 MHz for sub-resonance driving.

---

## 24.3 LIFU safety parameter space

The Mechanical Index (MI) is the internationally standardised LIFU safety
parameter (IEC 62359):

$$
\text{MI} = \frac{p^-_{derated}}{\sqrt{f_0\;[\text{MHz}]}}
$$

where $p^-_{derated}$ is the derated peak negative pressure in MPa.

Three critical boundaries in the $(f, \text{MI})$ plane:

| Regime | Approximate MI boundary | Physical mechanism |
|--------|------------------------|-------------------|
| SC onset with MBs | $\text{MI} \approx 0.18/\sqrt{f[\text{MHz}]}$ | Stable oscillation |
| BBB opening window | $0.20 \lesssim \text{MI} \lesssim 0.55$ | Microstreaming + shear |
| IC onset with MBs | $\text{MI} \approx 0.45/\sqrt{f[\text{MHz}]}$ | Blake threshold × sensitisation |

> **Theorem 24.2 (Blake threshold with MBs).**
> *The acoustic pressure threshold for inertial cavitation of a free bubble is:*
> $$p_{Blake} = P_0\sqrt{1 + \frac{4}{27}\left(\frac{2\sigma/R_0}{P_0}\right)^3}$$
> *Coated MBs lower this threshold by a factor of 2–4 due to shell pre-stress
> reducing the effective surface tension.*

**Proof.**  The Blake threshold follows from requiring a real positive maximum
of the quasi-static bubble potential energy (Leighton 1994, §4.4.1).  The MB
sensitisation factor is derived from the shell-modified equilibrium condition
$P_0 + 2\sigma_{eff}/R_0 = p_{gas,0}$ where $\sigma_{eff} < \sigma_{water}$.
$\square$

---

## 24.4 Permeability enhancement model

> **Definition 24.1 (BBB acoustic dose).**
> *$D = \text{MI}^2 \cdot t_{on} \cdot \text{PRF}\;[\text{MI}^2\text{·s}]$
> is the cumulative acoustic dose, proportional to the time-averaged acoustic
> energy deposited at the focus per unit area of the skull window.*

BBB permeability enhancement follows a Hill-function dose-response:

$$
P(D) = P_{max}\;\frac{D^n}{D_{50}^n + D^n}
$$

with $D_{50} \approx 1.2\;\text{MI}^2\text{·s}$ and $n \approx 2.5$ for stable
cavitation (fit to Evans-blue extravasation data, McDannold 2008).

> **Theorem 24.3 (Damage-free operating window).**
> *There exists a dose interval $[D_{min}, D_{max}]$ such that:*
> *(a) $P(D_{min}) > P_{baseline}$ (effective opening),*
> *(b) the tissue damage probability $P_{dam}(D_{max}) < \epsilon_{tol}$,*
> *provided the stable-cavitation regime is maintained ($\text{MI} < \text{MI}_{IC}$).*

**Proof.**  $P(D)$ is continuous, strictly increasing, and bounded above.
$P_{dam}(D)$ is a sigmoid with threshold $D_{dam} > D_{50}$ (empirically
$D_{dam}/D_{50} \approx 2.9$ in liver; brain tissue shows smaller margin due
to lower cavitation nucleation density).  The window $[D_{min}, D_{max}]$
is non-empty iff $D_{50} < D_{dam}$, which holds in the SC regime by
construction.  $\square$

---

## 24.5 Thermal safety

Pennes bioheat equation (linearised for small $\Delta T$):

$$
\rho_T C_p \frac{dT}{dt} = Q_{us} - W_b \rho_b C_b (T - T_{art})
$$

with acoustic heat source $Q_{us} = 2\alpha_T I_{SPTA}$ (W/m³) during
on-pulses and $Q_{us}=0$ during off-pulses.

The CEM43 thermal dose (Sapareto & Dewey 1984) accumulates during each
on-pulse:

$$
\text{CEM43} = \int_0^{t_{end}} R^{43-T(t)}\;\mathrm{d}t
\qquad R = \begin{cases} 0.5 & T > 43°C \\ 0.25 & T \leq 43°C \end{cases}
$$

> **Corollary 24.1 (LIFU thermal safety margin).**
> *For LIFU with $I_{SATA} \leq 10\;\text{W/cm}^2$, $DC \leq 5\%$, and
> blood perfusion $W_b = 0.01\;\text{s}^{-1}$ (brain), the steady-state
> focal temperature rise is $\Delta T_{ss} \approx 0.3°C$ and CEM43 < 0.01 min
> for sonication durations below 120 s — well below the 0.25 min brain
> damage threshold (O'Reilly & Hynynen 2012).*

---

## 24.6 BBB closure kinetics

Post-sonication, BBB permeability recovers as a bi-exponential (Deffieux &
Konofagou 2010):

$$
P(t) = P_{peak}\left[0.6\;e^{-t/\tau_{fast}} + 0.4\;e^{-t/\tau_{slow}}\right]
$$

where $\tau_{fast} \approx 0.5\;\text{h}$ (tight-junction re-assembly) and
$\tau_{slow} \approx 6\;\text{h}$ (vesicular transport clearance).

The *drug delivery window* — the interval during which BBB permeability
exceeds 50% of peak opening — spans approximately 1–8 hours post-sonication
for the stable-cavitation dose range.

---

## 24.7 Simulation workflow

```rust
use kwavers::plugin::*;
use kwavers::{Grid, AcousticSolver, BoundaryType, PhysicsModelType,
              PhysicsModelConfig, PhysicsConfig, PhysicsCatalog,
              BubbleModel};

let mut config = PhysicsConfig::new();

// Linear acoustic propagation through brain tissue
config.models.push(PhysicsModelConfig {
    model_type: PhysicsModelType::LinearAcoustics {
        solver_type: AcousticSolver::PSTD { spectral_accuracy: true },
        boundary_conditions: BoundaryType::Absorbing { pml_layers: 12 },
    },
    enabled: true,
    parameters: Default::default(),
});

// Thermal monitoring (Pennes bioheat)
config.models.push(PhysicsModelConfig {
    model_type: PhysicsModelType::ThermalDiffusion {
        bioheat: true,
        perfusion: true,
    },
    enabled: true,
    parameters: Default::default(),
});

let manager = PhysicsCatalog::build(&config, &grid, &brain_medium, dt)?;
```

The BubbleDynamics capability (Keller–Miksis plugin) is a near-term roadmap
item in `PhysicsCatalog::build_plugin`; the current implementation returns a
structured `ConfigError::InvalidValue` identifying the gap explicitly (see
Chapter 22 §4, Theorem 22.1).

---

## 24.8 Figure sources

```bash
python pykwavers/examples/book/ch24_bbb_lifu_opening.py
```

Outputs to `docs/book/figures/ch24/` (PDF and PNG).

| Figure | Content |
|--------|---------|
| fig01  | Keller–Miksis MB dynamics at three LIFU pressure levels |
| fig02  | MI vs frequency safety parameter space |
| fig03  | BBB permeability vs acoustic dose (Hill model) |
| fig04  | LIFU thermal safety: temperature rise + CEM43 |
| fig05  | CEUS backscatter signal vs MB concentration |
| fig06  | BBB opening window: bi-exponential closure kinetics |

---

## 24.9 References

- Hynynen K., McDannold N., Vykhodtseva N., Jolesz F.A. *Noninvasive MR
  imaging-guided focal opening of the blood-brain barrier in rabbits.*
  Radiology **220**(3), pp. 640–646, 2001. doi:10.1148/radiol.2202001804
- McDannold N., Arvanitis C.D., Vykhodtseva N., Livingstone M.S.
  *Temporary disruption of the BBB by use of ultrasound and microbubbles.*
  Ultrasound Med. Biol. **34**(6), pp. 930–937, 2008.
  doi:10.1016/j.ultrasmedbio.2007.11.005
- Deffieux T., Konofagou E.E. *Numerical study of a simple transcranial
  focused ultrasound system applied to blood-brain barrier opening.*
  IEEE Trans. Ultrason. Ferroelectr. Freq. Control **57**(12), 2010.
  doi:10.1109/TUFFC.2010.1738
- Tung Y.-S., Vlachos F., Choi J.J., Deffieux T., Selert K., Konofagou E.E.
  *In vivo transcranial cavitation threshold detection during ultrasound-
  induced blood-brain barrier opening in mice.* Phys. Med. Biol.
  **55**(20), pp. 6141–6155, 2010. doi:10.1088/0031-9155/55/20/007
- O'Reilly M.A., Hynynen K. *Blood-brain barrier: real-time feedback-
  controlled focused ultrasound disruption.* Radiology **263**(1),
  pp. 96–106, 2012. doi:10.1148/radiol.11111417
- Keller J.B., Miksis M. *Bubble oscillations of large amplitude.*
  J. Acoust. Soc. Am. **68**(2), pp. 628–633, 1980.
  doi:10.1121/1.384720
- Sapareto S.A., Dewey W.C. *Thermal dose determination in cancer therapy.*
  Int. J. Radiat. Oncol. Biol. Phys. **10**(6), pp. 787–800, 1984.
  doi:10.1016/0360-3016(84)90379-1
- Leighton T.G. *The Acoustic Bubble.* Academic Press, 1994. §4.4.1.
