# Chapter 25 — Low-Intensity Ultrasound Neuromodulation

> **Prerequisite:** Chapter 16 (Safety and Dosimetry), Chapter 16
> (Transcranial Ultrasound), Chapter 17 (Sonogenetics), Chapter 23
> (BBB Opening), and Chapter 24 (Transcranial HIFU and BBB Treatment
> Planning).

---

## 25.1 Scope

Low-intensity transcranial ultrasound stimulation (TUS/tFUS/LIFUS) uses
sub-ablative acoustic pulses to reversibly alter neural excitability.  This
chapter covers the non-genetic, microbubble-free neuromodulation case.  It is
therefore distinct from:

- HIFU ablation, where the endpoint is thermal lesioning.
- Histotripsy, where the endpoint is cavitation-mediated tissue fractionation.
- BBB opening, where systemically administered microbubbles are part of the
  intended mechanism.
- Sonogenetics, where genetic expression of ultrasound-sensitive channels
  supplies cell-type selectivity.

The executable chapter script is
`pykwavers/examples/book/ch26_neuromodulation.py`.  It simulates acoustic
focusing, MI and intensity, Pennes thermal dose, cavitation guardrails,
mechanochemical ion-channel gating, calcium accumulation, membrane-potential
response, and a closed-loop parameter guidance map.

This chapter is a simulation and research-planning specification.  It is not a
clinical treatment protocol.

---

## 25.2 Formal contract

Inputs:

- Patient or template head model with skull transmission estimate.
- Target coordinate and focal dimensions.
- Carrier frequency `f0`, peak rarefactional pressure `p-`, duty cycle, pulse
  repetition frequency, and sonication duration.
- Safety guardrails for MI, derated `I_SPTA`, temperature rise, CEM43, and
  unintended cavitation/BBB-opening risk.

Outputs:

- In-situ pressure, intensity, MI, and focal full-width at half maximum.
- Thermal trajectory `T(t)` and CEM43 dose.
- Mechanochemical drive from acoustic energy density to membrane tension.
- Channel open probabilities for endogenous mechanosensitive channels.
- Calcium and neural-response proxy time series.
- Feasible parameter region satisfying the safety inequalities.

Acceptance criteria:

$$
\mathrm{MI} = \frac{p^-_{\mathrm{MPa}}}{\sqrt{f_0[\mathrm{MHz}]}} \le 1.9
$$

$$
I_{\mathrm{SPTA}} = DC \frac{(p^-)^2}{2\rho c} \le 0.72\;\mathrm{W/cm^2}
$$

$$
\Delta T_{\max} < 2^\circ\mathrm{C}, \qquad
\mathrm{CEM43} < 0.25\;\mathrm{min}, \qquad
P_{\mathrm{cav}}(\mathrm{MI}) < 0.1.
$$

Reject a candidate protocol when any inequality fails, when skull attenuation is
not modeled, when sham/auditory masking is absent from a human-effect claim, or
when a claimed clinical endpoint is inferred from a mechanism-only simulation.

---

## 25.3 Acoustic exposure model

For the simulation chapter, the in-situ focus is represented as an ellipsoidal
Gaussian pressure envelope:

$$
p(x,y,z)=p_0
\exp\left[-\frac{1}{2}\left(
\frac{x^2+y^2}{\sigma_\perp^2}+\frac{z^2}{\sigma_z^2}
\right)\right].
$$

The defaults match the human neuromodulation scale reported in the literature:
`f0 = 500 kHz`, lateral FWHM `5 mm`, axial FWHM `30 mm`,
`p0 = 300 kPa`, `DC = 5%`, and `30 s` sonication.  The pressure and intensity
fields are in-situ values after skull transmission; free-field values must be
back-computed from a skull model rather than substituted.

Human neuromodulation studies commonly use approximately `0.25-0.65 MHz`,
`0.1-1.0 MPa`, pulsed delivery, and subject-specific skull modeling.  FDA
diagnostic limits (`MI <= 1.9`, `I_SPTA.3 <= 720 mW/cm^2`) are useful
benchmarks, but transcranial adult-skull heating requires additional modeling.

---

## 25.4 Mechanochemical coupling

The acoustic energy density is:

$$
E_a = \frac{p_{\mathrm{rms}}^2}{\rho c^2}.
$$

Approximating the cell membrane as a thin spherical shell with radius `R`,
the effective ultrasound-induced tension increment is:

$$
\Delta \gamma = \frac{E_a R}{2}.
$$

For channel `i`, open probability follows a two-state Boltzmann law:

$$
P_i(\Delta\gamma)=
\left[
1+\exp\left(-\frac{\Delta\gamma-\gamma_{1/2,i}}{s_i}\right)
\right]^{-1}.
$$

The channel drive combines calcium-permeable excitation and K2P leak-current
inhibition:

$$
u(t)=
\frac{\sum_i w_i P_i(\Delta\gamma(t))}
{\sum_i |w_i|}.
$$

The default channel set models the mechanisms reported for endogenous
ultrasound sensitivity:

| Channel family | Role in the chapter model | Mechanism |
|---|---:|---|
| TRPC1/TRPP2 cluster | Excitatory calcium entry | Mechanosensitive calcium accumulation |
| Piezo1 | Excitatory calcium entry | Membrane-tension-gated cation current |
| TREK/TRAAK K2P | Inhibitory leak current | Mechanosensitive potassium conductance |

The channel parameters are not patient-specific biomarkers.  They are a
mechanistic simulation layer used to rank protocols after acoustic and thermal
constraints have already been satisfied.

---

## 25.5 Chemical and neural response model

The chemical state is a calcium proxy:

$$
\frac{dC}{dt}=\frac{C_0 + G_C\max(u,0)-C}{\tau_C}.
$$

The membrane-potential proxy is:

$$
\frac{dV}{dt}=
\frac{V_0 + G_V C - G_K\max(-u,0)-V}{\tau_V}.
$$

The observable target-engagement probability is:

$$
P_{\mathrm{resp}}(t)=
\left[1+\exp\left(-\frac{V(t)-V_{50}}{s_V}\right)\right]^{-1}.
$$

The model encodes four mechanism constraints:

- Mechanical energy reaches the membrane before it becomes a chemical signal.
- Calcium accumulates over tens to hundreds of milliseconds, consistent with
  observed ultrasound-response latencies.
- Potassium leak can suppress excitability even when calcium channels activate.
- Thermal dose and cavitation risk remain separate rejection criteria, not
  explanatory shortcuts.

---

## 25.6 Safety model

Thermal dose uses the Pennes equation:

$$
\rho C_p \frac{dT}{dt}
=2\alpha I_{\mathrm{SPTA}}
-w_b\rho_b C_b(T-T_0).
$$

CEM43 follows the Chapter 15 convention:

$$
\mathrm{CEM43}=\int R^{43-T(t)}dt,
\qquad
R=\begin{cases}
0.5 & T>43^\circ\mathrm{C}\\
0.25 & T\le 43^\circ\mathrm{C}.
\end{cases}
$$

Cavitation risk is modeled as a steep guardrail around `MI = 0.7`, below the
diagnostic MI limit but above common neuromodulation operation.  The rejection
criterion is intentionally conservative because microbubble-free
neuromodulation should not depend on cavitation or BBB-opening mechanisms.

---

## 25.7 Clinical evidence guidance

Clinical evidence remains heterogeneous.  As of 2026-05-12, low-intensity
transcranial ultrasound neuromodulation should be treated as investigational in
the United States outside cleared or IDE-governed research contexts.  TPS has
jurisdiction-specific authorization for Alzheimer disease in Europe, while
neuronavigated tFUS systems are primarily research devices.

| Evidence class | Example | Guidance use |
|---|---|---|
| Controlled human physiology | S1 stimulation modulated sensory evoked EEG and discrimination performance | Target-engagement and sham-design reference |
| Human systematic review | 35 human studies, 677 participants through 2022 | Safety/event-rate and heterogeneity reference |
| Mechanism experiment | Focused ultrasound excited murine cortical neurons through mechanosensitive calcium channels | Mechanochemical model support |
| Early clinical population study | Chronic pain, dementia, epilepsy, depression, PD, and stroke studies | Hypothesis generation only unless randomized evidence exists |
| 2026 PD proof-of-concept | Four male PD participants, pallidal 130-Hz TUS, STN beta reduction and reaction-time improvement | Biomarker-guided trial-design reference, not efficacy proof |

Clinical planning must include sham control, auditory masking, MRI/CT
targeting, in-situ acoustic simulation, post-sonication adverse-event capture,
and imaging/neurocognitive follow-up when the protocol enters human research.

---

## 25.8 Simulation workflow

```bash
python pykwavers/examples/book/ch26_neuromodulation.py
```

Outputs to `docs/book/figures/ch26/`:

| Figure | Content |
|---|---|
| fig01 | In-situ focal pressure and half-maximum contour |
| fig02 | Calcium, voltage, and response probability for 150/300/500 kPa pulses |
| fig03 | Mechanochemical channel activation versus pressure |
| fig04 | Pennes temperature rise and CEM43 across pressure and duty cycle |
| fig05 | Clinical-study guidance space by frequency and MI |
| fig06 | Closed-loop parameter map constrained by MI, `I_SPTA`, heating, and cavitation |

The script also writes `metrics.json` containing default protocol parameters,
peak MI, peak `I_SPTA`, default thermal dose, and the best feasible point in
the guidance grid.

---

## 25.9 References

- Legon W. et al. *Transcranial focused ultrasound modulates the activity of
  primary somatosensory cortex in humans.* Nature Neuroscience **17**,
  322-329, 2014. doi:10.1038/nn.3620
- Sarica C. et al. *Human studies of transcranial ultrasound
  neuromodulation: a systematic review of effectiveness and safety.* Brain
  Stimulation **15**(3), 737-746, 2022. doi:10.1016/j.brs.2022.05.002
- Pasquinelli C. et al. *Safety of transcranial focused ultrasound
  stimulation: a systematic review of the state of knowledge from both human
  and animal studies.* Brain Stimulation **12**(6), 1367-1380, 2019.
  doi:10.1016/j.brs.2019.07.024
- Yoo S. et al. *Focused ultrasound excites cortical neurons via
  mechanosensitive calcium accumulation and ion channel amplification.*
  Nature Communications **13**, 493, 2022. doi:10.1038/s41467-022-28040-1
- Legon W., Strohman A. *Low-intensity focused ultrasound for human
  neuromodulation.* Nature Reviews Methods Primers **4**, 91, 2024.
  doi:10.1038/s43586-024-00368-6
- Martin E. et al. *ITRUSST Consensus on Standardised Reporting for
  Transcranial Ultrasound Stimulation.* arXiv:2402.10027, 2024.
  doi:10.48550/arXiv.2402.10027
- Beisteiner R. et al. *Transcranial Pulse Stimulation with Ultrasound in
  Alzheimer's Disease-A New Navigated Focal Brain Therapy.* Advanced Science
  **7**(3), 1902583, 2019. doi:10.1002/advs.201902583
- Eraifej J. et al. *Suppression of pathological oscillations with
  transcranial focused ultrasound in Parkinson's disease.* Nature
  Communications, early-access version, 2026.
- FDA. *Marketing Clearance of Diagnostic Ultrasound Systems and Transducers:
  Guidance for Industry and Food and Drug Administration Staff.* Acoustic
  output limits: `I_SPTA.3 <= 720 mW/cm^2`, `MI <= 1.9`, or
  `I_SPPA.3 <= 190 W/cm^2`.
