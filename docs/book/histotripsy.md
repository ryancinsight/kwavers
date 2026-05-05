# Chapter 21: Histotripsy — Classical vs Millisecond-Pulse Regimes

> **Module ownership.** Histotripsy modeling in kwavers is provided by
> `kwavers::clinical::therapy::modalities` (modality definitions, exposure
> parameters), `kwavers::physics::acoustics::bubble_dynamics` (Rayleigh–Plesset
> and Keller–Miksis), `kwavers::physics::acoustics::therapy::cavitation`
> (intrinsic-threshold detection), `kwavers::clinical::therapy::lithotripsy`
> (shock-wave propagation), and the bioheat solver under
> `kwavers::physics::thermodynamics::pennes`. The example figures in this
> chapter are produced by `pykwavers/examples/book/ch21_histotripsy_comparison.py`.

This chapter defines the two principal histotripsy exposure regimes used
clinically and compares them under matched fundamental frequency and matched
acoustic-energy targets:

1. **Classical (intrinsic-threshold) histotripsy.** Short pulses,
   typically 1–20 cycles ($\tau_p \in [0.5, 20]\,\mu\mathrm{s}$ at $f_0 = 1$ MHz),
   peak negative pressure (PNP) at the focus exceeding the intrinsic
   nucleation threshold $p^{-}_t \approx -24$ to $-30$ MPa for water-rich
   tissue. Cavitation cloud forms within a single negative half-cycle from
   pre-existing nuclei [Maxwell 2013, Vlaisavljevich 2015].
2. **Millisecond-pulse (boiling) histotripsy.** Long pulses,
   $\tau_p \sim 1$–$20$ ms with shock-formed waveforms (in-situ shock amplitude
   $\sigma$ such that nonlinear distortion produces a vapor bubble at the
   focus by the millisecond timescale). Mechanism is shock-scattering off
   the boiling vapor bubble seeded by absorption-driven heating
   [Khokhlova 2014, Khokhlova 2017].

Both regimes produce a sub-cellular fractionated lesion with sharp boundaries,
but their thermal footprint, cavitation onset criterion, and required source
parameters differ qualitatively. This chapter formalizes those differences.

---

## 21.1 Intrinsic Threshold Theorem (Classical Regime)

### Theorem 21.1 (Single-cycle nucleation probability)

Let $p^-(t)$ be the negative pressure waveform at a focal voxel and let
$p^-_{\min}$ be its minimum (most negative) value during a single pulse of
duration $\tau_p < T_\text{thermal}$, where $T_\text{thermal}$ is the
characteristic time over which heat conduction would alter local nucleation
statistics. Define the intrinsic-threshold cumulative distribution

$$
P_\text{cav}(p^-_{\min}) = \frac{1}{2}\left[1 + \mathrm{erf}\!\left(
\frac{|p^-_{\min}| - p_t}{\sigma_t \sqrt{2}}\right)\right],
$$

with parameters $(p_t, \sigma_t) = (28.2, 0.96)\,\mathrm{MPa}$ in
ex-vivo bovine liver at 1 MHz [Maxwell 2013, Table II]. Then

$$
P_\text{cav}(p^-_{\min}) \in [0,1]
\quad\text{is monotone non-decreasing in } |p^-_{\min}|.
$$

### Proof

The error function $\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-s^2}\,ds$
satisfies $\mathrm{erf}'(x) = \frac{2}{\sqrt{\pi}} e^{-x^2} > 0$ for all $x$,
hence $\mathrm{erf}$ is strictly increasing. The argument
$x(p^-_{\min}) = (|p^-_{\min}| - p_t) / (\sigma_t \sqrt{2})$ is linear and
non-decreasing in $|p^-_{\min}|$. Composition preserves monotonicity, and the
affine map $\frac{1}{2}[1 + \cdot]$ maps $[-1, 1]$ onto $[0, 1]$, so
$P_\text{cav} \in [0, 1]$ and is monotone non-decreasing. $\blacksquare$

**Implication.** Within the classical regime the cavitation event depends
only on the *amplitude* of the most negative pressure cycle, not on the
pulse duration $\tau_p$ or the cycle count $N_c = \tau_p f_0$ above the
threshold of one cycle. This decouples the mechanical mechanism from
thermal accumulation.

---

## 21.2 Shock-Scattering Theorem (Millisecond Regime)

### Theorem 21.2 (Boiling-onset time at a shock-formed focus)

Let the focal pressure waveform contain a fully-developed shock with peak
positive pressure $p^+_s$ and absorption coefficient at the shock spectrum
of order $\alpha_s$. Approximating the absorbed power density as
$Q_s = 2 \alpha_s I_s$ where $I_s$ is the cycle-averaged intensity in the
shock-rich part of the pulse, and using the bioheat balance

$$
\rho_0 c_p \frac{\partial T}{\partial t} = Q_s - K \nabla^2 T,
$$

then the time to reach $T_\text{boil} = 100^\circ\mathrm{C}$ from
$T_0 = 37^\circ\mathrm{C}$ at the focal point, in the limit
$\nabla^2 T \to 0$ (focal heating dominant), is

$$
t_\text{boil} = \frac{\rho_0 c_p (T_\text{boil} - T_0)}{Q_s}
= \frac{\rho_0 c_p \, \Delta T}{2 \alpha_s I_s}.
$$

### Proof

Setting $\nabla^2 T = 0$ (valid on the timescale where conductive losses are
small compared with absorbed deposition: $\tau \ll \rho_0 c_p w_f^2 / K$ where
$w_f$ is the focal radius) reduces the bioheat equation to the linear ODE
$\rho_0 c_p \, \dot T = Q_s$ with constant $Q_s$ in the shock-formed regime.
The solution is $T(t) = T_0 + (Q_s / \rho_0 c_p)\,t$. Solving for $T(t) =
T_\text{boil}$ yields the stated $t_\text{boil}$. $\blacksquare$

**Implication.** The vapor bubble that mediates shock-scattering nucleation
appears at $t_\text{boil}$, which is a property of $(I_s, \alpha_s)$ at the
focus. For 1-MHz histotripsy with $I_s \approx 25\,\mathrm{kW/cm}^2$ and
$\alpha_s \approx 100\,\mathrm{Np/m}$ (shock spectrum, liver), $t_\text{boil}
\approx 5$ ms. This *requires* $\tau_p \gtrsim t_\text{boil}$, hence the
millisecond-pulse regime.

---

## 21.3 Thermal Dose Scaling

### Theorem 21.3 (CEM43 ratio between regimes)

Let $\tau_p^{(C)}$ and $\tau_p^{(M)}$ be the pulse durations of the classical
and millisecond regimes, and let $\mathrm{PRF}^{(C)}$, $\mathrm{PRF}^{(M)}$
be the respective pulse-repetition frequencies. Define the duty cycle
$D = \tau_p\,\mathrm{PRF}$. Assume both regimes deliver the same in-situ
spatial-peak pulse-average intensity $I_\text{spta}$ history at the focus
during the on-time and matching focal volume. Then the ratio of thermal
doses satisfies

$$
\frac{\mathrm{CEM43}^{(M)}}{\mathrm{CEM43}^{(C)}}
= \frac{D^{(M)}}{D^{(C)}}\cdot
  \frac{\int R^{43 - T^{(M)}(t)}\,dt}{\int R^{43 - T^{(C)}(t)}\,dt},
$$

with $R = 0.5$ for $T \geq 43^\circ\mathrm{C}$ and $R = 0.25$ otherwise
[Sapareto-Dewey 1984].

### Proof

CEM43 is defined as $\int R^{43 - T(t)}\,dt$ over the duration of exposure.
Splitting the integral over on-time and off-time, the off-time contribution
between pulses is bounded above by an exponentially decaying transient governed
by thermal conduction; for $\tau_\text{off} \gg \tau_\text{thermal} \approx
\rho_0 c_p w_f^2 / K$ this contribution is negligible. The remaining
on-time integrals scale linearly with $D$ and are weighted by the regime-
specific peak temperatures, giving the stated ratio. $\blacksquare$

**Numerical example.** For $f_0 = 1$ MHz, focal $w_f = 0.5$ mm, liver
($\rho_0 c_p \approx 3.8 \times 10^6\,\mathrm{J/m^3/K}$, $K = 0.51\,\mathrm{W/m/K}$),
$\tau_\text{thermal} \approx 2$ ms. Classical regime: $\tau_p^{(C)} = 5\,\mu$s,
PRF $= 200$ Hz, $D^{(C)} = 10^{-3}$, focal $\Delta T \lesssim 1$ K per pulse.
Millisecond regime: $\tau_p^{(M)} = 10$ ms, PRF $= 1$ Hz, $D^{(M)} = 10^{-2}$,
focal $\Delta T \approx 70$–$100$ K *during the pulse*. CEM43 ratio is dominated
by the temperature term and is several orders of magnitude greater for the
millisecond regime, even though both produce purely mechanical lesions in
sub-cellular morphology.

---

## 21.4 Side-by-Side Comparison

The figures generated by `ch21_histotripsy_comparison.py` provide visual
discrepancy assessment across five quantitative axes.

### Figure 21.1 — Pulse waveforms

`figures/ch21/fig01_pulse_waveforms.{png,pdf}`

Side-by-side rendering of the focal pressure waveform $p(t)$ over one full
cycle for each regime. The classical regime shows a 5-cycle tone-burst with
PNP $\approx -28$ MPa; the millisecond regime shows a 10000-cycle envelope
with shock-formed positive peaks.

### Figure 21.2 — Cavitation probability

`figures/ch21/fig02_cavitation_probability.{png,pdf}`

$P_\text{cav}(|p^-_{\min}|)$ from Theorem 21.1 plotted against the scanned PNP,
with classical and millisecond operating points overlaid. The classical regime
operates on the steep slope of the erf transition (single-shot nucleation); the
millisecond regime operates *below* the intrinsic threshold and relies on
shock-scattering from the boiling bubble.

### Figure 21.3 — Bioheat temperature rise

`figures/ch21/fig03_bioheat_temperature.{png,pdf}`

Focal $T(t)$ from the linearized Theorem 21.2 ODE for both regimes over a
single pulse. The classical pulse ends before measurable temperature rise;
the millisecond pulse crosses $T_\text{boil}$ at $t \approx 5$ ms.

### Figure 21.4 — CEM43 accumulation

`figures/ch21/fig04_cem43_accumulation.{png,pdf}`

CEM43 per single pulse computed via Sapareto-Dewey on the bioheat
trajectories of Fig. 21.3. The classical regime stays at CEM43 $\ll 1\,$min;
the millisecond regime exceeds CEM43 $= 240\,$min within a fraction of a
single pulse, despite the lesion mechanism being mechanical.

### Figure 21.5 — Mechanism phase map

`figures/ch21/fig05_mechanism_phase_map.{png,pdf}`

Two-dimensional phase map in $(\tau_p, |p^-_{\min}|)$ coordinates with three
operating regions delineated:

- **Intrinsic-threshold (classical):** $|p^-_{\min}| > p_t$ and $\tau_p$ in the
  microsecond band.
- **Shock-scattering (millisecond / boiling):** $|p^-_{\min}| < p_t$ but
  $\tau_p > t_\text{boil}$ with shock-formed waveform.
- **Thermal ablation (HIFU):** $|p^-_{\min}|$ well below threshold and
  $\tau_p \gg t_\text{boil}$ with no inertial cavitation regime entered.

The classical and millisecond histotripsy operating points are marked.

---

## 21.5 Algorithm: Reproducing the Comparison

```text
Algorithm 21.1 (Histotripsy regime comparison)
INPUT:
  f0          : carrier frequency                 [Hz]
  rho0, c_p   : density, specific heat            [kg/m^3], [J/kg/K]
  alpha_s     : in-situ absorption (shock band)   [Np/m]
  I_s         : focal cycle-averaged intensity    [W/m^2]
  pnp_C, pnp_M: peak negative pressures (C,M)     [Pa]
  tau_C, tau_M: pulse durations (C,M)             [s]
  prf_C, prf_M: pulse-repetition frequencies      [Hz]

STEP 1: compute Q_s = 2 * alpha_s * I_s
STEP 2: compute t_boil = rho0 * c_p * (T_boil - T0) / Q_s
STEP 3: simulate p(t) for both regimes (carrier + envelope, shock model)
STEP 4: P_cav(C), P_cav(M) via erf-CDF (Theorem 21.1)
STEP 5: T(t) for both regimes via linear bioheat (Theorem 21.2)
STEP 6: CEM43 by Sapareto-Dewey integration of T(t)
STEP 7: phase map in (tau_p, |pnp|) overlaying p_t and t_boil contours

OUTPUT: five figures saved as PNG and PDF.
```

The implementation lives in `pykwavers/examples/book/ch21_histotripsy_comparison.py`
and is wired into `generate_all_figures.py` as Chapter 21.

---

## 21.6 Cross-References

- **Chapter 7 (Cavitation and Bubbles):** Rayleigh–Plesset / Keller–Miksis
  derivations underlying the inertial collapse model.
- **Chapter 6 (Therapeutic Ultrasound):** bioheat equation and CEM43.
- **Chapter 12 (Media and Tissue Models):** absorption $\alpha(\omega)$ and
  shock-spectrum coupling.
- **Chapter 15 (Safety and Dosimetry):** MI, TI, and the regulatory
  context for high-pressure short-duration exposures.

---

## 21.7 References

1. Maxwell A.D., Cain C.A., Hall T.L., Fowlkes J.B., Xu Z. (2013).
   "Probability of cavitation for single ultrasound pulses applied to
   tissues and tissue-mimicking materials."
   *Ultrasound in Medicine and Biology*, 39(3), 449–465.
2. Vlaisavljevich E., Lin K.W., Maxwell A., Warnez M.T., Mancia L.,
   Singh R., Putnam A.J., Fowlkes J.B., Johnsen E., Cain C., Xu Z. (2015).
   "Effects of ultrasound frequency and tissue stiffness on the histotripsy
   intrinsic threshold for cavitation."
   *Ultrasound in Medicine and Biology*, 41(6), 1651–1667.
3. Khokhlova T.D., Wang Y.N., Simon J.C., Cunitz B.W., Starr F., Paun M.,
   Crum L.A., Bailey M.R., Khokhlova V.A. (2014). "Ultrasound-guided
   tissue fractionation by high-intensity focused ultrasound in an in vivo
   porcine liver model." *PNAS*, 111(22), 8161–8166.
4. Khokhlova V.A., Fowlkes J.B., Roberts W.W., Schade G.R., Xu Z., Khokhlova
   T.D., Hall T.L., Maxwell A.D., Wang Y.N., Cain C.A. (2015).
   "Histotripsy methods in mechanical disintegration of tissue: towards
   clinical applications." *International Journal of Hyperthermia*, 31(2),
   145–162.
5. Sapareto S.A., Dewey W.C. (1984). "Thermal dose determination in cancer
   therapy." *Int. J. Radiation Oncology, Biology, Physics*, 10(6), 787–800.
6. Khokhlova T.D., Schade G.R., Wang Y.N., Buravkov S.V., Chernikov V.P.,
   Simon J.C., Starr F., Maxwell A.D., Bailey M.R., Kreider W., Khokhlova
   V.A. (2019). "Pilot in vivo studies on transcutaneous boiling histotripsy
   in porcine liver and kidney." *Scientific Reports*, 9, 20176.
7. Pennes H.H. (1948). "Analysis of tissue and arterial blood temperatures
   in the resting human forearm." *J. Appl. Physiol.*, 1(2), 93–122.
