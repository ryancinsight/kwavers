//! Histotripsy sonication planning: pulse-train timing, delivery losses, and
//! lesion-growth inversion for a rastered/interleaved sub-spot grid.
//!
//! These are the deterministic treatment-planning quantities behind the clinical
//! "Performing a Sonication" pipeline:
//! * [`build_sonication_schedule`] — the exact pulse timeline (pulse duration,
//!   repetition time, sonication duration) for sequential or interleaved firing
//!   of a sub-spot grid; the single source of truth for the pulsing-pattern
//!   diagram and the monitor time-base.
//! * [`forward_delivery_fraction`] / [`received_signal_fraction`] — the genuine
//!   reflection (impedance), tissue power-law attenuation, and residual-gas
//!   (Commander–Prosperetti) scattering losses along the path, one-way for the
//!   delivered drive and two-way for the passively measured emission.
//! * [`histotripsy_pulses_for_lesion_radius`] — inverse of the cavitation
//!   energy-balance lesion model: the pulse count needed to grow a lesion to a
//!   target radius, used both to size the dose for full tumour coverage and to
//!   cap it so the expanding lesion stays a safe margin from a sensitive
//!   structure (the per-spot pulse-duration compensation).

use crate::acoustics::bubble_dynamics::commander_prosperetti_attenuation;

/// Sub-spot grid firing order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SonicationOrder {
    /// All pulses at sub-spot 0, then all at sub-spot 1, … (one spot at a time).
    Sequential,
    /// Round-robin: one pulse at each sub-spot per repetition, then repeat — the
    /// rastered/interleaved pattern that lets each spot rest between its pulses.
    Interleaved,
}

/// Deterministic pulse timeline for a sub-spot grid sonication.
///
/// Index `k` of the flat arrays is the `k`-th fired pulse in time order.
#[derive(Debug, Clone)]
pub struct SonicationSchedule {
    /// Onset time of each fired pulse [s].
    pub onset_s: Vec<f64>,
    /// Sub-spot index fired at each pulse.
    pub subspot: Vec<usize>,
    /// Repetition index (pass number over the grid) of each pulse.
    pub repetition: Vec<usize>,
    /// Single-pulse duration [s] (the microsecond histotripsy pulse).
    pub pulse_duration_s: f64,
    /// Repetition time [s]: start of one grid pass to the next (interleaved) or
    /// the per-spot dwell `n_repetitions/PRF` (sequential).
    pub repetition_time_s: f64,
    /// Total sonication duration [s]: last pulse onset + one pulse duration.
    pub sonication_duration_s: f64,
    /// Number of repetitions (passes over the grid).
    pub n_repetitions: usize,
    /// Number of sub-spots.
    pub n_subspots: usize,
}

/// Build the pulse timeline for `n_subspots` fired over `n_repetitions` passes at
/// pulse repetition frequency `prf_hz` (the rate of *fired* pulses, any spot),
/// each pulse lasting `pulse_duration_s`.
///
/// * `Interleaved` — pulse `k = r·n + s` (repetition `r`, sub-spot `s`) fires at
///   `k/PRF`; each spot fires once per repetition, so its effective per-spot PRF
///   is `PRF/n` and the repetition time is `n/PRF`.
/// * `Sequential` — pulse `k = s·N + r` fires at `k/PRF`; a spot receives all `N`
///   pulses back-to-back (`1/PRF` apart) before the next spot.
///
/// Returns an empty schedule for non-physical inputs.
#[must_use]
pub fn build_sonication_schedule(
    n_subspots: usize,
    n_repetitions: usize,
    pulse_duration_s: f64,
    prf_hz: f64,
    order: SonicationOrder,
) -> SonicationSchedule {
    let n = n_subspots;
    let n_rep = n_repetitions;
    let total = n.saturating_mul(n_rep);
    if total == 0
        || !prf_hz.is_finite()
        || prf_hz <= 0.0
        || !pulse_duration_s.is_finite()
        || pulse_duration_s < 0.0
    {
        return SonicationSchedule {
            onset_s: Vec::new(),
            subspot: Vec::new(),
            repetition: Vec::new(),
            pulse_duration_s: pulse_duration_s.max(0.0),
            repetition_time_s: 0.0,
            sonication_duration_s: 0.0,
            n_repetitions: n_rep,
            n_subspots: n,
        };
    }
    let inter_pulse = 1.0 / prf_hz;
    let mut onset_s = Vec::with_capacity(total);
    let mut subspot = Vec::with_capacity(total);
    let mut repetition = Vec::with_capacity(total);

    match order {
        SonicationOrder::Interleaved => {
            for r in 0..n_rep {
                for s in 0..n {
                    let k = r * n + s;
                    onset_s.push(k as f64 * inter_pulse);
                    subspot.push(s);
                    repetition.push(r);
                }
            }
        }
        SonicationOrder::Sequential => {
            let mut k = 0usize;
            for s in 0..n {
                for r in 0..n_rep {
                    onset_s.push(k as f64 * inter_pulse);
                    subspot.push(s);
                    repetition.push(r);
                    k += 1;
                }
            }
        }
    }

    // Repetition time: interleaved = one grid pass (n pulses); sequential = the
    // per-spot dwell (n_rep pulses) — both expressed in real time.
    let repetition_time_s = match order {
        SonicationOrder::Interleaved => n as f64 * inter_pulse,
        SonicationOrder::Sequential => n_rep as f64 * inter_pulse,
    };
    let last_onset = onset_s.last().copied().unwrap_or(0.0);
    let sonication_duration_s = last_onset + pulse_duration_s;

    SonicationSchedule {
        onset_s,
        subspot,
        repetition,
        pulse_duration_s,
        repetition_time_s,
        sonication_duration_s,
        n_repetitions: n_rep,
        n_subspots: n,
    }
}

/// Normal-incidence pressure transmission coefficient between two media of
/// specific acoustic impedance `z1` (proximal) and `z2` (distal):
/// `T = 2·z2 / (z1 + z2)`.
#[must_use]
#[inline]
pub fn pressure_transmission_coefficient(z1: f64, z2: f64) -> f64 {
    let denom = z1 + z2;
    if denom > 0.0 {
        2.0 * z2 / denom
    } else {
        1.0
    }
}

/// One-way delivered-pressure fraction at the focus, combining
/// * electronic-steering efficiency `steering_eff` (already computed),
/// * a representative interface pressure transmission (reflection loss),
/// * tissue power-law attenuation `exp(−α_tissue·L)`, and
/// * residual-gas (Commander–Prosperetti) attenuation `exp(−α_gas·L)` from the
///   void fraction `void_beta` left by previous pulses.
///
/// `path_len_m` is the proximal→focus path; the gas attenuation uses the same
/// path as a conservative upper bound on the cloud extent.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn forward_delivery_fraction(
    steering_eff: f64,
    interface_z_prox: f64,
    interface_z_focal: f64,
    alpha_tissue_np_m: f64,
    path_len_m: f64,
    void_beta: f64,
    freq_hz: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> f64 {
    let l = path_len_m.max(0.0);
    let t_iface = pressure_transmission_coefficient(interface_z_prox, interface_z_focal);
    let alpha_gas = commander_prosperetti_attenuation(
        freq_hz, void_beta, r0_m, c_liquid, rho_liquid, mu_liquid, p0_pa, polytropic,
    );
    let atten = (-(alpha_tissue_np_m.max(0.0) + alpha_gas) * l).exp();
    (steering_eff.clamp(0.0, 1.0) * t_iface * atten).max(0.0)
}

/// Two-way (round-trip) amplitude fraction of a passive cavitation emission
/// measured back at the transducer: the emission generated at the focus
/// propagates back through the same tissue + residual-gas attenuation and the
/// interface, so the amplitude scales as the **square** of the path attenuation
/// and interface transmission.
///
/// This is the genuine reflection/scattering/attenuation loss that derates the
/// *measured* cavitation signal relative to the cavitation actually produced.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn received_signal_fraction(
    interface_z_prox: f64,
    interface_z_focal: f64,
    alpha_tissue_np_m: f64,
    path_len_m: f64,
    void_beta: f64,
    freq_hz: f64,
    r0_m: f64,
    c_liquid: f64,
    rho_liquid: f64,
    mu_liquid: f64,
    p0_pa: f64,
    polytropic: f64,
) -> f64 {
    let l = path_len_m.max(0.0);
    let t_iface = pressure_transmission_coefficient(interface_z_prox, interface_z_focal);
    let alpha_gas = commander_prosperetti_attenuation(
        freq_hz, void_beta, r0_m, c_liquid, rho_liquid, mu_liquid, p0_pa, polytropic,
    );
    let one_way = (-(alpha_tissue_np_m.max(0.0) + alpha_gas) * l).exp();
    // Round trip: forward attenuation × return attenuation (reciprocity) and the
    // interface is traversed on the way back as well.
    (t_iface * t_iface * one_way * one_way).max(0.0)
}

/// Inverse of [`crate::acoustics`-style] cavitation lesion energy balance: the
/// pulse count `N` whose accumulated inertial cavitation dose grows a lesion to
/// radius `target_radius_m`.
///
/// The forward model (Maxwell 2011; Vlaisavljevich 2015) is
/// `R_L = R₀·(P₀·ICD_total/σ_y)^(1/3)` with `ICD_total = N·icd_per_pulse`.
/// Solving for `N`:
/// ```text
///   N = (R_L/R₀)³ · σ_y / (P₀ · icd_per_pulse)
/// ```
/// Returns `0.0` for a non-positive target or non-physical inputs (a lesion of
/// zero radius needs no pulses); the result is a real (non-integer) pulse count,
/// the caller rounds up for a discrete schedule.
#[must_use]
pub fn histotripsy_pulses_for_lesion_radius(
    target_radius_m: f64,
    r0_m: f64,
    p0_pa: f64,
    tissue_yield_stress_pa: f64,
    icd_per_pulse: f64,
) -> f64 {
    if !(target_radius_m.is_finite()
        && r0_m.is_finite()
        && p0_pa.is_finite()
        && tissue_yield_stress_pa.is_finite()
        && icd_per_pulse.is_finite()
        && target_radius_m > 0.0
        && r0_m > 0.0
        && p0_pa > 0.0
        && tissue_yield_stress_pa > 0.0
        && icd_per_pulse > 0.0)
    {
        return 0.0;
    }
    let ratio = target_radius_m / r0_m;
    ratio * ratio * ratio * tissue_yield_stress_pa / (p0_pa * icd_per_pulse)
}

/// Histotripsy mechanical cell-kill fraction from cumulative cavitation dose, via a
/// Weibull (multi-hit) survival dose–response:
/// ```text
///   kill = 1 − exp(−(dose / d0)^k)
/// ```
/// The kill mechanism is mechanical fractionation (cavitation liquefies tissue to an
/// acellular homogenate), not DNA damage — but the cumulative dose–response is the
/// same sigmoidal cell-survival form used in radiobiology (the iso-effect basis of
/// biologically-effective dose). `d0` is the characteristic dose (kill = 1−1/e ≈ 63 %)
/// and the Weibull exponent `k > 1` reproduces the threshold/shoulder of measured
/// histotripsy fractionation curves (Vlaisavljevich 2015; Maxwell 2013). Iso-lethal
/// levels LD25/LD50/LD75/LD100 are the inverse, [`histotripsy_lethal_dose`].
#[must_use]
#[inline]
pub fn histotripsy_kill_fraction(dose: f64, d0: f64, weibull_k: f64) -> f64 {
    if !(dose.is_finite() && d0 > 0.0 && weibull_k > 0.0) || dose <= 0.0 {
        return 0.0;
    }
    (1.0 - (-(dose / d0).powf(weibull_k)).exp()).clamp(0.0, 1.0)
}

/// Lethal cumulative cavitation dose for a target cell-kill `fraction` (the LD_x of the
/// Weibull dose–response): `D = d0·(−ln(1 − fraction))^(1/k)`. LD50 ⇒ fraction = 0.5,
/// LD100 is asymptotic so callers use e.g. 0.99. Inverse of [`histotripsy_kill_fraction`].
#[must_use]
#[inline]
pub fn histotripsy_lethal_dose(fraction: f64, d0: f64, weibull_k: f64) -> f64 {
    let f = fraction.clamp(0.0, 1.0 - 1e-9);
    if !(d0 > 0.0 && weibull_k > 0.0) || f <= 0.0 {
        return 0.0;
    }
    d0 * (-(1.0 - f).ln()).powf(1.0 / weibull_k)
}

/// Local peak-pressure enhancement at a planar acoustic interface between media of
/// specific impedance `z1` and `z2`, from superposition of the incident and reflected
/// waves: `E = 1 + |R|`, `R = (z2 − z1)/(z2 + z1)`.
///
/// Cavitation nucleates preferentially at interfaces because (a) the peak
/// rarefactional pressure is locally raised by up to a factor of two at a strong
/// reflector, and (b) interfaces trap stabilising gas nuclei. The enhancement is
/// `1` for impedance-matched media (no boundary), `≈1.05–1.1` for soft-tissue
/// boundaries (liver/fat/tumour), and approaches `2` against a gas void — which is
/// why a liquefied **lacuna** (a gas-filled cavity) is such a strong cavitation site.
///
/// The caller multiplies the *effective* focal peak pressure by this factor (with a
/// spatial proximity weight) so the existing supralinear emission-vs-pressure curve
/// produces the enhanced cavitation; the function itself is the exact reflection law.
#[must_use]
#[inline]
pub fn interface_pressure_enhancement(z1: f64, z2: f64) -> f64 {
    let denom = z1 + z2;
    if denom > 0.0 && z1.is_finite() && z2.is_finite() {
        1.0 + ((z2 - z1) / denom).abs()
    } else {
        1.0
    }
}

/// Cavitation-susceptibility multiplier of tissue that has already been (partly)
/// fractionated by previous histotripsy pulses — the genuine "lesion memory" that
/// makes re-insonation of a forming or pre-existing lesion cavitate more readily.
///
/// `fractionation ∈ [0, 1]` is the local lesion completeness; `time_since_lesion_s`
/// is the elapsed time since that tissue was first fractionated; `tau_lacuna_s` is
/// the gas-evolution time constant over which dissolved gas diffuses out of the
/// surrounding tissue into the liquefied void to form a **lacuna** (a macroscopic
/// gas/fluid cavity, essentially zero cavitation threshold).
///
/// ```text
///   S = 1 + k_immediate·f + k_lacuna·f·(1 − exp(−t_since/τ_lacuna))
/// ```
/// * `k_immediate·f` — the prompt threshold reduction from residual bubble nuclei in
///   freshly fractionated tissue (present within one procedure).
/// * `k_lacuna·f·(1 − e^{−t/τ})` — the *delayed* enhancement as the lacuna degasses;
///   negligible at `t ≪ τ` (not apparent during the first procedure) and saturating
///   to `k_lacuna·f` at `t ≫ τ` (full cavity formed — strong cavitation on
///   re-treatment days later). `τ_lacuna` follows gas-diffusion scaling
///   `a²/(2·D·ζ)` for a void of radius `a` (tens of seconds to minutes intra-op,
///   hours–days post-op).
#[must_use]
#[inline]
pub fn lacuna_cavitation_susceptibility(
    fractionation: f64,
    time_since_lesion_s: f64,
    tau_lacuna_s: f64,
    k_immediate: f64,
    k_lacuna: f64,
) -> f64 {
    let f = fractionation.clamp(0.0, 1.0);
    if f <= 0.0 || !f.is_finite() {
        return 1.0;
    }
    let t = time_since_lesion_s.max(0.0);
    let tau = tau_lacuna_s.max(f64::EPSILON);
    let lacuna = (1.0 - (-t / tau).exp()).clamp(0.0, 1.0);
    (1.0 + k_immediate.max(0.0) * f + k_lacuna.max(0.0) * f * lacuna).max(1.0)
}

/// Local lacuna gas void fraction at fractionated tissue: the persistent gas/fluid
/// cavity that forms as dissolved gas diffuses out of the surrounding tissue into a
/// liquefied lesion. First-order gas-evolution growth toward a fractionation-scaled
/// saturation void fraction:
/// ```text
///   β_lacuna = β_max · fractionation · (1 − exp(−t_since_lesion/τ_lacuna))
/// ```
/// Distinct from the fast residual *bubble-cloud* dissolution (which only shrinks β
/// between pulses); this is the slow void *growth* that, once formed, both shields
/// and aberrates subsequent pulses (a strong gas/tissue impedance interface) and is
/// why a re-treated lesion (`t_since ≫ τ`) cavitates very differently from a virgin
/// one. `τ_lacuna ~ a²/(2 D ζ)` for a void of radius `a` (tens of seconds to minutes
/// intra-procedure; hours–days post-procedure).
#[must_use]
#[inline]
pub fn lacuna_void_fraction(
    fractionation: f64,
    time_since_lesion_s: f64,
    tau_lacuna_s: f64,
    beta_max: f64,
) -> f64 {
    let f = fractionation.clamp(0.0, 1.0);
    if f <= 0.0 || !f.is_finite() || beta_max <= 0.0 {
        return 0.0;
    }
    let t = time_since_lesion_s.max(0.0);
    let tau = tau_lacuna_s.max(f64::EPSILON);
    (beta_max.max(0.0) * f * (1.0 - (-t / tau).exp())).clamp(0.0, 1.0 - 1e-9)
}

#[cfg(test)]
#[path = "sonication_tests.rs"]
mod tests;
