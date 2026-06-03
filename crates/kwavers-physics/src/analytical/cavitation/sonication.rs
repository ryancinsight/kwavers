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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_schedule_timing_matches_diagram() {
        // 3 sub-spots, 4 repetitions, 10 µs pulse, 5 Hz fired-pulse rate.
        let n = 3usize;
        let n_rep = 4usize;
        let pd = 10e-6;
        let prf = 5.0;
        let s = build_sonication_schedule(n, n_rep, pd, prf, SonicationOrder::Interleaved);
        assert_eq!(s.onset_s.len(), n * n_rep);
        // First repetition fires sub-spots 0,1,2 at 0, 0.2, 0.4 s.
        assert_eq!(&s.subspot[0..3], &[0, 1, 2]);
        assert!((s.onset_s[1] - 0.2).abs() < 1e-12);
        // Repetition time = one grid pass = n/PRF = 0.6 s.
        assert!((s.repetition_time_s - n as f64 / prf).abs() < 1e-12);
        // Sub-spot 0 fires once per repetition → effective per-spot interval = 0.6 s.
        let spot0: Vec<f64> = s
            .onset_s
            .iter()
            .zip(&s.subspot)
            .filter(|(_, &sp)| sp == 0)
            .map(|(&t, _)| t)
            .collect();
        assert_eq!(spot0.len(), n_rep);
        assert!((spot0[1] - spot0[0] - 0.6).abs() < 1e-12);
        // Sonication duration = last onset + pulse duration.
        let last = (n * n_rep - 1) as f64 / prf;
        assert!((s.sonication_duration_s - (last + pd)).abs() < 1e-12);
    }

    #[test]
    fn sequential_schedule_fires_one_spot_at_a_time() {
        let n = 3usize;
        let n_rep = 4usize;
        let s = build_sonication_schedule(n, n_rep, 10e-6, 5.0, SonicationOrder::Sequential);
        // First n_rep pulses are all sub-spot 0.
        assert!(s.subspot[0..n_rep].iter().all(|&sp| sp == 0));
        assert_eq!(s.subspot[n_rep], 1);
        // Sequential per-spot dwell = n_rep/PRF.
        assert!((s.repetition_time_s - n_rep as f64 / 5.0).abs() < 1e-12);
    }

    #[test]
    fn delivery_fraction_decreases_with_gas_and_attenuation() {
        // Air-in-water residual cloud at 0.5 MHz, 2 µm bubbles.
        let base = forward_delivery_fraction(
            1.0, 1.5e6, 1.5e6, 5.0, 0.05, 0.0, 0.5e6, 2e-6, 1481.0, 998.0, 1e-3, 101_325.0, 1.4,
        );
        let with_gas = forward_delivery_fraction(
            1.0, 1.5e6, 1.5e6, 5.0, 0.05, 1e-4, 0.5e6, 2e-6, 1481.0, 998.0, 1e-3, 101_325.0, 1.4,
        );
        assert!(base > 0.0 && base <= 1.0);
        assert!(
            with_gas < base,
            "residual gas must reduce delivered pressure: {with_gas} < {base}"
        );
    }

    #[test]
    fn received_fraction_is_two_way_loss() {
        // With no gas and matched impedance, the received fraction is exp(−2αL).
        let alpha = 5.0;
        let l = 0.05;
        let recv = received_signal_fraction(
            1.5e6, 1.5e6, alpha, l, 0.0, 0.5e6, 2e-6, 1481.0, 998.0, 1e-3, 101_325.0, 1.4,
        );
        let expected = (-2.0 * alpha * l).exp(); // T_iface = 1 at matched impedance
        assert!(
            (recv - expected).abs() < 1e-12,
            "two-way: {recv} vs {expected}"
        );
    }

    #[test]
    fn interface_pressure_transmission_physics() {
        // Pressure transmission T = 2z2/(z1+z2): into a HIGHER impedance the
        // pressure amplitude rises (T>1, intensity still conserved); into a LOWER
        // impedance it drops (T<1). Matched impedance transmits unchanged.
        let into_higher = pressure_transmission_coefficient(1.38e6, 1.65e6); // fat→liver
        let into_lower = pressure_transmission_coefficient(1.65e6, 1.38e6); // liver→fat
        assert!(
            into_higher > 1.0,
            "into higher Z, pressure rises: {into_higher}"
        );
        assert!(
            into_lower < 1.0 && into_lower > 0.0,
            "into lower Z, pressure drops: {into_lower}"
        );
        assert!((pressure_transmission_coefficient(1.5e6, 1.5e6) - 1.0).abs() < 1e-12);
        // Strong mismatch toward a much lower impedance (gas) collapses transmission.
        let into_gas = pressure_transmission_coefficient(1.5e6, 4.1e2);
        assert!(
            into_gas < 0.01,
            "huge impedance drop nearly blocks transmission: {into_gas}"
        );
    }

    #[test]
    fn pulses_for_lesion_radius_inverts_forward_model() {
        use crate::analytical::cavitation::histotripsy_lesion_radius_m;
        let r0 = 3e-6;
        let p0 = 101_325.0;
        let sigma_y = 2.0e3;
        let icd_per_pulse = 50.0;
        let target = 1.0e-3; // 1 mm lesion
        let n = histotripsy_pulses_for_lesion_radius(target, r0, p0, sigma_y, icd_per_pulse);
        assert!(n > 0.0);
        // Round-trip: feeding N·icd_per_pulse back into the forward model recovers
        // the target radius.
        let r_back = histotripsy_lesion_radius_m(n * icd_per_pulse, r0, p0, sigma_y);
        assert!(
            (r_back - target).abs() / target < 1e-9,
            "inverse must recover target radius: {r_back} vs {target}"
        );
        // A tighter safety margin (smaller target) needs fewer pulses.
        let n_tight = histotripsy_pulses_for_lesion_radius(0.5e-3, r0, p0, sigma_y, icd_per_pulse);
        assert!(n_tight < n);
    }
}
