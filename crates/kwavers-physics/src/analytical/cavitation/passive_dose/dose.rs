//! Time-integrated stable / inertial cavitation dose and the closed-loop
//! pressure controller that drives clinical harmonic-dose monitoring.
//!
//! A passive-cavitation-dose controller integrates the band-resolved emission
//! power over the sonication and uses the two accumulated doses to steer drive
//! pressure: grow the *stable* cavitation dose (therapeutic, reversible) while
//! keeping the *inertial* cavitation dose below a damage limit
//! (McDannold 2006; O'Reilly & Hynynen 2012; Arvanitis 2012).

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Cumulative cavitation dose: trapezoidal time-integral of an emission-power
/// series.
///
/// ```text
///   D[m] = Σ_{i=1..m} ½·(P[i−1] + P[i])·Δt        [emission-power·s]
/// ```
/// `power_arr[k]` is the band emission power measured in the k-th monitoring
/// window (e.g. the stable emission `sub + ultra`, or the broadband emission)
/// and `dt_s` is the window duration. The returned array is the running dose,
/// the same length as `power_arr`, with `D[0] = 0` (no elapsed time yet).
///
/// Negative power samples are clamped to 0 — emission energy cannot reduce an
/// accumulated dose.
///
/// # Reference
/// O'Reilly M.A. & Hynynen K. (2012) *Radiology* 263, 96 (real-time dose control).
#[must_use]
pub fn cumulative_cavitation_dose(power_arr: &[f64], dt_s: f64) -> Vec<f64> {
    let n = power_arr.len();
    let mut out = vec![0.0_f64; n];
    if n < 2 || !(dt_s.is_finite() && dt_s > 0.0) {
        return out;
    }
    let mut acc = 0.0_f64;
    let mut prev = power_arr[0].max(0.0);
    for i in 1..n {
        let cur = power_arr[i].max(0.0);
        acc += 0.5 * (prev + cur) * dt_s;
        out[i] = acc;
        prev = cur;
    }
    out
}

/// Rust-owned dose traces for the Chapter 23 passive-cavitation dose panel.
#[derive(Clone, Debug, PartialEq)]
pub struct PassiveCavitationDoseFixture {
    /// Treatment-time samples [s].
    pub time_s: Vec<f64>,
    /// Normalized stable-cavitation cumulative dose.
    pub stable_dose: Vec<f64>,
    /// First seeded inertial-cavitation cumulative dose trial.
    pub inertial_trial1_dose: Vec<f64>,
    /// Second seeded inertial-cavitation cumulative dose trial.
    pub inertial_trial2_dose: Vec<f64>,
}

/// Generate stable and inertial passive-cavitation dose traces for book figures.
///
/// Stable cavitation is modeled as a deterministic per-pulse emission dose,
/// producing a normalized staircase over the sonication. Inertial cavitation is
/// modeled as a compound Poisson process: each pulse has a Poisson-distributed
/// number of collapse events, and each event contributes exponentially
/// distributed collapse energy. The RNG is deterministic and seed-controlled so
/// the generated book artifact is reproducible.
///
/// Returns an error when the time axis is empty, non-monotone, or contains
/// non-finite values, or when pulse parameters are non-physical.
pub fn passive_cavitation_dose_fixture(
    time_s: &[f64],
    prf_hz: f64,
    pulse_duration_s: f64,
    inertial_event_rate_fraction: f64,
    seed: u64,
) -> Result<PassiveCavitationDoseFixture, &'static str> {
    if time_s.len() < 2 {
        return Err("time axis must contain at least two samples");
    }
    if !(prf_hz.is_finite() && prf_hz > 0.0) {
        return Err("prf_hz must be finite and positive");
    }
    if !(pulse_duration_s.is_finite() && pulse_duration_s > 0.0) {
        return Err("pulse_duration_s must be finite and positive");
    }
    if !(inertial_event_rate_fraction.is_finite() && inertial_event_rate_fraction >= 0.0) {
        return Err("inertial_event_rate_fraction must be finite and non-negative");
    }
    if !time_s
        .windows(2)
        .all(|w| w[0].is_finite() && w[1].is_finite() && w[0] >= 0.0 && w[1] >= w[0])
    {
        return Err("time axis must be finite, non-negative, and monotone");
    }

    let time_end_s = *time_s.last().ok_or("time axis must be non-empty")?;
    let n_pulses = ((time_end_s * prf_hz).ceil() as usize).max(1);
    let stable_increment = pulse_duration_s * prf_hz;
    let stable_scale = n_pulses as f64 * stable_increment;
    let stable_dose = time_s
        .iter()
        .map(|&t| {
            let pulses_elapsed = pulse_count_at_time(t, prf_hz, n_pulses);
            pulses_elapsed as f64 * stable_increment / stable_scale
        })
        .collect();

    let mut rng1 = ChaCha8Rng::seed_from_u64(seed);
    let mut rng2 = ChaCha8Rng::seed_from_u64(seed.wrapping_add(1));
    let inertial_trial1_dose = compound_poisson_dose_trace(
        time_s,
        prf_hz,
        n_pulses,
        inertial_event_rate_fraction,
        &mut rng1,
    );
    let inertial_trial2_dose = compound_poisson_dose_trace(
        time_s,
        prf_hz,
        n_pulses,
        inertial_event_rate_fraction,
        &mut rng2,
    );

    Ok(PassiveCavitationDoseFixture {
        time_s: time_s.to_vec(),
        stable_dose,
        inertial_trial1_dose,
        inertial_trial2_dose,
    })
}

fn pulse_count_at_time(time_s: f64, prf_hz: f64, n_pulses: usize) -> usize {
    ((time_s * prf_hz).floor() as usize + 1).min(n_pulses)
}

fn compound_poisson_dose_trace(
    time_s: &[f64],
    prf_hz: f64,
    n_pulses: usize,
    event_rate_fraction: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let mut cumulative_by_pulse = Vec::with_capacity(n_pulses);
    let mut running = 0.0_f64;
    for _ in 0..n_pulses {
        for _ in 0..sample_poisson(event_rate_fraction, rng) {
            running += sample_exponential_unit(rng);
        }
        cumulative_by_pulse.push(running);
    }

    if running <= 0.0 {
        return vec![0.0; time_s.len()];
    }

    time_s
        .iter()
        .map(|&t| {
            let pulses_elapsed = pulse_count_at_time(t, prf_hz, n_pulses);
            cumulative_by_pulse[pulses_elapsed - 1] / running
        })
        .collect()
}

fn sample_poisson(lambda: f64, rng: &mut ChaCha8Rng) -> usize {
    if lambda <= 0.0 {
        return 0;
    }
    let limit = (-lambda).exp();
    let mut product = 1.0_f64;
    let mut count = 0_usize;
    loop {
        product *= rng.r#gen::<f64>();
        if product <= limit {
            return count;
        }
        count += 1;
    }
}

fn sample_exponential_unit(rng: &mut ChaCha8Rng) -> f64 {
    let u = rng
        .r#gen::<f64>()
        .clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON);
    -u.ln()
}

/// Indices defining a passive-cavitation therapeutic window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CavitationTherapeuticWindow {
    /// First sample where stable emission crosses the stable/harmonic threshold.
    pub stable_onset_index: usize,
    /// First sample where inertial emission crosses the inertial/harmonic threshold.
    pub inertial_onset_index: usize,
    /// First sample where inertial emission crosses the conservative controller cap.
    pub controller_cap_index: usize,
}

/// Classify the passive-cavitation therapeutic window from band-resolved powers.
///
/// Stable cavitation onset is the first sample where
/// `stable_power / harmonic_power > stable_ratio_threshold`. Inertial onset is
/// the first sample where `inertial_power / harmonic_power >
/// inertial_ratio_threshold`. The controller cap is the first sample where the
/// inertial ratio crosses `cap_ratio_threshold`.
///
/// If a threshold is never crossed, stable onset defaults to index `0`,
/// inertial onset defaults to the final shared sample, and controller cap
/// defaults to the inertial-onset index. The three input slices are evaluated
/// over their shared prefix.
///
/// # Arguments
/// * `harmonic_power` – harmonic-comb emission power samples
/// * `stable_power` – subharmonic + ultraharmonic emission power samples
/// * `inertial_power` – broadband emission power samples
/// * `stable_ratio_threshold` – stable/harmonic onset ratio
/// * `inertial_ratio_threshold` – inertial/harmonic onset ratio
/// * `cap_ratio_threshold` – conservative controller cap ratio
/// * `denominator_floor` – positive floor added to harmonic power
#[must_use]
pub fn cavitation_therapeutic_window_indices(
    harmonic_power: &[f64],
    stable_power: &[f64],
    inertial_power: &[f64],
    stable_ratio_threshold: f64,
    inertial_ratio_threshold: f64,
    cap_ratio_threshold: f64,
    denominator_floor: f64,
) -> CavitationTherapeuticWindow {
    let n = harmonic_power
        .len()
        .min(stable_power.len())
        .min(inertial_power.len());
    if n == 0 {
        return CavitationTherapeuticWindow {
            stable_onset_index: 0,
            inertial_onset_index: 0,
            controller_cap_index: 0,
        };
    }

    let stable_onset_index = first_ratio_crossing(
        stable_power,
        harmonic_power,
        n,
        stable_ratio_threshold,
        denominator_floor,
    )
    .unwrap_or(0);
    let inertial_onset_index = first_ratio_crossing(
        inertial_power,
        harmonic_power,
        n,
        inertial_ratio_threshold,
        denominator_floor,
    )
    .unwrap_or(n - 1);
    let controller_cap_index = first_ratio_crossing(
        inertial_power,
        harmonic_power,
        n,
        cap_ratio_threshold,
        denominator_floor,
    )
    .unwrap_or(inertial_onset_index);

    CavitationTherapeuticWindow {
        stable_onset_index,
        inertial_onset_index,
        controller_cap_index,
    }
}

/// First drive index where broadband emission occupies a target fraction of
/// total cavitation emission.
///
/// The inertial fraction is
/// ```text
/// f_inertial = inertial / (harmonic + stable + inertial + denominator_floor)
/// ```
/// evaluated over the shared prefix of the three input slices. The returned
/// index is the first sample where `f_inertial > threshold`. If no crossing is
/// found, the final shared sample is returned. The result is then clamped to at
/// least `min_index`, which is useful when a controller deliberately operates
/// one sampled drive step below the onset.
///
/// Empty input returns `0`. Non-finite or negative power samples are treated as
/// invalid and skipped. Negative finite power is clamped to zero because band
/// emission energy cannot be negative.
#[must_use]
pub fn cavitation_inertial_fraction_onset_index(
    harmonic_power: &[f64],
    stable_power: &[f64],
    inertial_power: &[f64],
    threshold: f64,
    denominator_floor: f64,
    min_index: usize,
) -> usize {
    let n = harmonic_power
        .len()
        .min(stable_power.len())
        .min(inertial_power.len());
    if n == 0 {
        return 0;
    }
    if !(threshold.is_finite()
        && threshold >= 0.0
        && denominator_floor.is_finite()
        && denominator_floor > 0.0)
    {
        return min_index.min(n - 1);
    }

    let crossing = harmonic_power
        .iter()
        .zip(stable_power.iter())
        .zip(inertial_power.iter())
        .take(n)
        .position(|((&harmonic, &stable), &inertial)| {
            if !(harmonic.is_finite() && stable.is_finite() && inertial.is_finite()) {
                return false;
            }
            let h = harmonic.max(0.0);
            let s = stable.max(0.0);
            let i = inertial.max(0.0);
            let total = h + s + i + denominator_floor;
            i / total > threshold
        })
        .unwrap_or(n - 1);

    crossing.max(min_index.min(n - 1))
}

fn first_ratio_crossing(
    numerator: &[f64],
    denominator: &[f64],
    n: usize,
    threshold: f64,
    denominator_floor: f64,
) -> Option<usize> {
    if !(threshold.is_finite()
        && threshold >= 0.0
        && denominator_floor.is_finite()
        && denominator_floor > 0.0)
    {
        return None;
    }

    numerator
        .iter()
        .zip(denominator.iter())
        .take(n)
        .position(|(&num, &den)| {
            if !(num.is_finite() && den.is_finite()) {
                return false;
            }
            let ratio = num.max(0.0) / (den.max(0.0) + denominator_floor);
            ratio > threshold
        })
}

/// One step of the InsighTec-style closed-loop cavitation-dose controller.
///
/// The controller adjusts drive pressure to maximise stable cavitation while
/// holding inertial cavitation below a safety limit. Given the most recent
/// monitoring-window emissions:
/// * **inertial emission above `inertial_limit`** → multiplicatively *back off*
///   pressure by `1 − gain` (clamped to `[p_min, p_max]`). Safety dominates.
/// * **otherwise, stable emission below `stable_target`** → *increase* pressure
///   by `1 + gain` to recruit more stable cavitation.
/// * **otherwise** → hold pressure (in the therapeutic window).
///
/// This is the discrete proportional law used for per-burst pressure stepping;
/// `gain` is the fractional step per burst (e.g. 0.05–0.1).
///
/// # Arguments
/// * `current_p_pa`     – drive pressure applied on the just-monitored burst [Pa]
/// * `stable_emission`  – measured sub+ultra-harmonic emission this burst
/// * `inertial_emission`– measured broadband emission this burst
/// * `stable_target`    – stable-emission set-point (recruit up to this level)
/// * `inertial_limit`   – broadband-emission ceiling (never exceed)
/// * `gain`             – fractional pressure step per burst (≥ 0)
/// * `p_min_pa`, `p_max_pa` – drive-pressure clamp [Pa]
///
/// Returns the drive pressure for the next burst [Pa].
///
/// # Reference
/// McDannold N. et al. (2006) *Phys. Med. Biol.* 51, 793.
/// Arvanitis C.D. et al. (2012) *PLoS ONE* 7, e45783.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn cavitation_controller_pressure(
    current_p_pa: f64,
    stable_emission: f64,
    inertial_emission: f64,
    stable_target: f64,
    inertial_limit: f64,
    gain: f64,
    p_min_pa: f64,
    p_max_pa: f64,
) -> f64 {
    let g = gain.max(0.0);
    let next = if inertial_emission > inertial_limit {
        current_p_pa * (1.0 - g) // safety back-off dominates
    } else if stable_emission < stable_target {
        current_p_pa * (1.0 + g) // recruit more stable cavitation
    } else {
        current_p_pa // hold inside the therapeutic window
    };
    next.clamp(p_min_pa, p_max_pa)
}
