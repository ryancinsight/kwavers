//! Triplet-pulse cavitation-bubble imaging (Iwasaki et al., JJAP 2018;
//! Kuji, Umemura & Yoshizawa, JJAP 2024).
//!
//! A triplet-pulse (3P) sequence transmits the same imaging burst three
//! times with the carrier phase shifted by 0°, 120° and 240°, then takes
//! the mean of the three received echoes. A harmonic of order `n`
//! (relative to the imaging fundamental `f0`) acquires a phase `n * phi`
//! per transmission, so the coherent sum retains a fraction
//! `|1 + e^(i*n*120°) + e^(i*n*240°)| / 3` of it:
//!
//! | harmonic | retention in 3P |
//! |----------|----------|
//! | 0.5th (subharmonic) | 2/3 |
//! | 1st (fundamental) | 0 |
//! | 1.5th (fractional, bubbles only) | 1/3 |
//! | 2nd | 0 |
//! | 3rd | 1 |
//!
//! Tissue echoes are dominated by the fundamental and even harmonics,
//! which cancel; cavitation bubbles radiate a 1.5th fractional harmonic
//! (subharmonic period-doubling), which survives. That is the whole
//! selectivity mechanism. Pulse inversion (PI, phases 0°/180°) is the
//! two-pulse analogue and is provided for comparison: it retains the
//! 1.5th at `|1 + e^(i*1.5*180°)| / 2 = 0.707` and the 2nd at 1.0.
//!
//! The prototype this was validated against (Python/pykwavers, 2D PSTD +
//! Keller-Miksis cloud) reproduced the paper's contrast ordering
//! 1P < PI < 3P at every bubble radius in the sweep; see
//! `Triplet_Pulse_Prototype_Validation.md` in the prototype work dir.
//! All experimental values the prototype used are chicken-breast
//! research values (Iwasaki 2018; Kuji 2024), not Suizenji clinical
//! settings.
//!
//! This module holds only the sequence definition and the coherent sums
//! (pure signal logic, no solver or bubble-physics dependency). The
//! bubble-cloud coupling (Keller-Miksis drive → monopole re-radiation)
//! lives in `kwavers-simulation`, which is the crate that can see both
//! the imaging sequence and the bubble physics.

/// Transmit phases of one triplet-pulse sequence, in degrees.
///
/// The 180° entry is the pulse-inversion partner, not part of the
/// triplet itself.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TripletPulsePhases {
    /// First transmission (reference, 0°).
    pub phase_0_deg: f64,
    /// Second transmission (120°).
    pub phase_120_deg: f64,
    /// Third transmission (240°).
    pub phase_240_deg: f64,
    /// Pulse-inversion partner (180°).
    pub phase_180_deg: f64,
}

impl Default for TripletPulsePhases {
    fn default() -> Self {
        Self {
            phase_0_deg: 0.0,
            phase_120_deg: 120.0,
            phase_240_deg: 240.0,
            phase_180_deg: 180.0,
        }
    }
}

/// Configuration of a triplet-pulse imaging acquisition.
#[derive(Debug, Clone)]
pub struct TripletPulseConfig {
    /// Imaging fundamental frequency (Hz), e.g. 1.74 MHz (Iwasaki 2018)
    /// or 2.45 MHz (Kuji 2024).
    pub frequency_hz: f64,
    /// Transmit phases.
    pub phases: TripletPulsePhases,
    /// Number of cycles in the imaging tone burst.
    pub burst_cycles: usize,
}

impl TripletPulseConfig {
    /// Iwasaki 2018 configuration: 1.74 MHz diverging burst.
    #[must_use]
    pub fn iwasaki_2018() -> Self {
        Self {
            frequency_hz: 1.74e6,
            phases: TripletPulsePhases::default(),
            burst_cycles: 3,
        }
    }

    /// Kuji 2024 configuration: 2.45 MHz imaging.
    #[must_use]
    pub fn kuji_2024() -> Self {
        Self {
            frequency_hz: 2.45e6,
            phases: TripletPulsePhases::default(),
            burst_cycles: 3,
        }
    }
}

/// Fraction of a harmonic of order `n` retained by the coherent 3P sum
/// (mean of the 0°/120°/240° transmissions).
///
/// `n` need not be an integer: the bubble signature is the 1.5th
/// fractional harmonic (`n = 1.5`). Computed as
/// `|1 + e^(i*n*120°) + e^(i*n*240°)| / 3` with a hand-rolled phasor
/// (real, imaginary) pair — no complex-number dependency.
#[must_use]
pub fn triplet_retention(order: f64) -> f64 {
    let (mut re, mut im) = (1.0, 0.0);
    for phase_deg in [120.0_f64, 240.0_f64] {
        let a = order * phase_deg.to_radians();
        re += a.cos();
        im += a.sin();
    }
    (re * re + im * im).sqrt() / 3.0
}

/// Fraction of a harmonic of order `n` retained by pulse inversion
/// (mean of the 0°/180° transmissions).
#[must_use]
pub fn pulse_inversion_retention(order: f64) -> f64 {
    let a = order * 180.0_f64.to_radians();
    ((1.0 + a.cos()).powi(2) + a.sin().powi(2)).sqrt() / 2.0
}

/// Mean of the phase-shifted echo traces.
///
/// `traces` holds one RF trace per transmission, all the same length,
/// in transmission order. Returns the coherent sum (the mean); callers
/// pass `&[ph0, ph120, ph240]` for 3P, `&[ph0, ph180]` for PI, or
/// `&[ph0]` for 1P.
#[must_use]
pub fn coherent_sum(traces: &[Vec<f64>]) -> Vec<f64> {
    assert!(!traces.is_empty(), "coherent_sum needs at least one trace");
    let n = traces[0].len();
    assert!(
        traces.iter().all(|t| t.len() == n),
        "all traces must have the same length"
    );
    let count = traces.len() as f64;
    (0..n)
        .map(|i| traces.iter().map(|t| t[i]).sum::<f64>() / count)
        .collect()
}

/// 1P / PI / 3P echoes from the four phase transmissions of one
/// acquisition. Each input is one receiver's (or one channel's) RF
/// trace; each output is the corresponding method's trace.
#[derive(Debug, Clone)]
pub struct TripletPulseEchoes {
    /// Single-pulse echo (phase 0° only).
    pub one_pulse: Vec<f64>,
    /// Pulse-inversion echo (mean of 0° and 180°).
    pub pulse_inversion: Vec<f64>,
    /// Triplet-pulse echo (mean of 0°, 120° and 240°).
    pub triplet_pulse: Vec<f64>,
}

/// Combine the four phase transmissions into 1P / PI / 3P echoes.
#[must_use]
pub fn combine_echoes(
    phase_0: &[f64],
    phase_120: &[f64],
    phase_240: &[f64],
    phase_180: &[f64],
) -> TripletPulseEchoes {
    TripletPulseEchoes {
        one_pulse: phase_0.to_vec(),
        pulse_inversion: coherent_sum(&[phase_0.to_vec(), phase_180.to_vec()]),
        triplet_pulse: coherent_sum(&[
            phase_0.to_vec(),
            phase_120.to_vec(),
            phase_240.to_vec(),
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// Stage 0 analytic proof: the phasor rule, exactly as verified in
    /// the Python prototype (2026-11-16).
    #[test]
    fn triplet_retention_matches_phasor_rule() {
        assert!(approx_eq(triplet_retention(0.5), 2.0 / 3.0, 1e-12));
        assert!(approx_eq(triplet_retention(1.0), 0.0, 1e-12));
        assert!(approx_eq(triplet_retention(1.5), 1.0 / 3.0, 1e-12));
        assert!(approx_eq(triplet_retention(2.0), 0.0, 1e-12));
        assert!(approx_eq(triplet_retention(3.0), 1.0, 1e-12));
    }

    #[test]
    fn pulse_inversion_retention_matches_phasor_rule() {
        assert!(approx_eq(pulse_inversion_retention(1.0), 0.0, 1e-12));
        assert!(approx_eq(
            pulse_inversion_retention(1.5),
            std::f64::consts::FRAC_1_SQRT_2,
            1e-12
        ));
        assert!(approx_eq(pulse_inversion_retention(2.0), 1.0, 1e-12));
    }

    /// The coherent sum does to a time trace what the phasor rule
    /// predicts: a fundamental cancels in 3P, a 1.5th harmonic survives
    /// at one third amplitude.
    #[test]
    fn coherent_sum_cancels_fundamental_keeps_fractional() {
        let f0 = 1.74e6;
        let dt = 1.0 / (20.0 * f0);
        let n = 2000;
        let mk = |order: f64, phase_deg: f64| -> Vec<f64> {
            let p = phase_deg.to_radians();
            (0..n)
                .map(|i| {
                    let t = i as f64 * dt;
                    (2.0 * std::f64::consts::PI * order * f0 * t + order * p).sin()
                })
                .collect()
        };
        for order in [1.0, 1.5] {
            let traces: Vec<Vec<f64>> = [0.0, 120.0, 240.0]
                .iter()
                .map(|&ph| mk(order, ph))
                .collect();
            let summed = coherent_sum(&traces);
            let peak = summed.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            let expected = triplet_retention(order);
            assert!(
                approx_eq(peak, expected, 0.02),
                "order {order}: peak {peak:.3}, expected {expected:.3}"
            );
        }
    }

    #[test]
    fn combine_echoes_returns_all_three_methods() {
        let zeros = vec![0.0; 8];
        let out = combine_echoes(&zeros, &zeros, &zeros, &zeros);
        assert_eq!(out.one_pulse.len(), 8);
        assert_eq!(out.pulse_inversion.len(), 8);
        assert_eq!(out.triplet_pulse.len(), 8);
    }
}
