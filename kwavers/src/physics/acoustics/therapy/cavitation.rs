//! Cavitation detection and monitoring for therapeutic ultrasound
//!
//! ## Physical Foundation
//!
//! ### Blake Threshold
//!
//! The critical negative pressure for cavitation inception in a liquid with a
//! spherical nucleus of equilibrium radius R₀ is (Apfel 1981, eq. 7):
//! ```text
//!   P_Blake = |P₀ + P_v − 2σ/R₀|
//! ```
//! where P₀ = ambient pressure, P_v = vapour pressure, σ = surface tension.
//! For R₀ = 1 µm in water at 20 °C:
//! ```text
//!   2σ/R₀ = 2 × 0.0728 / 1×10⁻⁶ = 145 600 Pa
//!   P_Blake ≈ |101 325 + 2 340 − 145 600| ≈ 41 935 Pa
//! ```
//!
//! ### Minnaert Resonance Frequency
//!
//! The natural frequency of a spherical gas bubble is (Minnaert 1933):
//! ```text
//!   f₀ = (1 / 2πR₀) √(3γP₀ / ρ_L)
//! ```
//! For R₀ = 1 µm, γ = 1.4 (air), P₀ = 101 325 Pa, ρ_L = 1000 kg/m³:
//!   f₀ ≈ 3.26 MHz
//!
//! ### Spectral (Resonance-Enhanced) Threshold
//!
//! Near Minnaert resonance, bubble oscillations are amplified by the mechanical
//! Q-factor of the oscillator (Leighton 1994, §4.4).  For a linearised bubble
//! with quality factor Q the enhancement at driving frequency f is:
//! ```text
//!   E(f) = max(1, 1 / sqrt((1−(f/f₀)²)² + (f/(Q·f₀))²))
//! ```
//! The effective threshold is P_eff = P_Blake / E(f).  Away from resonance
//! E → 1 and detection falls back to Blake threshold.
//!
//! ## Not Implemented
//!
//! - Passive acoustic mapping (PAM) for 3D spatial cavitation localisation
//! - Active dual-frequency cavitation detection
//! - Real-time feedback control
//! - Time-domain spectral analysis (subharmonics, broadband noise) for PAM
//!
//! ## References
//!
//! - Minnaert M (1933). Phil. Mag. **16**:235–248. (resonance frequency)
//! - Apfel RE (1981). J. Acoust. Soc. Am. **69**(6):1624–1633. (Blake threshold)
//! - Flynn HG (1982). J. Acoust. Soc. Am. **72**(6):1926–1932. (inertial cavitation)
//! - Leighton TG (1994). *The Acoustic Bubble*. Academic Press. §4.4 (Q-factor)

use ndarray::Array3;
use std::f64::consts::PI;

/// Physical constants (water at 20 °C)
const WATER_SURFACE_TENSION: f64 = 0.0728; // N/m
const WATER_VAPOR_PRESSURE: f64 = 2.34e3; // Pa
const ATMOSPHERIC_PRESSURE: f64 = 101_325.0; // Pa

/// Reference nucleus radius for threshold calculations (1 µm)
const DEFAULT_NUCLEUS_RADIUS: f64 = 1e-6; // m

/// Cavitation detection methods
#[derive(Debug, Clone, Copy)]
pub enum CavitationDetectionMethod {
    /// Pressure threshold: voxel cavitates iff |p| > P_Blake
    PressureThreshold,
    /// Resonance-enhanced threshold: P_eff = P_Blake / E(f) per voxel
    Spectral,
    /// Combined: Spectral method (encompasses threshold as E → 1 off-resonance)
    Combined,
}

/// Cavitation detector for therapeutic ultrasound
///
/// Provides per-voxel and aggregate cavitation metrics for HIFU/FUS therapy
/// planning and safety monitoring.  The Blake threshold is computed at
/// construction for a standard 1 µm air nucleus (adjustable via `new_with_radius`).
#[derive(Debug)]
pub struct TherapyCavitationDetector {
    /// Driving frequency (Hz)
    frequency: f64,
    /// Blake threshold pressure magnitude (Pa) — positive value
    pub blake_threshold: f64,
    /// Detection method
    method: CavitationDetectionMethod,
}

impl TherapyCavitationDetector {
    /// Create a cavitation detector for `frequency` (Hz) using a 1 µm reference nucleus.
    #[must_use]
    pub fn new(frequency: f64, _peak_negative_pressure: f64) -> Self {
        Self::new_with_radius(frequency, DEFAULT_NUCLEUS_RADIUS)
    }

    /// Create a cavitation detector for `frequency` (Hz) and nucleus radius `r0` (m).
    ///
    /// ## Theorem (Blake threshold, Apfel 1981 eq. 7)
    ///
    /// The critical negative pressure for cavitation inception is:
    /// ```text
    ///   P_Blake = |P₀ + P_v − 2σ/R₀|
    /// ```
    /// For R₀ = 1 µm: P_Blake ≈ 41 935 Pa.
    #[must_use]
    pub fn new_with_radius(frequency: f64, r0: f64) -> Self {
        let blake_threshold =
            (ATMOSPHERIC_PRESSURE + WATER_VAPOR_PRESSURE - 2.0 * WATER_SURFACE_TENSION / r0).abs();
        Self {
            frequency,
            blake_threshold,
            method: CavitationDetectionMethod::PressureThreshold,
        }
    }

    /// Detect cavitation in a snapshot pressure field.
    ///
    /// Returns a boolean array with the same shape as `pressure`.
    /// `true` indicates cavitation is predicted at that voxel.
    #[must_use]
    pub fn detect(&self, pressure: &Array3<f64>) -> Array3<bool> {
        let mut cavitation = Array3::from_elem(pressure.dim(), false);
        match self.method {
            CavitationDetectionMethod::PressureThreshold => {
                self.detect_by_threshold(pressure, &mut cavitation);
            }
            CavitationDetectionMethod::Spectral | CavitationDetectionMethod::Combined => {
                self.detect_by_spectral(pressure, &mut cavitation);
            }
        }
        cavitation
    }

    /// Per-voxel Blake threshold detection: cavitates iff `−p > P_Blake`.
    fn detect_by_threshold(&self, pressure: &Array3<f64>, cavitation: &mut Array3<bool>) {
        cavitation
            .iter_mut()
            .zip(pressure.iter())
            .for_each(|(cav, &p)| {
                *cav = -p > self.blake_threshold;
            });
    }

    /// Per-voxel resonance-enhanced threshold detection.
    ///
    /// ## Algorithm
    ///
    /// 1. Compute Minnaert resonance frequency f₀ for a 1 µm reference nucleus.
    /// 2. Compute resonance enhancement E(f) via the linearised bubble oscillator
    ///    with quality factor Q = 2.0 (Leighton 1994, §4.4):
    ///    ```text
    ///      E(f) = max(1, 1 / sqrt((1−(f/f₀)²)² + (f/(Q·f₀))²))
    ///    ```
    /// 3. Effective threshold: P_eff = P_Blake / E(f).
    /// 4. Voxel cavitates iff `−p > P_eff`.
    ///
    /// Away from resonance E = 1 and this reduces to Blake threshold detection.
    ///
    /// ## References
    /// - Minnaert (1933) for f₀; Leighton (1994) §4.4 for Q-factor model.
    fn detect_by_spectral(&self, pressure: &Array3<f64>, cavitation: &mut Array3<bool>) {
        let f0 = self.minnaert_frequency(DEFAULT_NUCLEUS_RADIUS);
        let f_over_f0 = self.frequency / f0;
        // Quality factor for µm air bubble in water (Leighton 1994, §4.4)
        const Q: f64 = 2.0;
        let detuning = 1.0 - f_over_f0 * f_over_f0;
        let dissipation = f_over_f0 / Q;
        let denominator = (detuning * detuning + dissipation * dissipation).sqrt();
        // E ≥ 1: resonance can only lower the threshold, never raise it
        let enhancement = if denominator < 1.0 { 1.0 / denominator } else { 1.0 };
        let effective_threshold = self.blake_threshold / enhancement;

        cavitation
            .iter_mut()
            .zip(pressure.iter())
            .for_each(|(cav, &p)| {
                *cav = -p > effective_threshold;
            });
    }

    /// Minnaert resonance frequency for a bubble of radius `r0` (m) (Hz).
    ///
    /// ## Theorem (Minnaert 1933)
    /// ```text
    ///   f₀ = (1 / 2πR₀) √(3γP₀ / ρ_L)
    /// ```
    /// For R₀ = 1 µm, γ = 1.4, P₀ = 101 325 Pa, ρ_L = 1000 kg/m³: f₀ ≈ 3.26 MHz.
    #[must_use]
    pub fn minnaert_frequency(&self, r0: f64) -> f64 {
        const GAMMA: f64 = 1.4; // polytropic index for air
        const LIQUID_DENSITY: f64 = 1000.0; // kg/m³
        let pressure_term = 3.0 * GAMMA * ATMOSPHERIC_PRESSURE / LIQUID_DENSITY;
        pressure_term.sqrt() / (2.0 * PI * r0)
    }

    /// Cavitation index CI = |P_neg| / P_Blake.
    ///
    /// CI < 0.5: no cavitation; 0.5 ≤ CI < 1.0: stable cavitation; CI ≥ 1.0: inertial cavitation.
    #[must_use]
    pub fn cavitation_index(&self, peak_negative_pressure: f64) -> f64 {
        peak_negative_pressure.abs() / self.blake_threshold
    }

    /// Logistic cavitation probability: sigmoid centred at CI = 1 with steepness k = 5.
    ///
    /// P(cav) = 1 / (1 + exp(−5·(CI − 1)))
    #[must_use]
    pub fn cavitation_probability(&self, peak_negative_pressure: f64) -> f64 {
        let ci = self.cavitation_index(peak_negative_pressure);
        1.0 / (1.0 + (-5.0 * (ci - 1.0)).exp())
    }

    /// Returns `true` when 0.5 < CI < 1.0 (stable/non-inertial cavitation).
    #[must_use]
    pub fn is_stable_cavitation(&self, peak_negative_pressure: f64) -> bool {
        let ci = self.cavitation_index(peak_negative_pressure);
        ci > 0.5 && ci < 1.0
    }

    /// Returns `true` when CI ≥ 1.0 (inertial / transient cavitation).
    #[must_use]
    pub fn is_inertial_cavitation(&self, peak_negative_pressure: f64) -> bool {
        self.cavitation_index(peak_negative_pressure) >= 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector() -> TherapyCavitationDetector {
        TherapyCavitationDetector::new(1e6, 0.0)
    }

    // ── Blake threshold ──────────────────────────────────────────────────────

    /// P_Blake for 1 µm nucleus: |(101325 + 2340 − 145600)| = 41935 Pa.
    #[test]
    fn test_blake_threshold_1um_value() {
        let det = detector();
        let expected = (ATMOSPHERIC_PRESSURE + WATER_VAPOR_PRESSURE
            - 2.0 * WATER_SURFACE_TENSION / 1e-6)
            .abs();
        assert!(
            (det.blake_threshold - expected).abs() < 1.0,
            "Blake threshold {:.1} Pa ≠ expected {expected:.1} Pa",
            det.blake_threshold
        );
    }

    /// Blake threshold must be positive (it is a pressure magnitude).
    #[test]
    fn test_blake_threshold_positive() {
        let det = detector();
        assert!(
            det.blake_threshold > 0.0,
            "Blake threshold must be positive; got {}",
            det.blake_threshold
        );
    }

    /// Larger nucleus → lower surface tension term → higher threshold.
    /// For R₀ = 10 µm: 2σ/R₀ = 14 560 Pa, P_Blake = |101325+2340−14560| = 89105 Pa.
    #[test]
    fn test_blake_threshold_larger_nucleus_higher_pressure() {
        let det_1um = TherapyCavitationDetector::new_with_radius(1e6, 1e-6);
        let det_10um = TherapyCavitationDetector::new_with_radius(1e6, 10e-6);
        assert!(
            det_10um.blake_threshold > det_1um.blake_threshold,
            "10 µm nucleus should have higher Blake threshold than 1 µm"
        );
    }

    // ── Minnaert frequency ───────────────────────────────────────────────────

    /// f₀(1 µm) ≈ 3.26 MHz (Minnaert 1933).
    #[test]
    fn test_minnaert_frequency_1um() {
        let det = detector();
        let f0 = det.minnaert_frequency(1e-6);
        // Literature value: ~3.26 MHz ± 5%
        assert!(
            (f0 - 3.26e6).abs() / 3.26e6 < 0.05,
            "Minnaert f₀(1µm) = {:.3e} Hz, expected ~3.26 MHz",
            f0
        );
    }

    /// Minnaert frequency scales as 1/R: halving radius doubles frequency.
    #[test]
    fn test_minnaert_frequency_scales_inversely_with_radius() {
        let det = detector();
        let f1 = det.minnaert_frequency(1e-6);
        let f2 = det.minnaert_frequency(2e-6);
        let ratio = f1 / f2;
        assert!(
            (ratio - 2.0).abs() < 1e-10,
            "f₀(R) should scale as 1/R: f(1µm)/f(2µm) = {ratio:.6}, expected 2.0"
        );
    }

    // ── Threshold detection ───────────────────────────────────────────────────

    /// Pressure below Blake threshold → no cavitation.
    #[test]
    fn test_threshold_detection_no_cavitation_below_threshold() {
        let det = detector();
        let p = Array3::from_elem((4, 4, 4), -0.5 * det.blake_threshold); // |p| < P_Blake
        let cav = det.detect(&p);
        assert!(
            cav.iter().all(|&c| !c),
            "pressure below Blake threshold should give no cavitation"
        );
    }

    /// Pressure above Blake threshold → cavitation everywhere.
    #[test]
    fn test_threshold_detection_cavitation_above_threshold() {
        let det = detector();
        let p = Array3::from_elem((4, 4, 4), -2.0 * det.blake_threshold); // |p| > P_Blake
        let cav = det.detect(&p);
        assert!(
            cav.iter().all(|&c| c),
            "pressure above Blake threshold should give cavitation everywhere"
        );
    }

    /// Spatial heterogeneity: only high-pressure voxels cavitate.
    #[test]
    fn test_threshold_detection_spatial_heterogeneity() {
        let det = detector();
        let p_high = -2.0 * det.blake_threshold; // cavitates
        let p_low = -0.1 * det.blake_threshold; // does not cavitate
        let mut p = Array3::from_elem((2, 2, 2), p_low);
        p[[0, 0, 0]] = p_high;
        p[[1, 1, 1]] = p_high;
        let cav = det.detect(&p);
        assert!(cav[[0, 0, 0]], "voxel 0,0,0 should cavitate");
        assert!(cav[[1, 1, 1]], "voxel 1,1,1 should cavitate");
        assert!(!cav[[0, 0, 1]], "voxel 0,0,1 should not cavitate");
    }

    // ── Spectral detection ────────────────────────────────────────────────────

    /// Zero pressure → no cavitation regardless of frequency.
    #[test]
    fn test_spectral_detection_zero_pressure_no_cavitation() {
        let mut det = TherapyCavitationDetector::new(3.26e6, 0.0); // at resonance
        det.method = CavitationDetectionMethod::Spectral;
        let p = Array3::zeros((4, 4, 4));
        let cav = det.detect(&p);
        assert!(
            cav.iter().all(|&c| !c),
            "zero pressure must never trigger cavitation"
        );
    }

    /// Near Minnaert resonance the effective threshold is lower, so moderately
    /// negative pressures that would not trigger off-resonance should trigger.
    #[test]
    fn test_spectral_detection_resonance_lowers_threshold() {
        let det_far = TherapyCavitationDetector::new(100e3, 0.0); // far off-resonance
        let mut det_near = TherapyCavitationDetector::new(3.26e6, 0.0); // ≈ resonance
        det_near.method = CavitationDetectionMethod::Spectral;

        // Pick a pressure that is 80% of Blake threshold (no cavitation off-resonance)
        let p_test = -0.80 * det_near.blake_threshold;
        let p = Array3::from_elem((4, 4, 4), p_test);

        let cav_far = det_far.detect(&p);
        let cav_near = det_near.detect(&p);

        assert!(
            cav_far.iter().all(|&c| !c),
            "off-resonance: 80% of threshold should not cavitate"
        );
        assert!(
            cav_near.iter().all(|&c| c),
            "at resonance: 80% of threshold should cavitate (threshold lowered by resonance)"
        );
    }

    // ── Cavitation index and classification ───────────────────────────────────

    /// CI = 1.0 at exactly Blake threshold.
    #[test]
    fn test_cavitation_index_at_threshold_is_one() {
        let det = detector();
        let ci = det.cavitation_index(det.blake_threshold);
        assert!(
            (ci - 1.0).abs() < 1e-12,
            "CI at P_Blake must equal 1.0; got {ci:.6e}"
        );
    }

    /// CI = 0 for zero pressure.
    #[test]
    fn test_cavitation_index_zero_pressure_is_zero() {
        let det = detector();
        assert_eq!(det.cavitation_index(0.0), 0.0);
    }

    /// Stable cavitation regime: 0.5 < CI < 1.0.
    #[test]
    fn test_stable_cavitation_in_correct_range() {
        let det = detector();
        assert!(det.is_stable_cavitation(0.7 * det.blake_threshold));
        assert!(!det.is_stable_cavitation(0.4 * det.blake_threshold));
        assert!(!det.is_stable_cavitation(1.1 * det.blake_threshold));
    }

    /// Inertial cavitation: CI ≥ 1.0.
    #[test]
    fn test_inertial_cavitation_above_threshold() {
        let det = detector();
        assert!(det.is_inertial_cavitation(det.blake_threshold));
        assert!(det.is_inertial_cavitation(2.0 * det.blake_threshold));
        assert!(!det.is_inertial_cavitation(0.9 * det.blake_threshold));
    }

    /// Cavitation probability is 0.5 at CI = 1 (sigmoid midpoint).
    #[test]
    fn test_cavitation_probability_at_threshold_is_half() {
        let det = detector();
        let p = det.cavitation_probability(det.blake_threshold);
        assert!(
            (p - 0.5).abs() < 1e-10,
            "probability at CI=1 must be 0.5; got {p:.6e}"
        );
    }

    /// Cavitation probability is monotone: higher pressure → higher probability.
    #[test]
    fn test_cavitation_probability_monotone() {
        let det = detector();
        let p1 = det.cavitation_probability(0.5 * det.blake_threshold);
        let p2 = det.cavitation_probability(1.0 * det.blake_threshold);
        let p3 = det.cavitation_probability(2.0 * det.blake_threshold);
        assert!(p1 < p2 && p2 < p3, "probability must increase with pressure");
    }
}
