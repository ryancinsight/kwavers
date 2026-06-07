//! Bulk piezoelectric **thickness-mode** resonator (e.g. a PZT therapy stack).
//!
//! Complements the flexural-plate MEMS models (`mems`): a bulk plate of thickness
//! `t` resonates in its thickness (half-wave) mode. From the stiffened elastic
//! constant `c₃₃^D`, density `ρ`, clamped permittivity `ε₃₃^S` and thickness
//! coupling `k_t²` this gives the (open-circuit) antiresonance, the series
//! resonance, the clamped capacitance, and the IEEE `k_t²`↔`(f_s, f_p)` relation.
//! Bulk PZT remains the workhorse for **high-pressure therapy** (Chapter 33 §33.9):
//! a thick active layer gives large strain volume and high output, unlike the
//! gap-limited CMUT.
//!
//! Thickness-mode coupling (IEEE Std 176-1987):
//!
//! ```text
//! k_t² = (π f_s)/(2 f_p) · tan[ π (f_p − f_s) / (2 f_p) ],
//! ```
//!
//! with `f_p = c_D/(2t)` the antiresonance (clamped, stiffened modulus) and
//! `f_s < f_p` the series resonance.
//!
//! # References
//! - IEEE Standard on Piezoelectricity, ANSI/IEEE Std 176-1987.
//! - Kino, G. S. (1987). *Acoustic Waves: Devices, Imaging, and Analog Signal
//!   Processing*, §1.3 (thickness-mode transducers).

use core::f64::consts::PI;
use kwavers_core::constants::fundamental::VACUUM_PERMITTIVITY;

/// A bulk piezoelectric thickness-mode resonator.
#[derive(Debug, Clone, Copy)]
pub struct BulkPiezoResonator {
    /// Plate thickness `t` \[m].
    pub thickness: f64,
    /// Electrode area `A` \[m²].
    pub area: f64,
    /// Density `ρ` \[kg·m⁻³].
    pub density: f64,
    /// Stiffened (open-circuit) elastic constant `c₃₃^D` \[Pa].
    pub stiffened_modulus: f64,
    /// Clamped relative permittivity `ε₃₃^S / ε₀`.
    pub clamped_rel_permittivity: f64,
    /// Thickness electromechanical coupling `k_t²` ∈ (0, 1).
    pub coupling_kt2: f64,
}

impl BulkPiezoResonator {
    /// PZT-5H thickness-mode preset (therapy-grade soft PZT).
    #[must_use]
    pub fn pzt5h(thickness: f64, area: f64) -> Option<Self> {
        if thickness > 0.0 && area > 0.0 {
            Some(Self {
                thickness,
                area,
                density: 7500.0,
                stiffened_modulus: 15.7e10, // c₃₃^D → c_D ≈ 4575 m/s
                clamped_rel_permittivity: 1470.0,
                coupling_kt2: 0.23, // k_t ≈ 0.48
            })
        } else {
            None
        }
    }

    /// Stiffened (open-circuit) thickness sound speed `c_D = √(c₃₃^D/ρ)` \[m·s⁻¹].
    #[must_use]
    pub fn sound_speed(&self) -> f64 {
        (self.stiffened_modulus / self.density).sqrt()
    }

    /// Antiresonance (parallel, open-circuit) frequency `f_p = c_D/(2t)` \[Hz].
    #[must_use]
    pub fn antiresonance_frequency(&self) -> f64 {
        self.sound_speed() / (2.0 * self.thickness)
    }

    /// Clamped capacitance `C₀ = ε₀ ε₃₃^S A / t` \[F].
    #[must_use]
    pub fn clamped_capacitance(&self) -> f64 {
        VACUUM_PERMITTIVITY * self.clamped_rel_permittivity * self.area / self.thickness
    }

    /// Thickness coupling `k_t²` from a `(f_s, f_p)` pair (IEEE relation).
    #[must_use]
    pub fn coupling_from_frequencies(f_s: f64, f_p: f64) -> f64 {
        if f_p <= 0.0 || f_s <= 0.0 || f_s >= f_p {
            return 0.0;
        }
        (PI * f_s / (2.0 * f_p)) * (PI * (f_p - f_s) / (2.0 * f_p)).tan()
    }

    /// Series resonance `f_s` \[Hz] for the configured `k_t²` and antiresonance,
    /// by bisection of the IEEE relation (`k_t²` decreases monotonically with `f_s`).
    #[must_use]
    pub fn resonance_frequency(&self) -> f64 {
        let f_p = self.antiresonance_frequency();
        let target = self.coupling_kt2;
        // kt²(f_s): →1 as f_s→0, →0 as f_s→f_p ; monotone decreasing.
        let (mut lo, mut hi) = (1e-6 * f_p, f_p * (1.0 - 1e-9));
        for _ in 0..80 {
            let mid = 0.5 * (lo + hi);
            let k2 = Self::coupling_from_frequencies(mid, f_p);
            if k2 > target {
                lo = mid; // need a higher f_s (smaller k²)
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Effective fractional gap `(f_p − f_s)/f_p` — a coupling indicator.
    #[must_use]
    pub fn resonance_gap(&self) -> f64 {
        let f_p = self.antiresonance_frequency();
        (f_p - self.resonance_frequency()) / f_p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1 MHz-class therapy PZT: f_p = c_D/(2t) ≈ 1 MHz → t ≈ 2.29 mm.
    fn therapy_pzt() -> BulkPiezoResonator {
        BulkPiezoResonator::pzt5h(2.29e-3, 1.0e-4).unwrap()
    }

    #[test]
    fn antiresonance_scales_inverse_thickness() {
        let a = BulkPiezoResonator::pzt5h(2.0e-3, 1e-4).unwrap();
        let b = BulkPiezoResonator::pzt5h(1.0e-3, 1e-4).unwrap(); // half thickness
        assert!((b.antiresonance_frequency() / a.antiresonance_frequency() - 2.0).abs() < 1e-9);
        // therapy-band resonance
        assert!(therapy_pzt().antiresonance_frequency() > 0.8e6);
        assert!(therapy_pzt().antiresonance_frequency() < 1.2e6);
    }

    #[test]
    fn sound_speed_and_capacitance_formulas() {
        let p = therapy_pzt();
        assert!((p.sound_speed() - (15.7e10_f64 / 7500.0).sqrt()).abs() < 1e-6);
        let expected_c = VACUUM_PERMITTIVITY * 1470.0 * p.area / p.thickness;
        assert!((p.clamped_capacitance() - expected_c).abs() / expected_c < 1e-12);
    }

    #[test]
    fn resonance_below_antiresonance_and_round_trips() {
        let p = therapy_pzt();
        let f_p = p.antiresonance_frequency();
        let f_s = p.resonance_frequency();
        assert!(f_s < f_p, "f_s {f_s} must be below f_p {f_p}");
        // recompute kt² from the recovered (f_s, f_p) → matches the configured value
        let kt2 = BulkPiezoResonator::coupling_from_frequencies(f_s, f_p);
        assert!((kt2 - p.coupling_kt2).abs() < 1e-4, "round-trip kt² {kt2} vs {}", p.coupling_kt2);
    }

    #[test]
    fn stronger_coupling_widens_the_resonance_gap() {
        let mut weak = therapy_pzt();
        weak.coupling_kt2 = 0.1;
        let mut strong = therapy_pzt();
        strong.coupling_kt2 = 0.45;
        assert!(strong.resonance_gap() > weak.resonance_gap());
        // zero coupling → f_s ≈ f_p (no gap)
        let mut none = therapy_pzt();
        none.coupling_kt2 = 1e-9;
        assert!(none.resonance_gap() < 1e-3);
    }
}
