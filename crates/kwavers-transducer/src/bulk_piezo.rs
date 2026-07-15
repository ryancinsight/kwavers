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
use eunomia::Complex64;
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

    /// Specific acoustic impedance `Z = ρ·c_D` \[Rayl = Pa·s·m⁻¹] of the plate,
    /// used to design quarter-wave matching layers (`Z_match = √(Z·Z_load)`).
    #[must_use]
    pub fn acoustic_impedance(&self) -> f64 {
        self.density * self.sound_speed()
    }

    /// Free (constant-stress) capacitance `C^T = C₀/(1 − k_t²)` \[F] — the
    /// low-frequency electrical capacitance, larger than the clamped `C₀`.
    #[must_use]
    pub fn free_capacitance(&self) -> f64 {
        self.clamped_capacitance() / (1.0 - self.coupling_kt2)
    }

    /// Electrical input impedance \[Ω] of the free (air-loaded both faces)
    /// thickness-mode plate at `frequency` \[Hz] — the Mason/KLM equivalent-circuit
    /// response.
    ///
    /// ```text
    /// Z_e(ω) = 1/(jωC₀) · [ 1 − k_t² · tan(X)/X ],   X = π f / (2 f_p)
    /// ```
    ///
    /// Lossless free plate ⇒ `Z_e` is purely reactive (`Re = 0`). The bracket
    /// vanishes at the series resonance `f_s` (`Z_e → 0`, identical to the IEEE
    /// `f_s` since `tan(X_s)/X_s = 1/k_t²`) and diverges at the antiresonance
    /// `f_p` (`X = π/2`). As `f → 0`, `Z_e → 1/(jω·C^T)` (free capacitance).
    /// Returns `Z_e = 0` at `frequency = 0` (DC short of the model's reactance).
    ///
    /// Air-loaded both faces is the open-circuit limit; matching/backing layers
    /// are modelled by the loaded acoustic transmission line ([`AcousticLayer`]).
    #[must_use]
    pub fn electrical_impedance(&self, frequency: f64) -> Complex64 {
        if frequency <= 0.0 {
            return Complex64::new(0.0, 0.0);
        }
        let f_p = self.antiresonance_frequency();
        let omega = 2.0 * PI * frequency;
        let c0 = self.clamped_capacitance();
        let x = PI * frequency / (2.0 * f_p);
        // tan(X)/X → 1 as X → 0 (removable singularity).
        let tanx_over_x = if x.abs() < 1e-12 { 1.0 } else { x.tan() / x };
        let bracket = 1.0 - self.coupling_kt2 * tanx_over_x;
        // 1/(jωC₀) = −i/(ωC₀); the lossless plate gives a purely reactive Z_e.
        Complex64::new(0.0, -1.0 / (omega * c0)) * bracket
    }

    /// Design a quarter-wave **matching layer** coupling this plate's acoustic
    /// impedance to an external acoustic `z_load` (e.g. water/tissue ≈ 1.5 MRayl).
    ///
    /// The layer impedance is the geometric mean `√(Z_plate·Z_load)`
    /// ([`quarter_wave_match_impedance`]) and the thickness is λ/4 at the
    /// antiresonance for the chosen layer `sound_speed` — at that frequency the
    /// layer transforms `z_load` back to `Z_plate`, eliminating the reflection.
    #[must_use]
    pub fn quarter_wave_matching_layer(
        &self,
        z_load: f64,
        layer_sound_speed: f64,
    ) -> AcousticLayer {
        let impedance = quarter_wave_match_impedance(self.acoustic_impedance(), z_load);
        AcousticLayer::quarter_wave(impedance, layer_sound_speed, self.antiresonance_frequency())
    }
}

/// Specific acoustic impedance of a single-layer quarter-wave matching transformer
/// coupling `z_source` to `z_load`: `Z = √(Z_source·Z_load)` \[Rayl].
///
/// A layer of this impedance, one quarter-wavelength thick at the design
/// frequency, presents `Z_in = Z_source` to the source — zero reflection
/// (the acoustic analogue of the λ/4 transmission-line transformer).
#[must_use]
pub fn quarter_wave_match_impedance(z_source: f64, z_load: f64) -> f64 {
    (z_source * z_load).sqrt()
}

/// A single lossless acoustic layer (matching layer or backing) in a 1-D
/// transmission-line stack.
#[derive(Debug, Clone, Copy)]
pub struct AcousticLayer {
    /// Specific acoustic impedance `Z = ρ·c` \[Rayl].
    pub impedance: f64,
    /// Layer thickness `d` \[m].
    pub thickness: f64,
    /// Layer longitudinal sound speed `c` \[m·s⁻¹].
    pub sound_speed: f64,
}

impl AcousticLayer {
    /// Construct a layer one quarter-wavelength thick at `frequency`
    /// (`d = c/(4f)`).
    #[must_use]
    pub fn quarter_wave(impedance: f64, sound_speed: f64, frequency: f64) -> Self {
        Self {
            impedance,
            sound_speed,
            thickness: sound_speed / (4.0 * frequency),
        }
    }

    /// Input acoustic impedance looking into the layer terminated by `z_load`, at
    /// `frequency` — the lossless transmission-line (telegrapher) transform:
    ///
    /// ```text
    /// Z_in = Z · (Z_load + j Z tan(kd)) / (Z + j Z_load tan(kd)),   k = 2πf/c.
    /// ```
    ///
    /// Quarter-wave (`kd = π/2`) ⇒ `Z_in = Z²/Z_load` (impedance inverter);
    /// half-wave (`kd = π`) ⇒ `Z_in = Z_load` (pass-through); a matched layer
    /// (`Z = Z_load`) ⇒ `Z_in = Z` at any thickness.
    #[must_use]
    pub fn input_impedance(&self, z_load: Complex64, frequency: f64) -> Complex64 {
        let kd = 2.0 * PI * frequency / self.sound_speed * self.thickness;
        let z = Complex64::new(self.impedance, 0.0);
        let jt = Complex64::new(0.0, kd.tan());
        z * (z_load + z * jt) / (z + z_load * jt)
    }

    /// Pressure reflection coefficient `Γ = (Z_in − Z_source)/(Z_in + Z_source)`
    /// seen by a source medium of impedance `z_source` looking into this layer
    /// terminated by `z_load`. `|Γ| = 0` is a perfect match.
    #[must_use]
    pub fn reflection_coefficient(
        &self,
        z_source: f64,
        z_load: Complex64,
        frequency: f64,
    ) -> Complex64 {
        let z_in = self.input_impedance(z_load, frequency);
        let zs = Complex64::new(z_source, 0.0);
        (z_in - zs) / (z_in + zs)
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
        assert!(
            (kt2 - p.coupling_kt2).abs() < 1e-4,
            "round-trip kt² {kt2} vs {}",
            p.coupling_kt2
        );
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

    #[test]
    fn acoustic_impedance_and_free_capacitance() {
        let p = therapy_pzt();
        assert!((p.acoustic_impedance() - 7500.0 * p.sound_speed()).abs() < 1e-6);
        // C^T = C₀/(1−k_t²) > C₀.
        let expected = p.clamped_capacitance() / (1.0 - p.coupling_kt2);
        assert!((p.free_capacitance() - expected).abs() / expected < 1e-12);
        assert!(p.free_capacitance() > p.clamped_capacitance());
    }

    #[test]
    fn electrical_impedance_is_purely_reactive_and_free_capacitive_at_low_freq() {
        let p = therapy_pzt();
        // Lossless free plate ⇒ Re{Z_e} = 0 at every frequency.
        for &f in &[1e3, 1e5, 0.5e6, 0.9e6] {
            assert_eq!(
                p.electrical_impedance(f).re,
                0.0,
                "Z_e must be purely reactive at {f} Hz"
            );
        }
        // f → 0: Z_e.im → −1/(ω·C^T) (free capacitance).
        let f = p.antiresonance_frequency() / 1.0e4;
        let omega = 2.0 * PI * f;
        let z = p.electrical_impedance(f);
        let expected_im = -1.0 / (omega * p.free_capacitance());
        assert!(
            (z.im - expected_im).abs() / expected_im.abs() < 1e-3,
            "low-freq reactance {} vs free-cap {expected_im}",
            z.im
        );
    }

    #[test]
    fn electrical_impedance_vanishes_at_series_resonance() {
        // The free-plate Z_e = 0 condition (1 − k_t² tan X/X = 0) is identical to
        // the IEEE series resonance, so |Z_e(f_s)| must be far below the midband.
        let p = therapy_pzt();
        let f_s = p.resonance_frequency();
        let z_res = p.electrical_impedance(f_s).norm();
        let z_mid = p.electrical_impedance(0.5 * f_s).norm();
        assert!(
            z_res < 1e-3 * z_mid,
            "|Z_e(f_s)|={z_res} should be ≪ midband |Z_e|={z_mid}"
        );
    }

    #[test]
    fn electrical_impedance_diverges_near_antiresonance() {
        let p = therapy_pzt();
        let f_p = p.antiresonance_frequency();
        // Approach f_p closely: tan(X)→∞ at X=π/2 dominates the 1/ω scaling.
        let z_near = p.electrical_impedance(0.9999 * f_p).norm();
        let z_mid = p.electrical_impedance(0.5 * f_p).norm();
        assert!(
            z_near > 100.0 * z_mid,
            "|Z_e| must blow up near f_p: {z_near} vs {z_mid}"
        );
    }

    // ── Loaded matching/backing transmission line (COV-6 follow-up) ──────────

    const Z_WATER: f64 = 1.5e6; // ≈ 1.5 MRayl

    #[test]
    fn match_impedance_is_geometric_mean() {
        let z = quarter_wave_match_impedance(30.0e6, Z_WATER); // PZT ~30 MRayl → water
        assert!((z - (30.0e6_f64 * Z_WATER).sqrt()).abs() / z < 1e-12);
    }

    #[test]
    fn half_wave_layer_is_a_passthrough() {
        // kd = π ⇒ tan = 0 ⇒ Z_in = Z_load, independent of the layer impedance.
        let f = 1.0e6;
        let c = 2000.0;
        let layer = AcousticLayer {
            impedance: 5.0e6,
            sound_speed: c,
            thickness: c / (2.0 * f), // λ/2
        };
        let z_load = Complex64::new(Z_WATER, 0.0);
        let z_in = layer.input_impedance(z_load, f);
        assert!(
            (z_in.re - Z_WATER).abs() / Z_WATER < 1e-6,
            "half-wave Re {z_in}"
        );
        assert!(z_in.im.abs() / Z_WATER < 1e-6, "half-wave Im {z_in}");
    }

    #[test]
    fn quarter_wave_layer_inverts_impedance() {
        // kd = π/2 ⇒ Z_in = Z²/Z_load.
        let f = 1.0e6;
        let layer = AcousticLayer::quarter_wave(5.0e6, 2000.0, f);
        let z_load = Complex64::new(Z_WATER, 0.0);
        let z_in = layer.input_impedance(z_load, f);
        let expected = 5.0e6_f64 * 5.0e6 / Z_WATER;
        assert!(
            (z_in.re - expected).abs() / expected < 1e-6,
            "λ/4 invert Re {z_in}"
        );
        assert!(z_in.im.abs() / expected < 1e-6, "λ/4 invert Im {z_in}");
    }

    #[test]
    fn matched_layer_transforms_to_its_own_impedance_at_any_thickness() {
        // Z = Z_load ⇒ no internal reflection ⇒ Z_in = Z for every kd.
        let f = 1.0e6;
        for &d_frac in &[0.1, 0.37, 0.5, 0.83] {
            let layer = AcousticLayer {
                impedance: Z_WATER,
                sound_speed: 2000.0,
                thickness: d_frac * 2000.0 / f,
            };
            let z_in = layer.input_impedance(Complex64::new(Z_WATER, 0.0), f);
            assert!(
                (z_in.re - Z_WATER).abs() / Z_WATER < 1e-9,
                "matched Re at d={d_frac}"
            );
            assert!(z_in.im.abs() / Z_WATER < 1e-9, "matched Im at d={d_frac}");
        }
    }

    #[test]
    fn quarter_wave_match_eliminates_reflection_into_water() {
        // A λ/4 layer of impedance √(Z_plate·Z_water) at f_p transforms the water
        // load back to the plate impedance ⇒ Z_in = Z_plate, Γ = 0.
        let p = therapy_pzt();
        let f_p = p.antiresonance_frequency();
        let z_plate = p.acoustic_impedance();
        let layer = p.quarter_wave_matching_layer(Z_WATER, 2000.0);
        // Designed impedance is the geometric mean.
        assert!((layer.impedance - (z_plate * Z_WATER).sqrt()).abs() / layer.impedance < 1e-12);
        let z_in = layer.input_impedance(Complex64::new(Z_WATER, 0.0), f_p);
        assert!(
            (z_in.re - z_plate).abs() / z_plate < 1e-6,
            "matched Z_in {z_in} vs Z_plate {z_plate}"
        );
        let gamma = layer.reflection_coefficient(z_plate, Complex64::new(Z_WATER, 0.0), f_p);
        assert!(
            gamma.norm() < 1e-6,
            "matched reflection |Γ|={}",
            gamma.norm()
        );
    }

    #[test]
    fn unmatched_plate_into_water_reflects_strongly() {
        // Sanity: without a matching layer the plate↔water mismatch is severe.
        // Direct interface Γ = (Z_water − Z_plate)/(Z_water + Z_plate).
        let p = therapy_pzt();
        let z_plate = p.acoustic_impedance();
        let gamma_direct = (Z_WATER - z_plate) / (Z_WATER + z_plate);
        assert!(
            gamma_direct.abs() > 0.8,
            "bare PZT↔water mismatch should reflect >80%: |Γ|={}",
            gamma_direct.abs()
        );
    }
}
