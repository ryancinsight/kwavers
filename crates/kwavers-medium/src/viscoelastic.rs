//! Kelvin–Voigt viscoelastic constitutive model (frequency domain).
//!
//! The Kelvin–Voigt model represents tissue as a spring and dashpot in parallel:
//! the shear stress is `τ = μ γ + η_s γ̇`. In the frequency domain this gives a
//! **complex shear modulus**
//!
//! ```text
//! G*(ω) = μ + i ω η_s            (storage μ,  loss ω η_s)
//! ```
//!
//! from which the (dispersive) shear-wave phase velocity and attenuation follow
//! through the complex wavenumber `k = ω √(ρ / G*)`. This is the medium-layer
//! counterpart of the analytical `voigt_complex_modulus` /
//! `voigt_shear_wave_dispersion` plotting helpers: solvers and inversion kernels
//! query it for the frequency-dependent modulus of a viscoelastic tissue.
//!
//! # References
//! - Catheline, S., et al. (2004). "Measurement of viscoelastic properties of
//!   homogeneous soft solid using transient elastography: An inverse problem
//!   approach." *J. Acoust. Soc. Am.*, 116(6), 3734–3741.
//! - Deffieux, T., et al. (2009). "Shear wave spectroscopy for in vivo
//!   quantification of human soft tissues viscoelasticity." *IEEE TMI*, 28(3).

use num_complex::Complex64;

/// A homogeneous Kelvin–Voigt viscoelastic solid for the shear channel.
#[derive(Debug, Clone, Copy)]
pub struct KelvinVoigtModel {
    /// Elastic shear modulus `μ` \[Pa].
    shear_modulus: f64,
    /// Shear viscosity `η_s` \[Pa·s].
    shear_viscosity: f64,
    /// Mass density `ρ` \[kg·m⁻³].
    density: f64,
}

impl KelvinVoigtModel {
    /// Create a Kelvin–Voigt model from shear modulus `μ` \[Pa], shear viscosity
    /// `η_s` \[Pa·s], and density `ρ` \[kg·m⁻³].
    ///
    /// Returns `None` if `μ ≤ 0`, `η_s < 0`, or `ρ ≤ 0` (unphysical).
    #[must_use]
    pub fn new(shear_modulus: f64, shear_viscosity: f64, density: f64) -> Option<Self> {
        if shear_modulus > 0.0 && shear_viscosity >= 0.0 && density > 0.0 {
            Some(Self {
                shear_modulus,
                shear_viscosity,
                density,
            })
        } else {
            None
        }
    }

    /// Complex shear modulus `G*(ω) = μ + i ω η_s` \[Pa].
    #[must_use]
    pub fn complex_modulus(&self, omega: f64) -> Complex64 {
        Complex64::new(self.shear_modulus, omega * self.shear_viscosity)
    }

    /// Storage (elastic) modulus `G'(ω) = μ` \[Pa].
    #[must_use]
    pub fn storage_modulus(&self, _omega: f64) -> f64 {
        self.shear_modulus
    }

    /// Loss (viscous) modulus `G''(ω) = ω η_s` \[Pa].
    #[must_use]
    pub fn loss_modulus(&self, omega: f64) -> f64 {
        omega * self.shear_viscosity
    }

    /// Loss tangent `tan δ = G''/G' = ω η_s / μ` (dimensionless).
    #[must_use]
    pub fn loss_tangent(&self, omega: f64) -> f64 {
        self.loss_modulus(omega) / self.shear_modulus
    }

    /// Shear quality factor `Q(ω) = G'/G'' = μ / (ω η_s)`; `+∞` at `ω → 0` or for
    /// a lossless (`η_s = 0`) solid.
    #[must_use]
    pub fn quality_factor(&self, omega: f64) -> f64 {
        let loss = self.loss_modulus(omega);
        if loss > 0.0 {
            self.shear_modulus / loss
        } else {
            f64::INFINITY
        }
    }

    /// Elastic-limit shear-wave speed `c_s = √(μ/ρ)` \[m·s⁻¹] (the `ω → 0` phase velocity).
    #[must_use]
    pub fn elastic_shear_speed(&self) -> f64 {
        (self.shear_modulus / self.density).sqrt()
    }

    /// Complex shear wavenumber `k = ω √(ρ / G*)` \[rad·m⁻¹] on the physical
    /// (positive real part) branch.
    #[must_use]
    pub fn complex_wavenumber(&self, omega: f64) -> Complex64 {
        if omega == 0.0 {
            return Complex64::new(0.0, 0.0);
        }
        let ratio = Complex64::new(self.density, 0.0) / self.complex_modulus(omega);
        let mut k = ratio.sqrt() * omega;
        if k.re < 0.0 {
            k = -k; // physical branch: forward-propagating wave
        }
        k
    }

    /// Dispersive shear-wave **phase velocity** `c_p(ω) = ω / Re(k)` \[m·s⁻¹].
    /// Increases with frequency (Kelvin–Voigt stiffening) and tends to
    /// [`Self::elastic_shear_speed`] as `ω → 0`.
    #[must_use]
    pub fn phase_velocity(&self, omega: f64) -> f64 {
        if omega == 0.0 {
            return self.elastic_shear_speed();
        }
        omega / self.complex_wavenumber(omega).re
    }

    /// Shear-wave **attenuation** `α(ω) = Im(k)` \[Np·m⁻¹] (amplitude decay rate).
    #[must_use]
    pub fn attenuation(&self, omega: f64) -> f64 {
        self.complex_wavenumber(omega).im.abs()
    }
}

/// Standard linear solid (**Zener**) viscoelastic model — a spring in parallel
/// with a Maxwell (spring + dashpot) arm.
///
/// Unlike the Kelvin–Voigt model (whose phase velocity and attenuation grow without
/// bound with frequency), the Zener model has **bounded dispersion**: the modulus
/// relaxes from the unrelaxed (instantaneous) value `G_u` at high frequency to the
/// relaxed value `G_r` at low frequency, with a single relaxation time `τ`:
///
/// ```text
/// G*(ω) = G_r + (G_u − G_r) · iωτ/(1 + iωτ)
/// G'(ω) = G_r + (G_u − G_r) (ωτ)²/(1+(ωτ)²)     (storage)
/// G''(ω) = (G_u − G_r) ωτ/(1+(ωτ)²)             (loss, peaks at ωτ = 1)
/// ```
///
/// This is the standard single-relaxation tissue model (also the building block of
/// generalized-Maxwell / fractional models).
#[derive(Debug, Clone, Copy)]
pub struct ZenerModel {
    /// Relaxed (low-frequency, ω→0) shear modulus `G_r` \[Pa].
    relaxed_modulus: f64,
    /// Unrelaxed (high-frequency, ω→∞) shear modulus `G_u ≥ G_r` \[Pa].
    unrelaxed_modulus: f64,
    /// Relaxation time `τ` \[s].
    relaxation_time: f64,
    /// Mass density `ρ` \[kg·m⁻³].
    density: f64,
}

impl ZenerModel {
    /// Create a Zener model. Returns `None` unless `0 < G_r ≤ G_u`, `τ > 0`, `ρ > 0`.
    #[must_use]
    pub fn new(relaxed_modulus: f64, unrelaxed_modulus: f64, relaxation_time: f64, density: f64) -> Option<Self> {
        if relaxed_modulus > 0.0
            && unrelaxed_modulus >= relaxed_modulus
            && relaxation_time > 0.0
            && density > 0.0
        {
            Some(Self { relaxed_modulus, unrelaxed_modulus, relaxation_time, density })
        } else {
            None
        }
    }

    /// Complex shear modulus `G*(ω)` \[Pa].
    #[must_use]
    pub fn complex_modulus(&self, omega: f64) -> Complex64 {
        let wt = omega * self.relaxation_time;
        let delta = self.unrelaxed_modulus - self.relaxed_modulus;
        let maxwell = Complex64::new(0.0, wt) / Complex64::new(1.0, wt); // iωτ/(1+iωτ)
        Complex64::new(self.relaxed_modulus, 0.0) + delta * maxwell
    }

    /// Storage modulus `G'(ω)` \[Pa] — rises monotonically from `G_r` to `G_u`.
    #[must_use]
    pub fn storage_modulus(&self, omega: f64) -> f64 {
        let wt = omega * self.relaxation_time;
        self.relaxed_modulus + (self.unrelaxed_modulus - self.relaxed_modulus) * (wt * wt) / (1.0 + wt * wt)
    }

    /// Loss modulus `G''(ω)` \[Pa] — Debye peak at `ωτ = 1`.
    #[must_use]
    pub fn loss_modulus(&self, omega: f64) -> f64 {
        let wt = omega * self.relaxation_time;
        (self.unrelaxed_modulus - self.relaxed_modulus) * wt / (1.0 + wt * wt)
    }

    /// Frequency \[Hz] of the loss (`G''`) peak, where `ωτ = 1`.
    #[must_use]
    pub fn loss_peak_frequency(&self) -> f64 {
        1.0 / (core::f64::consts::TAU * self.relaxation_time)
    }

    /// Relaxed (low-frequency) shear-wave speed `√(G_r/ρ)` \[m·s⁻¹].
    #[must_use]
    pub fn relaxed_shear_speed(&self) -> f64 {
        (self.relaxed_modulus / self.density).sqrt()
    }

    /// Unrelaxed (high-frequency) shear-wave speed `√(G_u/ρ)` \[m·s⁻¹].
    #[must_use]
    pub fn unrelaxed_shear_speed(&self) -> f64 {
        (self.unrelaxed_modulus / self.density).sqrt()
    }

    /// Dispersive shear-wave phase velocity `c_p(ω) = ω/Re(k)`, `k = ω√(ρ/G*)`
    /// \[m·s⁻¹] — bounded between the relaxed and unrelaxed speeds.
    #[must_use]
    pub fn phase_velocity(&self, omega: f64) -> f64 {
        if omega == 0.0 {
            return self.relaxed_shear_speed();
        }
        let ratio = Complex64::new(self.density, 0.0) / self.complex_modulus(omega);
        let mut k = ratio.sqrt() * omega;
        if k.re < 0.0 {
            k = -k;
        }
        omega / k.re
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Representative soft tissue: μ = 3 kPa, η_s = 1.5 Pa·s, ρ = 1000 kg/m³.
    fn liver() -> KelvinVoigtModel {
        KelvinVoigtModel::new(3000.0, 1.5, 1000.0).unwrap()
    }

    // Zener tissue: G_r = 2 kPa, G_u = 6 kPa, τ = 1 ms, ρ = 1000.
    fn zener() -> ZenerModel {
        ZenerModel::new(2000.0, 6000.0, 1.0e-3, 1000.0).unwrap()
    }

    #[test]
    fn zener_rejects_unphysical() {
        assert!(ZenerModel::new(6000.0, 2000.0, 1e-3, 1000.0).is_none()); // G_u < G_r
        assert!(ZenerModel::new(2000.0, 6000.0, 0.0, 1000.0).is_none()); // τ = 0
    }

    #[test]
    fn zener_storage_rises_from_relaxed_to_unrelaxed() {
        let z = zener();
        let lo = z.storage_modulus(1.0); // ωτ ≪ 1
        let hi = z.storage_modulus(1.0e7); // ωτ ≫ 1
        assert!((lo - 2000.0).abs() < 5.0, "low-ω storage {lo} ≈ G_r");
        assert!((hi - 6000.0).abs() < 5.0, "high-ω storage {hi} ≈ G_u");
        // monotone increasing
        assert!(z.storage_modulus(2000.0) > z.storage_modulus(500.0));
    }

    #[test]
    fn zener_loss_peaks_at_omega_tau_unity() {
        let z = zener();
        let f_peak = z.loss_peak_frequency();
        let w_peak = core::f64::consts::TAU * f_peak; // ωτ = 1
        let g_peak = z.loss_modulus(w_peak);
        // peak value is (G_u − G_r)/2
        assert!((g_peak - 2000.0).abs() < 1.0, "G''_peak {g_peak} ≈ (G_u−G_r)/2");
        // it is a maximum
        assert!(g_peak > z.loss_modulus(0.1 * w_peak));
        assert!(g_peak > z.loss_modulus(10.0 * w_peak));
    }

    #[test]
    fn zener_dispersion_is_bounded() {
        let z = zener();
        let lo = z.phase_velocity(1.0);
        let hi = z.phase_velocity(1.0e7);
        assert!((lo - z.relaxed_shear_speed()).abs() / lo < 1e-2, "low-ω → relaxed speed");
        assert!((hi - z.unrelaxed_shear_speed()).abs() / hi < 1e-2, "high-ω → unrelaxed speed");
        // every phase velocity stays within [relaxed, unrelaxed] (bounded, unlike Kelvin–Voigt)
        for &f in &[1.0, 1e2, 1e3, 1e4, 1e5, 1e6] {
            let c = z.phase_velocity(core::f64::consts::TAU * f);
            assert!(c >= z.relaxed_shear_speed() - 1e-9 && c <= z.unrelaxed_shear_speed() + 1e-9, "c={c}");
        }
    }

    #[test]
    fn rejects_unphysical_parameters() {
        assert!(KelvinVoigtModel::new(0.0, 1.0, 1000.0).is_none());
        assert!(KelvinVoigtModel::new(3000.0, -1.0, 1000.0).is_none());
        assert!(KelvinVoigtModel::new(3000.0, 1.0, 0.0).is_none());
    }

    #[test]
    fn complex_modulus_is_storage_plus_i_loss() {
        let m = liver();
        let omega = 2.0 * std::f64::consts::PI * 200.0; // 200 Hz drive
        let g = m.complex_modulus(omega);
        assert!((g.re - 3000.0).abs() < 1e-9, "storage = {}", g.re);
        assert!((g.im - omega * 1.5).abs() < 1e-9, "loss = {}", g.im);
        assert!((m.storage_modulus(omega) - 3000.0).abs() < 1e-9);
        assert!((m.loss_modulus(omega) - omega * 1.5).abs() < 1e-9);
        // tan δ and Q are reciprocals
        assert!((m.loss_tangent(omega) * m.quality_factor(omega) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn low_frequency_phase_velocity_tends_to_elastic_limit() {
        let m = liver();
        let c0 = m.elastic_shear_speed(); // √(3000/1000) = √3 ≈ 1.732 m/s
        assert!((c0 - 3.0_f64.sqrt()).abs() < 1e-12);
        // a very low drive frequency recovers the elastic speed
        let c_low = m.phase_velocity(2.0 * std::f64::consts::PI * 1.0);
        assert!((c_low - c0).abs() / c0 < 1e-3, "c_low = {c_low}, c0 = {c0}");
    }

    #[test]
    fn phase_velocity_and_attenuation_increase_with_frequency() {
        let m = liver();
        let w1 = 2.0 * std::f64::consts::PI * 100.0;
        let w2 = 2.0 * std::f64::consts::PI * 400.0;
        // viscous dispersion: faster and more attenuated at higher frequency
        assert!(m.phase_velocity(w2) > m.phase_velocity(w1));
        assert!(m.attenuation(w2) > m.attenuation(w1));
        // phase velocity always exceeds the elastic-limit speed
        assert!(m.phase_velocity(w1) >= m.elastic_shear_speed());
    }

    #[test]
    fn lossless_limit_matches_elastic_wave() {
        // η_s = 0 → no dispersion, no attenuation, c_p = √(μ/ρ) at all ω
        let m = KelvinVoigtModel::new(3000.0, 0.0, 1000.0).unwrap();
        let w = 2.0 * std::f64::consts::PI * 300.0;
        assert!((m.phase_velocity(w) - m.elastic_shear_speed()).abs() < 1e-9);
        assert!(m.attenuation(w) < 1e-9);
        assert!(m.quality_factor(w).is_infinite());
    }
}
