//! Kelvin–Voigt viscoelastic model (spring ∥ dashpot).

use super::{recover_complex_modulus, DispersionSample};
use eunomia::Complex64;

/// A homogeneous Kelvin–Voigt viscoelastic solid for the shear channel.
///
/// The shear stress is `τ = μ γ + η_s γ̇`, giving the complex shear modulus
/// `G*(ω) = μ + i ω η_s` (storage `μ`, loss `ω η_s`). Phase velocity and
/// attenuation follow from the complex wavenumber `k = ω √(ρ / G*)`.
#[derive(Debug, Clone, Copy)]
pub struct KelvinVoigtModel {
    /// Elastic shear modulus `μ` \`Pa`.
    shear_modulus: f64,
    /// Shear viscosity `η_s` \[Pa·s].
    shear_viscosity: f64,
    /// Mass density `ρ` \[kg·m⁻³].
    density: f64,
}

impl KelvinVoigtModel {
    /// Create a Kelvin–Voigt model from shear modulus `μ` \`Pa`, shear viscosity
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

    /// Complex shear modulus `G*(ω) = μ + i ω η_s` \`Pa`.
    #[must_use]
    pub fn complex_modulus(&self, omega: f64) -> Complex64 {
        Complex64::new(self.shear_modulus, omega * self.shear_viscosity)
    }

    /// Storage (elastic) modulus `G'(ω) = μ` \`Pa`.
    #[must_use]
    pub fn storage_modulus(&self, _omega: f64) -> f64 {
        self.shear_modulus
    }

    /// Loss (viscous) modulus `G''(ω) = ω η_s` \`Pa`.
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

    /// Shear viscosity `η_s` \[Pa·s].
    #[must_use]
    pub fn shear_viscosity(&self) -> f64 {
        self.shear_viscosity
    }

    /// **Dispersion-fitting inversion** (shear-wave spectroscopy): recover a
    /// Kelvin–Voigt model from measured shear-wave dispersion samples
    /// `(ω, c_p, α)` and a known density.
    ///
    /// Per frequency the complex modulus is recovered model-agnostically via
    /// [`recover_complex_modulus`] (`G* = ρ(ω/k)²`, `k = ω/c_p + iα`). For the
    /// Kelvin–Voigt model `G* = μ + iωη_s`, so the storage part is the constant
    /// `μ` and the loss part is `ωη_s`; the fit takes
    /// `μ = ⟨Re G*⟩` and `η_s = ⟨Im G* / ω⟩` (the maximum-likelihood estimate
    /// under i.i.d. sample noise). Returns `None` for empty input, non-positive
    /// density, or a non-physical recovered `μ ≤ 0`.
    ///
    /// # References
    /// - Catheline et al. (2004), *J. Acoust. Soc. Am.* 116(6) — inverse-problem
    ///   viscoelastic recovery from transient elastography.
    /// - Deffieux et al. (2009), *IEEE TMI* 28(3) — shear-wave spectroscopy.
    #[must_use]
    pub fn fit_dispersion(samples: &[DispersionSample], density: f64) -> Option<Self> {
        if samples.is_empty() || density <= 0.0 {
            return None;
        }
        let mut sum_mu = 0.0;
        let mut sum_eta = 0.0;
        let mut n = 0.0;
        for s in samples {
            if s.omega <= 0.0 {
                continue;
            }
            let g = recover_complex_modulus(s.omega, s.phase_velocity, s.attenuation, density);
            sum_mu += g.re; // G' = μ (frequency-independent for Kelvin–Voigt)
            sum_eta += g.im / s.omega; // G'' = ωη_s ⇒ η_s = G''/ω
            n += 1.0;
        }
        if n == 0.0 {
            return None;
        }
        Self::new(sum_mu / n, (sum_eta / n).max(0.0), density)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Representative soft tissue: μ = 3 kPa, η_s = 1.5 Pa·s, ρ = 1000 kg/m³.
    fn liver() -> KelvinVoigtModel {
        KelvinVoigtModel::new(3000.0, 1.5, 1000.0).unwrap()
    }

    /// `recover_complex_modulus` exactly inverts the forward dispersion: for a
    /// Kelvin–Voigt model, `G*` recovered from its own `(c_p, α)` equals
    /// `complex_modulus(ω)`.
    #[test]
    fn recover_complex_modulus_inverts_forward_dispersion() {
        let kv = liver();
        for &f in &[50.0, 100.0, 300.0, 800.0] {
            let omega = std::f64::consts::TAU * f;
            let g = recover_complex_modulus(
                omega,
                kv.phase_velocity(omega),
                kv.attenuation(omega),
                1000.0,
            );
            let truth = kv.complex_modulus(omega);
            assert!(
                (g - truth).norm() <= 1e-6 * truth.norm(),
                "G* recovery at {f} Hz: {g} vs {truth}"
            );
        }
        // Degenerate inputs → 0.
        assert_eq!(
            recover_complex_modulus(0.0, 2.0, 0.0, 1000.0),
            eunomia::Complex64::new(0.0, 0.0)
        );
    }

    /// Dispersion-fitting round-trip: synthetic `(ω, c_p, α)` from a known
    /// Kelvin–Voigt model are inverted back to (μ, η_s) — shear-wave spectroscopy.
    #[test]
    fn fit_dispersion_recovers_known_kelvin_voigt_parameters() {
        let truth = KelvinVoigtModel::new(4200.0, 2.3, 1000.0).unwrap(); // μ, η_s, ρ
        let samples: Vec<DispersionSample> = [40.0, 80.0, 160.0, 320.0, 640.0]
            .iter()
            .map(|&f| {
                let omega = std::f64::consts::TAU * f;
                DispersionSample {
                    omega,
                    phase_velocity: truth.phase_velocity(omega),
                    attenuation: truth.attenuation(omega),
                }
            })
            .collect();

        let fit = KelvinVoigtModel::fit_dispersion(&samples, 1000.0).expect("fit");
        assert!(
            (fit.storage_modulus(0.0) - 4200.0).abs() <= 1e-3 * 4200.0,
            "μ recovered = {}",
            fit.storage_modulus(0.0)
        );
        assert!(
            (fit.shear_viscosity() - 2.3).abs() <= 1e-3 * 2.3,
            "η_s recovered = {}",
            fit.shear_viscosity()
        );

        // Empty input ⇒ None.
        assert!(KelvinVoigtModel::fit_dispersion(&[], 1000.0).is_none());
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
