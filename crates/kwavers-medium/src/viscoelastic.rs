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

/// A measured shear-wave dispersion sample for [`KelvinVoigtModel::fit_dispersion`].
#[derive(Debug, Clone, Copy)]
pub struct DispersionSample {
    /// Angular frequency `ω` \[rad·s⁻¹].
    pub omega: f64,
    /// Measured phase velocity `c_p(ω)` \[m·s⁻¹].
    pub phase_velocity: f64,
    /// Measured attenuation `α(ω)` \[Np·m⁻¹].
    pub attenuation: f64,
}

/// Recover the complex shear modulus `G*(ω)` \[Pa] from a measured phase
/// velocity and attenuation (model-agnostic rheological inversion).
///
/// Inverts the dispersion relation `k = ω√(ρ/G*)` used by the forward models.
/// On the physical branch `Im(k) < 0` for a dissipative `G* = μ + iωη_s`
/// (`α = |Im k|`), so the measured complex wavenumber is `k = ω/c_p − iα` and
/// `G* = ρ(ω/k)²`. For a lossless medium (`α = 0`) this returns the real
/// `ρ c_p²`. Returns `0` for non-positive `ω`, `c_p`, or `ρ`.
#[must_use]
pub fn recover_complex_modulus(
    omega: f64,
    phase_velocity: f64,
    attenuation: f64,
    density: f64,
) -> Complex64 {
    if omega <= 0.0 || phase_velocity <= 0.0 || density <= 0.0 {
        return Complex64::new(0.0, 0.0);
    }
    let k = Complex64::new(omega / phase_velocity, -attenuation);
    let ratio = Complex64::new(omega, 0.0) / k;
    Complex64::new(density, 0.0) * ratio * ratio
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

    /// Shear-wave **attenuation** `α(ω) = |Im(k)|` \[Np·m⁻¹], `k = ω√(ρ/G*)`.
    #[must_use]
    pub fn attenuation(&self, omega: f64) -> f64 {
        if omega == 0.0 {
            return 0.0;
        }
        let ratio = Complex64::new(self.density, 0.0) / self.complex_modulus(omega);
        (ratio.sqrt() * omega).im.abs()
    }

    /// Relaxed (low-frequency) shear modulus `G_r` \[Pa].
    #[must_use]
    pub fn relaxed_modulus(&self) -> f64 {
        self.relaxed_modulus
    }

    /// Unrelaxed (high-frequency) shear modulus `G_u` \[Pa].
    #[must_use]
    pub fn unrelaxed_modulus(&self) -> f64 {
        self.unrelaxed_modulus
    }

    /// Relaxation time `τ` \[s].
    #[must_use]
    pub fn relaxation_time(&self) -> f64 {
        self.relaxation_time
    }

    /// **Dispersion-fitting inversion** of the Zener (standard-linear-solid)
    /// model from measured shear-wave dispersion `(ω, c_p, α)` and density.
    ///
    /// # Separable least squares
    ///
    /// Per frequency the complex modulus `G*(ω_k)` is recovered via
    /// [`recover_complex_modulus`]. For the Zener model the storage/loss parts
    /// are, with `x = ωτ`,
    /// `G' = G_r + Δ·x²/(1+x²)`, `G'' = Δ·x/(1+x²)`, `Δ = G_u − G_r`.
    /// For a **fixed `τ`** both are *linear* in `(G_r, Δ)`, so the optimal
    /// `(G_r, Δ)` is a closed-form 2×2 least-squares solve; the only nonlinear
    /// parameter `τ` is found by a 1-D search (coarse logarithmic scan +
    /// golden-section refinement) minimising the stacked `G'/G''` residual.
    ///
    /// Returns `None` for empty input, non-positive density, or a recovered
    /// model that violates `0 < G_r ≤ G_u`.
    ///
    /// # References
    /// - Catheline et al. (2004); Deffieux et al. (2009) — shear-wave spectroscopy.
    #[must_use]
    pub fn fit_dispersion(samples: &[DispersionSample], density: f64) -> Option<Self> {
        if samples.is_empty() || density <= 0.0 {
            return None;
        }
        // Recover G*(ω_k) once; the τ-search reuses these.
        let moduli: Vec<(f64, f64, f64)> = samples
            .iter()
            .filter(|s| s.omega > 0.0)
            .map(|s| {
                let g = recover_complex_modulus(s.omega, s.phase_velocity, s.attenuation, density);
                (s.omega, g.re, g.im) // (ω, G', G'')
            })
            .collect();
        if moduli.len() < 2 {
            return None;
        }

        // Residual of the best (G_r, Δ) for a given τ (separable LS).
        let residual_at = |tau: f64| -> (f64, f64, f64) {
            // Normal-equations accumulation for p = [G_r, Δ].
            let (mut m00, mut m01, mut m11) = (0.0, 0.0, 0.0);
            let (mut v0, mut v1) = (0.0, 0.0);
            for &(omega, gp, gpp) in &moduli {
                let x = omega * tau;
                let denom = 1.0 + x * x;
                let a = x * x / denom; // G' coefficient on Δ
                let b = x / denom; // G'' coefficient on Δ
                // G' equation: [1, a]·p = G'
                m00 += 1.0;
                m01 += a;
                m11 += a * a;
                v0 += gp;
                v1 += a * gp;
                // G'' equation: [0, b]·p = G''
                m11 += b * b;
                v1 += b * gpp;
            }
            let det = m00 * m11 - m01 * m01;
            if det.abs() < 1e-30 {
                return (f64::INFINITY, 0.0, 0.0);
            }
            let g_r = (v0 * m11 - v1 * m01) / det;
            let delta = (m00 * v1 - m01 * v0) / det;
            // Residual ‖G'_pred − G'‖² + ‖G''_pred − G''‖².
            let mut res = 0.0;
            for &(omega, gp, gpp) in &moduli {
                let x = omega * tau;
                let denom = 1.0 + x * x;
                let gp_pred = (g_r) + delta * (x * x / denom);
                let gpp_pred = delta * (x / denom);
                res += (gp_pred - gp).powi(2) + (gpp_pred - gpp).powi(2);
            }
            (res, g_r, delta)
        };

        // Coarse logarithmic scan over τ ∈ [1e-6, 1] s (60 points), then refine.
        let (lo_exp, hi_exp, steps) = (-6.0_f64, 0.0_f64, 60usize);
        let mut best_tau = 10f64.powf(lo_exp);
        let mut best_res = f64::INFINITY;
        for i in 0..=steps {
            let tau = 10f64.powf((hi_exp - lo_exp).mul_add(i as f64 / steps as f64, lo_exp));
            let (res, _, _) = residual_at(tau);
            if res < best_res {
                best_res = res;
                best_tau = tau;
            }
        }
        // Golden-section refinement in the bracketing decade around best_tau.
        let mut a = best_tau / 3.0;
        let mut b = best_tau * 3.0;
        const PHI_INV: f64 = 0.618_033_988_749_895;
        let mut c = b - (b - a) * PHI_INV;
        let mut d = a + (b - a) * PHI_INV;
        for _ in 0..60 {
            if residual_at(c).0 < residual_at(d).0 {
                b = d;
            } else {
                a = c;
            }
            c = b - (b - a) * PHI_INV;
            d = a + (b - a) * PHI_INV;
        }
        let tau = 0.5 * (a + b);
        let (_, g_r, delta) = residual_at(tau);
        ZenerModel::new(g_r, g_r + delta.max(0.0), tau, density)
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
            num_complex::Complex64::new(0.0, 0.0)
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

    /// Zener separable-LS dispersion fit recovers the three SLS parameters
    /// `(G_r, G_u, τ)` from synthetic `(ω, c_p, α)` spanning the loss peak.
    #[test]
    fn zener_fit_dispersion_recovers_three_parameters() {
        let truth = ZenerModel::new(2000.0, 6000.0, 1.0e-3, 1000.0).unwrap(); // G_r,G_u,τ,ρ
        // Frequencies spanning ωτ ≈ 0.06 … 6 (around the ωτ=1 Debye peak).
        let samples: Vec<DispersionSample> = [10.0, 30.0, 80.0, 160.0, 320.0, 640.0, 1000.0]
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

        let fit = ZenerModel::fit_dispersion(&samples, 1000.0).expect("zener fit");
        assert!(
            (fit.relaxed_modulus() - 2000.0).abs() <= 0.01 * 2000.0,
            "G_r = {}",
            fit.relaxed_modulus()
        );
        assert!(
            (fit.unrelaxed_modulus() - 6000.0).abs() <= 0.01 * 6000.0,
            "G_u = {}",
            fit.unrelaxed_modulus()
        );
        assert!(
            (fit.relaxation_time() - 1.0e-3).abs() <= 0.02 * 1.0e-3,
            "τ = {}",
            fit.relaxation_time()
        );

        // Recovered model reproduces the dispersion it was fit from.
        for &f in &[20.0, 200.0, 800.0] {
            let omega = std::f64::consts::TAU * f;
            assert!(
                (fit.phase_velocity(omega) - truth.phase_velocity(omega)).abs()
                    <= 1e-2 * truth.phase_velocity(omega),
                "c_p mismatch at {f} Hz"
            );
        }
        assert!(ZenerModel::fit_dispersion(&[], 1000.0).is_none());
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
