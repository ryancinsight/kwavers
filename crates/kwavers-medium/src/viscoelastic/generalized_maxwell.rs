//! Generalized Maxwell (Wiechert) viscoelastic model — `N` relaxation arms.
//!
//! A spring `E_∞` in parallel with `N` Maxwell arms (spring `E_j` + dashpot,
//! relaxation time `τ_j`) gives the complex shear modulus
//!
//! ```text
//! G*(ω) = E_∞ + Σ_{j=1}^N E_j · iωτ_j / (1 + iωτ_j).
//! ```
//!
//! A single arm is exactly the [`ZenerModel`](super::ZenerModel). With many arms
//! whose weights follow `E_j ∝ τ_j^{1-y}` over a log-spaced set of relaxation
//! times, the loss modulus follows `G''(ω) ∝ ω^{y-1}` — i.e. the shear
//! attenuation follows the power law `α ∝ ω^y` (Fung 1993). This is the discrete,
//! time-domain-realizable analog of the fractional-Laplacian absorption operator
//! (book §4.4.3 / §4.8.3).
//!
//! # References
//! - Fung, Y.C. (1993). *Biomechanics: Mechanical Properties of Living Tissues*,
//!   2nd ed. Springer — power-law absorption from a relaxation-time distribution.
//! - Tschoegl, N.W. (1989). *The Phenomenological Theory of Linear Viscoelastic
//!   Behavior.* Springer.

use eunomia::Complex64;

/// A generalized Maxwell (Wiechert) viscoelastic solid for the shear channel.
#[derive(Debug, Clone)]
pub struct GeneralizedMaxwellModel {
    /// Equilibrium (relaxed, ω→0) modulus `E_∞` \`Pa`.
    equilibrium_modulus: f64,
    /// Relaxation arms `(E_j, τ_j)` — weight \`Pa` and relaxation time \`s`.
    arms: Vec<(f64, f64)>,
    /// Mass density `ρ` \[kg·m⁻³].
    density: f64,
}

impl GeneralizedMaxwellModel {
    /// Build from an equilibrium modulus, explicit relaxation arms `(E_j, τ_j)`,
    /// and density. Returns `None` unless `E_∞ > 0`, every `E_j > 0` and
    /// `τ_j > 0`, `ρ > 0`, and at least one arm is supplied.
    #[must_use]
    pub fn new(equilibrium_modulus: f64, arms: Vec<(f64, f64)>, density: f64) -> Option<Self> {
        let valid = equilibrium_modulus > 0.0
            && density > 0.0
            && !arms.is_empty()
            && arms
                .iter()
                .all(|&(e, t)| e > 0.0 && t > 0.0 && e.is_finite() && t.is_finite());
        valid.then_some(Self {
            equilibrium_modulus,
            arms,
            density,
        })
    }

    /// Construct a model whose shear attenuation follows a target power law
    /// `α ∝ ω^y` across `[f_min, f_max]` (Fung 1993).
    ///
    /// `n_arms` relaxation times are placed logarithmically over the band and the
    /// weights set to `E_j ∝ τ_j^{1-y}`, normalised so the total relaxation
    /// strength `Σ E_j` equals `delta_total` \`Pa` (the modulus rises from
    /// `equilibrium_modulus` at ω→0 to `equilibrium_modulus + delta_total` at
    /// ω→∞). Returns `None` for invalid bounds (`0 < f_min < f_max`),
    /// `n_arms == 0`, non-positive `E_∞`/`delta_total`/`ρ`, or non-finite `y`.
    #[must_use]
    pub fn power_law(
        equilibrium_modulus: f64,
        delta_total: f64,
        f_min: f64,
        f_max: f64,
        n_arms: usize,
        y: f64,
        density: f64,
    ) -> Option<Self> {
        if !(equilibrium_modulus > 0.0
            && delta_total > 0.0
            && density > 0.0
            && n_arms >= 1
            && f_min > 0.0
            && f_max > f_min
            && y.is_finite())
        {
            return None;
        }
        let tau_max = 1.0 / (core::f64::consts::TAU * f_min); // low f ⇒ long τ
        let tau_min = 1.0 / (core::f64::consts::TAU * f_max);
        let ln_lo = tau_min.ln();
        let ln_hi = tau_max.ln();

        // Log-spaced τ_j and raw weights τ_j^{1-y}.
        let exponent = 1.0 - y;
        let taus: Vec<f64> = (0..n_arms)
            .map(|j| {
                let frac = if n_arms == 1 {
                    0.5
                } else {
                    j as f64 / (n_arms - 1) as f64
                };
                ((ln_hi - ln_lo) * frac + ln_lo).exp()
            })
            .collect();
        let raw: Vec<f64> = taus.iter().map(|&t| t.powf(exponent)).collect();
        let raw_sum: f64 = raw.iter().sum();
        if !(raw_sum > 0.0 && raw_sum.is_finite()) {
            return None;
        }

        // Normalise so Σ E_j = delta_total.
        let arms: Vec<(f64, f64)> = taus
            .iter()
            .zip(&raw)
            .map(|(&tau, &w)| (delta_total * w / raw_sum, tau))
            .collect();
        Self::new(equilibrium_modulus, arms, density)
    }

    /// Complex shear modulus `G*(ω) = E_∞ + Σ E_j iωτ_j/(1+iωτ_j)` \`Pa`.
    #[must_use]
    pub fn complex_modulus(&self, omega: f64) -> Complex64 {
        let mut g = Complex64::new(self.equilibrium_modulus, 0.0);
        for &(e, tau) in &self.arms {
            let wt = omega * tau;
            g += e * (Complex64::new(0.0, wt) / Complex64::new(1.0, wt));
        }
        g
    }

    /// Storage modulus `G'(ω) = E_∞ + Σ E_j (ωτ_j)²/(1+(ωτ_j)²)` \`Pa`.
    #[must_use]
    pub fn storage_modulus(&self, omega: f64) -> f64 {
        let mut g = self.equilibrium_modulus;
        for &(e, tau) in &self.arms {
            let wt = omega * tau;
            g += e * (wt * wt) / (1.0 + wt * wt);
        }
        g
    }

    /// Loss modulus `G''(ω) = Σ E_j ωτ_j/(1+(ωτ_j)²)` \`Pa`.
    #[must_use]
    pub fn loss_modulus(&self, omega: f64) -> f64 {
        let mut g = 0.0;
        for &(e, tau) in &self.arms {
            let wt = omega * tau;
            g += e * wt / (1.0 + wt * wt);
        }
        g
    }

    /// Equilibrium (relaxed) shear-wave speed `√(E_∞/ρ)` \[m·s⁻¹].
    #[must_use]
    pub fn relaxed_shear_speed(&self) -> f64 {
        (self.equilibrium_modulus / self.density).sqrt()
    }

    /// Instantaneous (unrelaxed) shear-wave speed `√((E_∞ + Σ E_j)/ρ)` \[m·s⁻¹].
    #[must_use]
    pub fn unrelaxed_shear_speed(&self) -> f64 {
        let g_u = self.equilibrium_modulus + self.arms.iter().map(|&(e, _)| e).sum::<f64>();
        (g_u / self.density).sqrt()
    }

    /// Dispersive shear-wave phase velocity `c_p(ω) = ω/Re(k)`, `k = ω√(ρ/G*)`
    /// \[m·s⁻¹].
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

    /// Equilibrium (relaxed) modulus `E_∞` \`Pa`.
    #[must_use]
    pub fn equilibrium_modulus(&self) -> f64 {
        self.equilibrium_modulus
    }

    /// Number of relaxation arms `N`.
    #[must_use]
    pub fn num_arms(&self) -> usize {
        self.arms.len()
    }

    /// The relaxation arms `(E_j, τ_j)`.
    #[must_use]
    pub fn arms(&self) -> &[(f64, f64)] {
        &self.arms
    }

    /// Mass density `ρ` \[kg·m⁻³].
    #[must_use]
    pub fn density(&self) -> f64 {
        self.density
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viscoelastic::ZenerModel;

    /// A single Maxwell arm `(E_u − E_r, τ)` over equilibrium `E_r` is exactly the
    /// Zener (standard linear solid) model — verified against [`ZenerModel`].
    #[test]
    fn single_arm_equals_zener() {
        let (g_r, g_u, tau, rho) = (2000.0, 6000.0, 1.0e-3, 1000.0);
        let gm = GeneralizedMaxwellModel::new(g_r, vec![(g_u - g_r, tau)], rho).unwrap();
        let z = ZenerModel::new(g_r, g_u, tau, rho).unwrap();
        for &f in &[1.0, 50.0, 159.155, 500.0, 5000.0] {
            let w = std::f64::consts::TAU * f;
            assert!((gm.complex_modulus(w) - z.complex_modulus(w)).norm() < 1e-9);
            assert!((gm.storage_modulus(w) - z.storage_modulus(w)).abs() < 1e-9);
            assert!((gm.loss_modulus(w) - z.loss_modulus(w)).abs() < 1e-9);
            assert!((gm.phase_velocity(w) - z.phase_velocity(w)).abs() < 1e-9);
            assert!((gm.attenuation(w) - z.attenuation(w)).abs() < 1e-9);
        }
    }

    /// The `power_law` constructor reproduces the target frequency exponent: the
    /// loss modulus `G''(ω)` follows `ω^{y-1}` across the band interior, so the
    /// log-log slope of `G''` equals `y-1` within tolerance (Fung 1993).
    #[test]
    fn power_law_loss_modulus_follows_target_exponent() {
        let y = 1.5_f64; // ⇒ G'' ∝ ω^0.5
        let m = GeneralizedMaxwellModel::power_law(5000.0, 3000.0, 10.0, 10_000.0, 16, y, 1000.0)
            .expect("valid power-law spec");
        assert_eq!(m.num_arms(), 16);

        // Sample G'' over the band interior (avoid the τ-spectrum roll-off edges).
        let (f1, f2) = (60.0_f64, 1500.0_f64);
        let w1 = std::f64::consts::TAU * f1;
        let w2 = std::f64::consts::TAU * f2;
        let g1 = m.loss_modulus(w1);
        let g2 = m.loss_modulus(w2);
        let slope = (g2.ln() - g1.ln()) / (w2.ln() - w1.ln());
        assert!(
            (slope - (y - 1.0)).abs() < 0.1,
            "G'' log-log slope {slope} should be ≈ y-1 = {}",
            y - 1.0
        );

        // Storage rises from E_∞ to E_∞ + Σ E_j; modulus is well ordered.
        assert!((m.storage_modulus(0.0) - 5000.0).abs() < 1.0);
        let g_u = m.storage_modulus(1.0e9);
        assert!(
            (g_u - 8000.0).abs() < 50.0,
            "unrelaxed storage {g_u} ≈ 8000"
        );
        assert!(m.unrelaxed_shear_speed() > m.relaxed_shear_speed());
    }

    /// Complex modulus decomposes as storage + i·loss, and attenuation rises with
    /// frequency over the design band (a lossy, dispersive solid).
    #[test]
    fn complex_modulus_and_dispersion_are_consistent() {
        let m = GeneralizedMaxwellModel::power_law(4000.0, 2000.0, 20.0, 5000.0, 10, 1.3, 1000.0)
            .unwrap();
        for &f in &[40.0, 200.0, 1000.0] {
            let w = std::f64::consts::TAU * f;
            let g = m.complex_modulus(w);
            assert!((g.re - m.storage_modulus(w)).abs() < 1e-9);
            assert!((g.im - m.loss_modulus(w)).abs() < 1e-9);
            assert!(m.attenuation(w) > 0.0);
        }
        // attenuation increases with frequency for α ∝ ω^1.3
        let a_lo = m.attenuation(std::f64::consts::TAU * 50.0);
        let a_hi = m.attenuation(std::f64::consts::TAU * 2000.0);
        assert!(a_hi > a_lo, "attenuation must rise: {a_lo} → {a_hi}");
    }

    #[test]
    fn rejects_unphysical_specs() {
        // No arms.
        assert!(GeneralizedMaxwellModel::new(1000.0, vec![], 1000.0).is_none());
        // Negative arm weight / time.
        assert!(GeneralizedMaxwellModel::new(1000.0, vec![(-1.0, 1e-3)], 1000.0).is_none());
        assert!(GeneralizedMaxwellModel::new(1000.0, vec![(1.0, 0.0)], 1000.0).is_none());
        // Non-positive equilibrium / density.
        assert!(GeneralizedMaxwellModel::new(0.0, vec![(1.0, 1e-3)], 1000.0).is_none());
        // Invalid power-law band.
        assert!(
            GeneralizedMaxwellModel::power_law(1000.0, 1000.0, 100.0, 50.0, 8, 1.5, 1000.0)
                .is_none()
        );
        assert!(
            GeneralizedMaxwellModel::power_law(1000.0, 1000.0, 10.0, 1000.0, 0, 1.5, 1000.0)
                .is_none()
        );
    }
}
