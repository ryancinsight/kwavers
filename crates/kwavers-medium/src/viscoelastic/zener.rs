//! Zener / standard-linear-solid viscoelastic model (single relaxation arm).

use super::{recover_complex_modulus, DispersionSample};
use eunomia::Complex64;

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
    pub fn new(
        relaxed_modulus: f64,
        unrelaxed_modulus: f64,
        relaxation_time: f64,
        density: f64,
    ) -> Option<Self> {
        if relaxed_modulus > 0.0
            && unrelaxed_modulus >= relaxed_modulus
            && relaxation_time > 0.0
            && density > 0.0
        {
            Some(Self {
                relaxed_modulus,
                unrelaxed_modulus,
                relaxation_time,
                density,
            })
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
        self.relaxed_modulus
            + (self.unrelaxed_modulus - self.relaxed_modulus) * (wt * wt) / (1.0 + wt * wt)
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

    // Zener tissue: G_r = 2 kPa, G_u = 6 kPa, τ = 1 ms, ρ = 1000.
    fn zener() -> ZenerModel {
        ZenerModel::new(2000.0, 6000.0, 1.0e-3, 1000.0).unwrap()
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
        assert!(
            (g_peak - 2000.0).abs() < 1.0,
            "G''_peak {g_peak} ≈ (G_u−G_r)/2"
        );
        // it is a maximum
        assert!(g_peak > z.loss_modulus(0.1 * w_peak));
        assert!(g_peak > z.loss_modulus(10.0 * w_peak));
    }

    #[test]
    fn zener_dispersion_is_bounded() {
        let z = zener();
        let lo = z.phase_velocity(1.0);
        let hi = z.phase_velocity(1.0e7);
        assert!(
            (lo - z.relaxed_shear_speed()).abs() / lo < 1e-2,
            "low-ω → relaxed speed"
        );
        assert!(
            (hi - z.unrelaxed_shear_speed()).abs() / hi < 1e-2,
            "high-ω → unrelaxed speed"
        );
        // every phase velocity stays within [relaxed, unrelaxed] (bounded, unlike Kelvin–Voigt)
        for &f in &[1.0, 1e2, 1e3, 1e4, 1e5, 1e6] {
            let c = z.phase_velocity(core::f64::consts::TAU * f);
            assert!(
                c >= z.relaxed_shear_speed() - 1e-9 && c <= z.unrelaxed_shear_speed() + 1e-9,
                "c={c}"
            );
        }
    }
}

