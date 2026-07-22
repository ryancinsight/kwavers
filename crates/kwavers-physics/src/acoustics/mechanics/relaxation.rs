//! Multi-relaxation acoustic absorption (Nachman–Smith–Waag / Szabo).
//!
//! A medium with a discrete set of relaxation processes `(τ_l, w_l)` has the
//! frequency-dependent amplitude absorption (book §4.4.2, discrete form of the
//! relaxation-spectrum integral)
//!
//! ```text
//!   α(ω) = (ω² / 2c₀) · Σ_l  w_l τ_l / (1 + ω²τ_l²)   [Np·m⁻¹].
//! ```
//!
//! Limits: at `ωτ ≪ 1` each Lorentzian → `w_l τ_l`, giving `α ∝ ω²` (viscous,
//! exponent 2); at `ωτ ≫ 1` → `w_l/(ω²τ_l)`, so `α → (1/2c₀) Σ w_l/τ_l` (a
//! frequency-independent plateau, exponent 0). A power-law distribution of
//! relaxation times `w(τ) ∝ τ^{y-2}` reproduces `α ∝ ω^y` with `0 < y < 2`
//! (Corollary 4.1) — the physical mechanism the fractional-Laplacian operator of
//! §4.4.3 encodes.
//!
//! This type provides the exact `α(ω)` and its local power-law exponent
//! `d ln α / d ln ω`, used by the spectral solver to realize the
//! `AbsorptionMode::{MultiRelaxation, Causal}` modes through the validated
//! fractional-Laplacian path at the drive frequency.
//!
//! # References
//! - Nachman, A.I., Smith, J.F. & Waag, R.C. (1990). "An equation for acoustic
//!   propagation in inhomogeneous media with relaxation losses." *J. Acoust.
//!   Soc. Am.* 88(3), 1584–1595.
//! - Szabo, T.L. (1995). "Time domain wave equations for lossy media obeying a
//!   frequency power law." *J. Acoust. Soc. Am.* 96(1), 491–500.

/// A discrete multi-relaxation absorption spectrum `(τ_l, w_l)`.
#[derive(Debug, Clone, PartialEq)]
pub struct RelaxationAbsorption {
    /// Relaxation times `τ_l` \`s` (each `> 0`).
    tau: Vec<f64>,
    /// Relaxation weights `w_l` \[s⁻¹] (each `> 0`); set the strength of each process.
    weights: Vec<f64>,
}

impl RelaxationAbsorption {
    /// Build from relaxation times and weights. Returns `None` unless the two
    /// slices are the same non-zero length and every `τ_l > 0`, `w_l > 0`.
    #[must_use]
    pub fn new(tau: Vec<f64>, weights: Vec<f64>) -> Option<Self> {
        if tau.is_empty()
            || tau.len() != weights.len()
            || tau.iter().any(|&t| !t.is_finite() || t <= 0.0)
            || weights.iter().any(|&w| !w.is_finite() || w <= 0.0)
        {
            return None;
        }
        Some(Self { tau, weights })
    }

    /// Build from relaxation times with unit weights (the natural choice when the
    /// overall absorption scale is fixed separately, e.g. the `Causal` mode's
    /// `alpha_0`). Returns `None` for empty or non-positive times.
    #[must_use]
    pub fn unit_weights(tau: Vec<f64>) -> Option<Self> {
        let weights = vec![1.0; tau.len()];
        Self::new(tau, weights)
    }

    /// Relaxation-spectrum sum `S(ω) = Σ_l w_l τ_l / (1 + ω²τ_l²)` \[dimensionless·s⁻¹·s].
    #[must_use]
    fn spectrum(&self, omega: f64) -> f64 {
        self.tau
            .iter()
            .zip(&self.weights)
            .map(|(&t, &w)| {
                let wt = omega * t;
                w * t / (1.0 + wt * wt)
            })
            .sum()
    }

    /// Amplitude absorption `α(ω) = (ω²/2c₀) · S(ω)` \[Np·m⁻¹] at sound speed `c₀`.
    #[must_use]
    pub fn attenuation(&self, omega: f64, c0: f64) -> f64 {
        if omega <= 0.0 || c0 <= 0.0 {
            return 0.0;
        }
        omega * omega / (2.0 * c0) * self.spectrum(omega)
    }

    /// Local power-law exponent `y(ω) = d ln α / d ln ω` — the slope of the
    /// absorption on a log–log plot (independent of `c₀`). Lies in `(0, 2]`:
    /// 2 in the viscous regime (`ωτ ≪ 1`), → 0 in the plateau (`ωτ ≫ 1`).
    ///
    /// With `α = (ω²/2c₀) S(ω)`, `ln α = 2 ln ω + ln S + const`, so
    /// `y = 2 + (ω/S) dS/dω`,
    /// `dS/dω = Σ_l w_l τ_l · (−2ωτ_l²)/(1+ω²τ_l²)²`.
    #[must_use]
    pub fn local_exponent(&self, omega: f64) -> f64 {
        if omega <= 0.0 {
            return 2.0;
        }
        let s = self.spectrum(omega);
        if s <= 0.0 {
            return 2.0;
        }
        let ds_domega: f64 = self
            .tau
            .iter()
            .zip(&self.weights)
            .map(|(&t, &w)| {
                let denom = 1.0 + (omega * t) * (omega * t);
                w * t * (-2.0 * omega * t * t) / (denom * denom)
            })
            .sum();
        2.0 + omega / s * ds_domega
    }
}

#[cfg(test)]
mod tests {
    use super::RelaxationAbsorption;

    /// In the viscous regime (`ωτ ≪ 1`) a single relaxation gives the Stokes
    /// `α = ω²τw/(2c₀)` and a local exponent of 2 (book §4.4.2, Theorem 4.2).
    #[test]
    fn single_relaxation_low_frequency_is_viscous() {
        let tau = 1.0e-9; // 1 ns ⇒ ωτ ≪ 1 across the MHz band
        let w = 5.0e8;
        let r = RelaxationAbsorption::new(vec![tau], vec![w]).unwrap();
        let c0 = 1500.0;
        let omega = std::f64::consts::TAU * 1.0e6; // 1 MHz, ωτ ≈ 6.3e-3
        let expected = omega * omega * tau * w / (2.0 * c0);
        let got = r.attenuation(omega, c0);
        assert!(
            (got - expected).abs() <= 1e-4 * expected,
            "low-freq α {got} vs Stokes {expected}"
        );
        assert!(
            (r.local_exponent(omega) - 2.0).abs() < 1e-3,
            "viscous exponent {} ≈ 2",
            r.local_exponent(omega)
        );
    }

    /// In the plateau regime (`ωτ ≫ 1`) the absorption saturates to
    /// `(1/2c₀) Σ w_l/τ_l` and the local exponent tends to 0.
    #[test]
    fn single_relaxation_high_frequency_plateaus() {
        let tau = 1.0e-6;
        let w = 1.0e6;
        let r = RelaxationAbsorption::new(vec![tau], vec![w]).unwrap();
        let c0 = 1500.0;
        let omega = std::f64::consts::TAU * 1.0e9; // ωτ ≈ 6.3e3 ≫ 1
        let plateau = w / tau / (2.0 * c0);
        assert!(
            (r.attenuation(omega, c0) - plateau).abs() <= 1e-3 * plateau,
            "plateau α {} vs {plateau}",
            r.attenuation(omega, c0)
        );
        assert!(r.local_exponent(omega).abs() < 1e-2, "plateau exponent → 0");
    }

    /// The analytic local exponent matches a centered finite-difference of
    /// `ln α(ln ω)` across the band — the derivative is correct.
    #[test]
    fn local_exponent_matches_numerical_slope() {
        let r = RelaxationAbsorption::new(vec![1.0e-7, 3.0e-7, 1.0e-6], vec![2.0e6, 1.0e6, 5.0e5])
            .unwrap();
        let c0 = 1540.0;
        for &f in &[1.0e5, 5.0e5, 1.0e6, 5.0e6] {
            let omega = std::f64::consts::TAU * f;
            let h = 1.0e-4;
            let a_hi = r.attenuation(omega * (1.0 + h), c0).ln();
            let a_lo = r.attenuation(omega * (1.0 - h), c0).ln();
            let numeric = (a_hi - a_lo) / ((1.0 + h).ln() - (1.0 - h).ln());
            let analytic = r.local_exponent(omega);
            assert!(
                (analytic - numeric).abs() < 1e-3,
                "exponent at {f} Hz: analytic {analytic} vs numeric {numeric}"
            );
            // physical bound
            assert!(analytic > 0.0 && analytic <= 2.0 + 1e-9);
        }
    }

    #[test]
    fn rejects_invalid_spectra() {
        assert!(RelaxationAbsorption::new(vec![], vec![]).is_none());
        assert!(RelaxationAbsorption::new(vec![1e-6], vec![1.0, 2.0]).is_none());
        assert!(RelaxationAbsorption::new(vec![-1e-6], vec![1.0]).is_none());
        assert!(RelaxationAbsorption::new(vec![1e-6], vec![0.0]).is_none());
        assert!(RelaxationAbsorption::unit_weights(vec![1e-6, 2e-6]).is_some());
    }
}
