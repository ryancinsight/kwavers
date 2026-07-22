//! Cavitation-nuclei size distribution and resonance mapping.
//!
//! Vascularised soft tissue holds a *population* of stabilised gas nuclei whose
//! equilibrium radii span roughly a decade; the population is well described by
//! a log-normal distribution in radius (Church 1988; Apfel 1984; Fowlkes & Crum
//! 1988). Each radius has a Minnaert resonance `f(R₀) = 1/(2πR₀)·√(3κP₀/ρ)`, so
//! a drive frequency engages the size band whose resonance it matches. The
//! number-weighted fraction of nuclei in a radius (or, via the inverse Minnaert
//! map, a frequency) band is the quantity the swept-engagement model integrates.

use kwavers_math::special::erf;

use super::super::dynamics::minnaert_resonance_hz;

/// Standard-normal CDF `Φ(x) = ½(1 + erf(x/√2))`.
#[inline]
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Log-normal distribution of cavitation-nuclei equilibrium radii.
///
/// Parameterised by the *median* (geometric-mean) radius and the geometric
/// standard deviation `σ_g > 1` (dimensionless multiplicative spread). The
/// number density of nuclei per unit volume scales the absolute count but not
/// the engaged *fraction*, which is what the engagement model needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NucleiSizeDistribution {
    /// Median (geometric-mean) equilibrium radius `m`.
    pub median_radius_m: f64,
    /// Geometric standard deviation `σ_g` (> 1) — multiplicative spread.
    pub geometric_std: f64,
}

impl NucleiSizeDistribution {
    /// Construct a distribution, returning `None` for non-physical parameters.
    #[must_use]
    pub fn new(median_radius_m: f64, geometric_std: f64) -> Option<Self> {
        if !(median_radius_m.is_finite()
            && geometric_std.is_finite()
            && median_radius_m > 0.0
            && geometric_std > 1.0)
        {
            return None;
        }
        Some(Self {
            median_radius_m,
            geometric_std,
        })
    }

    /// Log-standard-deviation `ln σ_g` (the σ of the underlying normal in `ln R`).
    #[inline]
    fn log_sigma(&self) -> f64 {
        self.geometric_std.ln()
    }

    /// Probability density `f(R)` of the radius distribution [1/m].
    ///
    /// `f(R) = 1/(R·ln σ_g·√(2π))·exp(−(ln(R/R_med))² / (2 ln²σ_g))`.
    #[must_use]
    pub fn pdf(&self, radius_m: f64) -> f64 {
        if !(radius_m.is_finite() && radius_m > 0.0) {
            return 0.0;
        }
        let s = self.log_sigma();
        let z = (radius_m / self.median_radius_m).ln() / s;
        (-0.5 * z * z).exp() / (radius_m * s * (2.0 * std::f64::consts::PI).sqrt())
    }

    /// Number-weighted fraction of nuclei with radius in `[r_lo, r_hi]` ∈ [0, 1].
    ///
    /// Closed form via the log-normal CDF `Φ((ln(R/R_med))/ln σ_g)`; order of the
    /// bounds is normalised so the result is always non-negative.
    #[must_use]
    pub fn number_fraction_in_radius_band(&self, r_lo_m: f64, r_hi_m: f64) -> f64 {
        let (lo, hi) = (r_lo_m.min(r_hi_m), r_lo_m.max(r_hi_m));
        if !(lo.is_finite() && hi.is_finite()) || hi <= 0.0 {
            return 0.0;
        }
        let s = self.log_sigma();
        let lo = lo.max(0.0);
        // Φ(−∞)=0 at R=0 (ln→−∞); cap the lower tail there.
        let cdf_lo = if lo <= 0.0 {
            0.0
        } else {
            standard_normal_cdf((lo / self.median_radius_m).ln() / s)
        };
        let cdf_hi = standard_normal_cdf((hi / self.median_radius_m).ln() / s);
        (cdf_hi - cdf_lo).clamp(0.0, 1.0)
    }

    /// Equilibrium radius whose Minnaert resonance equals `freq_hz` `m`.
    ///
    /// Inverts `f = 1/(2πR)·√(3κP₀/ρ)` ⇒ `R = 1/(2πf)·√(3κP₀/ρ)`. Returns 0 for
    /// non-physical inputs.
    #[must_use]
    pub fn resonant_radius_for_frequency(freq_hz: f64, kappa: f64, p0_pa: f64, rho: f64) -> f64 {
        if !(freq_hz.is_finite()
            && kappa.is_finite()
            && p0_pa.is_finite()
            && rho.is_finite()
            && freq_hz > 0.0
            && kappa > 0.0
            && p0_pa > 0.0
            && rho > 0.0)
        {
            return 0.0;
        }
        (3.0 * kappa * p0_pa / rho).sqrt() / (2.0 * std::f64::consts::PI * freq_hz)
    }

    /// Number-weighted fraction of nuclei whose Minnaert resonance lies in the
    /// frequency band `[f_lo, f_hi]` ∈ [0, 1].
    ///
    /// The inverse Minnaert map sends a frequency band to a radius band
    /// (`R ∝ 1/f`, so the ordering reverses); the fraction is then the radius-band
    /// fraction. This is the size population a drive covering `[f_lo, f_hi]`
    /// resonantly engages.
    #[must_use]
    pub fn number_fraction_resonant_in_band(
        &self,
        f_lo_hz: f64,
        f_hi_hz: f64,
        kappa: f64,
        p0_pa: f64,
        rho: f64,
    ) -> f64 {
        let r_a = Self::resonant_radius_for_frequency(f_lo_hz, kappa, p0_pa, rho);
        let r_b = Self::resonant_radius_for_frequency(f_hi_hz, kappa, p0_pa, rho);
        if r_a <= 0.0 && r_b <= 0.0 {
            return 0.0;
        }
        self.number_fraction_in_radius_band(r_a, r_b)
    }

    /// Minnaert resonance frequency of the median nucleus `Hz`.
    #[must_use]
    pub fn median_resonance_hz(&self, kappa: f64, p0_pa: f64, rho: f64) -> f64 {
        minnaert_resonance_hz(self.median_radius_m, kappa, p0_pa, rho)
    }

    /// `n` geometric-spaced radii spanning `±n_sigma` log-standard-deviations
    /// about the median, with the trapezoidal number-weight of each sample.
    ///
    /// Returns `(radii, weights)` where `weights` sum to the covered probability
    /// mass — used to numerically integrate a per-size response (e.g. the
    /// chirped-KM expansion ratio) over the population.
    #[must_use]
    pub fn sample_radii(&self, n: usize, n_sigma: f64) -> (Vec<f64>, Vec<f64>) {
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let s = self.log_sigma();
        let ln_med = self.median_radius_m.ln();
        let ln_lo = ln_med - n_sigma * s;
        let ln_hi = ln_med + n_sigma * s;
        let radii: Vec<f64> = (0..n)
            .map(|i| {
                let frac = if n == 1 {
                    0.5
                } else {
                    i as f64 / (n - 1) as f64
                };
                (ln_lo + frac * (ln_hi - ln_lo)).exp()
            })
            .collect();
        // Trapezoidal number-weight: fraction of the population each sample
        // represents, via the CDF over the mid-points between samples.
        let weights: Vec<f64> = (0..n)
            .map(|i| {
                let lo = if i == 0 {
                    radii[0]
                } else {
                    (radii[i - 1] * radii[i]).sqrt()
                };
                let hi = if i + 1 == n {
                    radii[n - 1]
                } else {
                    (radii[i] * radii[i + 1]).sqrt()
                };
                self.number_fraction_in_radius_band(lo, hi)
            })
            .collect();
        (radii, weights)
    }
}
