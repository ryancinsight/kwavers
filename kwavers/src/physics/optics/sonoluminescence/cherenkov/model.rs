use super::emission::{
    COMPRESSION_REFRACTIVE_COEFFICIENT, REFERENCE_TEMPERATURE, THERMAL_REFRACTIVE_COEFFICIENT,
};
use crate::core::constants::fundamental::SPEED_OF_LIGHT;
use ndarray::Array1;
use std::f64::consts::PI;

/// Cherenkov radiation model with refractive index and coherence parameters
///
/// # Mathematical Specification
///
/// ## Theorem: Frank-Tamm Formula (1937)
/// A charged particle moving through a dielectric medium with velocity
/// $v > c/n$ (where $n$ is the refractive index) emits electromagnetic radiation
/// at an angle $\theta$ given by:
///
/// $$ \cos\theta = \frac{1}{n \beta} $$
///
/// where $\beta = v/c$. The spectral energy radiated per unit path length is:
///
/// $$ \frac{dW}{dx\,d\omega} = \frac{q^2}{4\pi} \mu(\omega)\omega\left(1 - \frac{1}{n^2(\omega)\beta^2}\right) $$
///
/// **Invariant:** No radiation is emitted when $v \le c/n$ (sub-threshold).
///
/// # References
/// - Frank, I. M., & Tamm, I. E. (1937). "Coherent visible radiation of fast
///   electrons passing through matter". Doklady Akademii Nauk SSSR, 14(3), 109–114.
#[derive(Debug, Clone)]
pub struct CherenkovModel {
    /// Base refractive index at ambient conditions
    pub refractive_index_base: f64,
    /// Current refractive index (may vary with compression/temperature)
    pub refractive_index: f64,
    /// Coherence enhancement factor (phenomenological)
    pub coherence_factor: f64,
    /// Critical velocity `v_c = c/n`
    pub critical_velocity: f64,
}

impl CherenkovModel {
    /// Create a new Cherenkov model
    #[must_use]
    pub fn new(refractive_index: f64, coherence_factor: f64) -> Self {
        let n = refractive_index.max(1.0);
        let critical = SPEED_OF_LIGHT / n;
        Self {
            refractive_index_base: n,
            refractive_index: n,
            coherence_factor,
            critical_velocity: critical,
        }
    }

    /// Update refractive index based on compression ratio and temperature
    /// - Compression increases `n`: ~2% per unit compression beyond ambient
    /// - High temperature decreases `n`: ~1e-5 per Kelvin above 300 K
    pub fn update_refractive_index(&mut self, compression_ratio: f64, temperature: f64) {
        let comp = compression_ratio.max(0.0);
        let temp = temperature.max(0.0);

        // Empirical model: n(comp, T) = n0 * (1 + coef*(ρ/ρ0 - 1)) - coef*(T - T_ref)
        let increased_n = self.refractive_index_base
            * (1.0 + COMPRESSION_REFRACTIVE_COEFFICIENT * (comp - 1.0));
        let decreased_n =
            increased_n - THERMAL_REFRACTIVE_COEFFICIENT * (temp - REFERENCE_TEMPERATURE);

        self.refractive_index = decreased_n.max(1.0);
        self.critical_velocity = SPEED_OF_LIGHT / self.refractive_index;
    }

    /// Check Cherenkov threshold condition `v > c/n`
    #[must_use]
    pub fn exceeds_threshold(&self, velocity: f64) -> bool {
        velocity > self.critical_velocity
    }

    /// Cherenkov angle `θ = arccos(1/(nβ))` in radians (0..π/2)
    #[must_use]
    pub fn cherenkov_angle(&self, velocity: f64) -> f64 {
        if !self.exceeds_threshold(velocity) {
            return 0.0;
        }
        let beta = (velocity / SPEED_OF_LIGHT).min(1.0);
        let cos_theta = (1.0 / (self.refractive_index * beta)).clamp(-1.0, 1.0);
        cos_theta.acos().min(PI / 2.0)
    }

    /// Frank-Tamm factor: $(1 - 1/(n^2 \beta^2))$
    ///
    /// **SSOT**: This is the single authoritative computation of the
    /// Cherenkov radiation threshold factor, used by `spectral_intensity`,
    /// `total_power_density`, and `emission_spectrum`.
    ///
    /// Returns 0 for sub-threshold velocities.
    #[must_use]
    fn frank_tamm_factor(&self, velocity: f64) -> f64 {
        if !self.exceeds_threshold(velocity) {
            return 0.0;
        }
        let beta = velocity / SPEED_OF_LIGHT;
        (1.0 - 1.0 / (self.refractive_index.powi(2) * beta.powi(2))).max(0.0)
    }

    /// Spectral intensity with inverse frequency dependence
    ///
    /// Frank-Tamm spectral intensity: $I(f) \propto q \cdot (1 - 1/(n^2\beta^2)) / f$
    ///
    /// Returns 0 below threshold.
    #[must_use]
    pub fn spectral_intensity(&self, frequency_hz: f64, velocity: f64, charge: f64) -> f64 {
        if frequency_hz <= 0.0 {
            return 0.0;
        }
        let ft = self.frank_tamm_factor(velocity);
        self.coherence_factor * charge.abs() * ft / frequency_hz
    }

    /// Total power density (phenomenological): scales with charge density and temperature
    #[must_use]
    pub fn total_power_density(&self, velocity: f64, charge_density: f64, temperature: f64) -> f64 {
        let ft = self.frank_tamm_factor(velocity);
        let temp_scale = (temperature / 300.0).sqrt().max(0.0);
        self.coherence_factor * charge_density.max(0.0) * ft * temp_scale
    }

    /// Emission spectrum over wavelengths
    ///
    /// For Cherenkov radiation, $I(\lambda) \propto 1/\lambda$ (peaks at UV/blue).
    ///
    /// Reference: Frank & Tamm (1937)
    #[must_use]
    pub fn emission_spectrum(
        &self,
        velocity: f64,
        charge: f64,
        wavelengths: &Array1<f64>,
    ) -> Array1<f64> {
        let ft = self.frank_tamm_factor(velocity);
        if ft == 0.0 {
            return Array1::zeros(wavelengths.len());
        }
        let scale = self.coherence_factor * charge.abs() * ft;
        wavelengths.mapv(|lambda| {
            if lambda > 0.0 {
                scale / lambda
            } else {
                0.0
            }
        })
    }
}

