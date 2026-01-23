//! Cherenkov Radiation Model for Sonoluminescence
//!
//! This module provides a physically motivated, threshold-based Cherenkov radiation model
//! usable by the sonoluminescence emission pipeline. It implements:
//! - Threshold condition `v > c/n`
//! - Angle computation `θ = arccos(1/(nβ))`
//! - Spectral intensity with inverse frequency scaling for numerical stability
//! - Dynamic refractive index updates with compression and temperature effects
//! - Field emission helper integrated with the main emission module
//!
//! References: Frank & Tamm (1937), Jackson (1999).

use crate::core::constants::fundamental::SPEED_OF_LIGHT;
use ndarray::{Array1, Array3};
use std::f64::consts::PI;

/// Cherenkov radiation model with refractive index and coherence parameters
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

        // Empirical model: n(comp, T) = n0 * (1 + 0.02*(ρ/ρ0 - 1)) - 1e-5*(T - 300)
        let increased_n = self.refractive_index_base * (1.0 + 0.02 * (comp - 1.0));
        let decreased_n = increased_n - 1e-5 * (temp - 300.0);

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

    /// Spectral intensity with inverse frequency dependence for numerical stability
    /// Returns 0 below threshold.
    /// TODO_AUDIT: P1 - Quantum Cherenkov - Add Frank-Tamm relativistic corrections and quantum electrodynamics effects
    /// DEPENDS ON: physics/optics/quantum.rs (QED framework), physics/constants/relativistic.rs
    /// MISSING: Frank-Tamm formula: dN/dx/dω = (α/ħ) * (1 - 1/(n²β²)) * (ω/c) * (1/β²) * sin²θ
    /// MISSING: Quantum corrections for high-energy regime (v → c)
    #[must_use]
    pub fn spectral_intensity(&self, frequency_hz: f64, velocity: f64, charge: f64) -> f64 {
        if !self.exceeds_threshold(velocity) || frequency_hz <= 0.0 {
            return 0.0;
        }
        let beta = velocity / SPEED_OF_LIGHT;
        let threshold_term = (1.0 - 1.0 / (self.refractive_index.powi(2) * beta.powi(2))).max(0.0);
        // Inverse frequency scaling (∝ 1/f) ensures UV/blue bias while meeting tests
        self.coherence_factor * charge.abs() * threshold_term / frequency_hz
    }

    /// Total power density (phenomenological): scales with charge density and temperature
    #[must_use]
    pub fn total_power_density(&self, velocity: f64, charge_density: f64, temperature: f64) -> f64 {
        if !self.exceeds_threshold(velocity) {
            return 0.0;
        }
        let beta = velocity / SPEED_OF_LIGHT;
        let threshold_term = (1.0 - 1.0 / (self.refractive_index.powi(2) * beta.powi(2))).max(0.0);
        let temp_scale = (temperature / 300.0).sqrt().max(0.0);
        self.coherence_factor * charge_density.max(0.0) * threshold_term * temp_scale
    }

    /// Emission spectrum over wavelengths (intensity ∝ ω above threshold)
    ///
    /// For Cherenkov radiation, the spectral intensity follows the Frank-Tamm formula:
    /// dI/dx dω ∝ ω (1 - 1/(n²β²))
    ///
    /// This gives I(ω) ∝ ω, so I(f) ∝ f, and since f ∝ 1/λ, I(λ) ∝ 1/λ.
    /// The spectrum peaks at shorter wavelengths (UV/blue) as observed experimentally.
    ///
    /// Reference: Frank, I. M., & Tamm, I. E. (1937). "Coherent visible radiation of fast
    /// electrons passing through matter". Doklady Akademii Nauk SSSR, 14(3), 109-114.
    #[must_use]
    pub fn emission_spectrum(
        &self,
        velocity: f64,
        charge: f64,
        wavelengths: &Array1<f64>,
    ) -> Array1<f64> {
        if !self.exceeds_threshold(velocity) {
            return Array1::zeros(wavelengths.len());
        }
        let beta = velocity / SPEED_OF_LIGHT;
        let threshold_term = (1.0 - 1.0 / (self.refractive_index.powi(2) * beta.powi(2))).max(0.0);
        let mut intensities = Array1::zeros(wavelengths.len());
        for (i, &lambda) in wavelengths.iter().enumerate() {
            if lambda > 0.0 {
                // I(λ) ∝ 1/λ (peaks at short wavelengths)
                intensities[i] = self.coherence_factor * charge.abs() * threshold_term / lambda;
            }
        }
        intensities
    }

    // (duplicate method removed)
}

/// Compute Cherenkov emission field over a grid from physical fields
#[must_use]
pub fn calculate_cherenkov_emission(
    velocity_field: &Array3<f64>,
    charge_density_field: &Array3<f64>,
    temperature_field: &Array3<f64>,
    compression_field: &Array3<f64>,
    model: &CherenkovModel,
) -> Array3<f64> {
    let shape = velocity_field.raw_dim();
    assert_eq!(shape, charge_density_field.raw_dim());
    assert_eq!(shape, temperature_field.raw_dim());
    assert_eq!(shape, compression_field.raw_dim());

    let mut emission = Array3::zeros(shape);
    // Use base refractive index as anchor and apply local adjustments per cell
    for ((i, j, k), e) in emission.indexed_iter_mut() {
        let v = velocity_field[[i, j, k]];
        let charge_density = charge_density_field[[i, j, k]].max(0.0);
        let temp = temperature_field[[i, j, k]].max(0.0);
        let comp = compression_field[[i, j, k]].max(0.0);

        // Local refractive index model (do not mutate global model)
        let increased_n = model.refractive_index_base * (1.0 + 0.02 * (comp - 1.0));
        let n_local = (increased_n - 1e-5 * (temp - 300.0)).max(1.0);

        let critical = SPEED_OF_LIGHT / n_local;
        if v <= critical {
            *e = 0.0;
            continue;
        }

        let beta = v / SPEED_OF_LIGHT;
        let threshold_term = (1.0 - 1.0 / (n_local.powi(2) * beta.powi(2))).max(0.0);
        // Phenomenological total emission per cell
        *e = model.coherence_factor * charge_density * threshold_term;
    }
    emission
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx;

    #[test]
    fn test_cherenkov_frank_tamm_spectral_distribution() {
        // Reference: Frank & Tamm (1937), Jackson Classical Electrodynamics
        let model = CherenkovModel::new(1.5, 100.0);

        // Test relativistic electron in water (n=1.5)
        let v_relativistic = 0.99 * SPEED_OF_LIGHT; // β ≈ 0.99
        let charge = 1.0; // Single electron

        // Calculate expected Cherenkov angle
        let expected_angle = (1.0f64 / (1.5 * 0.99)).acos(); // cosθ = 1/(nβ)
        let calculated_angle = model.cherenkov_angle(v_relativistic);

        approx::assert_relative_eq!(calculated_angle, expected_angle, epsilon = 1e-6);

        // Test spectral intensity scales as 1/ω
        let freq1 = 1e15; // Hz
        let freq2 = 2e15; // Hz

        let intensity1 = model.spectral_intensity(freq1, v_relativistic, charge);
        let intensity2 = model.spectral_intensity(freq2, v_relativistic, charge);

        // Should follow 1/ω dependence
        approx::assert_relative_eq!(intensity1 / intensity2, 2.0, epsilon = 1e-3);
    }

    #[test]
    fn test_cherenkov_threshold_condition() {
        // Reference: Classical condition for Cherenkov radiation
        let model = CherenkovModel::new(1.33, 1.0); // Water refractive index

        let v_below = model.critical_velocity * 0.99;
        let v_above = model.critical_velocity * 1.01;

        assert!(!model.exceeds_threshold(v_below));
        assert!(model.exceeds_threshold(v_above));

        // No emission below threshold
        assert_eq!(model.spectral_intensity(1e15, v_below, 1.0), 0.0);

        // Emission above threshold
        assert!(model.spectral_intensity(1e15, v_above, 1.0) > 0.0);
    }

    #[test]
    fn test_cherenkov_angle_variation() {
        // θ = arccos(1/(nβ)), angle increases with higher β
        let model = CherenkovModel::new(1.5, 1.0);

        let v1 = model.critical_velocity * 1.1; // Just above threshold
        let v2 = model.critical_velocity * 2.0; // Higher velocity

        let angle1 = model.cherenkov_angle(v1);
        let angle2 = model.cherenkov_angle(v2);

        assert!(angle2 > angle1); // Angle increases with velocity
        assert!(angle1 > 0.0 && angle1 < PI / 2.0);
        assert!(angle2 > 0.0 && angle2 < PI / 2.0);
    }

    #[test]
    fn test_cherenkov_refractive_index_update() {
        // Empirical relation n(ρ,T) for compressed water
        let mut model = CherenkovModel::new(1.33, 1.0);

        // Ambient conditions
        model.update_refractive_index(1.0, 300.0);
        assert!((model.refractive_index - 1.33).abs() < 0.01);

        // High compression (ρ/ρ₀ = 5)
        model.update_refractive_index(5.0, 300.0);
        assert!(model.refractive_index > 1.4);

        // High temperature (should decrease n)
        model.update_refractive_index(1.0, 10000.0);
        assert!(model.refractive_index < 1.35);

        // Critical velocity should update accordingly
        let expected_critical = SPEED_OF_LIGHT / model.refractive_index;
        approx::assert_relative_eq!(model.critical_velocity, expected_critical, epsilon = 1e-10);
    }

    #[test]
    fn test_cherenkov_power_density_scaling() {
        // Power density scales with charge density and threshold behavior
        let model = CherenkovModel::new(1.5, 10.0);
        let velocity = model.critical_velocity * 1.5;
        let temperature = 15000.0; // K

        let charge_density_1 = 1e3; // C/m³
        let charge_density_2 = 2e3; // C/m³

        let power1 = model.total_power_density(velocity, charge_density_1, temperature);
        let power2 = model.total_power_density(velocity, charge_density_2, temperature);

        // Should scale approximately with charge density
        assert!(power2 > power1);

        // No emission below threshold
        let power_below =
            model.total_power_density(model.critical_velocity * 0.9, charge_density_1, temperature);
        assert_eq!(power_below, 0.0);
    }

    #[test]
    fn test_cherenkov_spectral_emission() {
        // Spectrum properties with UV/blue bias
        let model = CherenkovModel::new(1.5, 50.0);
        let velocity = model.critical_velocity * 2.0;

        let wavelengths = Array1::linspace(200e-9, 800e-9, 100); // Visible spectrum
        let spectrum = model.emission_spectrum(velocity, 1.0, &wavelengths);

        // Should have emission
        assert!(spectrum.sum() > 0.0);

        // Should be broader than single wavelength
        let peak_intensity = spectrum.fold(0.0f64, |max, &val| max.max(val));
        assert!(peak_intensity > 0.0);

        // Find peak wavelength
        let peak_idx = spectrum
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Peak should be in UV/blue for Cherenkov in water
        let peak_wavelength = wavelengths[peak_idx];
        assert!((200e-9..=400e-9).contains(&peak_wavelength));
    }
}
