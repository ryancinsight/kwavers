//! Blood perfusion models for bioheat transfer
//!
//! References:
//! - Kolios et al. (2003) "Blood flow cooling and ultrasonic lesion formation"
//! - Curra et al. (2000) "Numerical simulations of heating patterns"

use ndarray::Array3;

/// Temperature-dependent perfusion model
pub struct PerfusionModel {
    /// Baseline perfusion rate (kg/m³/s)
    w_b0: f64,
    /// Temperature threshold for perfusion shutdown (°C)
    T_shutdown: f64,
    /// Temperature for maximum perfusion (°C)
    T_max: f64,
    /// Maximum perfusion multiplier
    max_multiplier: f64,
}

impl PerfusionModel {
    /// Create new perfusion model
    pub fn new(baseline_perfusion: f64) -> Self {
        Self {
            w_b0: baseline_perfusion,
            T_shutdown: 50.0,    // Perfusion stops above 50°C
            T_max: 42.0,         // Maximum perfusion at mild hyperthermia
            max_multiplier: 2.0, // Double perfusion at peak
        }
    }

    /// Calculate perfusion rate based on temperature
    pub fn perfusion_rate(&self, temperature: f64) -> f64 {
        if temperature > self.T_shutdown {
            // Perfusion shutdown due to vascular damage
            0.0
        } else if temperature > self.T_max {
            // Linear decrease from max to shutdown
            let fraction = (self.T_shutdown - temperature) / (self.T_shutdown - self.T_max);
            self.w_b0 * self.max_multiplier * fraction
        } else if temperature > 37.0 {
            // Linear increase from baseline to max
            let fraction = (temperature - 37.0) / (self.T_max - 37.0);
            self.w_b0 * (1.0 + (self.max_multiplier - 1.0) * fraction)
        } else {
            // Below body temperature
            self.w_b0
        }
    }

    /// Update perfusion field based on temperature field
    pub fn update_perfusion_field(&self, temperature: &Array3<f64>) -> Array3<f64> {
        temperature.mapv(|t| self.perfusion_rate(t))
    }

    /// Check if perfusion is shut down
    pub fn is_shutdown(&self, temperature: f64) -> bool {
        temperature > self.T_shutdown
    }
}

/// Vessel cooling model for large blood vessels
pub struct VesselCooling {
    /// Vessel locations (i, j, k, radius)
    vessels: Vec<(usize, usize, usize, f64)>,
    /// Blood flow velocity (m/s)
    velocity: f64,
    /// Blood temperature (°C)
    blood_temp: f64,
}

impl VesselCooling {
    /// Create new vessel cooling model
    pub fn new() -> Self {
        Self {
            vessels: Vec::new(),
            velocity: 0.1, // 10 cm/s typical
            blood_temp: 37.0,
        }
    }

    /// Add a vessel
    pub fn add_vessel(&mut self, i: usize, j: usize, k: usize, radius: f64) {
        self.vessels.push((i, j, k, radius));
    }

    /// Calculate cooling effect at a point
    pub fn cooling_rate(&self, i: usize, j: usize, k: usize, dx: f64, temperature: f64) -> f64 {
        let mut total_cooling = 0.0;

        for &(vi, vj, vk, radius) in &self.vessels {
            let distance = (((i as f64 - vi as f64) * dx).powi(2)
                + ((j as f64 - vj as f64) * dx).powi(2)
                + ((k as f64 - vk as f64) * dx).powi(2))
            .sqrt();

            if distance < radius {
                // Inside vessel - strong cooling
                let h = 1000.0; // Heat transfer coefficient (W/m²/K)
                                // Positive cooling when tissue is hotter than blood
                total_cooling += h * (temperature - self.blood_temp).abs();
            } else if distance < 2.0 * radius {
                // Near vessel - moderate cooling
                let h = 100.0 * (2.0 - distance / radius);
                // Positive cooling when tissue is hotter than blood
                total_cooling += h * (temperature - self.blood_temp).abs();
            }
        }

        total_cooling
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfusion_temperature_dependence() {
        let model = PerfusionModel::new(1.0);

        // Normal temperature
        assert_eq!(model.perfusion_rate(37.0), 1.0);

        // Mild hyperthermia - increased perfusion
        let rate_42 = model.perfusion_rate(42.0);
        assert!(rate_42 > 1.5 && rate_42 <= 2.0);

        // High temperature - shutdown
        assert_eq!(model.perfusion_rate(55.0), 0.0);
    }

    #[test]
    fn test_vessel_cooling() {
        let mut vessel_model = VesselCooling::new();
        vessel_model.add_vessel(5, 5, 5, 2.0);

        // At vessel center
        let cooling_center = vessel_model.cooling_rate(5, 5, 5, 1.0, 45.0);
        assert!(cooling_center > 0.0);

        // Far from vessel
        let cooling_far = vessel_model.cooling_rate(20, 20, 20, 1.0, 45.0);
        assert_eq!(cooling_far, 0.0);
    }
}
