//! Unified ultrasound therapy physics module
//!
//! This module consolidates all therapeutic ultrasound modalities including:
//! - HIFU (High-Intensity Focused Ultrasound)
//! - LIFU (Low-Intensity Focused Ultrasound)
//! - Histotripsy (mechanical tissue ablation)
//! - BBB (Blood-Brain Barrier) opening
//! - Sonodynamic therapy
//! - Sonoporation
//! - Microbubble-mediated therapies
//!
//! # Literature References
//!
//! 1. **ter Haar, G. (2016)**. "HIFU tissue ablation: concept and devices."
//!    *Advances in Experimental Medicine and Biology*, 880, 3-20.
//!    DOI: 10.1007/978-3-319-22536-4_1
//!
//! 2. **Khokhlova, V. A., et al. (2015)**. "Histotripsy methods in mechanical
//!    disintegration of tissue: towards clinical applications." *International
//!    Journal of Hyperthermia*, 31(2), 145-162.
//!
//! 3. **Hynynen, K., et al. (2001)**. "Noninvasive MR imaging-guided focal
//!    opening of the blood-brain barrier in rabbits." *Radiology*, 220(3), 640-646.
//!
//! 4. **McHale, A. P., et al. (2016)**. "Sonodynamic therapy: concept, mechanism
//!    and application to cancer treatment." *Advances in Experimental Medicine
//!    and Biology*, 880, 429-450.
//!
//! 5. **Bader, K. B., & Holland, C. K. (2013)**. "Gauging the likelihood of stable
//!    cavitation from ultrasound contrast agents." *Physics in Medicine & Biology*,
//!    58(1), 127-144.
//!
//! ## Design Principles
//! - **SOLID**: Single responsibility per module
//! - **GRASP**: Modular organization under 200 lines per file
//! - **CUPID**: Composable therapy components
//! - **Zero-Cost**: Efficient abstractions

pub mod cavitation;
pub mod metrics;
pub mod modalities;
pub mod parameters;

// Re-export main types for convenience
pub use cavitation::{CavitationDetectionMethod, TherapyCavitationDetector};
pub use metrics::TreatmentMetrics;
pub use modalities::{TherapyMechanism, TherapyModality};
pub use parameters::TherapyParameters;

use crate::{
    error::KwaversResult,
    grid::Grid,
    medium::Medium,
    physics::thermal::{HeatSource, ThermalCalculator, ThermalConfig},
};
use ndarray::{Array3, Zip};
use std::sync::Arc;

/// Main therapy calculator
pub struct TherapyCalculator {
    /// Treatment modality
    pub modality: TherapyModality,
    /// Treatment parameters
    pub parameters: TherapyParameters,
    /// Thermal calculator (optional)
    pub thermal: Option<ThermalCalculator>,
    /// Cavitation detector (optional)
    pub cavitation: Option<TherapyCavitationDetector>,
    /// Treatment metrics
    pub metrics: TreatmentMetrics,
    /// Grid reference
    grid_shape: (usize, usize, usize),
}

impl TherapyCalculator {
    /// Create a new therapy calculator
    pub fn new(modality: TherapyModality, parameters: TherapyParameters, grid: &Grid) -> Self {
        // Initialize components based on modality
        let thermal = if modality.has_thermal_effects() {
            let config = ThermalConfig {
                use_bioheat: true,
                bioheat: true,
                blood_temperature: 310.15,
                blood_perfusion: 0.5e-3,
                perfusion_rate: 0.5e-3,
                blood_specific_heat: 3617.0,
                thermal_diffusivity: 1.4e-7,
                hyperbolic: false,
                relaxation_time: 20.0,
                reference_temperature: 316.15, // 43°C
            };
            Some(ThermalCalculator::new(grid, 310.15).with_config(config))
        } else {
            None
        };

        let cavitation = if modality.has_cavitation() {
            Some(TherapyCavitationDetector::new(
                parameters.frequency,
                parameters.peak_negative_pressure,
            ))
        } else {
            None
        };

        Self {
            modality,
            parameters,
            thermal,
            cavitation,
            metrics: TreatmentMetrics::default(),
            grid_shape: (grid.nx, grid.ny, grid.nz),
        }
    }

    /// Calculate therapy effects
    pub fn calculate(
        &mut self,
        pressure: &Array3<f64>,
        temperature: &mut Array3<f64>,
        dt: f64,
        medium: &Arc<dyn Medium>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        // Calculate thermal effects if applicable
        if let Some(ref mut thermal_calc) = self.thermal {
            // Calculate heat source from acoustic absorption
            let heat_source = self.calculate_heat_source(pressure, medium, grid)?;

            // Update temperature
            thermal_calc.step(temperature, &heat_source, dt, medium, grid)?;

            // Update thermal dose
            self.metrics.thermal_dose += TreatmentMetrics::calculate_thermal_dose(temperature, dt);
            self.metrics.update_peak_temperature(temperature);
        }

        // Detect and track cavitation if applicable
        if let Some(ref cavitation_detector) = self.cavitation {
            let cavitation_field = cavitation_detector.detect(pressure);
            self.metrics.cavitation_dose +=
                TreatmentMetrics::calculate_cavitation_dose(&cavitation_field, dt);
        }

        // Update safety and efficiency metrics
        self.metrics.calculate_safety_index();
        self.metrics.calculate_efficiency(self.get_target_dose());

        Ok(())
    }

    /// Calculate heat source from acoustic absorption
    fn calculate_heat_source(
        &self,
        pressure: &Array3<f64>,
        medium: &Arc<dyn Medium>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let mut heat_source = Array3::zeros(self.grid_shape);

        // Q = 2 * α * I, where I = p²/(2*ρ*c)
        Zip::indexed(&mut heat_source)
            .and(pressure)
            .for_each(|(i, j, k), q, &p| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = medium.density(x, y, z, grid);
                let c = medium.sound_speed(x, y, z, grid);
                let alpha = medium.absorption_coefficient(x, y, z, grid, self.parameters.frequency);

                // Acoustic intensity
                let intensity = p * p / (2.0 * rho * c);

                // Heat generation rate [W/m³]
                *q = 2.0 * alpha * intensity;
            });

        Ok(heat_source)
    }

    /// Get target dose based on modality
    fn get_target_dose(&self) -> f64 {
        match self.modality {
            TherapyModality::HIFU => 240.0,      // 240 CEM43 for ablation
            TherapyModality::LIFU => 0.0,        // No thermal goal
            TherapyModality::Histotripsy => 0.0, // Mechanical disruption
            TherapyModality::BBBOpening => 0.0,  // Mechanical opening
            _ => 10.0,                           // Default mild hyperthermia
        }
    }

    /// Check if treatment is complete
    pub fn is_complete(&self) -> bool {
        self.metrics.is_successful(self.get_target_dose(), 0.8)
    }

    /// Get treatment summary
    pub fn summary(&self) -> String {
        format!(
            "Therapy: {:?}\n\
             Frequency: {:.1} MHz\n\
             Pressure: {:.1} MPa\n\
             Duration: {:.1} s\n\
             {}",
            self.modality,
            self.parameters.frequency / 1e6,
            self.parameters.peak_negative_pressure / 1e6,
            self.parameters.treatment_duration,
            self.metrics.summary()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_therapy_modality() {
        let modality = TherapyModality::HIFU;
        assert_eq!(modality.primary_mechanism(), TherapyMechanism::Thermal);
        assert!(modality.has_thermal_effects());
        assert!(!modality.has_cavitation());
    }

    #[test]
    fn test_therapy_parameters() {
        let mut params = TherapyParameters::hifu();
        params.calculate_mechanical_index();
        assert!(params.mechanical_index > 0.0);
        assert!(params.validate_safety());
    }

    #[test]
    fn test_cavitation_detector() {
        let detector = TherapyCavitationDetector::new(1e6, 1e6);
        assert!(detector.blake_threshold > 0.0);

        let ci = detector.cavitation_index(2e6);
        assert!(ci > 0.0);

        let prob = detector.cavitation_probability(2e6);
        assert!(prob >= 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_treatment_metrics() {
        let mut metrics = TreatmentMetrics::default();
        assert_eq!(metrics.thermal_dose, 0.0);
        assert_eq!(metrics.safety_index, 1.0);

        metrics.thermal_dose = 240.0;
        metrics.calculate_efficiency(240.0);
        assert!(metrics.efficiency > 0.0);
    }
}
