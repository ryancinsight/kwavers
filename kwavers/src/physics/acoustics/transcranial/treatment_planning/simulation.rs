//! Fast acoustic intensity and bioheat calculations

use super::planner::TreatmentPlanner;
use super::types::{TargetVolume, TransducerSetup};
use crate::core::error::KwaversResult;
use ndarray::Array3;
use num_complex::Complex;

impl TreatmentPlanner {
    /// Simulate acoustic field through skull
    pub(crate) fn simulate_acoustic_field(
        &self,
        setup: &TransducerSetup,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.brain_grid.dimensions();
        let mut acoustic_field = Array3::zeros((nx, ny, nz));

        // Simplified field calculation - would need full wave propagation
        // Compute acoustic field for transcranial therapy planning
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * self.brain_grid.dx;
                    let y = j as f64 * self.brain_grid.dy;
                    let z = k as f64 * self.brain_grid.dz;

                    // Calculate field contribution from each element
                    let mut total_field: f64 = 0.0;

                    for element in &setup.element_positions {
                        let dx = x - element[0];
                        let dy = y - element[1];
                        let dz = z - element[2];
                        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                        if distance > 0.0 {
                            let k = 2.0 * std::f64::consts::PI * setup.frequency / 1500.0;
                            let phase = k * distance;
                            let complex_phase = Complex::new(0.0, phase);
                            total_field += complex_phase.exp().norm_sqr().sqrt() / distance;
                        }
                    }

                    acoustic_field[[i, j, k]] = total_field * total_field; // Intensity
                }
            }
        }

        Ok(acoustic_field)
    }

    /// Calculate thermal response to acoustic field
    pub(crate) fn calculate_thermal_response(
        &self,
        acoustic_field: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = acoustic_field.dim();
        let mut temperature_field = Array3::zeros((nx, ny, nz));

        // Simplified bioheat equation solution
        // T = T0 + (α * I * t) / (ρ * c)
        // where α = absorption coefficient, I = intensity, t = time

        let absorption_coeff = 0.5; // dB/MHz/cm converted to appropriate units
        let perfusion_rate = 0.01; // 1/s (brain perfusion)
        let specific_heat = 3600.0; // J/kg/K
        let density = 1000.0; // kg/m³
        let exposure_time = 10.0; // seconds

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let intensity = acoustic_field[[i, j, k]];
                    let absorbed_power = absorption_coeff * intensity;

                    // Steady-state temperature rise calculation
                    let temp_rise = absorbed_power * exposure_time
                        / (density * specific_heat * (perfusion_rate + absorption_coeff));

                    temperature_field[[i, j, k]] = 37.0 + temp_rise; // Body temp + rise
                }
            }
        }

        Ok(temperature_field)
    }

    /// Estimate treatment time
    pub(crate) fn estimate_treatment_time(
        &self,
        _targets: &[TargetVolume],
        acoustic_field: &Array3<f64>,
    ) -> f64 {
        // Estimate based on required thermal dose
        let thermal_dose_target = 240.0; // CEM43
        let max_intensity = acoustic_field.iter().fold(0.0_f64, |a, &b| a.max(b));

        if max_intensity > 0.0 {
            // Simplified: t = thermal_dose / (absorption_rate * intensity)
            thermal_dose_target / (0.5 * max_intensity)
        } else {
            0.0
        }
    }
}
