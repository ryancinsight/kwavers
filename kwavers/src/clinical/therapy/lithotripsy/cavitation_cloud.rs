//! Cavitation cloud dynamics for lithotripsy simulation.
//!
//! This module implements bubble cloud formation, growth, and collapse dynamics
//! relevant to shock wave lithotripsy, where cavitation plays a key role in
//! stone erosion and tissue bioeffects.

use crate::core::constants::cavitation::{SURFACE_TENSION_WATER, VISCOSITY_WATER};
use crate::core::constants::fundamental::{ATMOSPHERIC_PRESSURE, DENSITY_WATER_NOMINAL};
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;
use std::f64::consts::PI;

/// Parameters for cavitation cloud dynamics.
#[derive(Debug, Clone)]
pub struct CloudParameters {
    /// Initial bubble radius (m)
    pub initial_bubble_radius: f64,
    /// Bubble number density (#/m³)
    pub bubble_density: f64,
    /// Ambient pressure (Pa)
    pub ambient_pressure: f64,
    /// Surface tension (N/m)
    pub surface_tension: f64,
    /// Viscosity (Pa·s)
    pub viscosity: f64,
    /// Erosion efficiency (kg/J)
    pub erosion_efficiency: f64,
}

impl Default for CloudParameters {
    fn default() -> Self {
        Self {
            initial_bubble_radius: 1e-6, // 1 micron
            bubble_density: 1e12,        // 10^12 bubbles/m³
            ambient_pressure: ATMOSPHERIC_PRESSURE,
            surface_tension: SURFACE_TENSION_WATER,
            viscosity: VISCOSITY_WATER,
            erosion_efficiency: 1e-12, // kg/J (empirical, Sapozhnikov et al. 2002)
        }
    }
}

/// Cavitation cloud dynamics model.
#[derive(Debug, Clone)]
pub struct CavitationCloudDynamics {
    /// Cloud parameters
    parameters: CloudParameters,
    /// Cloud density field (void fraction or bubble count)
    density_field: Array3<f64>,
    /// Total eroded mass accumulated
    accumulated_eroded_mass: f64,
}

impl CavitationCloudDynamics {
    /// Create new cavitation cloud dynamics model with parameters and grid dimensions.
    #[must_use]
    pub fn new(parameters: CloudParameters, dimensions: (usize, usize, usize)) -> Self {
        Self {
            parameters,
            density_field: Array3::zeros(dimensions),
            accumulated_eroded_mass: 0.0,
        }
    }

    /// Get cloud parameters.
    #[must_use]
    pub fn parameters(&self) -> &CloudParameters {
        &self.parameters
    }

    /// Initialize cloud based on geometry and pressure field.
    pub fn initialize_cloud(&mut self, geometry: &Array3<f64>, pressure: &Array3<f64>) {
        if self.density_field.dim() != pressure.dim() {
            self.density_field = Array3::zeros(pressure.dim());
        }
        // Simple init: nucleate bubbles where pressure < threshold (-1 MPa) and near stone
        let threshold = -1e6;
        for ((i, j, k), p) in pressure.indexed_iter() {
            if *p < threshold && geometry[[i, j, k]] < 0.5 {
                // Near stone but not inside?
                self.density_field[[i, j, k]] = self.parameters.bubble_density;
            } else {
                self.density_field[[i, j, k]] = 0.0;
            }
        }
    }

    /// Evolve cloud dynamics for a time step using a pressure field.
    ///
    /// This implements a simplified pressure-driven model:
    /// - Nucleation/growth when pressure drops below a Blake-like threshold.
    /// - Collapse when pressure exceeds ambient.
    /// - Erosion proportional to collapse energy.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn evolve_cloud(
        &mut self,
        dt: f64,
        _time: f64,
        pressure: &Array3<f64>,
    ) -> KwaversResult<()> {
        if pressure.dim() != self.density_field.dim() {
            return Err(KwaversError::InvalidInput(
                "Pressure field dimensions must match cavitation cloud".to_owned(),
            ));
        }

        // Not yet implemented: full compressible cloud dynamics. Absent: Gilmore equation
        // with Mach-number corrections (Gilmore 1952); shock-bubble interactions with
        // Richtmyer-Meshkov instability; Rayleigh-Taylor cloud stability analysis;
        // multi-bubble acoustic coupling and emission back-reaction; and cloud
        // expansion/collapse with inter-phase mass transfer (Brenner et al. 2002, Rev Mod Phys).

        let ambient = self.parameters.ambient_pressure.max(1.0);
        let r0 = self.parameters.initial_bubble_radius.max(1e-12);
        let max_density = self.parameters.bubble_density.max(0.0);

        // Rayleigh collapse time scale: t_c = R0 * sqrt(rho / Delta_p).
        let t_char = r0 * (DENSITY_WATER_NOMINAL / ambient).sqrt();
        let growth_rate = 1.0 / t_char;
        let collapse_rate = 2.0 / t_char;

        let p_crit = ambient - 2.0 * self.parameters.surface_tension / r0;

        for ((i, j, k), density) in self.density_field.indexed_iter_mut() {
            let p = pressure[[i, j, k]];
            let drive = if p < p_crit {
                ((p_crit - p) / p_crit.max(1.0)).clamp(0.0, 1.0)
            } else {
                0.0
            };

            let current_density = *density;
            let growth = drive * (max_density - current_density).max(0.0) * growth_rate;
            let collapse = (1.0 - drive) * current_density * collapse_rate;
            let updated = (growth - collapse)
                .mul_add(dt, current_density)
                .clamp(0.0, max_density)
                .max(0.0);

            let collapse_pressure = (p - ambient).max(0.0);
            let bubble_volume = (4.0 / 3.0) * PI * r0.powi(3);
            let energy_per_bubble = bubble_volume * collapse_pressure;
            let erosion_increment =
                current_density * energy_per_bubble * self.parameters.erosion_efficiency * dt;

            self.accumulated_eroded_mass += erosion_increment.max(0.0);
            *density = updated;
        }

        Ok(())
    }

    /// Get total eroded mass at specific time (time is ignored in this simple stateful model).
    #[must_use]
    pub fn total_eroded_mass(&self, _time: f64) -> f64 {
        self.accumulated_eroded_mass
    }

    /// Get cloud density field.
    #[must_use]
    pub fn cloud_density(&self) -> &Array3<f64> {
        &self.density_field
    }
}

impl Default for CavitationCloudDynamics {
    fn default() -> Self {
        Self::new(CloudParameters::default(), (1, 1, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_growth_under_tension() {
        let params = CloudParameters::default();
        let mut cloud = CavitationCloudDynamics::new(params, (2, 2, 2));
        cloud.density_field.fill(0.0);

        let pressure = Array3::from_elem((2, 2, 2), -2.0e6);
        cloud.evolve_cloud(1e-6, 0.0, &pressure).unwrap();

        assert!(cloud.density_field.sum() > 0.0);
    }

    #[test]
    fn test_cloud_collapse_under_compression() {
        let params = CloudParameters::default();
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 2, 2));
        cloud.density_field.fill(params.bubble_density * 0.5);

        let pressure = Array3::from_elem((2, 2, 2), 2.0e6);
        cloud.evolve_cloud(1e-6, 0.0, &pressure).unwrap();

        assert!(cloud.density_field.sum() < params.bubble_density * 0.5 * 8.0);
    }

    #[test]
    fn test_erosion_accumulates_on_collapse() {
        let params = CloudParameters::default();
        let mut cloud = CavitationCloudDynamics::new(params.clone(), (2, 2, 2));
        cloud.density_field.fill(params.bubble_density * 0.5);

        let pressure = Array3::from_elem((2, 2, 2), 2.0e6);
        cloud.evolve_cloud(1e-6, 0.0, &pressure).unwrap();

        assert!(cloud.total_eroded_mass(0.0) > 0.0);
    }
}
