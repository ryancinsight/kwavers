//! Pennes Bioheat Equation Implementation
//!
//! Reference: Pennes, H. H. (1948). "Analysis of tissue and arterial blood temperatures
//! in the resting human forearm." Journal of Applied Physiology, 1(2), 93-122.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use ndarray::{Array3, Zip};

/// Pennes bioheat equation parameters
#[derive(Debug, Clone)]
pub struct BioheatParameters {
    /// Blood perfusion rate [1/s]
    pub perfusion_rate: f64,
    /// Blood density [kg/m³]
    pub blood_density: f64,
    /// Blood specific heat [J/(kg·K)]
    pub blood_specific_heat: f64,
    /// Arterial blood temperature [K]
    pub arterial_temperature: f64,
}

impl Default for BioheatParameters {
    fn default() -> Self {
        Self {
            perfusion_rate: 0.5e-3,      // 0.5 mL/g/min typical tissue
            blood_density: 1050.0,       // kg/m³
            blood_specific_heat: 3840.0, // J/(kg·K)
            arterial_temperature: 310.15, // 37°C in Kelvin
        }
    }
}

/// Pennes bioheat equation solver
#[derive(Debug)]
pub struct PennesBioheat {
    params: BioheatParameters,
}

impl PennesBioheat {
    pub fn new(params: BioheatParameters) -> Self {
        Self { params }
    }

    /// Calculate the perfusion heat source term
    /// Q_perfusion = ω_b * ρ_b * c_b * (T_a - T)
    pub fn perfusion_source(
        &self,
        temperature: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let mut source = Array3::zeros(temperature.raw_dim());

        // Calculate perfusion term at each point
        Zip::indexed(&mut source)
            .and(temperature)
            .for_each(|(i, j, k), q, &t| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                // Get local tissue properties
                let rho = medium.density(x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);

                // Perfusion heat source
                let perfusion_coeff = self.params.perfusion_rate
                    * self.params.blood_density
                    * self.params.blood_specific_heat
                    / (rho * cp);

                *q = perfusion_coeff * (self.params.arterial_temperature - t);
            });

        Ok(source)
    }

    /// Update temperature using Pennes bioheat equation
    /// ∂T/∂t = α∇²T + Q_perfusion/(ρc) + Q_source/(ρc)
    pub fn update(
        &self,
        temperature: &mut Array3<f64>,
        laplacian: &Array3<f64>,
        external_source: Option<&Array3<f64>>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        // Get perfusion source
        let perfusion = self.perfusion_source(temperature, medium, grid)?;

        // Update temperature
        Zip::indexed(temperature)
            .and(laplacian)
            .and(&perfusion)
            .for_each(|(i, j, k), t, &lap, &perf| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                // Get thermal diffusivity
                let alpha = medium.thermal_diffusivity(x, y, z, grid);

                // Add external source if provided
                let ext_source = external_source.map(|s| s[[i, j, k]]).unwrap_or(0.0);

                // Update using forward Euler
                *t += dt * (alpha * lap + perf + ext_source);
            });

        Ok(())
    }
}
