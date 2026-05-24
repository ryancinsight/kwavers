//! Pennes Bioheat Equation Implementation
//!
//! Reference: Pennes, H. H. (1948). "Analysis of tissue and arterial blood temperatures
//! in the resting human forearm." Journal of Applied Physiology, 1(2), 93-122.

use crate::core::constants::tissue_acoustics::DENSITY_BLOOD;
use crate::core::constants::medical::BLOOD_SPECIFIC_HEAT;
use crate::core::constants::thermodynamic::BODY_TEMPERATURE_K;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::{Array3, ArrayView3, Zip};

/// Pennes bioheat equation parameters
#[derive(Debug, Clone)]
pub struct BioheatParameters {
    /// Blood perfusion rate [1/s]
    pub perfusion_rate: f64,
    /// Blood density [kg/m³]
    pub blood_density: f64,
    /// Blood specific heat [J/(kg·K)]
    pub blood_specific_heat: f64,
    /// Arterial blood temperature (K)
    pub arterial_temperature: f64,
}

impl Default for BioheatParameters {
    fn default() -> Self {
        Self {
            perfusion_rate: 0.5e-3,
            blood_density: DENSITY_BLOOD,
            blood_specific_heat: BLOOD_SPECIFIC_HEAT,
            arterial_temperature: BODY_TEMPERATURE_K,
        }
    }
}

/// Pennes bioheat equation solver
#[derive(Debug)]
pub struct PennesBioheat {
    params: BioheatParameters,
}

impl PennesBioheat {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(params: BioheatParameters) -> Self {
        Self { params }
    }
    /// Perfusion source.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn perfusion_source(
        &self,
        temperature: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let mut source = Array3::zeros(temperature.raw_dim());

        Zip::indexed(&mut source)
            .and(temperature)
            .par_for_each(|(i, j, k), q, &t| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = crate::domain::medium::density_at(medium, x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);

                let perfusion_coeff = self.params.perfusion_rate
                    * self.params.blood_density
                    * self.params.blood_specific_heat
                    / (rho * cp);

                *q = perfusion_coeff * (self.params.arterial_temperature - t);
            });

        Ok(source)
    }

    /// Update temperature in place without allocating a perfusion field.
    ///
    /// # Contract
    /// The Pennes source term
    /// `ω_b ρ_b c_b (T_a - T) / (ρ c_p)` is point-local. Therefore the update
    /// can compute perfusion inside the same traversal that applies diffusion
    /// and external heating. This preserves the explicit Euler equation while
    /// removing one `Array3<f64>` allocation per bioheat step.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn update(
        &self,
        temperature: &mut Array3<f64>,
        laplacian: &Array3<f64>,
        external_source: Option<ArrayView3<'_, f64>>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        Zip::indexed(temperature)
            .and(laplacian)
            .par_for_each(|(i, j, k), t, &lap| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = crate::domain::medium::density_at(medium, x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);
                let alpha = medium.thermal_diffusivity(x, y, z, grid);
                let perfusion = self.params.perfusion_rate
                    * self.params.blood_density
                    * self.params.blood_specific_heat
                    * (self.params.arterial_temperature - *t)
                    / (rho * cp);
                let ext_source = external_source.as_ref().map_or(0.0, |s| s[[i, j, k]]);

                *t += dt * (alpha.mul_add(lap, perfusion) + ext_source);
            });

        Ok(())
    }
}
