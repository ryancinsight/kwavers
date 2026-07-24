//! Pennes Bioheat Equation Implementation
//!
//! Reference: Pennes, H. H. (1948). "Analysis of tissue and arterial blood temperatures
//! in the resting human forearm." Journal of Applied Physiology, 1(2), 93-122.

use aequitas::systems::si::quantities::{
    MassDensity, ReciprocalTime, SpecificHeatCapacity, ThermodynamicTemperature,
};
use kwavers_core::constants::medical::{BLOOD_SPECIFIC_HEAT, TISSUE_PERFUSION_RATE};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;
use kwavers_core::constants::tissue_acoustics::DENSITY_BLOOD;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::{Array3, ArrayView3};

/// Pennes bioheat equation parameters
#[derive(Debug, Clone)]
pub struct BioheatParameters {
    /// Blood perfusion rate [1/s].
    pub perfusion_rate: ReciprocalTime<f64>,
    /// Blood density [kg/m³].
    pub blood_density: MassDensity<f64>,
    /// Blood specific heat [J/(kg·K)].
    pub blood_specific_heat: SpecificHeatCapacity<f64>,
    /// Arterial blood temperature [K].
    pub arterial_temperature: ThermodynamicTemperature<f64>,
}

impl Default for BioheatParameters {
    fn default() -> Self {
        Self {
            // TISSUE_PERFUSION_RATE = 5×10⁻⁴ 1/s — generic soft tissue value
            // (Pennes 1948; Duck 1990). See `kwavers_core::constants::medical`.
            perfusion_rate: ReciprocalTime::from_base(TISSUE_PERFUSION_RATE),
            blood_density: MassDensity::from_base(DENSITY_BLOOD),
            blood_specific_heat: SpecificHeatCapacity::from_base(BLOOD_SPECIFIC_HEAT),
            arterial_temperature: ThermodynamicTemperature::from_base(BODY_TEMPERATURE_K),
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
        let mut source = Array3::zeros(temperature.shape());

        crate::parallel::for_each_indexed_pair_mut(
            source.view_mut(),
            temperature.view(),
            |(i, j, k), q, &t| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = kwavers_medium::density_at(medium, x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);

                let perfusion_coeff = (self.params.perfusion_rate
                    * self.params.blood_density
                    * self.params.blood_specific_heat
                    / (MassDensity::from_base(rho) * SpecificHeatCapacity::from_base(cp)))
                .into_base();

                *q = perfusion_coeff * (self.params.arterial_temperature.into_base() - t);
            },
        );

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
    /// - Propagates any `KwaversError` returned by called functions.
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
        crate::parallel::for_each_indexed_pair_mut(
            temperature.view_mut(),
            laplacian.view(),
            |(i, j, k), t, &lap| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = kwavers_medium::density_at(medium, x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);
                let alpha = medium.thermal_diffusivity(x, y, z, grid);
                let perfusion_coeff = (self.params.perfusion_rate
                    * self.params.blood_density
                    * self.params.blood_specific_heat
                    / (MassDensity::from_base(rho) * SpecificHeatCapacity::from_base(cp)))
                .into_base();
                let perfusion =
                    perfusion_coeff * (self.params.arterial_temperature.into_base() - *t);
                let ext_source = external_source.as_ref().map_or(0.0, |s| s[[i, j, k]]);

                *t += dt * (alpha.mul_add(lap, perfusion) + ext_source);
            },
        );

        Ok(())
    }
}
