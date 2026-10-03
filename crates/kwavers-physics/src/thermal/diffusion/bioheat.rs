//! Pennes Bioheat Equation Implementation
//!
//! Reference: Pennes, H. H. (1948). "Analysis of tissue and arterial blood temperatures
//! in the resting human forearm." Journal of Applied Physiology, 1(2), 93-122.

use aequitas::systems::si::quantities::{
    MassDensity, ReciprocalTime, SpecificHeatCapacity, ThermodynamicTemperature, Time,
};
use kwavers_core::constants::medical::{BLOOD_SPECIFIC_HEAT, TISSUE_PERFUSION_RATE};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;
use kwavers_core::constants::tissue_acoustics::DENSITY_BLOOD;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::{Medium, ThermalMaterialFields};
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
    /// Arterial blood temperature (K).
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

        kwavers_core::traversal::zip_mut_indexed(
            source.view_mut(),
            temperature.view(),
            |[i, j, k], q, &t| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = kwavers_medium::density_at(medium, x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);

                let perfusion_coeff = self.params.perfusion_rate.into_base()
                    * self.params.blood_density.into_base()
                    * self.params.blood_specific_heat.into_base()
                    / (rho * cp);

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
        dt: Time<f64>,
    ) -> KwaversResult<()> {
        let dt_s = dt.into_base();

        // rho, cp and alpha are properties of the medium's static spatial
        // field, not of the evolving temperature, so they are read once here
        // instead of through the `&dyn Medium` accessors on every cell of every
        // step. `rho * cp` is hoisted for the same reason: it is constant across
        // the traversal, so one division per cell becomes one division.
        let fields = ThermalMaterialFields::sample(medium, grid);
        let heat_capacity = fields.volumetric_heat_capacity();
        let perfusion_coefficient = self.params.perfusion_rate.into_base()
            * self.params.blood_density.into_base()
            * self.params.blood_specific_heat.into_base();

        kwavers_core::traversal::zip_mut_indexed(
            temperature.view_mut(),
            laplacian.view(),
            |[i, j, k], t, &lap| {
                let alpha = fields.alpha[[i, j, k]];
                let perfusion = perfusion_coefficient
                    * (self.params.arterial_temperature.into_base() - *t)
                    / heat_capacity[[i, j, k]];
                let ext_source = external_source.as_ref().map_or(0.0, |s| s[[i, j, k]]);

                *t += dt_s * (alpha.mul_add(lap, perfusion) + ext_source);
            },
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use kwavers_medium::{HomogeneousMedium, ThermalProperties};

    fn parameters() -> BioheatParameters {
        BioheatParameters {
            perfusion_rate: ReciprocalTime::from_base(TISSUE_PERFUSION_RATE),
            blood_density: MassDensity::from_base(DENSITY_BLOOD),
            blood_specific_heat: SpecificHeatCapacity::from_base(BLOOD_SPECIFIC_HEAT),
            arterial_temperature: ThermodynamicTemperature::from_base(BODY_TEMPERATURE_K),
        }
    }

    /// The oracle for the sampled-property hoist in [`PennesBioheat::update`].
    ///
    /// The update no longer reads `density`, `specific_heat` and
    /// `thermal_diffusivity` through `&dyn Medium` per cell; it reads
    /// `ThermalMaterialFields`. This recomputes the same step with the
    /// original per-cell accessors and asserts bit-equality, which is the only
    /// claim the hoist is entitled to make.
    ///
    /// A curved field is used because a centered stencil annihilates a linear
    /// one: a ramp would give a zero Laplacian and the perfusion term alone
    /// would dominate, making the comparison much weaker.
    #[test]
    fn update_matches_per_cell_medium_accessors_exactly() {
        let grid = Grid::new(5, 4, 3, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
        let bioheat = PennesBioheat::new(parameters());
        let dt = Time::from_base(1.0e3);

        let mut initial = Array3::zeros((5, 4, 3));
        for k in 0..3 {
            for j in 0..4 {
                for i in 0..5 {
                    let (ii, jj, kk) = (i as f64, j as f64, k as f64);
                    initial[[i, j, k]] = 310.0 + ii * ii + 2.0 * jj * jj + 3.0 * kk * kk;
                }
            }
        }

        // A non-trivial Laplacian so the diffusion term participates.
        let mut laplacian = Array3::zeros((5, 4, 3));
        for k in 0..3 {
            for j in 0..4 {
                for i in 0..5 {
                    laplacian[[i, j, k]] =
                        if i > 0 { 1.0 } else { 0.0 } + if j > 0 { -0.5 } else { 0.0 };
                }
            }
        }

        // Reference: the pre-hoist arithmetic.
        let mut reference = initial.clone();
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    let rho = kwavers_medium::density_at(&medium, x, y, z, &grid);
                    let cp = medium.specific_heat(x, y, z, &grid);
                    let alpha = medium.thermal_diffusivity(x, y, z, &grid);
                    let t = reference[[i, j, k]];
                    let perfusion = parameters().perfusion_rate.into_base()
                        * parameters().blood_density.into_base()
                        * parameters().blood_specific_heat.into_base()
                        * (parameters().arterial_temperature.into_base() - t)
                        / (rho * cp);
                    reference[[i, j, k]] =
                        t + dt.into_base() * alpha.mul_add(laplacian[[i, j, k]], perfusion);
                }
            }
        }

        let mut actual = initial;
        bioheat
            .update(&mut actual, &laplacian, None, &medium, &grid, dt)
            .unwrap();

        assert_eq!(actual.as_slice(), reference.as_slice());
    }

    /// A curved field must actually move, or the equality above would hold for
    /// an implementation that does nothing.
    #[test]
    fn update_changes_a_curved_temperature_field() {
        let grid = Grid::new(5, 4, 3, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
        let medium =
            HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
        let bioheat = PennesBioheat::new(parameters());

        let mut temperature = Array3::zeros((5, 4, 3));
        for k in 0..3 {
            for j in 0..4 {
                for i in 0..5 {
                    let (ii, jj, kk) = (i as f64, j as f64, k as f64);
                    temperature[[i, j, k]] = 310.0 + ii * ii + 2.0 * jj * jj + 3.0 * kk * kk;
                }
            }
        }
        let before = temperature.clone();
        let laplacian = Array3::zeros((5, 4, 3));

        bioheat
            .update(
                &mut temperature,
                &laplacian,
                None,
                &medium,
                &grid,
                Time::from_base(1.0e3),
            )
            .unwrap();

        let changed = match (before.as_slice(), temperature.as_slice()) {
            (Some(a), Some(b)) => a.iter().zip(b).any(|(x, y)| x != y),
            _ => panic!("invariant: contiguous owned temperature arrays"),
        };
        assert!(changed, "perfusion must change the field");
    }
}
