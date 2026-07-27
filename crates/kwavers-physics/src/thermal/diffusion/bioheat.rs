//! Pennes Bioheat Equation Implementation
//!
//! Reference: Pennes, H. H. (1948). "Analysis of tissue and arterial blood temperatures
//! in the resting human forearm." Journal of Applied Physiology, 1(2), 93-122.

use crate::thermal::source::VolumetricHeatSource;
use aequitas::systems::si::quantities::{
    MassDensity, ReciprocalTime, SpecificHeatCapacity, ThermodynamicTemperature,
};
use aequitas::systems::si::units::{
    JoulePerKilogramKelvin, Kelvin, KilogramPerCubicMeter, PerSecond,
};
use kwavers_core::constants::medical::{BLOOD_SPECIFIC_HEAT, TISSUE_PERFUSION_RATE};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;
use kwavers_core::constants::tissue_acoustics::DENSITY_BLOOD;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::Array3;

/// Pennes bioheat equation parameters.
///
/// Each term carries its Aequitas dimension, so a perfusion rate cannot be
/// substituted for a specific heat and the `ω_b ρ_b c_b` product is checked to
/// be `W/(m³·K)` at compile time rather than asserted in a comment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BioheatParameters {
    perfusion_rate: ReciprocalTime,
    blood_density: MassDensity,
    blood_specific_heat: SpecificHeatCapacity,
    arterial_temperature: ThermodynamicTemperature,
}

impl BioheatParameters {
    /// Assemble Pennes parameters from dimensional quantities.
    #[must_use]
    pub const fn new(
        perfusion_rate: ReciprocalTime,
        blood_density: MassDensity,
        blood_specific_heat: SpecificHeatCapacity,
        arterial_temperature: ThermodynamicTemperature,
    ) -> Self {
        Self {
            perfusion_rate,
            blood_density,
            blood_specific_heat,
            arterial_temperature,
        }
    }

    /// Assemble Pennes parameters from coherent SI magnitudes.
    ///
    /// The units are `1/s`, `kg/m³`, `J/(kg·K)`, and `K` respectively.
    #[must_use]
    pub fn from_si(
        perfusion_rate: f64,
        blood_density: f64,
        blood_specific_heat: f64,
        arterial_temperature: f64,
    ) -> Self {
        Self::new(
            ReciprocalTime::from_unit::<PerSecond>(perfusion_rate),
            MassDensity::from_unit::<KilogramPerCubicMeter>(blood_density),
            SpecificHeatCapacity::from_unit::<JoulePerKilogramKelvin>(blood_specific_heat),
            ThermodynamicTemperature::from_unit::<Kelvin>(arterial_temperature),
        )
    }

    /// Blood perfusion rate `ω_b`.
    #[must_use]
    pub const fn perfusion_rate(&self) -> ReciprocalTime {
        self.perfusion_rate
    }

    /// Blood mass density `ρ_b`.
    #[must_use]
    pub const fn blood_density(&self) -> MassDensity {
        self.blood_density
    }

    /// Blood specific heat capacity `c_b`.
    #[must_use]
    pub const fn blood_specific_heat(&self) -> SpecificHeatCapacity {
        self.blood_specific_heat
    }

    /// Arterial blood temperature `T_a`.
    #[must_use]
    pub const fn arterial_temperature(&self) -> ThermodynamicTemperature {
        self.arterial_temperature
    }

    /// Volumetric perfusion heat-transfer coefficient `ω_b ρ_b c_b`, in
    /// `W/(m³·K)`.
    ///
    /// Loop-invariant across a voxel traversal, so the Pennes update evaluates
    /// it once per step rather than per voxel.
    #[must_use]
    fn perfusion_heat_transfer_coefficient(&self) -> f64 {
        self.perfusion_rate.into_base()
            * self.blood_density.into_base()
            * self.blood_specific_heat.into_base()
    }
}

impl Default for BioheatParameters {
    fn default() -> Self {
        // TISSUE_PERFUSION_RATE = 5×10⁻⁴ 1/s — generic soft tissue value
        // (Pennes 1948; Duck 1990). See `kwavers_core::constants::medical`.
        Self::from_si(
            TISSUE_PERFUSION_RATE,
            DENSITY_BLOOD,
            BLOOD_SPECIFIC_HEAT,
            BODY_TEMPERATURE_K,
        )
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
        let blood_coefficient = self.params.perfusion_heat_transfer_coefficient();
        let arterial_temperature = self.params.arterial_temperature.into_base();

        crate::parallel::for_each_indexed_pair_mut(
            source.view_mut(),
            temperature.view(),
            |(i, j, k), q, &t| {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;

                let rho = kwavers_medium::density_at(medium, x, y, z, grid);
                let cp = medium.specific_heat(x, y, z, grid);

                *q = blood_coefficient * (arterial_temperature - t) / (rho * cp);
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
    /// `external_source` is a volumetric power density `Q` in `W/m³`. It is
    /// divided by the *local* `ρ c_p` here, alongside the perfusion term, so
    /// deposition and perfusion always reference the same material state. A
    /// caller that pre-divides by a uniform `ρ c_p` silently disagrees with
    /// perfusion wherever the medium is heterogeneous.
    ///
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn update(
        &self,
        temperature: &mut Array3<f64>,
        laplacian: &Array3<f64>,
        external_source: Option<VolumetricHeatSource<'_>>,
        medium: &dyn Medium,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        let blood_coefficient = self.params.perfusion_heat_transfer_coefficient();
        let arterial_temperature = self.params.arterial_temperature.into_base();

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
                let deposition = external_source
                    .as_ref()
                    .map_or(0.0, |source| source.as_view()[[i, j, k]]);
                // Both terms divide by the same local ρ c_p, so a heterogeneous
                // medium cannot make deposition and perfusion disagree.
                let heating =
                    blood_coefficient.mul_add(arterial_temperature - *t, deposition) / (rho * cp);

                *t += dt * alpha.mul_add(lap, heating);
            },
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{BioheatParameters, BLOOD_SPECIFIC_HEAT, BODY_TEMPERATURE_K, DENSITY_BLOOD};
    use aequitas::systems::si::quantities::{TemperatureDifference, VolumetricPowerDensity};
    use aequitas::systems::si::units::{
        JoulePerKilogramKelvin, Kelvin, KilogramPerCubicMeter, PerSecond, WattPerCubicMeter,
    };
    use kwavers_core::constants::medical::TISSUE_PERFUSION_RATE;

    #[test]
    fn parameters_round_trip_through_their_declared_units() {
        let params = BioheatParameters::default();

        assert_eq!(
            params.perfusion_rate().in_unit::<PerSecond>().to_bits(),
            TISSUE_PERFUSION_RATE.to_bits()
        );
        assert_eq!(
            params
                .blood_density()
                .in_unit::<KilogramPerCubicMeter>()
                .to_bits(),
            DENSITY_BLOOD.to_bits()
        );
        assert_eq!(
            params
                .blood_specific_heat()
                .in_unit::<JoulePerKilogramKelvin>()
                .to_bits(),
            BLOOD_SPECIFIC_HEAT.to_bits()
        );
        assert_eq!(
            params.arterial_temperature().in_unit::<Kelvin>().to_bits(),
            BODY_TEMPERATURE_K.to_bits()
        );
    }

    /// `ω_b ρ_b c_b ΔT` must be a volumetric power density. The annotation is
    /// the assertion: a wrong factor fails to compile rather than producing a
    /// plausible number.
    #[test]
    fn perfusion_term_composes_to_a_volumetric_power_density() {
        let params = BioheatParameters::from_si(2.0, 1_000.0, 4.0, 310.0);
        let difference = TemperatureDifference::from_unit::<Kelvin>(3.0);

        let deposition: VolumetricPowerDensity = params.perfusion_rate()
            * params.blood_density()
            * params.blood_specific_heat()
            * difference;

        assert_eq!(
            deposition.in_unit::<WattPerCubicMeter>().to_bits(),
            24_000.0_f64.to_bits()
        );
    }

    #[test]
    fn perfusion_coefficient_is_the_product_of_the_blood_terms() {
        let params = BioheatParameters::from_si(2.0, 1_000.0, 4.0, 310.0);

        assert_eq!(
            params.perfusion_heat_transfer_coefficient().to_bits(),
            8_000.0_f64.to_bits()
        );
    }
}
