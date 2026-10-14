//! `ThermalProperties` and `ThermalField` impls for `HeterogeneousTissueMedium`

use super::HeterogeneousTissueMedium;
use crate::thermal::{ThermalField, ThermalProperties};
use aequitas::systems::si::quantities::{
    ReciprocalTemperature, SpecificHeatCapacity, ThermalConductivity, ThermalDiffusivity,
};
use kwavers_grid::Grid;
use leto::Array3;

impl ThermalProperties for HeterogeneousTissueMedium {
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> SpecificHeatCapacity {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        SpecificHeatCapacity::from_base(self.get_tissue_properties(i, j, k).specific_heat)
    }

    fn specific_heat_capacity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> SpecificHeatCapacity {
        self.specific_heat(x, y, z, grid)
    }

    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> ThermalConductivity {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        ThermalConductivity::from_base(self.get_tissue_properties(i, j, k).thermal_conductivity)
    }

    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> ThermalDiffusivity {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        let props = self.get_tissue_properties(i, j, k);
        // alpha = k / (rho * Cp)
        ThermalDiffusivity::from_base(
            props.thermal_conductivity / (props.density * props.specific_heat),
        )
    }

    fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> ReciprocalTemperature {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        ReciprocalTemperature::from_base(self.get_tissue_properties(i, j, k).thermal_expansion)
    }
}

impl ThermalField for HeterogeneousTissueMedium {
    fn update_thermal_field(&mut self, temperature: &Array3<f64>) {
        self.temperature = temperature.clone();
    }

    fn thermal_field(&self) -> &Array3<f64> {
        &self.temperature
    }
}
