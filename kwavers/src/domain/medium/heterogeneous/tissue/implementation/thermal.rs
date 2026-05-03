//! `ThermalProperties` and `ThermalField` impls for `HeterogeneousTissueMedium`

use super::HeterogeneousTissueMedium;
use crate::domain::grid::Grid;
use crate::domain::medium::thermal::{ThermalField, ThermalProperties};
use ndarray::Array3;

impl ThermalProperties for HeterogeneousTissueMedium {
    fn specific_heat(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).specific_heat
    }

    fn specific_heat_capacity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        self.specific_heat(x, y, z, grid)
    }

    fn thermal_conductivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).thermal_conductivity
    }

    fn thermal_diffusivity(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        let props = self.get_tissue_properties(i, j, k);
        // alpha = k / (rho * Cp)
        props.thermal_conductivity / (props.density * props.specific_heat)
    }

    fn thermal_expansion(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).thermal_expansion
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
