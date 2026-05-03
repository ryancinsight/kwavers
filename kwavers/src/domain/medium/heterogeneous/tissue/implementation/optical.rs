//! `OpticalProperties` impl for `HeterogeneousTissueMedium`

use super::HeterogeneousTissueMedium;
use crate::domain::grid::Grid;
use crate::domain::medium::optical::OpticalProperties;

impl OpticalProperties for HeterogeneousTissueMedium {
    fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).optical_absorption_coeff
    }

    fn optical_scattering_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).optical_scattering_coeff
    }
}
