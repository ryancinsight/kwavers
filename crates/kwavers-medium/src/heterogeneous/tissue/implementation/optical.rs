//! `MediumOpticalProperties` impl for `HeterogeneousTissueMedium`

use super::HeterogeneousTissueMedium;
use crate::optical::{interaction_from_si, MediumOpticalProperties};
use hyperion::{
    coefficient::{InteractionCoefficient, ReducedScattering},
    TransportError,
};
use kwavers_grid::Grid;

impl MediumOpticalProperties for HeterogeneousTissueMedium {
    fn optical_absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).optical_absorption_coeff
    }

    fn optical_reduced_scattering_coefficient(
        &self,
        x: f64,
        y: f64,
        z: f64,
        grid: &Grid,
    ) -> Result<InteractionCoefficient<f64, ReducedScattering>, TransportError<f64>> {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        interaction_from_si(
            self.get_tissue_properties(i, j, k)
                .optical_reduced_scattering_coeff,
        )
    }
}
