//! `AcousticProperties` impl for `HeterogeneousTissueMedium`

use super::HeterogeneousTissueMedium;
use crate::absorption::AbsorptionTissueType;
use crate::acoustic::AcousticProperties;
use aequitas::systems::si::quantities::{Dimensionless, ReciprocalLength};
use kwavers_core::constants::{DB_TO_NP, MHZ_TO_HZ};
use kwavers_grid::Grid;

impl AcousticProperties for HeterogeneousTissueMedium {
    fn absorption_coefficient(
        &self,
        x: f64,
        y: f64,
        z: f64,
        grid: &Grid,
        frequency: f64,
    ) -> ReciprocalLength<f64> {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        let props = self.get_tissue_properties(i, j, k);
        // Power law: α(f) [dB/cm] = α₀ · (f/1 MHz)^y
        // Unit conversion to Np/m: × DB_TO_NP [Np/dB] × 100 [cm/m]
        // Reference: Szabo (1994) J.Acoust.Soc.Am. 96(1), eq. (2), (4)
        let alpha_db_per_cm = props.alpha_0 * (frequency / MHZ_TO_HZ).powf(props.y);
        ReciprocalLength::from_base(alpha_db_per_cm * DB_TO_NP * 100.0)
    }

    fn alpha_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).alpha_0
    }

    fn alpha_power(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).y
    }

    fn nonlinearity_parameter(&self, x: f64, y: f64, z: f64, grid: &Grid) -> Dimensionless<f64> {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        Dimensionless::from_base(self.get_tissue_properties(i, j, k).nonlinearity)
    }

    fn tissue_type(&self, x: f64, y: f64, z: f64, grid: &Grid) -> Option<AbsorptionTissueType> {
        let (i, j, k) = crate::continuous_to_discrete(x, y, z, grid);
        Some(self.tissue_map[[i, j, k]])
    }
}
