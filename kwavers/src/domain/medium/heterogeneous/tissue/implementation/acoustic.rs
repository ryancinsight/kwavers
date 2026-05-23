//! `AcousticProperties` impl for `HeterogeneousTissueMedium`

use super::HeterogeneousTissueMedium;
use crate::core::constants::{DB_TO_NP, MHZ_TO_HZ};
use crate::domain::grid::Grid;
use crate::domain::medium::absorption::AbsorptionTissueType;
use crate::domain::medium::acoustic::AcousticProperties;

impl AcousticProperties for HeterogeneousTissueMedium {
    fn absorption_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid, frequency: f64) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        let props = self.get_tissue_properties(i, j, k);
        // Power law: α(f) [dB/cm] = α₀ · (f/1 MHz)^y
        // Unit conversion to Np/m: × DB_TO_NP [Np/dB] × 100 [cm/m]
        // Reference: Szabo (1994) J.Acoust.Soc.Am. 96(1), eq. (2), (4)
        let alpha_db_per_cm = props.alpha_0 * (frequency / MHZ_TO_HZ).powf(props.y);
        alpha_db_per_cm * DB_TO_NP * 100.0
    }

    fn alpha_coefficient(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).alpha_0
    }

    fn alpha_power(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).y
    }

    fn nonlinearity_parameter(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        self.get_tissue_properties(i, j, k).nonlinearity
    }

    fn acoustic_diffusivity(&self, _x: f64, _y: f64, _z: f64, _grid: &Grid) -> f64 {
        0.0 // Default/Placeholder
    }

    fn tissue_type(&self, x: f64, y: f64, z: f64, grid: &Grid) -> Option<AbsorptionTissueType> {
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        Some(self.tissue_map[[i, j, k]])
    }
}
