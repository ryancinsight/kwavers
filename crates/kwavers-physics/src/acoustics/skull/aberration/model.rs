//! Aberration-correction model construction.

use kwavers_grid::Grid;
use crate::acoustics::skull::HeterogeneousSkull;

use super::constants::C_WATER_DEFAULT;

/// Aberration correction calculator using the CT-based phase-screen model.
///
/// Implements volumetric phase integration, aperture phase-map extraction, and
/// scalar phase corrections for planar transducer elements.
#[derive(Debug)]
pub struct AberrationCorrection<'a> {
    pub(super) grid: &'a Grid,
    pub(super) skull: &'a HeterogeneousSkull,
    /// Water sound speed [m/s].
    pub c_water: f64,
}

impl<'a> AberrationCorrection<'a> {
    /// Construct an aberration correction calculator with 22 deg C water speed.
    #[must_use]
    pub fn new(grid: &'a Grid, skull: &'a HeterogeneousSkull) -> Self {
        Self {
            grid,
            skull,
            c_water: C_WATER_DEFAULT,
        }
    }

    /// Override the reference water sound speed [m/s].
    #[must_use]
    pub fn with_water_speed(mut self, c_water: f64) -> Self {
        self.c_water = c_water;
        self
    }
}
