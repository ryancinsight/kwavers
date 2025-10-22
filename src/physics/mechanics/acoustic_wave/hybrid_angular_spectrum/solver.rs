//! HAS solver with operator splitting
//!
//! Reference: Zemp et al. (2003) "k-space pseudospectral method"

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::mechanics::acoustic_wave::hybrid_angular_spectrum::{
    AbsorptionOperator, DiffractionOperator, HASConfig, NonlinearOperator,
};
use ndarray::Array3;

/// Hybrid Angular Spectrum solver with operator splitting
#[derive(Debug)]
pub struct HybridAngularSpectrumSolver {
    diffraction: DiffractionOperator,
    nonlinearity: NonlinearOperator,
    absorption: AbsorptionOperator,
}

impl HybridAngularSpectrumSolver {
    /// Create new HAS solver
    pub fn new(grid: &Grid, config: &HASConfig) -> KwaversResult<Self> {
        let diffraction = DiffractionOperator::new(grid, config)?;
        let nonlinearity = NonlinearOperator::new(config);
        let absorption = AbsorptionOperator::new(config);

        Ok(Self {
            diffraction,
            nonlinearity,
            absorption,
        })
    }

    /// Propagate pressure field for specified number of steps
    ///
    /// Uses Strang splitting for second-order accuracy:
    /// 1. Half-step diffraction
    /// 2. Full-step nonlinearity
    /// 3. Full-step absorption
    /// 4. Half-step diffraction
    pub fn propagate_steps(
        &self,
        initial: &Array3<f64>,
        num_steps: usize,
        dz: f64,
    ) -> KwaversResult<Array3<f64>> {
        let mut pressure = initial.clone();

        for _step in 0..num_steps {
            // Strang splitting for 2nd order accuracy
            pressure = self.diffraction.apply(&pressure, dz / 2.0)?;
            pressure = self.nonlinearity.apply(&pressure, dz)?;
            pressure = self.absorption.apply(&pressure, dz)?;
            pressure = self.diffraction.apply(&pressure, dz / 2.0)?;
        }

        Ok(pressure)
    }
}
