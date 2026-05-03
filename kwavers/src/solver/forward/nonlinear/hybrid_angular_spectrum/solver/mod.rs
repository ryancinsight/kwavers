//! HAS solver with operator splitting.
//!
//! ## Algorithm — Strang Splitting (Strang 1968)
//!
//! For each propagation step Δz the pressure field is advanced by:
//!
//! ```text
//! p_{n+1} = D(Δz/2) · N(Δz) · A(Δz) · D(Δz/2)  [p_n]
//! ```
//!
//! where:
//! - D = diffraction operator (FFT-based angular spectrum, Goodman 2005)
//! - N = nonlinearity operator (Burgers equation, Hamilton & Blackstock 2008)
//! - A = absorption operator (power-law exponential decay, Szabo 1994)
//!
//! The Strang (1968) form achieves global 2nd-order accuracy in Δz:
//! the splitting error is O(Δz²) per step.
//!
//! ## References
//!
//! - Strang G (1968). "On the construction and comparison of difference schemes."
//!   SIAM J. Numer. Anal. 5(3), 506–517. DOI: 10.1137/0705041
//! - Goodman JW (2005). Introduction to Fourier Optics. Roberts & Co., §3.
//! - Hamilton MF, Blackstock DT (2008). Nonlinear Acoustics. ASA Press.
//! - Szabo TL (1994). J. Acoust. Soc. Am. 96(1), 491–500. DOI: 10.1121/1.410434
//! - Zemp RJ et al. (2003). J. Acoust. Soc. Am. 113(1), 139–152.
//!   DOI: 10.1121/1.1528928

#[cfg(test)]
mod tests;

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;

use super::{AbsorptionOperator, DiffractionOperator, HASConfig, NonlinearOperator};

/// Hybrid Angular Spectrum solver with Strang operator splitting.
#[derive(Debug)]
pub struct HybridAngularSpectrumSolver {
    diffraction: DiffractionOperator,
    nonlinearity: NonlinearOperator,
    absorption: AbsorptionOperator,
}

impl HybridAngularSpectrumSolver {
    /// Construct from grid and configuration.
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

    /// Propagate pressure field for `num_steps` steps of size `dz`.
    ///
    /// ## Algorithm (Strang 1968, Zemp et al. 2003)
    ///
    /// Each step applies the Strang-split operators in the order:
    ///
    /// ```text
    /// p_{n+1} = D(Δz/2) · N(Δz) · A(Δz) · D(Δz/2)  [p_n]
    /// ```
    ///
    /// The half-steps in D bracket the full nonlinear and absorption steps,
    /// yielding 2nd-order accuracy in Δz (Strang 1968).
    ///
    /// ## Parameters
    /// - `initial`   — initial pressure field `p[ix, iy, iz]` [Pa]
    /// - `num_steps` — number of Δz steps to advance
    /// - `dz`        — axial step size [m]
    ///
    /// ## References
    /// - Strang G (1968). SIAM J. Numer. Anal. 5(3), 506–517.
    ///   DOI: 10.1137/0705041
    /// - Zemp RJ et al. (2003). J. Acoust. Soc. Am. 113(1), 139–152.
    ///   DOI: 10.1121/1.1528928
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
