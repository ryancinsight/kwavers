//! `FemHelmholtzBackend` construction and state.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_math::fft::Complex64;
use kwavers_medium::Medium;
use kwavers_solver::forward::helmholtz::fem::{FemHelmholtzConfig, FemHelmholtzSolver};

/// Frequency-domain acoustic backend backed by P1 tetrahedral FEM.
#[derive(Debug)]
pub struct FemHelmholtzBackend<'a> {
    pub(super) solver: FemHelmholtzSolver,
    pub(super) medium: &'a dyn Medium,
    pub(super) pending_nodal_loads: Vec<(usize, Complex64)>,
    pub(super) wavenumber: f64,
}

impl<'a> FemHelmholtzBackend<'a> {
    /// Construct a FEM Helmholtz backend from a structured Cartesian vertex grid.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn from_grid(
        grid: &Grid,
        medium: &'a dyn Medium,
        config: FemHelmholtzConfig,
    ) -> KwaversResult<Self> {
        if !config.wavenumber.is_finite() || config.wavenumber < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "FEM Helmholtz wavenumber must be finite and non-negative, got {}",
                config.wavenumber
            )));
        }

        let wavenumber = config.wavenumber;
        let solver = FemHelmholtzSolver::from_grid(config, grid)?;
        Ok(Self {
            solver,
            medium,
            pending_nodal_loads: Vec::new(),
            wavenumber,
        })
    }
}