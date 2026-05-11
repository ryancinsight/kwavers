//! Diagonal preconditioner for Helmholtz solvers
//!
//! This module implements a simple diagonal preconditioner that approximates
//! the inverse of the Helmholtz operator using only the diagonal elements.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::solver::forward::helmholtz::Preconditioner;
use ndarray::{Array3, ArrayView3, ArrayViewMut3, Zip};
use rayon::prelude::*;
use num_complex::Complex64;

/// Diagonal preconditioner for Helmholtz equation
#[derive(Debug)]
pub struct DiagonalPreconditioner {
    /// Diagonal elements of the preconditioner
    diagonal: Array3<Complex64>,
    /// Wavenumber for which preconditioner is set up
    wavenumber: Option<f64>,
}

impl DiagonalPreconditioner {
    /// Create a new diagonal preconditioner
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(grid: &Grid) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        Self {
            diagonal: Array3::zeros(shape),
            wavenumber: None,
        }
    }
}

impl Preconditioner for DiagonalPreconditioner {
    /// Apply diagonal preconditioner: M⁻¹x where M is diagonal
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply(
        &self,
        input: &ArrayView3<Complex64>,
        output: &mut ArrayViewMut3<Complex64>,
    ) -> KwaversResult<()> {
        Zip::from(output)
            .and(input)
            .and(&self.diagonal)
            .par_for_each(|out, &inp, &diag| {
                if diag.norm_sqr() > 1e-12 {
                    *out = inp / diag;
                } else {
                    *out = inp * Complex64::new(1e6, 0.0); // Fallback for near-zero diagonal
                }
            });

        Ok(())
    }

    /// Setup diagonal preconditioner for Helmholtz operator
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if `diagonal contiguous`.
    ///
    fn setup(&mut self, wavenumber: f64, medium: &dyn Medium, grid: &Grid) -> KwaversResult<()> {
        let k_squared = wavenumber * wavenumber;
        let (nx, ny, nz) = self.diagonal.dim();
        let c0 = 1500.0_f64;
        let rho0 = 1000.0_f64;
        let laplacian_diag =
            -6.0 / (grid.dx * grid.dx) - 6.0 / (grid.dy * grid.dy) - 6.0 / (grid.dz * grid.dz);

        // Phase 1: sequential — dyn Medium not guaranteed Sync; collect heterogeneity values.
        let heterogeneities: Vec<f64> = (0..nx)
            .flat_map(|i| {
                (0..ny).flat_map(move |j| {
                    (0..nz).map(move |k| {
                        let c = medium.sound_speed(i, j, k);
                        let rho = medium.density(i, j, k);
                        let contrast = (rho * c * c) / (rho0 * c0 * c0);
                        1.0 - contrast
                    })
                })
            })
            .collect();

        // Phase 2: parallel — compute diagonal elements from pre-collected values.
        self.diagonal
            .as_slice_mut()
            .expect("diagonal contiguous")
            .par_iter_mut()
            .zip(heterogeneities.par_iter())
            .for_each(|(diag, &heterogeneity)| {
                let helmholtz_diag = k_squared.mul_add(-heterogeneity, laplacian_diag);
                *diag = Complex64::new(helmholtz_diag, 0.0);
            });

        self.wavenumber = Some(wavenumber);
        Ok(())
    }
}
