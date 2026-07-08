//! Gradient-based field updater for FDTD solvers with split-field PML.
//!
//! ## Mathematical Background
//!
//! FDTD acoustic wave equation:
//! ```text
//! ∂v/∂t = -(1/ρ) ∇p
//! ∂p/∂t = -ρc² ∇·v
//! ```
//!
//! With CPML, gradients are modified by memory variables:
//! ```text
//! ∇p → ∇p + ψ (memory correction)
//! ```

use kwavers_grid::GridTopology;
use leto::{Array3, ArrayView3, ArrayViewMut3};

/// Gradient-based field updater for FDTD solvers.
///
/// Handles gradient computation and boundary correction common in FDTD
/// solvers with split-field PML.
#[derive(Debug)]
pub struct GradientFieldUpdater {
    /// Temporary storage for x-gradient.
    pub(super) grad_x: Array3<f64>,
    /// Temporary storage for y-gradient.
    pub(super) grad_y: Array3<f64>,
    /// Temporary storage for z-gradient.
    pub(super) grad_z: Array3<f64>,
}

impl GradientFieldUpdater {
    /// Create a new gradient field updater, allocating arrays from grid dimensions.
    pub fn new(grid: &dyn GridTopology) -> Self {
        let dims = grid.dimensions();
        Self {
            grad_x: Array3::zeros([dims[0], dims[1], dims[2]]),
            grad_y: Array3::zeros([dims[0], dims[1], dims[2]]),
            grad_z: Array3::zeros([dims[0], dims[1], dims[2]]),
        }
    }

    /// Compute spatial gradients using central differences (one-sided at boundaries).
    pub fn compute_gradients(&mut self, field: &Array3<f64>, grid: &dyn GridTopology) {
        let dims = grid.dimensions();
        let spacing = grid.spacing();
        let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
        let (dx, dy, dz) = (spacing[0], spacing[1], spacing[2]);

        // X-gradient — interior
        for i in 1..nx - 1 {
            for j in 0..ny {
                for k in 0..nz {
                    self.grad_x[[i, j, k]] =
                        (field[[i + 1, j, k]] - field[[i - 1, j, k]]) / (2.0 * dx);
                }
            }
        }
        // X boundaries (one-sided)
        for j in 0..ny {
            for k in 0..nz {
                self.grad_x[[0, j, k]] = (field[[1, j, k]] - field[[0, j, k]]) / dx;
                self.grad_x[[nx - 1, j, k]] = (field[[nx - 1, j, k]] - field[[nx - 2, j, k]]) / dx;
            }
        }

        // Y-gradient — interior
        for i in 0..nx {
            for j in 1..ny - 1 {
                for k in 0..nz {
                    self.grad_y[[i, j, k]] =
                        (field[[i, j + 1, k]] - field[[i, j - 1, k]]) / (2.0 * dy);
                }
            }
        }
        // Y boundaries
        for i in 0..nx {
            for k in 0..nz {
                self.grad_y[[i, 0, k]] = (field[[i, 1, k]] - field[[i, 0, k]]) / dy;
                self.grad_y[[i, ny - 1, k]] = (field[[i, ny - 1, k]] - field[[i, ny - 2, k]]) / dy;
            }
        }

        // Z-gradient — interior
        for i in 0..nx {
            for j in 0..ny {
                for k in 1..nz - 1 {
                    self.grad_z[[i, j, k]] =
                        (field[[i, j, k + 1]] - field[[i, j, k - 1]]) / (2.0 * dz);
                }
            }
        }
        // Z boundaries
        for i in 0..nx {
            for j in 0..ny {
                self.grad_z[[i, j, 0]] = (field[[i, j, 1]] - field[[i, j, 0]]) / dz;
                self.grad_z[[i, j, nz - 1]] = (field[[i, j, nz - 1]] - field[[i, j, nz - 2]]) / dz;
            }
        }
    }

    /// Immutable views of (grad_x, grad_y, grad_z).
    #[must_use]
    pub fn gradients(
        &self,
    ) -> (
        ArrayView3<'_, f64>,
        ArrayView3<'_, f64>,
        ArrayView3<'_, f64>,
    ) {
        (self.grad_x.view(), self.grad_y.view(), self.grad_z.view())
    }

    /// Mutable views of (grad_x, grad_y, grad_z).
    pub fn gradients_mut(
        &mut self,
    ) -> (
        ArrayViewMut3<'_, f64>,
        ArrayViewMut3<'_, f64>,
        ArrayViewMut3<'_, f64>,
    ) {
        (
            self.grad_x.view_mut(),
            self.grad_y.view_mut(),
            self.grad_z.view_mut(),
        )
    }

    /// Apply CPML memory correction to computed gradients in-place.
    ///
    /// `correction_fn(field, axis)` modifies the gradient array for the given axis.
    pub fn apply_cpml_correction<F>(&mut self, mut correction_fn: F)
    where
        F: FnMut(&mut Array3<f64>, usize),
    {
        correction_fn(&mut self.grad_x, 0);
        correction_fn(&mut self.grad_y, 1);
        correction_fn(&mut self.grad_z, 2);
    }

    /// Compute divergence ∇·v = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z using central differences.
    #[must_use]
    pub fn compute_divergence(
        vx: &Array3<f64>,
        vy: &Array3<f64>,
        vz: &Array3<f64>,
        grid: &dyn GridTopology,
    ) -> Array3<f64> {
        let dims = grid.dimensions();
        let spacing = grid.spacing();
        let (nx, ny, nz) = (dims[0], dims[1], dims[2]);
        let (dx, dy, dz) = (spacing[0], spacing[1], spacing[2]);

        let mut div = Array3::zeros([nx, ny, nz]);

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let dvx_dx = (vx[[i + 1, j, k]] - vx[[i - 1, j, k]]) / (2.0 * dx);
                    let dvy_dy = (vy[[i, j + 1, k]] - vy[[i, j - 1, k]]) / (2.0 * dy);
                    let dvz_dz = (vz[[i, j, k + 1]] - vz[[i, j, k - 1]]) / (2.0 * dz);

                    div[[i, j, k]] = dvx_dx + dvy_dy + dvz_dz;
                }
            }
        }

        div
    }
}
