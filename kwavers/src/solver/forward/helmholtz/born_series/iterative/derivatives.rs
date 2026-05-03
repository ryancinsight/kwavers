//! Finite-difference Laplacian and second-derivative methods for [`IterativeBornSolver`].

use super::IterativeBornSolver;
use num_complex::Complex64;

impl IterativeBornSolver {
    /// Compute Laplacian using finite differences (3D central difference)
    pub(super) fn compute_laplacian(&self, i: usize, j: usize, k: usize) -> Complex64 {
        let d2x = self.second_derivative_x(i, j, k);
        let d2y = self.second_derivative_y(i, j, k);
        let d2z = self.second_derivative_z(i, j, k);
        d2x + d2y + d2z
    }

    /// Compute second derivative in x-direction with proper boundary handling
    fn second_derivative_x(&self, i: usize, j: usize, k: usize) -> Complex64 {
        let field = &self.current_field;
        let dx2 = self.grid.dx * self.grid.dx;

        if i == 0 {
            let f0 = field[[0, j, k]];
            let f1 = field[[1, j, k]];
            let f2 = field[[2, j, k]];
            (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f2) / dx2
        } else if i == self.grid.nx - 1 {
            let f0 = field[[i, j, k]];
            let fm1 = field[[i - 1, j, k]];
            let fm2 = field[[i - 2, j, k]];
            (2.0 * f0 - 5.0 * fm1 + 4.0 * fm2 - fm2) / dx2
        } else {
            let fm1 = field[[i - 1, j, k]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i + 1, j, k]];
            (fm1 - 2.0 * f0 + fp1) / dx2
        }
    }

    /// Compute second derivative in y-direction with proper boundary handling
    fn second_derivative_y(&self, i: usize, j: usize, k: usize) -> Complex64 {
        let field = &self.current_field;
        let dy2 = self.grid.dy * self.grid.dy;

        if j == 0 {
            let f0 = field[[i, 0, k]];
            let f1 = field[[i, 1, k]];
            let f2 = field[[i, 2, k]];
            (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f2) / dy2
        } else if j == self.grid.ny - 1 {
            let f0 = field[[i, j, k]];
            let fm1 = field[[i, j - 1, k]];
            let fm2 = field[[i, j - 2, k]];
            (2.0 * f0 - 5.0 * fm1 + 4.0 * fm2 - fm2) / dy2
        } else {
            let fm1 = field[[i, j - 1, k]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i, j + 1, k]];
            (fm1 - 2.0 * f0 + fp1) / dy2
        }
    }

    /// Compute second derivative in z-direction with proper boundary handling
    fn second_derivative_z(&self, i: usize, j: usize, k: usize) -> Complex64 {
        let field = &self.current_field;
        let dz2 = self.grid.dz * self.grid.dz;

        if k == 0 {
            let f0 = field[[i, j, 0]];
            let f1 = field[[i, j, 1]];
            let f2 = field[[i, j, 2]];
            (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f2) / dz2
        } else if k == self.grid.nz - 1 {
            let f0 = field[[i, j, k]];
            let fm1 = field[[i, j, k - 1]];
            let fm2 = field[[i, j, k - 2]];
            (2.0 * f0 - 5.0 * fm1 + 4.0 * fm2 - fm2) / dz2
        } else {
            let fm1 = field[[i, j, k - 1]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i, j, k + 1]];
            (fm1 - 2.0 * f0 + fp1) / dz2
        }
    }
}
