use ndarray::ArrayView3;
use num_complex::Complex64;

use super::ModifiedBornSolver;

impl ModifiedBornSolver {
    pub(super) fn compute_laplacian(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let d2x = self.fourth_order_derivative_x(field, i, j, k);
        let d2y = self.fourth_order_derivative_y(field, i, j, k);
        let d2z = self.fourth_order_derivative_z(field, i, j, k);

        d2x + d2y + d2z
    }

    fn second_derivative_x(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let dx2 = self.grid.dx * self.grid.dx;

        if i == 0 {
            let f0 = field[[0, j, k]];
            let f1 = field[[1, j, k]];
            let f2 = field[[2, j, k]];
            let f3 = field[[3, j, k]];
            (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f3) / dx2
        } else if i == self.grid.nx - 1 {
            let f0 = field[[i, j, k]];
            let fm1 = field[[i - 1, j, k]];
            let fm2 = field[[i - 2, j, k]];
            let fm3 = field[[i - 3, j, k]];
            (2.0 * f0 - 5.0 * fm1 + 4.0 * fm2 - fm3) / dx2
        } else {
            let fm1 = field[[i - 1, j, k]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i + 1, j, k]];
            (fm1 - 2.0 * f0 + fp1) / dx2
        }
    }

    fn second_derivative_y(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let dy2 = self.grid.dy * self.grid.dy;

        if j == 0 {
            let f0 = field[[i, 0, k]];
            let f1 = field[[i, 1, k]];
            let f2 = field[[i, 2, k]];
            let f3 = field[[i, 3, k]];
            (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f3) / dy2
        } else if j == self.grid.ny - 1 {
            let f0 = field[[i, j, k]];
            let fm1 = field[[i, j - 1, k]];
            let fm2 = field[[i, j - 2, k]];
            let fm3 = field[[i, j - 3, k]];
            (2.0 * f0 - 5.0 * fm1 + 4.0 * fm2 - fm3) / dy2
        } else {
            let fm1 = field[[i, j - 1, k]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i, j + 1, k]];
            (fm1 - 2.0 * f0 + fp1) / dy2
        }
    }

    fn second_derivative_z(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let dz2 = self.grid.dz * self.grid.dz;

        if k == 0 {
            let f0 = field[[i, j, 0]];
            let f1 = field[[i, j, 1]];
            let f2 = field[[i, j, 2]];
            let f3 = field[[i, j, 3]];
            (2.0 * f0 - 5.0 * f1 + 4.0 * f2 - f3) / dz2
        } else if k == self.grid.nz - 1 {
            let f0 = field[[i, j, k]];
            let fm1 = field[[i, j, k - 1]];
            let fm2 = field[[i, j, k - 2]];
            let fm3 = field[[i, j, k - 3]];
            (2.0 * f0 - 5.0 * fm1 + 4.0 * fm2 - fm3) / dz2
        } else {
            let fm1 = field[[i, j, k - 1]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i, j, k + 1]];
            (fm1 - 2.0 * f0 + fp1) / dz2
        }
    }

    fn fourth_order_derivative_x(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let dx2 = self.grid.dx * self.grid.dx;

        if i >= 2 && i < self.grid.nx - 2 {
            let fm2 = field[[i - 2, j, k]];
            let fm1 = field[[i - 1, j, k]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i + 1, j, k]];
            let fp2 = field[[i + 2, j, k]];

            (-fm2 + 16.0 * fm1 - 30.0 * f0 + 16.0 * fp1 - fp2) / (12.0 * dx2)
        } else {
            self.second_derivative_x(field, i, j, k)
        }
    }

    fn fourth_order_derivative_y(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let dy2 = self.grid.dy * self.grid.dy;

        if j >= 2 && j < self.grid.ny - 2 {
            let fm2 = field[[i, j - 2, k]];
            let fm1 = field[[i, j - 1, k]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i, j + 1, k]];
            let fp2 = field[[i, j + 2, k]];

            (-fm2 + 16.0 * fm1 - 30.0 * f0 + 16.0 * fp1 - fp2) / (12.0 * dy2)
        } else {
            self.second_derivative_y(field, i, j, k)
        }
    }

    fn fourth_order_derivative_z(
        &self,
        field: ArrayView3<Complex64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> Complex64 {
        let dz2 = self.grid.dz * self.grid.dz;

        if k >= 2 && k < self.grid.nz - 2 {
            let fm2 = field[[i, j, k - 2]];
            let fm1 = field[[i, j, k - 1]];
            let f0 = field[[i, j, k]];
            let fp1 = field[[i, j, k + 1]];
            let fp2 = field[[i, j, k + 2]];

            (-fm2 + 16.0 * fm1 - 30.0 * f0 + 16.0 * fp1 - fp2) / (12.0 * dz2)
        } else {
            self.second_derivative_z(field, i, j, k)
        }
    }
}
