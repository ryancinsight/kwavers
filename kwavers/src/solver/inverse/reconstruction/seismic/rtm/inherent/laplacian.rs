//! 2nd-order isotropic Laplacian operator for RTM image filtering.
//!
//! # Theorem: parallel Zip-slice Laplacian (2nd-order, unit stencil)
//!
//! For an interior point `(i,j,k)` of an array `field` with uniform spacing:
//! ```text
//! ∇²field[i,j,k] = field[i+1,j,k] + field[i-1,j,k]
//!                 + field[i,j+1,k] + field[i,j-1,k]
//!                 + field[i,j,k+1] + field[i,j,k-1]
//!                 − 6·field[i,j,k]
//! ```
//! Boundary cells remain zero.
//!
//! ## Implementation
//!
//! Three sequential `Zip::par_for_each` passes accumulate into the interior
//! laplacian slice:
//! 1. x-neighbours: `+= field[i+1] + field[i-1]`
//! 2. y-neighbours: `+= field[j+1] + field[j-1]`
//! 3. z-neighbours and centre: `+= field[k+1] + field[k-1] − 6·field[c]`
//!
//! Each pass uses 3 or 4 arrays (well within ndarray Zip's limit).

use crate::core::error::KwaversResult;
use ndarray::{s, Array3, Zip};

use super::super::types::ReverseTimeMigration;

impl ReverseTimeMigration {
    /// Compute the 2nd-order isotropic Laplacian of `field`.
    ///
    /// Returns an array of the same shape; boundary elements are zero.
    pub(super) fn compute_laplacian(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut laplacian = Array3::<f64>::zeros((nx, ny, nz));
        let inn = s![1..nx - 1, 1..ny - 1, 1..nz - 1];

        // x-neighbours
        Zip::from(laplacian.slice_mut(inn))
            .and(&field.slice(s![2..nx, 1..ny - 1, 1..nz - 1]))
            .and(&field.slice(s![..nx - 2, 1..ny - 1, 1..nz - 1]))
            .par_for_each(|lap, &xp, &xm| *lap += xp + xm);

        // y-neighbours
        Zip::from(laplacian.slice_mut(inn))
            .and(&field.slice(s![1..nx - 1, 2..ny, 1..nz - 1]))
            .and(&field.slice(s![1..nx - 1, ..ny - 2, 1..nz - 1]))
            .par_for_each(|lap, &yp, &ym| *lap += yp + ym);

        // z-neighbours and subtract 6×centre
        Zip::from(laplacian.slice_mut(inn))
            .and(&field.slice(s![1..nx - 1, 1..ny - 1, 2..nz]))
            .and(&field.slice(s![1..nx - 1, 1..ny - 1, ..nz - 2]))
            .and(&field.slice(s![1..nx - 1, 1..ny - 1, 1..nz - 1]))
            .par_for_each(|lap, &zp, &zm, &c| *lap += zp + zm - 6.0 * c);

        Ok(laplacian)
    }
}
