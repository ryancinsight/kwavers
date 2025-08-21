//! High-order interpolation for staggered grids
//!
//! This module provides interpolation schemes that match the order of
//! the spatial derivatives to maintain consistent accuracy.
//!
//! References:
//! - Fornberg, B. (1998). "A practical guide to pseudospectral methods"
//! - Boyd, J. P. (2001). "Chebyshev and Fourier spectral methods"

use crate::error::KwaversResult;
use ndarray::Array3;

/// High-order interpolation schemes for staggered grids
pub struct StaggeredInterpolation;

impl StaggeredInterpolation {
    /// 4th-order cubic interpolation for half-grid points
    /// f(i+1/2) ≈ (-1/16)f(i-1) + (9/16)f(i) + (9/16)f(i+1) - (1/16)f(i+2)
    pub fn cubic_interpolate(
        field: &ndarray::ArrayView3<f64>,
        axis: usize,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        match axis {
            0 => {
                Ok(Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                    if i > 0 && i < nx - 2 {
                        // Full 4-point cubic interpolation
                        -1.0 / 16.0 * field[[i - 1, j, k]]
                            + 9.0 / 16.0 * field[[i, j, k]]
                            + 9.0 / 16.0 * field[[i + 1, j, k]]
                            - 1.0 / 16.0 * field[[i + 2, j, k]]
                    } else if i == 0 && nx > 2 {
                        // Near left boundary: use forward-biased stencil
                        5.0 / 16.0 * field[[0, j, k]] + 15.0 / 16.0 * field[[1, j, k]]
                            - 5.0 / 16.0 * field[[2, j, k]]
                            + 1.0 / 16.0 * field[[3.min(nx - 1), j, k]]
                    } else if i == nx - 2 && nx > 2 {
                        // Near right boundary: use backward-biased stencil
                        1.0 / 16.0 * field[[(nx - 4).max(0), j, k]]
                            - 5.0 / 16.0 * field[[nx - 3, j, k]]
                            + 15.0 / 16.0 * field[[nx - 2, j, k]]
                            + 5.0 / 16.0 * field[[nx - 1, j, k]]
                    } else if i < nx - 1 {
                        // Fallback to linear interpolation
                        0.5 * (field[[i, j, k]] + field[[i + 1, j, k]])
                    } else {
                        field[[i, j, k]]
                    }
                }))
            }
            1 => {
                Ok(Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                    if j > 0 && j < ny - 2 {
                        // Full 4-point cubic interpolation
                        -1.0 / 16.0 * field[[i, j - 1, k]]
                            + 9.0 / 16.0 * field[[i, j, k]]
                            + 9.0 / 16.0 * field[[i, j + 1, k]]
                            - 1.0 / 16.0 * field[[i, j + 2, k]]
                    } else if j == 0 && ny > 2 {
                        // Near bottom boundary
                        5.0 / 16.0 * field[[i, 0, k]] + 15.0 / 16.0 * field[[i, 1, k]]
                            - 5.0 / 16.0 * field[[i, 2, k]]
                            + 1.0 / 16.0 * field[[i, 3.min(ny - 1), k]]
                    } else if j == ny - 2 && ny > 2 {
                        // Near top boundary
                        1.0 / 16.0 * field[[i, (ny - 4).max(0), k]]
                            - 5.0 / 16.0 * field[[i, ny - 3, k]]
                            + 15.0 / 16.0 * field[[i, ny - 2, k]]
                            + 5.0 / 16.0 * field[[i, ny - 1, k]]
                    } else if j < ny - 1 {
                        // Fallback to linear
                        0.5 * (field[[i, j, k]] + field[[i, j + 1, k]])
                    } else {
                        field[[i, j, k]]
                    }
                }))
            }
            2 => {
                Ok(Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                    if k > 0 && k < nz - 2 {
                        // Full 4-point cubic interpolation
                        -1.0 / 16.0 * field[[i, j, k - 1]]
                            + 9.0 / 16.0 * field[[i, j, k]]
                            + 9.0 / 16.0 * field[[i, j, k + 1]]
                            - 1.0 / 16.0 * field[[i, j, k + 2]]
                    } else if k == 0 && nz > 2 {
                        // Near front boundary
                        5.0 / 16.0 * field[[i, j, 0]] + 15.0 / 16.0 * field[[i, j, 1]]
                            - 5.0 / 16.0 * field[[i, j, 2]]
                            + 1.0 / 16.0 * field[[i, j, 3.min(nz - 1)]]
                    } else if k == nz - 2 && nz > 2 {
                        // Near back boundary
                        1.0 / 16.0 * field[[i, j, (nz - 4).max(0)]]
                            - 5.0 / 16.0 * field[[i, j, nz - 3]]
                            + 15.0 / 16.0 * field[[i, j, nz - 2]]
                            + 5.0 / 16.0 * field[[i, j, nz - 1]]
                    } else if k < nz - 1 {
                        // Fallback to linear
                        0.5 * (field[[i, j, k]] + field[[i, j, k + 1]])
                    } else {
                        field[[i, j, k]]
                    }
                }))
            }
            _ => Err(crate::error::KwaversError::Grid(
                crate::error::GridError::ValidationFailed {
                    field: "axis".to_string(),
                    value: axis.to_string(),
                    constraint: "must be 0, 1, or 2".to_string(),
                },
            )),
        }
    }

    /// 6th-order quintic interpolation for half-grid points
    /// Uses a 6-point stencil for 5th-order accurate interpolation
    pub fn quintic_interpolate(
        field: &ndarray::ArrayView3<f64>,
        axis: usize,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();

        // Quintic interpolation coefficients for f(i+1/2)
        // Derived from Lagrange interpolation
        const C0: f64 = 3.0 / 256.0;
        const C1: f64 = -25.0 / 256.0;
        const C2: f64 = 150.0 / 256.0;
        const C3: f64 = 150.0 / 256.0;
        const C4: f64 = -25.0 / 256.0;
        const C5: f64 = 3.0 / 256.0;

        match axis {
            0 => {
                Ok(Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                    if i >= 2 && i < nx - 3 {
                        // Full 6-point quintic interpolation
                        C0 * field[[i - 2, j, k]]
                            + C1 * field[[i - 1, j, k]]
                            + C2 * field[[i, j, k]]
                            + C3 * field[[i + 1, j, k]]
                            + C4 * field[[i + 2, j, k]]
                            + C5 * field[[i + 3, j, k]]
                    } else if i > 0 && i < nx - 2 {
                        // Fall back to cubic near boundaries
                        -1.0 / 16.0 * field[[i - 1, j, k]]
                            + 9.0 / 16.0 * field[[i, j, k]]
                            + 9.0 / 16.0 * field[[i + 1, j, k]]
                            - 1.0 / 16.0 * field[[i + 2, j, k]]
                    } else if i < nx - 1 {
                        // Fall back to linear at boundaries
                        0.5 * (field[[i, j, k]] + field[[i + 1, j, k]])
                    } else {
                        field[[i, j, k]]
                    }
                }))
            }
            1 => {
                Ok(Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                    if j >= 2 && j < ny - 3 {
                        // Full 6-point quintic interpolation
                        C0 * field[[i, j - 2, k]]
                            + C1 * field[[i, j - 1, k]]
                            + C2 * field[[i, j, k]]
                            + C3 * field[[i, j + 1, k]]
                            + C4 * field[[i, j + 2, k]]
                            + C5 * field[[i, j + 3, k]]
                    } else if j > 0 && j < ny - 2 {
                        // Fall back to cubic
                        -1.0 / 16.0 * field[[i, j - 1, k]]
                            + 9.0 / 16.0 * field[[i, j, k]]
                            + 9.0 / 16.0 * field[[i, j + 1, k]]
                            - 1.0 / 16.0 * field[[i, j + 2, k]]
                    } else if j < ny - 1 {
                        // Fall back to linear
                        0.5 * (field[[i, j, k]] + field[[i, j + 1, k]])
                    } else {
                        field[[i, j, k]]
                    }
                }))
            }
            2 => {
                Ok(Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                    if k >= 2 && k < nz - 3 {
                        // Full 6-point quintic interpolation
                        C0 * field[[i, j, k - 2]]
                            + C1 * field[[i, j, k - 1]]
                            + C2 * field[[i, j, k]]
                            + C3 * field[[i, j, k + 1]]
                            + C4 * field[[i, j, k + 2]]
                            + C5 * field[[i, j, k + 3]]
                    } else if k > 0 && k < nz - 2 {
                        // Fall back to cubic
                        -1.0 / 16.0 * field[[i, j, k - 1]]
                            + 9.0 / 16.0 * field[[i, j, k]]
                            + 9.0 / 16.0 * field[[i, j, k + 1]]
                            - 1.0 / 16.0 * field[[i, j, k + 2]]
                    } else if k < nz - 1 {
                        // Fall back to linear
                        0.5 * (field[[i, j, k]] + field[[i, j, k + 1]])
                    } else {
                        field[[i, j, k]]
                    }
                }))
            }
            _ => Err(crate::error::KwaversError::Grid(
                crate::error::GridError::ValidationFailed {
                    field: "axis".to_string(),
                    value: axis.to_string(),
                    constraint: "must be 0, 1, or 2".to_string(),
                },
            )),
        }
    }
}
