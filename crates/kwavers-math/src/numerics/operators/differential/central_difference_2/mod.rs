//! # Second-Order Central Difference Operator
//!
//! This module implements the second-order accurate central difference scheme
//! for computing spatial derivatives on uniform Cartesian grids.
//!
//! ## Mathematical Specification
//!
//! For a smooth function u(x), the first derivative is approximated by:
//!
//! ```text
//! du/dx ≈ (u[i+1] - u[i-1]) / (2Δx) + O(Δx²)
//! ```
//!
//! ## Stencil
//!
//! Interior points use a 3-point stencil:
//! ```text
//! [-1, 0, +1] with coefficients [-1/2Δx, 0, +1/2Δx]
//! ```
//!
//! Boundary points use first-order forward/backward differences:
//! ```text
//! Left boundary:  (u[1] - u[0]) / Δx
//! Right boundary: (u[n-1] - u[n-2]) / Δx
//! ```
//!
//! ## Properties
//!
//! - **Order**: 2 (interior), 1 (boundaries)
//! - **Stencil Width**: 3 points
//! - **Conservation**: No (standard central difference)
//! - **Adjoint Consistency**: Yes (symmetric stencil)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use kwavers::math::numerics::operators::differential::{DifferentialOperator, CentralDifference2};
//! use ndarray::Array3;
//!
//! let dx = 0.001; // 1 mm grid spacing
//! let op = CentralDifference2::new(dx, dx, dx)?;
//!
//! let field = Array3::zeros((100, 100, 100));
//! let gradient_x = op.apply_x(field.view())?;
//! ```
//!
//! ## References
//!
//! - Fornberg, B. (1988). "Generation of finite difference formulas on arbitrarily
//!   spaced grids." *Mathematics of Computation*, 51(184), 699-706.
//!   DOI: 10.1090/S0025-5718-1988-0935077-0

use kwavers_core::error::{KwaversResult, NumericalError};
use ndarray::{s, Array3, ArrayView3, Zip};

use super::{traversal, DifferentialOperator};

#[cfg(test)]
mod tests;

/// Second-order central difference operator
///
/// This operator computes spatial derivatives using the classical second-order
/// central difference formula. It is the most commonly used finite difference
/// scheme due to its simplicity and adequate accuracy for many applications.
///
/// # Grid Spacing
///
/// The operator requires uniform grid spacing in each direction. Grid spacings
/// can differ between directions (anisotropic grids), but must be constant
/// within each direction.
///
/// # Boundary Treatment
///
/// Boundaries are handled using first-order forward/backward differences,
/// which reduces the global order of accuracy from 2 to 1 near boundaries.
/// For applications requiring higher accuracy at boundaries, consider using
/// higher-order operators or specialized boundary stencils.
#[derive(Debug, Clone)]
pub struct CentralDifference2 {
    /// Grid spacing in X direction (meters)
    dx: f64,
    /// Grid spacing in Y direction (meters)
    dy: f64,
    /// Grid spacing in Z direction (meters)
    dz: f64,
}

impl CentralDifference2 {
    /// Create a new second-order central difference operator
    ///
    /// # Arguments
    ///
    /// * `dx` - Grid spacing in X direction (meters)
    /// * `dy` - Grid spacing in Y direction (meters)
    /// * `dz` - Grid spacing in Z direction (meters)
    ///
    /// # Returns
    ///
    /// New operator instance
    ///
    /// # Errors
    ///
    /// Returns error if any grid spacing is non-positive
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let op = CentralDifference2::new(0.001, 0.001, 0.001)?; // Isotropic 1mm grid
    /// let op = CentralDifference2::new(0.001, 0.001, 0.002)?; // Anisotropic grid
    /// ```
    pub fn new(dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(NumericalError::InvalidGridSpacing { dx, dy, dz }.into());
        }
        Ok(Self { dx, dy, dz })
    }
}

impl CentralDifference2 {
    /// Apply ∂/∂x into a pre-allocated destination — zero heap allocation.
    ///
    /// Interior: `dst[i] = (f[i+1] − f[i−1]) / (2Δx)` for i ∈ [1, nx−2].
    /// Left boundary (i=0): forward difference `(f[1] − f[0]) / Δx`.
    /// Right boundary (i=nx−1): backward difference `(f[nx−1] − f[nx−2]) / Δx`.
    ///
    /// Uses `Zip` slice-pair to expose element-wise independence to LLVM SIMD
    /// autovectorisation (same pattern as `CentralDifference4/6::apply_x_into`).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn apply_x_into(&self, field: ArrayView3<f64>, dst: &mut Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if nx < 3 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 3,
                actual: nx,
                direction: "X".to_owned(),
            }
            .into());
        }
        debug_assert_eq!(dst.dim(), (nx, ny, nz), "dst shape must match field shape");
        let inv2dx = 0.5 / self.dx;
        let inv_dx = 1.0 / self.dx;

        if field.is_standard_layout() {
            if let Some(field_values) = field.as_slice() {
                if traversal::try_fill_standard_layout(dst, |i, j, k| {
                    let center = traversal::row_major_index(i, j, k, ny, nz);
                    if i == 0 {
                        (field_values[traversal::row_major_index(1, j, k, ny, nz)]
                            - field_values[center])
                            * inv_dx
                    } else if i == nx - 1 {
                        (field_values[center]
                            - field_values[traversal::row_major_index(nx - 2, j, k, ny, nz)])
                            * inv_dx
                    } else {
                        (field_values[traversal::row_major_index(i + 1, j, k, ny, nz)]
                            - field_values[traversal::row_major_index(i - 1, j, k, ny, nz)])
                            * inv2dx
                    }
                }) {
                    return Ok(());
                }
            }
        }

        // Interior: central difference via contiguous slice pairs
        Zip::from(dst.slice_mut(s![1..nx - 1, .., ..]))
            .and(field.slice(s![2..nx, .., ..]))
            .and(field.slice(s![0..nx - 2, .., ..]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) * inv2dx);

        // Left boundary (i=0): forward difference
        Zip::from(dst.slice_mut(s![0, .., ..]))
            .and(field.slice(s![1, .., ..]))
            .and(field.slice(s![0, .., ..]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) * inv_dx);

        // Right boundary (i=nx−1): backward difference
        Zip::from(dst.slice_mut(s![nx - 1, .., ..]))
            .and(field.slice(s![nx - 1, .., ..]))
            .and(field.slice(s![nx - 2, .., ..]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) * inv_dx);

        Ok(())
    }

    /// Apply ∂/∂y into a pre-allocated destination — zero heap allocation.
    ///
    /// Interior: `dst[j] = (f[j+1] − f[j−1]) / (2Δy)` for j ∈ [1, ny−2].
    /// Boundaries: first/last-order forward/backward difference.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn apply_y_into(&self, field: ArrayView3<f64>, dst: &mut Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if ny < 3 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 3,
                actual: ny,
                direction: "Y".to_owned(),
            }
            .into());
        }
        debug_assert_eq!(dst.dim(), (nx, ny, nz), "dst shape must match field shape");
        let inv2dy = 0.5 / self.dy;
        let inv_dy = 1.0 / self.dy;

        if field.is_standard_layout() {
            if let Some(field_values) = field.as_slice() {
                if traversal::try_fill_standard_layout(dst, |i, j, k| {
                    let center = traversal::row_major_index(i, j, k, ny, nz);
                    if j == 0 {
                        (field_values[traversal::row_major_index(i, 1, k, ny, nz)]
                            - field_values[center])
                            * inv_dy
                    } else if j == ny - 1 {
                        (field_values[center]
                            - field_values[traversal::row_major_index(i, ny - 2, k, ny, nz)])
                            * inv_dy
                    } else {
                        (field_values[traversal::row_major_index(i, j + 1, k, ny, nz)]
                            - field_values[traversal::row_major_index(i, j - 1, k, ny, nz)])
                            * inv2dy
                    }
                }) {
                    return Ok(());
                }
            }
        }

        // Interior
        Zip::from(dst.slice_mut(s![.., 1..ny - 1, ..]))
            .and(field.slice(s![.., 2..ny, ..]))
            .and(field.slice(s![.., 0..ny - 2, ..]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) * inv2dy);

        // Bottom boundary (j=0)
        Zip::from(dst.slice_mut(s![.., 0, ..]))
            .and(field.slice(s![.., 1, ..]))
            .and(field.slice(s![.., 0, ..]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) * inv_dy);

        // Top boundary (j=ny−1)
        Zip::from(dst.slice_mut(s![.., ny - 1, ..]))
            .and(field.slice(s![.., ny - 1, ..]))
            .and(field.slice(s![.., ny - 2, ..]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) * inv_dy);

        Ok(())
    }

    /// Apply ∂/∂z into a pre-allocated destination — zero heap allocation.
    ///
    /// Interior: `dst[k] = (f[k+1] − f[k−1]) / (2Δz)` for k ∈ [1, nz−2].
    /// Boundaries: first/last-order forward/backward difference.
    /// The innermost (contiguous) dimension gives the best autovectorisation here.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    pub fn apply_z_into(&self, field: ArrayView3<f64>, dst: &mut Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        if nz < 3 {
            return Err(NumericalError::InsufficientGridPoints {
                required: 3,
                actual: nz,
                direction: "Z".to_owned(),
            }
            .into());
        }
        debug_assert_eq!(dst.dim(), (nx, ny, nz), "dst shape must match field shape");
        let inv2dz = 0.5 / self.dz;
        let inv_dz = 1.0 / self.dz;

        if field.is_standard_layout() {
            if let Some(field_values) = field.as_slice() {
                if traversal::try_fill_standard_layout(dst, |i, j, k| {
                    let center = traversal::row_major_index(i, j, k, ny, nz);
                    if k == 0 {
                        (field_values[traversal::row_major_index(i, j, 1, ny, nz)]
                            - field_values[center])
                            * inv_dz
                    } else if k == nz - 1 {
                        (field_values[center]
                            - field_values[traversal::row_major_index(i, j, nz - 2, ny, nz)])
                            * inv_dz
                    } else {
                        (field_values[traversal::row_major_index(i, j, k + 1, ny, nz)]
                            - field_values[traversal::row_major_index(i, j, k - 1, ny, nz)])
                            * inv2dz
                    }
                }) {
                    return Ok(());
                }
            }
        }

        // Interior (innermost dimension — highest SIMD throughput)
        Zip::from(dst.slice_mut(s![.., .., 1..nz - 1]))
            .and(field.slice(s![.., .., 2..nz]))
            .and(field.slice(s![.., .., 0..nz - 2]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) * inv2dz);

        // Near boundary (k=0)
        Zip::from(dst.slice_mut(s![.., .., 0]))
            .and(field.slice(s![.., .., 1]))
            .and(field.slice(s![.., .., 0]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) * inv_dz);

        // Far boundary (k=nz−1)
        Zip::from(dst.slice_mut(s![.., .., nz - 1]))
            .and(field.slice(s![.., .., nz - 1]))
            .and(field.slice(s![.., .., nz - 2]))
            .for_each(|r, &hi, &lo| *r = (hi - lo) * inv_dz);

        Ok(())
    }
}

impl DifferentialOperator for CentralDifference2 {
    fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let mut result = Array3::zeros(field.dim());
        self.apply_x_into(field, &mut result)?;
        Ok(result)
    }

    fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let mut result = Array3::zeros(field.dim());
        self.apply_y_into(field, &mut result)?;
        Ok(result)
    }

    fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        let mut result = Array3::zeros(field.dim());
        self.apply_z_into(field, &mut result)?;
        Ok(result)
    }

    fn order(&self) -> usize {
        2
    }

    fn stencil_width(&self) -> usize {
        3
    }

    fn is_adjoint_consistent(&self) -> bool {
        true // Symmetric stencil implies adjoint consistency
    }
}
