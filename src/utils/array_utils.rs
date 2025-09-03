//! Array utility functions to reduce code duplication
//!
//! Implements DRY principle by providing reusable array initialization functions

use ndarray::{Array3, Array4};

/// Create a zero-initialized 3D array with the given dimensions
///
/// # Arguments
/// * `nx` - Size in x dimension
/// * `ny` - Size in y dimension  
/// * `nz` - Size in z dimension
#[inline]
#[must_use]
pub fn zeros_3d(nx: usize, ny: usize, nz: usize) -> Array3<f64> {
    Array3::zeros((nx, ny, nz))
}

/// Create a zero-initialized 4D array with the given dimensions
///
/// # Arguments
/// * `n0` - Size in first dimension
/// * `n1` - Size in second dimension
/// * `n2` - Size in third dimension
/// * `n3` - Size in fourth dimension
#[inline]
#[must_use]
pub fn zeros_4d(n0: usize, n1: usize, n2: usize, n3: usize) -> Array4<f64> {
    Array4::zeros((n0, n1, n2, n3))
}

/// Create a 3D array filled with a specific value
///
/// # Arguments
/// * `nx` - Size in x dimension
/// * `ny` - Size in y dimension
/// * `nz` - Size in z dimension
/// * `value` - Value to fill the array with
#[inline]
#[must_use]
pub fn filled_3d(nx: usize, ny: usize, nz: usize, value: f64) -> Array3<f64> {
    Array3::from_elem((nx, ny, nz), value)
}

/// Create a 3D array from a grid's dimensions
///
/// # Arguments
/// * `grid` - Grid reference to get dimensions from
#[inline]
pub fn zeros_from_grid(grid: &crate::grid::Grid) -> Array3<f64> {
    zeros_3d(grid.nx, grid.ny, grid.nz)
}

/// Clone and convert a 3D array to ensure contiguous memory layout
///
/// # Arguments
/// * `array` - Array to clone and make contiguous
#[inline]
#[must_use]
pub fn to_contiguous_3d(array: &Array3<f64>) -> Array3<f64> {
    if array.is_standard_layout() {
        array.clone()
    } else {
        array.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_3d() {
        let arr = zeros_3d(2, 3, 4);
        assert_eq!(arr.shape(), &[2, 3, 4]);
        assert!(arr.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_filled_3d() {
        let arr = filled_3d(2, 2, 2, 5.0);
        assert_eq!(arr.shape(), &[2, 2, 2]);
        assert!(arr.iter().all(|&x| x == 5.0));
    }
}
