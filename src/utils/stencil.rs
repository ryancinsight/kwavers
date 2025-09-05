//! Zero-cost abstractions for stencil operations
//!
//! This module provides efficient, generic stencil operations that compile
//! down to optimal machine code with no runtime overhead.

use ndarray::{ArrayView3, ArrayViewMut3};
use std::ops::Add;

/// Trait for types that can be used in stencil operations
pub trait StencilValue: Copy + Add<Output = Self> + Default {
    fn mul_f64(self, rhs: f64) -> Self;
}

impl StencilValue for f64 {
    #[inline(always)]
    fn mul_f64(self, rhs: f64) -> Self {
        self * rhs
    }
}

impl StencilValue for f32 {
    #[inline(always)]
    fn mul_f64(self, rhs: f64) -> Self {
        self * (rhs as f32)
    }
}

/// A zero-cost abstraction for 3D Laplacian operations
#[inline(always)]
pub fn laplacian_3d<T: StencilValue>(
    input: ArrayView3<T>,
    mut output: ArrayViewMut3<T>,
    dx_inv2: f64,
    dy_inv2: f64,
    dz_inv2: f64,
) {
    let (nx, ny, nz) = input.dim();

    // Process interior points
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let center = input[[i, j, k]];

                // Compute Laplacian with optimal memory access pattern
                let lap_x = (input[[i + 1, j, k]] + input[[i - 1, j, k]]).mul_f64(dx_inv2);
                let lap_y = (input[[i, j + 1, k]] + input[[i, j - 1, k]]).mul_f64(dy_inv2);
                let lap_z = (input[[i, j, k + 1]] + input[[i, j, k - 1]]).mul_f64(dz_inv2);
                let center_term = center.mul_f64((dx_inv2 + dy_inv2 + dz_inv2) * -2.0);

                output[[i, j, k]] = lap_x + lap_y + lap_z + center_term;
            }
        }
    }
}

/// A zero-cost abstraction for gradient computation
#[inline(always)]
pub fn gradient_3d<T: StencilValue>(
    input: ArrayView3<T>,
    grad_x: ArrayViewMut3<T>,
    grad_y: ArrayViewMut3<T>,
    grad_z: ArrayViewMut3<T>,
    dx_inv: f64,
    dy_inv: f64,
    dz_inv: f64,
    order: StencilOrder,
) {
    match order {
        StencilOrder::Second => {
            gradient_3d_order2(input, grad_x, grad_y, grad_z, dx_inv, dy_inv, dz_inv);
        }
        StencilOrder::Fourth => {
            gradient_3d_order4(input, grad_x, grad_y, grad_z, dx_inv, dy_inv, dz_inv);
        }
    }
}

/// Stencil order for finite difference schemes
#[derive(Debug, Clone, Copy)]
pub enum StencilOrder {
    Second,
    Fourth,
}

#[inline(always)]
fn gradient_3d_order2<T: StencilValue>(
    input: ArrayView3<T>,
    mut grad_x: ArrayViewMut3<T>,
    mut grad_y: ArrayViewMut3<T>,
    mut grad_z: ArrayViewMut3<T>,
    dx_inv: f64,
    dy_inv: f64,
    dz_inv: f64,
) {
    let (nx, ny, nz) = input.dim();

    // X-gradient
    use ndarray::s;
    let x_interior = s![1..nx - 1, .., ..];
    ndarray::Zip::indexed(&mut grad_x.slice_mut(x_interior)).for_each(|(i, j, k), grad| {
        let i = i + 1; // Adjust for slice offset
        let diff = input[[i + 1, j, k]] + input[[i - 1, j, k]].mul_f64(-1.0);
        *grad = diff.mul_f64(0.5 * dx_inv);
    });

    // Y-gradient
    let y_interior = s![.., 1..ny - 1, ..];
    ndarray::Zip::indexed(&mut grad_y.slice_mut(y_interior)).for_each(|(i, j, k), grad| {
        let j = j + 1; // Adjust for slice offset
        let diff = input[[i, j + 1, k]] + input[[i, j - 1, k]].mul_f64(-1.0);
        *grad = diff.mul_f64(0.5 * dy_inv);
    });

    // Z-gradient
    let z_interior = s![.., .., 1..nz - 1];
    ndarray::Zip::indexed(&mut grad_z.slice_mut(z_interior)).for_each(|(i, j, k), grad| {
        let k = k + 1; // Adjust for slice offset
        let diff = input[[i, j, k + 1]] + input[[i, j, k - 1]].mul_f64(-1.0);
        *grad = diff.mul_f64(0.5 * dz_inv);
    });
}

#[inline(always)]
fn gradient_3d_order4<T: StencilValue>(
    input: ArrayView3<T>,
    mut grad_x: ArrayViewMut3<T>,
    mut grad_y: ArrayViewMut3<T>,
    mut grad_z: ArrayViewMut3<T>,
    dx_inv: f64,
    dy_inv: f64,
    dz_inv: f64,
) {
    let (nx, ny, nz) = input.dim();

    // Fourth-order central differences coefficients
    const C1: f64 = 8.0 / 12.0;
    const C2: f64 = -1.0 / 12.0;

    // X-gradient (interior points only for 4th order)
    for i in 2..nx - 2 {
        for j in 0..ny {
            for k in 0..nz {
                let val = input[[i + 2, j, k]].mul_f64(C2)
                    + input[[i + 1, j, k]].mul_f64(C1)
                    + input[[i - 1, j, k]].mul_f64(-C1)
                    + input[[i - 2, j, k]].mul_f64(-C2);
                grad_x[[i, j, k]] = val.mul_f64(dx_inv);
            }
        }
    }

    // Y-gradient (interior points only for 4th order)
    for i in 0..nx {
        for j in 2..ny - 2 {
            for k in 0..nz {
                let val = input[[i, j + 2, k]].mul_f64(C2)
                    + input[[i, j + 1, k]].mul_f64(C1)
                    + input[[i, j - 1, k]].mul_f64(-C1)
                    + input[[i, j - 2, k]].mul_f64(-C2);
                grad_y[[i, j, k]] = val.mul_f64(dy_inv);
            }
        }
    }

    // Z-gradient (interior points only for 4th order)
    for i in 0..nx {
        for j in 0..ny {
            for k in 2..nz - 2 {
                let val = input[[i, j, k + 2]].mul_f64(C2)
                    + input[[i, j, k + 1]].mul_f64(C1)
                    + input[[i, j, k - 1]].mul_f64(-C1)
                    + input[[i, j, k - 2]].mul_f64(-C2);
                grad_z[[i, j, k]] = val.mul_f64(dz_inv);
            }
        }
    }
}

/// Generic stencil application with compile-time optimization
#[inline(always)]
pub fn apply_stencil<T, S, const N: usize>(
    input: ArrayView3<T>,
    output: ArrayViewMut3<T>,
    stencil: S,
) where
    T: StencilValue,
    S: Stencil<T, N>,
{
    stencil.apply(input, output);
}

/// Trait for custom stencil operations
pub trait Stencil<T: StencilValue, const N: usize> {
    fn apply(&self, input: ArrayView3<T>, output: ArrayViewMut3<T>);
}

/// A 7-point stencil for 3D Laplacian
/// Configuration for 3D Laplacian computation
#[derive(Debug, Clone, Copy)]
pub struct Laplacian3D {
    pub dx_inv2: f64,
    pub dy_inv2: f64,
    pub dz_inv2: f64,
}

impl<T: StencilValue> Stencil<T, 7> for Laplacian3D {
    #[inline(always)]
    fn apply(&self, input: ArrayView3<T>, output: ArrayViewMut3<T>) {
        laplacian_3d(input, output, self.dx_inv2, self.dy_inv2, self.dz_inv2);
    }
}

/// Iterator-based stencil operations for maximum flexibility
#[derive(Debug)]
pub struct StencilIterator3D<'a, T> {
    array: ArrayView3<'a, T>,
    position: (usize, usize, usize),
    bounds: (usize, usize, usize),
}

impl<'a, T: StencilValue> StencilIterator3D<'a, T> {
    #[must_use]
    pub fn new(array: ArrayView3<'a, T>) -> Self {
        let bounds = array.dim();
        Self {
            array,
            position: (1, 1, 1),
            bounds: (bounds.0 - 1, bounds.1 - 1, bounds.2 - 1),
        }
    }

    /// Get the value at the current position
    #[inline(always)]
    #[must_use]
    pub fn center(&self) -> T {
        let (i, j, k) = self.position;
        self.array[[i, j, k]]
    }

    /// Get neighboring values for a 7-point stencil
    #[inline(always)]
    #[must_use]
    pub fn neighbors_7point(&self) -> [T; 6] {
        let (i, j, k) = self.position;
        [
            self.array[[i - 1, j, k]],
            self.array[[i + 1, j, k]],
            self.array[[i, j - 1, k]],
            self.array[[i, j + 1, k]],
            self.array[[i, j, k - 1]],
            self.array[[i, j, k + 1]],
        ]
    }
}

impl<'a, T: StencilValue> Iterator for StencilIterator3D<'a, T> {
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let (i, j, k) = self.position;
        let (max_i, max_j, max_k) = self.bounds;

        if k >= max_k {
            return None;
        }

        let result = Some(self.position);

        // Advance position
        self.position = if i < max_i - 1 {
            (i + 1, j, k)
        } else if j < max_j - 1 {
            (1, j + 1, k)
        } else {
            (1, 1, k + 1)
        };

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_laplacian_3d() {
        let mut input = Array3::<f64>::zeros((5, 5, 5));
        let mut output = Array3::<f64>::zeros((5, 5, 5));

        // Set center value
        input[[2, 2, 2]] = 1.0;

        laplacian_3d(input.view(), output.view_mut(), 1.0, 1.0, 1.0);

        // Check that Laplacian is computed correctly
        assert_eq!(output[[2, 2, 2]], -6.0);
        assert_eq!(output[[1, 2, 2]], 1.0);
        assert_eq!(output[[3, 2, 2]], 1.0);
    }
}
