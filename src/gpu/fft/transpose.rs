//! Matrix transpose operations for FFT
//!
//! Provides efficient transpose operations needed for multi-dimensional FFTs

use ndarray::{Array3, Axis};
use num_complex::Complex;

/// Transpose operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransposeOperation {
    /// Transpose XY dimensions
    XY,
    /// Transpose XZ dimensions
    XZ,
    /// Transpose YZ dimensions
    YZ,
}

/// Perform 3D array transpose
pub fn transpose_3d(data: &mut Array3<Complex<f64>>, operation: TransposeOperation) {
    match operation {
        TransposeOperation::XY => transpose_xy(data),
        TransposeOperation::XZ => transpose_xz(data),
        TransposeOperation::YZ => transpose_yz(data),
    }
}

/// Transpose XY dimensions (swap first two axes)
fn transpose_xy(data: &mut Array3<Complex<f64>>) {
    let shape = data.dim();
    let (nx, ny, nz) = shape;

    // Create transposed array
    let mut transposed = Array3::zeros((ny, nx, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                transposed[[j, i, k]] = data[[i, j, k]];
            }
        }
    }

    *data = transposed;
}

/// Transpose XZ dimensions (swap first and third axes)
fn transpose_xz(data: &mut Array3<Complex<f64>>) {
    let shape = data.dim();
    let (nx, ny, nz) = shape;

    // Create transposed array
    let mut transposed = Array3::zeros((nz, ny, nx));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                transposed[[k, j, i]] = data[[i, j, k]];
            }
        }
    }

    *data = transposed;
}

/// Transpose YZ dimensions (swap second and third axes)
fn transpose_yz(data: &mut Array3<Complex<f64>>) {
    let shape = data.dim();
    let (nx, ny, nz) = shape;

    // Create transposed array
    let mut transposed = Array3::zeros((nx, nz, ny));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                transposed[[i, k, j]] = data[[i, j, k]];
            }
        }
    }

    *data = transposed;
}

/// In-place transpose for contiguous memory (cache-friendly)
pub fn transpose_inplace_2d(data: &mut [Complex<f64>], rows: usize, cols: usize) {
    assert_eq!(data.len(), rows * cols);

    // Use cycle-following algorithm for in-place transpose
    let mut visited = vec![false; data.len()];

    for start in 0..data.len() {
        if visited[start] {
            continue;
        }

        let mut current = start;
        loop {
            visited[current] = true;
            let next = (current % rows) * cols + current / rows;

            if next == start {
                break;
            }

            data.swap(current, next);
            current = next;
        }
    }
}
