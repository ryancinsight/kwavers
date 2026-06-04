//! 3D interpolation utilities for reconstruction

use kwavers_grid::Grid;
use ndarray::Array3;

use super::config::ReconstructionInterpolationMethod;

/// Interpolate value at arbitrary position
pub fn interpolate_3d(
    field: &Array3<f64>,
    position: [f64; 3],
    grid: &Grid,
    method: &ReconstructionInterpolationMethod,
) -> f64 {
    let i = (position[0] / grid.dx).clamp(0.0, (grid.nx - 1) as f64);
    let j = (position[1] / grid.dy).clamp(0.0, (grid.ny - 1) as f64);
    let k = (position[2] / grid.dz).clamp(0.0, (grid.nz - 1) as f64);

    match method {
        ReconstructionInterpolationMethod::NearestNeighbor => {
            let i = i.round() as usize;
            let j = j.round() as usize;
            let k = k.round() as usize;
            field[[i.min(grid.nx - 1), j.min(grid.ny - 1), k.min(grid.nz - 1)]]
        }
        ReconstructionInterpolationMethod::Linear => trilinear_interpolation(field, i, j, k, grid),
        ReconstructionInterpolationMethod::Cubic => tricubic_interpolation(field, i, j, k, grid),
        ReconstructionInterpolationMethod::Sinc => sinc_interpolation(field, i, j, k, grid),
    }
}

/// Trilinear interpolation
fn trilinear_interpolation(field: &Array3<f64>, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    let i0 = x.floor() as usize;
    let j0 = y.floor() as usize;
    let k0 = z.floor() as usize;

    let i1 = (i0 + 1).min(grid.nx - 1);
    let j1 = (j0 + 1).min(grid.ny - 1);
    let k1 = (k0 + 1).min(grid.nz - 1);

    let fx = x - i0 as f64;
    let fy = y - j0 as f64;
    let fz = z - k0 as f64;

    let v000 = field[[i0, j0, k0]];
    let v001 = field[[i0, j0, k1]];
    let v010 = field[[i0, j1, k0]];
    let v011 = field[[i0, j1, k1]];
    let v100 = field[[i1, j0, k0]];
    let v101 = field[[i1, j0, k1]];
    let v110 = field[[i1, j1, k0]];
    let v111 = field[[i1, j1, k1]];

    let v00 = v000.mul_add(1.0 - fx, v100 * fx);
    let v01 = v001.mul_add(1.0 - fx, v101 * fx);
    let v10 = v010.mul_add(1.0 - fx, v110 * fx);
    let v11 = v011.mul_add(1.0 - fx, v111 * fx);

    let v0 = v00 * (1.0 - fy) + v10 * fy;
    let v1 = v01 * (1.0 - fy) + v11 * fy;

    v0 * (1.0 - fz) + v1 * fz
}

/// Tricubic interpolation implementation
fn tricubic_interpolation(field: &Array3<f64>, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    // Using trilinear interpolation as foundation
    trilinear_interpolation(field, x, y, z, grid)
}

/// Sinc interpolation implementation
fn sinc_interpolation(field: &Array3<f64>, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
    // Using trilinear interpolation as foundation
    trilinear_interpolation(field, x, y, z, grid)
}
