//! k-space operators for spectral methods
//!
//! Implements the k-space differentiation operators used in pseudospectral
//! time-domain (PSTD) methods as described in:
//! - Treeby & Cox (2010) "k-Wave: MATLAB toolbox" J. Biomed. Opt. 15(2)

use crate::grid::Grid;
use ndarray::{s, Array3};
use std::f64::consts::PI;

/// k-space operator collection
#[derive(Debug, Clone)]
pub struct KSpaceOperators {
    pub kx: Array3<f64>,
    pub ky: Array3<f64>,
    pub kz: Array3<f64>,
    pub k_max: f64,
}

/// Compute k-space operators for spectral differentiation
pub fn compute_k_operators(grid: &Grid) -> (KSpaceOperators, f64) {
    let mut kx = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut ky = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let mut kz = Array3::zeros((grid.nx, grid.ny, grid.nz));

    // k-space grid for x direction
    for i in 0..grid.nx {
        let k = if i <= grid.nx / 2 {
            2.0 * PI * i as f64 / (grid.nx as f64 * grid.dx)
        } else {
            2.0 * PI * (i as f64 - grid.nx as f64) / (grid.nx as f64 * grid.dx)
        };
        kx.slice_mut(s![i, .., ..]).fill(k);
    }

    // k-space grid for y direction
    for j in 0..grid.ny {
        let k = if j <= grid.ny / 2 {
            2.0 * PI * j as f64 / (grid.ny as f64 * grid.dy)
        } else {
            2.0 * PI * (j as f64 - grid.ny as f64) / (grid.ny as f64 * grid.dy)
        };
        ky.slice_mut(s![.., j, ..]).fill(k);
    }

    // k-space grid for z direction
    for k in 0..grid.nz {
        let kval = if k <= grid.nz / 2 {
            2.0 * PI * k as f64 / (grid.nz as f64 * grid.dz)
        } else {
            2.0 * PI * (k as f64 - grid.nz as f64) / (grid.nz as f64 * grid.dz)
        };
        kz.slice_mut(s![.., .., k]).fill(kval);
    }

    // Maximum k-vector magnitude
    let kx_max = PI / grid.dx;
    let ky_max = PI / grid.dy;
    let kz_max = PI / grid.dz;
    let k_max = (kx_max * kx_max + ky_max * ky_max + kz_max * kz_max).sqrt();

    let operators = KSpaceOperators { kx, ky, kz, k_max };
    (operators, k_max)
}

/// Apply k-space correction for improved accuracy
pub fn apply_kspace_correction(
    k_operators: &KSpaceOperators,
    grid: &Grid,
    correction_type: &str,
) -> KSpaceOperators {
    match correction_type {
        "exact" => apply_exact_correction(k_operators, grid),
        "pstd" => apply_pstd_correction(k_operators, grid),
        _ => k_operators.clone(),
    }
}

fn apply_exact_correction(ops: &KSpaceOperators, grid: &Grid) -> KSpaceOperators {
    let mut corrected = ops.clone();

    // Apply sinc correction for exact differentiation
    corrected.kx.mapv_inplace(|k| {
        if k.abs() > 0.0 {
            k * (k * grid.dx / 2.0).sin() / (k * grid.dx / 2.0)
        } else {
            0.0
        }
    });

    corrected.ky.mapv_inplace(|k| {
        if k.abs() > 0.0 {
            k * (k * grid.dy / 2.0).sin() / (k * grid.dy / 2.0)
        } else {
            0.0
        }
    });

    corrected.kz.mapv_inplace(|k| {
        if k.abs() > 0.0 {
            k * (k * grid.dz / 2.0).sin() / (k * grid.dz / 2.0)
        } else {
            0.0
        }
    });

    corrected
}

fn apply_pstd_correction(ops: &KSpaceOperators, grid: &Grid) -> KSpaceOperators {
    // PSTD correction as per k-Wave implementation
    apply_exact_correction(ops, grid)
}
