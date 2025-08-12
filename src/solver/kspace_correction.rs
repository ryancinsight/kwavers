//! Unified k-space correction module for spectral methods
//!
//! This module provides a consistent, theoretically sound k-space correction
//! approach for all spectral-based solvers, based on the k-Wave toolbox methodology.
//!
//! # Theory
//!
//! The k-space pseudospectral method introduces numerical errors due to:
//! 1. Finite difference approximation of temporal derivatives
//! 2. Spatial discretization on a finite grid
//!
//! The k-space correction compensates for these errors using a modified
//! dispersion relation that accounts for the discrete nature of the grid.
//!
//! # References
//!
//! - Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the simulation
//!   and reconstruction of photoacoustic wave fields." Journal of Biomedical Optics, 15(2).
//! - Tabei, M., Mast, T. D., & Waag, R. C. (2002). "A k-space method for coupled
//!   first-order acoustic propagation equations." JASA, 111(1), 53-63.

use ndarray::{Array3, Zip};
use std::f64::consts::PI;
use crate::grid::Grid;

/// K-space correction configuration
#[derive(Debug, Clone, Copy)]
pub struct KSpaceCorrectionConfig {
    /// Enable k-space correction
    pub enabled: bool,
    /// Correction order (typically 2 for most applications)
    pub order: usize,
    /// Use exact k-Wave formulation
    pub use_kwave_exact: bool,
    /// CFL number for stability
    pub cfl_number: f64,
}

impl Default for KSpaceCorrectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            order: 2,
            use_kwave_exact: true,
            cfl_number: 0.3,
        }
    }
}

/// Compute unified k-space correction factors using k-Wave methodology
///
/// This implements the exact k-space correction as used in k-Wave, which
/// accounts for both spatial and temporal discretization errors.
///
/// # Arguments
/// * `grid` - Computational grid
/// * `config` - K-space correction configuration
/// * `dt` - Time step
/// * `c_ref` - Reference sound speed
///
/// # Returns
/// Array of k-space correction factors (kappa)
pub fn compute_kspace_correction(
    grid: &Grid,
    config: &KSpaceCorrectionConfig,
    dt: f64,
    c_ref: f64,
) -> Array3<f64> {
    if !config.enabled {
        return Array3::ones((grid.nx, grid.ny, grid.nz));
    }

    if config.use_kwave_exact {
        compute_kwave_correction(grid, dt, c_ref, config.cfl_number)
    } else {
        compute_sinc_correction(grid, config.order)
    }
}

/// Compute k-Wave exact k-space correction
///
/// This implements Equation 11 from Treeby & Cox (2010), which provides
/// exact dispersion compensation for the k-space pseudospectral method.
fn compute_kwave_correction(
    grid: &Grid,
    dt: f64,
    c_ref: f64,
    cfl: f64,
) -> Array3<f64> {
    let mut kappa = Array3::ones((grid.nx, grid.ny, grid.nz));
    
    // Compute CFL-based correction factor
    let c_ref_dt = c_ref * dt;
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                // Compute wavenumber components
                let kx = compute_wavenumber_component(i, grid.nx, grid.dx);
                let ky = compute_wavenumber_component(j, grid.ny, grid.dy);
                let kz = compute_wavenumber_component(k, grid.nz, grid.dz);
                
                // Compute k-space operator for each dimension
                let kx_op = 2.0 * (kx * grid.dx / 2.0).sin() / grid.dx;
                let ky_op = 2.0 * (ky * grid.dy / 2.0).sin() / grid.dy;
                let kz_op = 2.0 * (kz * grid.dz / 2.0).sin() / grid.dz;
                
                // Total k-space operator magnitude
                let k_op_sq = kx_op * kx_op + ky_op * ky_op + kz_op * kz_op;
                
                if k_op_sq > 0.0 {
                    // Compute exact k-space correction factor
                    // This accounts for both spatial and temporal discretization
                    let k_mag_sq = kx * kx + ky * ky + kz * kz;
                    
                    // Temporal correction term
                    let temporal_term = (c_ref_dt * k_op_sq.sqrt() / 2.0).sin();
                    let temporal_correction = if temporal_term.abs() > 1e-12 {
                        2.0 * temporal_term / (c_ref_dt * k_op_sq.sqrt())
                    } else {
                        1.0
                    };
                    
                    // Spatial correction term (sinc function)
                    let spatial_correction = if k_mag_sq > 0.0 {
                        k_op_sq.sqrt() / k_mag_sq.sqrt()
                    } else {
                        1.0
                    };
                    
                    // Combined correction factor
                    kappa[[i, j, k]] = temporal_correction * spatial_correction;
                    
                    // Apply stability limit
                    let max_correction = 1.0 / cfl;
                    if kappa[[i, j, k]] > max_correction {
                        kappa[[i, j, k]] = max_correction;
                    }
                } else {
                    kappa[[i, j, k]] = 1.0;
                }
            }
        }
    }
    
    kappa
}

/// Compute sinc-based k-space correction (simpler alternative)
///
/// This implements a simpler sinc-based correction that only accounts
/// for spatial discretization errors.
fn compute_sinc_correction(grid: &Grid, order: usize) -> Array3<f64> {
    let mut kappa = Array3::ones((grid.nx, grid.ny, grid.nz));
    
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                // Compute wavenumber components
                let kx = compute_wavenumber_component(i, grid.nx, grid.dx);
                let ky = compute_wavenumber_component(j, grid.ny, grid.dy);
                let kz = compute_wavenumber_component(k, grid.nz, grid.dz);
                
                // Compute sinc correction for each dimension
                let sinc_x = compute_sinc_factor(kx, grid.dx, order);
                let sinc_y = compute_sinc_factor(ky, grid.dy, order);
                let sinc_z = compute_sinc_factor(kz, grid.dz, order);
                
                // Combined correction factor
                kappa[[i, j, k]] = sinc_x * sinc_y * sinc_z;
            }
        }
    }
    
    kappa
}

/// Compute wavenumber component for a given index
#[inline]
fn compute_wavenumber_component(index: usize, n: usize, dx: f64) -> f64 {
    if index <= n / 2 {
        2.0 * PI * index as f64 / (n as f64 * dx)
    } else {
        2.0 * PI * (index as f64 - n as f64) / (n as f64 * dx)
    }
}

/// Compute sinc correction factor for a single dimension
#[inline]
fn compute_sinc_factor(k: f64, dx: f64, order: usize) -> f64 {
    let arg = k * dx / 2.0;
    
    if arg.abs() < 1e-12 {
        return 1.0;
    }
    
    let sinc = arg.sin() / arg;
    
    // Apply higher-order correction if requested
    match order {
        1 => sinc,
        2 => sinc.powi(2),
        3 => sinc.powi(3),
        4 => sinc.powi(4),
        _ => sinc,
    }
}

/// Apply k-space correction to spectral field
pub fn apply_correction(
    field_k: &mut Array3<num_complex::Complex<f64>>,
    kappa: &Array3<f64>,
) {
    Zip::from(field_k)
        .and(kappa)
        .for_each(|f, &k| {
            *f *= num_complex::Complex::new(k, 0.0);
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kspace_correction_unity_at_origin() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let config = KSpaceCorrectionConfig::default();
        let kappa = compute_kspace_correction(&grid, &config, 1e-6, 1500.0);
        
        // Correction should be 1.0 at k=0
        assert!((kappa[[0, 0, 0]] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_sinc_correction_symmetry() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let kappa = compute_sinc_correction(&grid, 2);
        
        // Check symmetry
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        
        // Should be symmetric about the center
        assert!((kappa[[1, 0, 0]] - kappa[[nx-1, 0, 0]]).abs() < 1e-10);
        assert!((kappa[[0, 1, 0]] - kappa[[0, ny-1, 0]]).abs() < 1e-10);
        assert!((kappa[[0, 0, 1]] - kappa[[0, 0, nz-1]]).abs() < 1e-10);
    }
}