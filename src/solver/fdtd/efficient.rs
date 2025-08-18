//! Efficient FDTD solver implementation with correctness fixes
//!
//! This module provides an efficient version of the FDTD solver that addresses:
//! - Single-pass divergence calculation
//! - Matched interpolation order
//! - Deprecated subgridding API
//! - Unified boundary handling

use crate::grid::Grid;
use crate::medium::Medium;
use crate::error::{KwaversResult, KwaversError, GridError};
use ndarray::{Array3, ArrayView3, ArrayViewMut3, Zip, s};
use std::collections::HashMap;

/// Efficient FDTD solver implementation
pub struct EfficientFdtdSolver {
    grid: Grid,
    fd_coeffs: HashMap<usize, Vec<f64>>,
    spatial_order: usize,
}

impl EfficientFdtdSolver {
    /// Create a new efficient FDTD solver
    pub fn new(grid: Grid, spatial_order: usize) -> KwaversResult<Self> {
        if ![2, 4, 6].contains(&spatial_order) {
            return Err(KwaversError::Grid(GridError::ValidationFailed {
                field: "spatial_order".to_string(),
                value: spatial_order.to_string(),
                constraint: "must be 2, 4, or 6".to_string(),
            }));
        }
        
        let mut fd_coeffs = HashMap::new();
        
        // 2nd order coefficients
        fd_coeffs.insert(2, vec![-0.5, 0.0, 0.5]);
        
        // 4th order coefficients
        fd_coeffs.insert(4, vec![
            1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0
        ]);
        
        // 6th order coefficients  
        fd_coeffs.insert(6, vec![
            -1.0/60.0, 3.0/20.0, -3.0/4.0, 0.0, 3.0/4.0, -3.0/20.0, 1.0/60.0
        ]);
        
        Ok(Self {
            grid,
            fd_coeffs,
            spatial_order,
        })
    }
    
    /// Compute divergence in a single pass for optimal cache performance
    pub fn compute_divergence_single_pass(
        &self,
        vx: &ArrayView3<f64>,
        vy: &ArrayView3<f64>,
        vz: &ArrayView3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = vx.dim();
        let mut divergence = Array3::zeros((nx, ny, nz));
        
        let coeffs = &self.fd_coeffs[&self.spatial_order];
        let half_stencil = coeffs.len() / 2;
        
        // Single pass over interior points
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mut div = 0.0;
                    
                    // X-derivative of vx
                    if i >= half_stencil && i < nx - half_stencil {
                        let mut dvx_dx = 0.0;
                        for (m, &coeff) in coeffs.iter().enumerate() {
                            let idx = i + m - half_stencil;
                            dvx_dx += coeff * vx[[idx, j, k]];
                        }
                        div += dvx_dx / self.grid.dx;
                    } else {
                        // Use lower-order stencil at boundaries
                        div += self.compute_boundary_derivative_x(vx, i, j, k);
                    }
                    
                    // Y-derivative of vy
                    if j >= half_stencil && j < ny - half_stencil {
                        let mut dvy_dy = 0.0;
                        for (m, &coeff) in coeffs.iter().enumerate() {
                            let idx = j + m - half_stencil;
                            dvy_dy += coeff * vy[[i, idx, k]];
                        }
                        div += dvy_dy / self.grid.dy;
                    } else {
                        // Use lower-order stencil at boundaries
                        div += self.compute_boundary_derivative_y(vy, i, j, k);
                    }
                    
                    // Z-derivative of vz
                    if k >= half_stencil && k < nz - half_stencil {
                        let mut dvz_dz = 0.0;
                        for (m, &coeff) in coeffs.iter().enumerate() {
                            let idx = k + m - half_stencil;
                            dvz_dz += coeff * vz[[i, j, idx]];
                        }
                        div += dvz_dz / self.grid.dz;
                    } else {
                        // Use lower-order stencil at boundaries
                        div += self.compute_boundary_derivative_z(vz, i, j, k);
                    }
                    
                    divergence[[i, j, k]] = div;
                }
            }
        }
        
        Ok(divergence)
    }
    
    /// Compute boundary derivative in X direction
    fn compute_boundary_derivative_x(
        &self,
        field: &ArrayView3<f64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> f64 {
        let nx = field.dim().0;
        
        if i == 0 {
            // Forward difference at left boundary
            (field[[1, j, k]] - field[[0, j, k]]) / self.grid.dx
        } else if i == nx - 1 {
            // Backward difference at right boundary
            (field[[nx-1, j, k]] - field[[nx-2, j, k]]) / self.grid.dx
        } else if i == 1 || i == nx - 2 {
            // Centered difference near boundaries
            (field[[i+1, j, k]] - field[[i-1, j, k]]) / (2.0 * self.grid.dx)
        } else {
            // Should not reach here, but use centered difference as fallback
            (field[[i+1, j, k]] - field[[i-1, j, k]]) / (2.0 * self.grid.dx)
        }
    }
    
    /// Compute boundary derivative in Y direction
    fn compute_boundary_derivative_y(
        &self,
        field: &ArrayView3<f64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> f64 {
        let ny = field.dim().1;
        
        if j == 0 {
            (field[[i, 1, k]] - field[[i, 0, k]]) / self.grid.dy
        } else if j == ny - 1 {
            (field[[i, ny-1, k]] - field[[i, ny-2, k]]) / self.grid.dy
        } else if j == 1 || j == ny - 2 {
            (field[[i, j+1, k]] - field[[i, j-1, k]]) / (2.0 * self.grid.dy)
        } else {
            (field[[i, j+1, k]] - field[[i, j-1, k]]) / (2.0 * self.grid.dy)
        }
    }
    
    /// Compute boundary derivative in Z direction
    fn compute_boundary_derivative_z(
        &self,
        field: &ArrayView3<f64>,
        i: usize,
        j: usize,
        k: usize,
    ) -> f64 {
        let nz = field.dim().2;
        
        if k == 0 {
            (field[[i, j, 1]] - field[[i, j, 0]]) / self.grid.dz
        } else if k == nz - 1 {
            (field[[i, j, nz-1]] - field[[i, j, nz-2]]) / self.grid.dz
        } else if k == 1 || k == nz - 2 {
            (field[[i, j, k+1]] - field[[i, j, k-1]]) / (2.0 * self.grid.dz)
        } else {
            (field[[i, j, k+1]] - field[[i, j, k-1]]) / (2.0 * self.grid.dz)
        }
    }
    
    /// Interpolate to staggered grid with order matching spatial derivatives
    pub fn interpolate_to_staggered_matched(
        &self,
        field: &ArrayView3<f64>,
        axis: usize,
        offset: f64,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut result = Array3::zeros((nx, ny, nz));
        
        match self.spatial_order {
            2 => {
                // 2nd-order: Linear interpolation
                self.linear_interpolation(field, &mut result, axis, offset)?;
            }
            4 => {
                // 4th-order: Cubic interpolation
                self.cubic_interpolation(field, &mut result, axis, offset)?;
            }
            6 => {
                // 6th-order: Quintic interpolation
                self.quintic_interpolation(field, &mut result, axis, offset)?;
            }
            _ => {
                return Err(KwaversError::Grid(GridError::ValidationFailed {
                    field: "spatial_order".to_string(),
                    value: self.spatial_order.to_string(),
                    constraint: "must be 2, 4, or 6".to_string(),
                }));
            }
        }
        
        Ok(result)
    }
    
    /// Linear interpolation (2nd-order)
    fn linear_interpolation(
        &self,
        field: &ArrayView3<f64>,
        result: &mut Array3<f64>,
        axis: usize,
        offset: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        
        match axis {
            0 => {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            if i < nx - 1 {
                                result[[i, j, k]] = (1.0 - offset) * field[[i, j, k]] 
                                                  + offset * field[[i + 1, j, k]];
                            } else {
                                result[[i, j, k]] = field[[i, j, k]];
                            }
                        }
                    }
                }
            }
            1 => {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            if j < ny - 1 {
                                result[[i, j, k]] = (1.0 - offset) * field[[i, j, k]] 
                                                  + offset * field[[i, j + 1, k]];
                            } else {
                                result[[i, j, k]] = field[[i, j, k]];
                            }
                        }
                    }
                }
            }
            2 => {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            if k < nz - 1 {
                                result[[i, j, k]] = (1.0 - offset) * field[[i, j, k]] 
                                                  + offset * field[[i, j, k + 1]];
                            } else {
                                result[[i, j, k]] = field[[i, j, k]];
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(KwaversError::Grid(GridError::ValidationFailed {
                    field: "axis".to_string(),
                    value: axis.to_string(),
                    constraint: "must be 0, 1, or 2".to_string(),
                }));
            }
        }
        
        Ok(())
    }
    
    /// Cubic interpolation (4th-order) using Lagrange polynomials
    fn cubic_interpolation(
        &self,
        field: &ArrayView3<f64>,
        result: &mut Array3<f64>,
        axis: usize,
        offset: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        
        // Lagrange polynomial coefficients for cubic interpolation
        // Points at x = -1, 0, 1, 2; interpolate at x = offset
        let x = offset;
        let c0 = -x * (x - 1.0) * (x - 2.0) / 6.0;
        let c1 = (x + 1.0) * (x - 1.0) * (x - 2.0) / 2.0;
        let c2 = -(x + 1.0) * x * (x - 2.0) / 2.0;
        let c3 = (x + 1.0) * x * (x - 1.0) / 6.0;
        
        match axis {
            0 => {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            if i >= 1 && i < nx - 2 {
                                result[[i, j, k]] = c0 * field[[i - 1, j, k]]
                                                  + c1 * field[[i, j, k]]
                                                  + c2 * field[[i + 1, j, k]]
                                                  + c3 * field[[i + 2, j, k]];
                            } else {
                                // Fall back to linear at boundaries
                                if i < nx - 1 {
                                    result[[i, j, k]] = (1.0 - offset) * field[[i, j, k]] 
                                                      + offset * field[[i + 1, j, k]];
                                } else {
                                    result[[i, j, k]] = field[[i, j, k]];
                                }
                            }
                        }
                    }
                }
            }
            1 => {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            if j >= 1 && j < ny - 2 {
                                result[[i, j, k]] = c0 * field[[i, j - 1, k]]
                                                  + c1 * field[[i, j, k]]
                                                  + c2 * field[[i, j + 1, k]]
                                                  + c3 * field[[i, j + 2, k]];
                            } else {
                                // Fall back to linear at boundaries
                                if j < ny - 1 {
                                    result[[i, j, k]] = (1.0 - offset) * field[[i, j, k]] 
                                                      + offset * field[[i, j + 1, k]];
                                } else {
                                    result[[i, j, k]] = field[[i, j, k]];
                                }
                            }
                        }
                    }
                }
            }
            2 => {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            if k >= 1 && k < nz - 2 {
                                result[[i, j, k]] = c0 * field[[i, j, k - 1]]
                                                  + c1 * field[[i, j, k]]
                                                  + c2 * field[[i, j, k + 1]]
                                                  + c3 * field[[i, j, k + 2]];
                            } else {
                                // Fall back to linear at boundaries
                                if k < nz - 1 {
                                    result[[i, j, k]] = (1.0 - offset) * field[[i, j, k]] 
                                                      + offset * field[[i, j, k + 1]];
                                } else {
                                    result[[i, j, k]] = field[[i, j, k]];
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(KwaversError::Grid(GridError::ValidationFailed {
                    field: "axis".to_string(),
                    value: axis.to_string(),
                    constraint: "must be 0, 1, or 2".to_string(),
                }));
            }
        }
        
        Ok(())
    }
    
    /// Quintic interpolation (6th-order) using Lagrange polynomials
    fn quintic_interpolation(
        &self,
        field: &ArrayView3<f64>,
        result: &mut Array3<f64>,
        axis: usize,
        offset: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        
        // Lagrange polynomial coefficients for quintic interpolation
        // Points at x = -2, -1, 0, 1, 2, 3; interpolate at x = offset
        let x = offset;
        let c0 = -x * (x - 1.0) * (x - 2.0) * (x - 3.0) * (x + 1.0) / 120.0;
        let c1 = x * (x - 1.0) * (x - 2.0) * (x - 3.0) * (x + 2.0) / 24.0;
        let c2 = -x * (x - 1.0) * (x - 2.0) * (x + 1.0) * (x + 2.0) / 12.0;
        let c3 = x * (x - 1.0) * (x - 3.0) * (x + 1.0) * (x + 2.0) / 12.0;
        let c4 = -x * (x - 2.0) * (x - 3.0) * (x + 1.0) * (x + 2.0) / 24.0;
        let c5 = (x - 1.0) * (x - 2.0) * (x - 3.0) * (x + 1.0) * (x + 2.0) / 120.0;
        
        match axis {
            0 => {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            if i >= 2 && i < nx - 3 {
                                result[[i, j, k]] = c0 * field[[i - 2, j, k]]
                                                  + c1 * field[[i - 1, j, k]]
                                                  + c2 * field[[i, j, k]]
                                                  + c3 * field[[i + 1, j, k]]
                                                  + c4 * field[[i + 2, j, k]]
                                                  + c5 * field[[i + 3, j, k]];
                            } else {
                                // Fall back to linear at boundaries
                                if i < nx - 1 {
                                    result[[i, j, k]] = (1.0 - offset) * field[[i, j, k]] 
                                                      + offset * field[[i + 1, j, k]];
                                } else {
                                    result[[i, j, k]] = field[[i, j, k]];
                                }
                            }
                        }
                    }
                }
            }
            1 => {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            if j >= 2 && j < ny - 3 {
                                result[[i, j, k]] = c0 * field[[i, j - 2, k]]
                                                  + c1 * field[[i, j - 1, k]]
                                                  + c2 * field[[i, j, k]]
                                                  + c3 * field[[i, j + 1, k]]
                                                  + c4 * field[[i, j + 2, k]]
                                                  + c5 * field[[i, j + 3, k]];
                            } else {
                                // Fall back to linear at boundaries
                                if j < ny - 1 {
                                    result[[i, j, k]] = (1.0 - offset) * field[[i, j, k]] 
                                                      + offset * field[[i, j + 1, k]];
                                } else {
                                    result[[i, j, k]] = field[[i, j, k]];
                                }
                            }
                        }
                    }
                }
            }
            2 => {
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            if k >= 2 && k < nz - 3 {
                                result[[i, j, k]] = c0 * field[[i, j, k - 2]]
                                                  + c1 * field[[i, j, k - 1]]
                                                  + c2 * field[[i, j, k]]
                                                  + c3 * field[[i, j, k + 1]]
                                                  + c4 * field[[i, j, k + 2]]
                                                  + c5 * field[[i, j, k + 3]];
                            } else {
                                // Fall back to linear at boundaries
                                if k < nz - 1 {
                                    result[[i, j, k]] = (1.0 - offset) * field[[i, j, k]] 
                                                      + offset * field[[i, j, k + 1]];
                                } else {
                                    result[[i, j, k]] = field[[i, j, k]];
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(KwaversError::Grid(GridError::ValidationFailed {
                    field: "axis".to_string(),
                    value: axis.to_string(),
                    constraint: "must be 0, 1, or 2".to_string(),
                }));
            }
        }
        
        Ok(())
    }
}

/// Deprecated subgridding functionality
/// 
/// This feature is not fully implemented and should not be used in production.
/// The interpolation and restriction methods are too simplistic for stable
/// coarse-fine grid interfaces.
pub mod deprecated_subgridding {
    use super::*;
    
    /// Add a subgrid region for local refinement
    /// 
    /// **WARNING**: This feature is not fully implemented and will not work correctly.
    /// The update loop does not handle multiple time steps on fine grids, and the
    /// interpolation/restriction methods are inadequate for stability.
    #[deprecated(
        since = "0.4.0",
        note = "Subgridding feature is not fully implemented and should not be used. \
                It will be removed or properly implemented in a future version."
    )]
    pub fn add_subgrid(
        _start: (usize, usize, usize),
        _end: (usize, usize, usize),
    ) -> KwaversResult<()> {
        Err(KwaversError::Grid(GridError::ValidationFailed {
            field: "subgridding".to_string(),
            value: "not implemented".to_string(),
            constraint: "This feature is incomplete and has been deprecated".to_string(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_single_pass_divergence() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let solver = EfficientFdtdSolver::new(grid, 2).unwrap();
        
        let vx = Array3::ones((10, 10, 10));
        let vy = Array3::ones((10, 10, 10));
        let vz = Array3::ones((10, 10, 10));
        
        let div = solver.compute_divergence_single_pass(&vx.view(), &vy.view(), &vz.view()).unwrap();
        
        // Interior points should have zero divergence for constant fields
        for i in 1..9 {
            for j in 1..9 {
                for k in 1..9 {
                    assert!((div[[i, j, k]]).abs() < 1e-10);
                }
            }
        }
    }
    
    #[test]
    fn test_matched_interpolation_order() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        
        // Test 2nd order
        let solver2 = EfficientFdtdSolver::new(grid.clone(), 2).unwrap();
        let field = Array3::ones((10, 10, 10));
        let interp2 = solver2.interpolate_to_staggered_matched(&field.view(), 0, 0.5).unwrap();
        assert_eq!(interp2.dim(), (10, 10, 10));
        
        // Test 4th order
        let solver4 = EfficientFdtdSolver::new(grid.clone(), 4).unwrap();
        let interp4 = solver4.interpolate_to_staggered_matched(&field.view(), 0, 0.5).unwrap();
        assert_eq!(interp4.dim(), (10, 10, 10));
        
        // Test 6th order
        let solver6 = EfficientFdtdSolver::new(grid, 6).unwrap();
        let interp6 = solver6.interpolate_to_staggered_matched(&field.view(), 0, 0.5).unwrap();
        assert_eq!(interp6.dim(), (10, 10, 10));
    }
    
    #[test]
    fn test_deprecated_subgridding() {
        // Should return an error
        let result = deprecated_subgridding::add_subgrid((0, 0, 0), (5, 5, 5));
        assert!(result.is_err());
    }
}