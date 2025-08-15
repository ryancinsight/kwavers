//! Consolidated differential operators for the entire codebase
//! 
//! This module provides the single source of truth for all differential
//! operators (gradient, divergence, curl, Laplacian) to ensure consistency
//! and avoid duplicate implementations across the codebase.
//!
//! # Design Principles
//! - **SSOT/SPOT**: Single implementation for each operator
//! - **Zero-copy**: Uses ArrayView for input, efficient memory access
//! - **Configurable accuracy**: Support for 2nd, 4th, and 6th order schemes
//! - **Domain-agnostic**: Works with any Grid configuration

use ndarray::{Array3, ArrayView3};
use crate::grid::Grid;
use crate::error::KwaversResult;

/// Spatial accuracy order for finite difference schemes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpatialOrder {
    /// Second-order accurate (3-point stencil)
    Second,
    /// Fourth-order accurate (5-point stencil)
    Fourth,
    /// Sixth-order accurate (7-point stencil)
    Sixth,
}

/// Finite difference coefficients for different orders
pub struct FDCoefficients;

impl FDCoefficients {
    /// Get coefficients for first derivative
    pub fn first_derivative(order: SpatialOrder) -> Vec<f64> {
        match order {
            SpatialOrder::Second => vec![0.5],
            SpatialOrder::Fourth => vec![2.0/3.0, -1.0/12.0],
            SpatialOrder::Sixth => vec![3.0/4.0, -3.0/20.0, 1.0/60.0],
        }
    }
    
    /// Get coefficients for second derivative
    pub fn second_derivative(order: SpatialOrder) -> Vec<f64> {
        match order {
            SpatialOrder::Second => vec![1.0],
            SpatialOrder::Fourth => vec![4.0/3.0, -1.0/12.0],
            SpatialOrder::Sixth => vec![3.0/2.0, -3.0/20.0, 1.0/90.0],
        }
    }
}

/// Compute gradient of a scalar field
/// 
/// Returns (∂f/∂x, ∂f/∂y, ∂f/∂z)
pub fn gradient(
    field: ArrayView3<f64>,
    grid: &Grid,
    order: SpatialOrder,
) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
    let (nx, ny, nz) = field.dim();
    let mut grad_x = Array3::zeros((nx, ny, nz));
    let mut grad_y = Array3::zeros((nx, ny, nz));
    let mut grad_z = Array3::zeros((nx, ny, nz));
    
    let coeffs = FDCoefficients::first_derivative(order);
    let stencil_size = coeffs.len();
    
    // X-direction gradient
    for k in 0..nz {
        for j in 0..ny {
            for i in stencil_size..nx-stencil_size {
                let mut sum = 0.0;
                for (s, &coeff) in coeffs.iter().enumerate() {
                    let offset = s + 1;
                    sum += coeff * (field[[i+offset, j, k]] - field[[i-offset, j, k]]);
                }
                grad_x[[i, j, k]] = sum / grid.dx;
            }
        }
    }
    
    // Y-direction gradient
    for k in 0..nz {
        for j in stencil_size..ny-stencil_size {
            for i in 0..nx {
                let mut sum = 0.0;
                for (s, &coeff) in coeffs.iter().enumerate() {
                    let offset = s + 1;
                    sum += coeff * (field[[i, j+offset, k]] - field[[i, j-offset, k]]);
                }
                grad_y[[i, j, k]] = sum / grid.dy;
            }
        }
    }
    
    // Z-direction gradient
    for k in stencil_size..nz-stencil_size {
        for j in 0..ny {
            for i in 0..nx {
                let mut sum = 0.0;
                for (s, &coeff) in coeffs.iter().enumerate() {
                    let offset = s + 1;
                    sum += coeff * (field[[i, j, k+offset]] - field[[i, j, k-offset]]);
                }
                grad_z[[i, j, k]] = sum / grid.dz;
            }
        }
    }
    
    Ok((grad_x, grad_y, grad_z))
}

/// Compute divergence of a vector field
/// 
/// Returns ∇·v = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z
pub fn divergence(
    vx: ArrayView3<f64>,
    vy: ArrayView3<f64>,
    vz: ArrayView3<f64>,
    grid: &Grid,
    order: SpatialOrder,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = vx.dim();
    let mut div = Array3::zeros((nx, ny, nz));
    
    let coeffs = FDCoefficients::first_derivative(order);
    let stencil_size = coeffs.len();
    
    // Compute divergence at interior points
    for k in stencil_size..nz-stencil_size {
        for j in stencil_size..ny-stencil_size {
            for i in stencil_size..nx-stencil_size {
                let mut dvx_dx = 0.0;
                let mut dvy_dy = 0.0;
                let mut dvz_dz = 0.0;
                
                for (s, &coeff) in coeffs.iter().enumerate() {
                    let offset = s + 1;
                    dvx_dx += coeff * (vx[[i+offset, j, k]] - vx[[i-offset, j, k]]);
                    dvy_dy += coeff * (vy[[i, j+offset, k]] - vy[[i, j-offset, k]]);
                    dvz_dz += coeff * (vz[[i, j, k+offset]] - vz[[i, j, k-offset]]);
                }
                
                div[[i, j, k]] = dvx_dx / grid.dx + dvy_dy / grid.dy + dvz_dz / grid.dz;
            }
        }
    }
    
    Ok(div)
}

/// Compute Laplacian of a scalar field
/// 
/// Returns ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
pub fn laplacian(
    field: ArrayView3<f64>,
    grid: &Grid,
    order: SpatialOrder,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = field.dim();
    let mut lap = Array3::zeros((nx, ny, nz));
    
    let coeffs = FDCoefficients::second_derivative(order);
    let stencil_size = coeffs.len();
    
    let dx2_inv = 1.0 / (grid.dx * grid.dx);
    let dy2_inv = 1.0 / (grid.dy * grid.dy);
    let dz2_inv = 1.0 / (grid.dz * grid.dz);
    
    // Compute Laplacian at interior points
    for k in stencil_size..nz-stencil_size {
        for j in stencil_size..ny-stencil_size {
            for i in stencil_size..nx-stencil_size {
                let mut d2f_dx2 = -2.0 * field[[i, j, k]];
                let mut d2f_dy2 = -2.0 * field[[i, j, k]];
                let mut d2f_dz2 = -2.0 * field[[i, j, k]];
                
                for (s, &coeff) in coeffs.iter().enumerate() {
                    let offset = s + 1;
                    d2f_dx2 += coeff * (field[[i+offset, j, k]] + field[[i-offset, j, k]]);
                    d2f_dy2 += coeff * (field[[i, j+offset, k]] + field[[i, j-offset, k]]);
                    d2f_dz2 += coeff * (field[[i, j, k+offset]] + field[[i, j, k-offset]]);
                }
                
                lap[[i, j, k]] = d2f_dx2 * dx2_inv + d2f_dy2 * dy2_inv + d2f_dz2 * dz2_inv;
            }
        }
    }
    
    Ok(lap)
}

/// Compute curl of a vector field
/// 
/// Returns ∇×v = (∂vz/∂y - ∂vy/∂z, ∂vx/∂z - ∂vz/∂x, ∂vy/∂x - ∂vx/∂y)
pub fn curl(
    vx: ArrayView3<f64>,
    vy: ArrayView3<f64>,
    vz: ArrayView3<f64>,
    grid: &Grid,
    order: SpatialOrder,
) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
    let (nx, ny, nz) = vx.dim();
    let mut curl_x = Array3::zeros((nx, ny, nz));
    let mut curl_y = Array3::zeros((nx, ny, nz));
    let mut curl_z = Array3::zeros((nx, ny, nz));
    
    let coeffs = FDCoefficients::first_derivative(order);
    let stencil_size = coeffs.len();
    
    // Compute curl at interior points
    for k in stencil_size..nz-stencil_size {
        for j in stencil_size..ny-stencil_size {
            for i in stencil_size..nx-stencil_size {
                let mut dvz_dy = 0.0;
                let mut dvy_dz = 0.0;
                let mut dvx_dz = 0.0;
                let mut dvz_dx = 0.0;
                let mut dvy_dx = 0.0;
                let mut dvx_dy = 0.0;
                
                for (s, &coeff) in coeffs.iter().enumerate() {
                    let offset = s + 1;
                    dvz_dy += coeff * (vz[[i, j+offset, k]] - vz[[i, j-offset, k]]);
                    dvy_dz += coeff * (vy[[i, j, k+offset]] - vy[[i, j, k-offset]]);
                    dvx_dz += coeff * (vx[[i, j, k+offset]] - vx[[i, j, k-offset]]);
                    dvz_dx += coeff * (vz[[i+offset, j, k]] - vz[[i-offset, j, k]]);
                    dvy_dx += coeff * (vy[[i+offset, j, k]] - vy[[i-offset, j, k]]);
                    dvx_dy += coeff * (vx[[i, j+offset, k]] - vx[[i, j-offset, k]]);
                }
                
                curl_x[[i, j, k]] = dvz_dy / grid.dy - dvy_dz / grid.dz;
                curl_y[[i, j, k]] = dvx_dz / grid.dz - dvz_dx / grid.dx;
                curl_z[[i, j, k]] = dvy_dx / grid.dx - dvx_dy / grid.dy;
            }
        }
    }
    
    Ok((curl_x, curl_y, curl_z))
}

/// Compute spectral Laplacian using FFT
/// 
/// More accurate for smooth fields, uses k-space representation
pub fn spectral_laplacian(
    field: ArrayView3<f64>,
    grid: &Grid,
) -> KwaversResult<Array3<f64>> {
    use num_complex::Complex;
    
    let (nx, ny, nz) = field.dim();
    
    // Convert to complex for FFT
    let mut field_complex = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                field_complex[[i, j, k]] = Complex::new(field[[i, j, k]], 0.0);
            }
        }
    }
    
    // Use the existing FFT infrastructure
    let mut fft = crate::fft::Fft3d::new(nx, ny, nz);
    fft.process(&mut field_complex, grid);
    
    // Get k-space grid
    let k_squared = grid.k_squared();
    
    // Apply Laplacian in k-space: ∇²f = -k²F
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                field_complex[[i, j, k]] *= -k_squared[[i, j, k]];
            }
        }
    }
    
    // Transform back using IFFT
    let mut ifft = crate::fft::Ifft3d::new(nx, ny, nz);
    ifft.process(&mut field_complex, grid);
    
    // Extract real part
    let mut result = Array3::zeros((nx, ny, nz));
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                result[[i, j, k]] = field_complex[[i, j, k]].re;
            }
        }
    }
    
    Ok(result)
}

/// Compute transverse Laplacian (for KZK equation)
/// 
/// Returns ∂²f/∂x² + ∂²f/∂y² (no z-component)
pub fn transverse_laplacian(
    field: ArrayView3<f64>,
    grid: &Grid,
    order: SpatialOrder,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = field.dim();
    let mut lap = Array3::zeros((nx, ny, nz));
    
    let coeffs = FDCoefficients::second_derivative(order);
    let stencil_size = coeffs.len();
    
    let dx2_inv = 1.0 / (grid.dx * grid.dx);
    let dy2_inv = 1.0 / (grid.dy * grid.dy);
    
    // Compute transverse Laplacian at interior points
    for k in 0..nz {
        for j in stencil_size..ny-stencil_size {
            for i in stencil_size..nx-stencil_size {
                let mut d2f_dx2 = -2.0 * field[[i, j, k]];
                let mut d2f_dy2 = -2.0 * field[[i, j, k]];
                
                for (s, &coeff) in coeffs.iter().enumerate() {
                    let offset = s + 1;
                    d2f_dx2 += coeff * (field[[i+offset, j, k]] + field[[i-offset, j, k]]);
                    d2f_dy2 += coeff * (field[[i, j+offset, k]] + field[[i, j-offset, k]]);
                }
                
                lap[[i, j, k]] = d2f_dx2 * dx2_inv + d2f_dy2 * dy2_inv;
            }
        }
    }
    
    Ok(lap)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_laplacian_constant_field() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let field = Array3::ones((10, 10, 10));
        
        let lap = laplacian(field.view(), &grid, SpatialOrder::Second).unwrap();
        
        // Laplacian of constant field should be zero (at interior points)
        for k in 1..9 {
            for j in 1..9 {
                for i in 1..9 {
                    assert_abs_diff_eq!(lap[[i, j, k]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }
    
    #[test]
    fn test_divergence_uniform_flow() {
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        let vx = Array3::ones((10, 10, 10));
        let vy = Array3::zeros((10, 10, 10));
        let vz = Array3::zeros((10, 10, 10));
        
        let div = divergence(vx.view(), vy.view(), vz.view(), &grid, SpatialOrder::Second).unwrap();
        
        // Divergence of uniform flow should be zero
        for k in 1..9 {
            for j in 1..9 {
                for i in 1..9 {
                    assert_abs_diff_eq!(div[[i, j, k]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }
}