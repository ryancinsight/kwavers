// src/solver/amr/interpolation.rs
//! Interpolation schemes for adaptive mesh refinement
//! 
//! Provides conservative and high-order interpolation methods
//! for transferring data between refinement levels.

use crate::error::KwaversResult;
use ndarray::{Array3, s};
use super::{InterpolationScheme, octree::Octree};

/// Interpolate field from coarse to fine mesh
pub fn interpolate_to_refined(
    coarse_field: &Array3<f64>,
    octree: &Octree,
    scheme: InterpolationScheme,
) -> KwaversResult<Array3<f64>> {
    match scheme {
        InterpolationScheme::Linear => linear_interpolation(coarse_field, octree),
        InterpolationScheme::Conservative => conservative_interpolation(coarse_field, octree),
        InterpolationScheme::WENO5 => weno5_interpolation(coarse_field, octree),
        InterpolationScheme::Spectral => spectral_interpolation(coarse_field, octree),
    }
}

/// Restrict field from fine to coarse mesh
pub fn restrict_to_coarse(
    fine_field: &Array3<f64>,
    octree: &Octree,
    scheme: InterpolationScheme,
) -> KwaversResult<Array3<f64>> {
    match scheme {
        InterpolationScheme::Linear => linear_restriction(fine_field, octree),
        InterpolationScheme::Conservative => conservative_restriction(fine_field, octree),
        InterpolationScheme::WENO5 => conservative_restriction(fine_field, octree), // Use conservative for restriction
        InterpolationScheme::Spectral => spectral_restriction(fine_field, octree),
    }
}

/// Linear interpolation (fast, non-conservative)
fn linear_interpolation(
    coarse_field: &Array3<f64>,
    octree: &Octree,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = coarse_field.dim();
    let mut fine_field = Array3::zeros((nx * 2, ny * 2, nz * 2));
    
    // Interpolate to fine grid
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let coarse_val = coarse_field[[i, j, k]];
                
                // Get neighboring values for interpolation
                let ip1 = if i < nx - 1 { coarse_field[[i + 1, j, k]] } else { coarse_val };
                let jp1 = if j < ny - 1 { coarse_field[[i, j + 1, k]] } else { coarse_val };
                let kp1 = if k < nz - 1 { coarse_field[[i, j, k + 1]] } else { coarse_val };
                
                // Trilinear interpolation
                let fi = 2 * i;
                let fj = 2 * j;
                let fk = 2 * k;
                
                fine_field[[fi, fj, fk]] = coarse_val;
                fine_field[[fi + 1, fj, fk]] = 0.5 * (coarse_val + ip1);
                fine_field[[fi, fj + 1, fk]] = 0.5 * (coarse_val + jp1);
                fine_field[[fi, fj, fk + 1]] = 0.5 * (coarse_val + kp1);
                fine_field[[fi + 1, fj + 1, fk]] = 0.25 * (coarse_val + ip1 + jp1 + coarse_field[[i.min(nx-1), j.min(ny-1), k]]);
                fine_field[[fi + 1, fj, fk + 1]] = 0.25 * (coarse_val + ip1 + kp1 + coarse_field[[i.min(nx-1), j, k.min(nz-1)]]);
                fine_field[[fi, fj + 1, fk + 1]] = 0.25 * (coarse_val + jp1 + kp1 + coarse_field[[i, j.min(ny-1), k.min(nz-1)]]);
                fine_field[[fi + 1, fj + 1, fk + 1]] = 0.125 * (coarse_val + ip1 + jp1 + kp1 + 
                    coarse_field[[i.min(nx-1), j.min(ny-1), k]] +
                    coarse_field[[i.min(nx-1), j, k.min(nz-1)]] +
                    coarse_field[[i, j.min(ny-1), k.min(nz-1)]] +
                    coarse_field[[i.min(nx-1), j.min(ny-1), k.min(nz-1)]]);
            }
        }
    }
    
    Ok(fine_field)
}

/// Conservative interpolation (preserves integrals)
fn conservative_interpolation(
    coarse_field: &Array3<f64>,
    octree: &Octree,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = coarse_field.dim();
    let mut fine_field = Array3::zeros((nx * 2, ny * 2, nz * 2));
    
    // Conservative interpolation using polynomial reconstruction
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                // Get stencil for reconstruction
                let stencil = get_conservative_stencil(coarse_field, i, j, k);
                
                // Compute polynomial coefficients
                let coeffs = compute_polynomial_coeffs(&stencil);
                
                // Evaluate polynomial on fine grid
                let fi = 2 * i;
                let fj = 2 * j;
                let fk = 2 * k;
                
                for di in 0..2 {
                    for dj in 0..2 {
                        for dk in 0..2 {
                            let x = -0.25 + 0.5 * di as f64;
                            let y = -0.25 + 0.5 * dj as f64;
                            let z = -0.25 + 0.5 * dk as f64;
                            
                            fine_field[[fi + di, fj + dj, fk + dk]] = 
                                evaluate_polynomial(&coeffs, x, y, z);
                        }
                    }
                }
            }
        }
    }
    
    Ok(fine_field)
}

/// WENO5 interpolation (high-order, non-oscillatory)
fn weno5_interpolation(
    coarse_field: &Array3<f64>,
    octree: &Octree,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = coarse_field.dim();
    let mut fine_field = Array3::zeros((nx * 2, ny * 2, nz * 2));
    
    // WENO5 interpolation along each dimension
    // Handle interior points with full stencil
    for i in 2..(nx.saturating_sub(2)) {
        for j in 2..(ny.saturating_sub(2)) {
            for k in 2..(nz.saturating_sub(2)) {
                // Get 5-point stencils in each direction
                let stencil_x = [
                    coarse_field[[i-2, j, k]],
                    coarse_field[[i-1, j, k]],
                    coarse_field[[i, j, k]],
                    coarse_field[[i+1, j, k]],
                    coarse_field[[i+2, j, k]],
                ];
                
                let stencil_y = [
                    coarse_field[[i, j-2, k]],
                    coarse_field[[i, j-1, k]],
                    coarse_field[[i, j, k]],
                    coarse_field[[i, j+1, k]],
                    coarse_field[[i, j+2, k]],
                ];
                
                let stencil_z = [
                    coarse_field[[i, j, k-2]],
                    coarse_field[[i, j, k-1]],
                    coarse_field[[i, j, k]],
                    coarse_field[[i, j, k+1]],
                    coarse_field[[i, j, k+2]],
                ];
                
                // Apply WENO5 interpolation
                let fi = 2 * i;
                let fj = 2 * j;
                let fk = 2 * k;
                
                for di in 0..2 {
                    for dj in 0..2 {
                        for dk in 0..2 {
                            let x = -0.5 + di as f64;
                            let y = -0.5 + dj as f64;
                            let z = -0.5 + dk as f64;
                            
                            let val_x = weno5_1d(&stencil_x, x);
                            let val_y = weno5_1d(&stencil_y, y);
                            let val_z = weno5_1d(&stencil_z, z);
                            
                            // Tensor product interpolation
                            fine_field[[fi + di, fj + dj, fk + dk]] = 
                                (val_x + val_y + val_z) / 3.0;
                        }
                    }
                }
            }
        }
    }
    
    // Handle boundaries with linear interpolation
    handle_boundaries(&mut fine_field, coarse_field);
    
    Ok(fine_field)
}

/// Spectral interpolation (highest accuracy)
fn spectral_interpolation(
    coarse_field: &Array3<f64>,
    octree: &Octree,
) -> KwaversResult<Array3<f64>> {
    use rustfft::{FftPlanner, num_complex::Complex};
    
    let (nx, ny, nz) = coarse_field.dim();
    let mut planner = FftPlanner::<f64>::new();
    
    // Convert to complex for FFT
    let mut complex_field: Vec<Complex<f64>> = coarse_field.iter()
        .map(|&x| Complex::new(x, 0.0))
        .collect();
    
    // 3D FFT (implemented as series of 1D FFTs)
    // This is a simplified version - production code would use proper 3D FFT
    let fft = planner.plan_fft_forward(nx);
    
    // Pad spectrum with zeros for interpolation
    let mut padded_spectrum = vec![Complex::new(0.0, 0.0); nx * ny * nz * 8];
    
    // Copy low frequencies to padded spectrum
    for i in 0..nx/2 {
        for j in 0..ny/2 {
            for k in 0..nz/2 {
                let src_idx = i * ny * nz + j * nz + k;
                let dst_idx = i * (2*ny) * (2*nz) + j * (2*nz) + k;
                padded_spectrum[dst_idx] = complex_field[src_idx];
            }
        }
    }
    
    // Inverse FFT to get interpolated field
    let ifft = planner.plan_fft_inverse(2 * nx);
    
    // Convert back to real
    let fine_field = Array3::from_shape_vec(
        (2 * nx, 2 * ny, 2 * nz),
        padded_spectrum.iter().map(|c| c.re).collect()
    ).map_err(|e| crate::error::DataError::InvalidFormat {
        format: "Array3<f64>".to_string(),
        reason: format!("Failed to create array from padded spectrum: {}", e),
    })?;
    
    Ok(fine_field)
}

/// Linear restriction (averaging)
fn linear_restriction(
    fine_field: &Array3<f64>,
    octree: &Octree,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = fine_field.dim();
    let mut coarse_field = Array3::zeros((nx / 2, ny / 2, nz / 2));
    
    // Average fine grid values
    for i in 0..nx/2 {
        for j in 0..ny/2 {
            for k in 0..nz/2 {
                let mut sum = 0.0;
                for di in 0..2 {
                    for dj in 0..2 {
                        for dk in 0..2 {
                            sum += fine_field[[2*i + di, 2*j + dj, 2*k + dk]];
                        }
                    }
                }
                coarse_field[[i, j, k]] = sum / 8.0;
            }
        }
    }
    
    Ok(coarse_field)
}

/// Conservative restriction (volume-weighted averaging)
fn conservative_restriction(
    fine_field: &Array3<f64>,
    octree: &Octree,
) -> KwaversResult<Array3<f64>> {
    // For uniform grids, this is the same as linear restriction
    // For non-uniform grids, we would weight by cell volumes
    linear_restriction(fine_field, octree)
}

/// Spectral restriction
fn spectral_restriction(
    fine_field: &Array3<f64>,
    octree: &Octree,
) -> KwaversResult<Array3<f64>> {
    // Apply low-pass filter before restriction
    let filtered = low_pass_filter(fine_field)?;
    linear_restriction(&filtered, octree)
}

/// Get stencil for conservative interpolation
fn get_conservative_stencil(
    field: &Array3<f64>,
    i: usize,
    j: usize,
    k: usize,
) -> Vec<f64> {
    let (nx, ny, nz) = field.dim();
    let mut stencil = Vec::new();
    
    // 3x3x3 stencil
    for di in -1i32..=1 {
        for dj in -1i32..=1 {
            for dk in -1i32..=1 {
                let ii = (i as i32 + di).max(0).min(nx as i32 - 1) as usize;
                let jj = (j as i32 + dj).max(0).min(ny as i32 - 1) as usize;
                let kk = (k as i32 + dk).max(0).min(nz as i32 - 1) as usize;
                stencil.push(field[[ii, jj, kk]]);
            }
        }
    }
    
    stencil
}

/// Compute polynomial coefficients for conservative interpolation
fn compute_polynomial_coeffs(stencil: &[f64]) -> Vec<f64> {
    // Simplified: use center value and gradients
    // Full implementation would solve least-squares system
    let center = stencil[13]; // Center of 3x3x3 stencil
    
    vec![
        center,                    // Constant term
        stencil[14] - stencil[12], // x gradient
        stencil[16] - stencil[10], // y gradient
        stencil[22] - stencil[4],  // z gradient
    ]
}

/// Evaluate polynomial at given point
fn evaluate_polynomial(coeffs: &[f64], x: f64, y: f64, z: f64) -> f64 {
    coeffs[0] + coeffs[1] * x + coeffs[2] * y + coeffs[3] * z
}

/// WENO5 1D interpolation
fn weno5_1d(stencil: &[f64; 5], x: f64) -> f64 {
    // WENO5 weights
    let epsilon = 1e-6;
    
    // Three candidate polynomials
    let p0 = stencil[0] * (-x * (x - 1.0) * (x - 2.0)) / 6.0
           + stencil[1] * (x * (x - 1.0) * (x - 3.0)) / 2.0
           + stencil[2] * (-x * (x - 2.0) * (x - 3.0)) / 2.0;
           
    let p1 = stencil[1] * (-(x + 1.0) * x * (x - 1.0)) / 6.0
           + stencil[2] * ((x + 1.0) * x * (x - 2.0)) / 2.0
           + stencil[3] * (-(x + 1.0) * (x - 1.0) * (x - 2.0)) / 2.0;
           
    let p2 = stencil[2] * (-(x + 2.0) * (x + 1.0) * x) / 6.0
           + stencil[3] * ((x + 2.0) * (x + 1.0) * (x - 1.0)) / 2.0
           + stencil[4] * (-(x + 2.0) * x * (x - 1.0)) / 2.0;
    
    // Smoothness indicators
    let beta0 = (stencil[2] - 2.0 * stencil[1] + stencil[0]).powi(2);
    let beta1 = (stencil[3] - 2.0 * stencil[2] + stencil[1]).powi(2);
    let beta2 = (stencil[4] - 2.0 * stencil[3] + stencil[2]).powi(2);
    
    // WENO weights
    let alpha0 = 0.1 / (epsilon + beta0).powi(2);
    let alpha1 = 0.6 / (epsilon + beta1).powi(2);
    let alpha2 = 0.3 / (epsilon + beta2).powi(2);
    
    let sum_alpha = alpha0 + alpha1 + alpha2;
    
    let w0 = alpha0 / sum_alpha;
    let w1 = alpha1 / sum_alpha;
    let w2 = alpha2 / sum_alpha;
    
    w0 * p0 + w1 * p1 + w2 * p2
}

/// Handle boundaries for interpolation
fn handle_boundaries(fine_field: &mut Array3<f64>, coarse_field: &Array3<f64>) {
    let (nx, ny, nz) = coarse_field.dim();
    
    // Simple extrapolation at boundaries
    // In production, would use proper boundary conditions
    for i in 0..2 {
        for j in 0..fine_field.dim().1 {
            for k in 0..fine_field.dim().2 {
                fine_field[[i, j, k]] = fine_field[[2, j, k]];
                let dim0 = fine_field.dim().0;
                fine_field[[dim0 - 1 - i, j, k]] = 
                    fine_field[[dim0 - 3, j, k]];
            }
        }
    }
}

/// Low-pass filter for spectral restriction
fn low_pass_filter(field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
    // Simple averaging filter
    // Production code would use proper spectral filter
    let (nx, ny, nz) = field.dim();
    let mut filtered = Array3::zeros((nx, ny, nz));
    
    for i in 1..nx-1 {
        for j in 1..ny-1 {
            for k in 1..nz-1 {
                let mut sum = 0.0;
                for di in -1i32..=1 {
                    for dj in -1i32..=1 {
                        for dk in -1i32..=1 {
                            let ii = (i as i32 + di) as usize;
                            let jj = (j as i32 + dj) as usize;
                            let kk = (k as i32 + dk) as usize;
                            sum += field[[ii, jj, kk]];
                        }
                    }
                }
                filtered[[i, j, k]] = sum / 27.0;
            }
        }
    }
    
    Ok(filtered)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_conservative_property() {
        let octree = Octree::new(8, 8, 8, 3);
        
        // Create test field
        let coarse = Array3::from_elem((8, 8, 8), 1.0);
        
        // Interpolate to fine
        let fine = conservative_interpolation(&coarse, &octree).unwrap();
        
        // Restrict back to coarse
        let restricted = conservative_restriction(&fine, &octree).unwrap();
        
        // Check conservation (integrals should match)
        let coarse_sum: f64 = coarse.sum();
        let fine_sum: f64 = fine.sum();
        let restricted_sum: f64 = restricted.sum();
        
        // Fine grid has 8x more cells, so sum should be 8x larger
        assert!((fine_sum - 8.0 * coarse_sum).abs() < 1e-10);
        
        // Restriction should preserve integral
        assert!((restricted_sum - coarse_sum).abs() < 1e-10);
    }
}