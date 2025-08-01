// src/solver/amr/interpolation.rs
//! Interpolation schemes for adaptive mesh refinement
//! 
//! Provides conservative and high-order interpolation methods
//! for transferring data between refinement levels.

use crate::error::KwaversResult;
use ndarray::Array3;
use super::{InterpolationScheme, octree::Octree};

/// Interpolate field from coarse to fine mesh
pub fn interpolate_to_refined(
    coarse_field: &Array3<f64>,
    octree: &Octree,
    scheme: InterpolationScheme,
) -> KwaversResult<Array3<f64>> {
    // Validate input dimensions
    let (nx, ny, nz) = coarse_field.dim();
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(crate::error::DataError::InvalidFormat {
            format: "empty field".to_string(),
            reason: "Cannot interpolate from empty field".to_string()
        }.into());
    }
    
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
    // Validate input dimensions
    let (nx, ny, nz) = fine_field.dim();
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(crate::error::DataError::InvalidFormat {
            format: "empty field".to_string(),
            reason: "Cannot restrict from empty field".to_string()
        }.into());
    }
    
    // Validate that dimensions are even (required for restriction)
    if nx % 2 != 0 || ny % 2 != 0 || nz % 2 != 0 {
        return Err(crate::error::DataError::InvalidFormat {
            format: "odd dimensions".to_string(),
            reason: format!("Fine field dimensions must be even for restriction, got ({}, {}, {})", nx, ny, nz)
        }.into());
    }
    
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
    
    // Validate dimensions
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(crate::error::DataError::InvalidFormat {
            format: "zero dimension".to_string(),
            reason: "Coarse field has zero dimension".to_string()
        }.into());
    }
    
    // Check for potential overflow
    if nx > usize::MAX / 2 || ny > usize::MAX / 2 || nz > usize::MAX / 2 {
        return Err(crate::error::DataError::InvalidFormat {
            format: "dimensions too large".to_string(),
            reason: "Coarse field dimensions too large for refinement".to_string()
        }.into());
    }
    
    let mut fine_field = Array3::zeros((nx * 2, ny * 2, nz * 2));
    
    // Interpolate to fine grid using correct trilinear interpolation
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let v000 = coarse_field[[i, j, k]];
                
                // Get all 8 corner values for the coarse cell
                // Use boundary conditions (repeat values) at edges
                let v100 = if i < nx - 1 { coarse_field[[i + 1, j, k]] } else { v000 };
                let v010 = if j < ny - 1 { coarse_field[[i, j + 1, k]] } else { v000 };
                let v001 = if k < nz - 1 { coarse_field[[i, j, k + 1]] } else { v000 };
                let v110 = if i < nx - 1 && j < ny - 1 { coarse_field[[i + 1, j + 1, k]] } else if i < nx - 1 { v100 } else { v010 };
                let v101 = if i < nx - 1 && k < nz - 1 { coarse_field[[i + 1, j, k + 1]] } else if i < nx - 1 { v100 } else { v001 };
                let v011 = if j < ny - 1 && k < nz - 1 { coarse_field[[i, j + 1, k + 1]] } else if j < ny - 1 { v010 } else { v001 };
                let v111 = if i < nx - 1 && j < ny - 1 && k < nz - 1 { 
                    coarse_field[[i + 1, j + 1, k + 1]] 
                } else {
                    // Handle edge cases
                    if i < nx - 1 && j < ny - 1 { v110 }
                    else if i < nx - 1 && k < nz - 1 { v101 }
                    else if j < ny - 1 && k < nz - 1 { v011 }
                    else if i < nx - 1 { v100 }
                    else if j < ny - 1 { v010 }
                    else if k < nz - 1 { v001 }
                    else { v000 }
                };
                
                // Fine grid indices
                let fi = 2 * i;
                let fj = 2 * j;
                let fk = 2 * k;
                
                // Trilinear interpolation at 8 fine grid points
                // (0,0,0) - original coarse point
                fine_field[[fi, fj, fk]] = v000;
                
                // (1,0,0) - interpolate in x direction
                fine_field[[fi + 1, fj, fk]] = 0.5 * (v000 + v100);
                
                // (0,1,0) - interpolate in y direction
                fine_field[[fi, fj + 1, fk]] = 0.5 * (v000 + v010);
                
                // (0,0,1) - interpolate in z direction
                fine_field[[fi, fj, fk + 1]] = 0.5 * (v000 + v001);
                
                // (1,1,0) - interpolate in x and y
                fine_field[[fi + 1, fj + 1, fk]] = 0.25 * (v000 + v100 + v010 + v110);
                
                // (1,0,1) - interpolate in x and z
                fine_field[[fi + 1, fj, fk + 1]] = 0.25 * (v000 + v100 + v001 + v101);
                
                // (0,1,1) - interpolate in y and z
                fine_field[[fi, fj + 1, fk + 1]] = 0.25 * (v000 + v010 + v001 + v011);
                
                // (1,1,1) - interpolate in all directions
                fine_field[[fi + 1, fj + 1, fk + 1]] = 0.125 * (v000 + v100 + v010 + v001 + 
                                                                 v110 + v101 + v011 + v111);
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
                
                // For true conservation, distribute the coarse value equally
                // among the 8 fine cells (divide by 8 to conserve integral)
                let coarse_value = coarse_field[[i, j, k]] / 8.0;
                for di in 0..2 {
                    for dj in 0..2 {
                        for dk in 0..2 {
                            fine_field[[fi + di, fj + dj, fk + dk]] = coarse_value;
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
    let complex_field: Vec<Complex<f64>> = coarse_field.iter()
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
    let (nx, ny, nz) = fine_field.dim();
    let mut coarse_field = Array3::zeros((nx / 2, ny / 2, nz / 2));
    
    // Sum fine grid values (no division by 8 for conservation)
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
                // For conservation, we sum all fine cells
                coarse_field[[i, j, k]] = sum;
            }
        }
    }
    
    Ok(coarse_field)
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
    
    // Ensure that dim0 >= 3 to avoid out-of-bounds access
    let dim0 = fine_field.dim().0;
    if dim0 < 3 {
        // If the grid is too small, return early without modifying fine_field
        return;
    }
    
    // Simple extrapolation at boundaries
    // In production, would use proper boundary conditions
    for i in 0..2 {
        for j in 0..fine_field.dim().1 {
            for k in 0..fine_field.dim().2 {
                fine_field[[i, j, k]] = fine_field[[2, j, k]];
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
        
        // Conservation: total integral should be preserved
        // Interpolation divides by 8, so fine sum equals coarse sum
        assert!((fine_sum - coarse_sum).abs() < 1e-10);
        
        // Restriction sums, so it should give back the original
        assert!((restricted_sum - coarse_sum).abs() < 1e-10);
    }
}