// src/solver/amr/error_estimator.rs
//! Error estimation for adaptive mesh refinement
//! 
//! Uses wavelet transforms to estimate local truncation error
//! and determine regions requiring refinement or coarsening.

use crate::error::KwaversResult;
use ndarray::{Array3, s};
use super::{WaveletType, wavelet::WaveletTransform};

/// Error estimator for AMR
#[derive(Debug)]
pub struct ErrorEstimator {
    /// Wavelet transform for analysis
    wavelet_transform: WaveletTransform,
    /// Threshold for refinement
    refine_threshold: f64,
    /// Threshold for coarsening
    coarsen_threshold: f64,
    /// Smoothing radius for error field
    smoothing_radius: usize,
}

impl ErrorEstimator {
    /// Create a new error estimator
    pub fn new(
        wavelet_type: WaveletType,
        refine_threshold: f64,
        coarsen_threshold: f64,
    ) -> Self {
        Self {
            wavelet_transform: WaveletTransform::new(wavelet_type),
            refine_threshold,
            coarsen_threshold,
            smoothing_radius: 2,
        }
    }
    
    /// Estimate error field from solution
    pub fn estimate_error(&self, solution: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        // Compute wavelet coefficients
        let coeffs = self.wavelet_transform.forward_transform(solution)?;
        
        // Get detail coefficient magnitudes
        let detail_mag = self.wavelet_transform.detail_magnitude(&coeffs);
        
        // Apply smoothing to avoid isolated refinements
        let smoothed = self.smooth_error_field(&detail_mag);
        
        // Normalize by local solution magnitude
        let normalized = self.normalize_error(&smoothed, solution);
        
        Ok(normalized)
    }
    
    /// Estimate error using gradient-based indicator
    pub fn gradient_error(&self, solution: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = solution.dim();
        let mut error = Array3::zeros((nx, ny, nz));
        
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    // Compute gradients using central differences
                    let dx = (solution[[i+1, j, k]] - solution[[i-1, j, k]]) / 2.0;
                    let dy = (solution[[i, j+1, k]] - solution[[i, j-1, k]]) / 2.0;
                    let dz = (solution[[i, j, k+1]] - solution[[i, j, k-1]]) / 2.0;
                    
                    // Gradient magnitude as error indicator
                    error[[i, j, k]] = (dx*dx + dy*dy + dz*dz).sqrt();
                }
            }
        }
        
        error
    }
    
    /// Estimate error using Hessian-based indicator (second derivatives)
    pub fn hessian_error(&self, solution: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = solution.dim();
        let mut error = Array3::zeros((nx, ny, nz));
        
        for i in 2..nx-2 {
            for j in 2..ny-2 {
                for k in 2..nz-2 {
                    // Compute second derivatives
                    let dxx = solution[[i+1, j, k]] - 2.0*solution[[i, j, k]] + solution[[i-1, j, k]];
                    let dyy = solution[[i, j+1, k]] - 2.0*solution[[i, j, k]] + solution[[i, j-1, k]];
                    let dzz = solution[[i, j, k+1]] - 2.0*solution[[i, j, k]] + solution[[i, j, k-1]];
                    
                    let dxy = (solution[[i+1, j+1, k]] - solution[[i+1, j-1, k]]
                             - solution[[i-1, j+1, k]] + solution[[i-1, j-1, k]]) / 4.0;
                    let dxz = (solution[[i+1, j, k+1]] - solution[[i+1, j, k-1]]
                             - solution[[i-1, j, k+1]] + solution[[i-1, j, k-1]]) / 4.0;
                    let dyz = (solution[[i, j+1, k+1]] - solution[[i, j+1, k-1]]
                             - solution[[i, j-1, k+1]] + solution[[i, j-1, k-1]]) / 4.0;
                    
                    // Frobenius norm of Hessian
                    error[[i, j, k]] = (dxx*dxx + dyy*dyy + dzz*dzz 
                                     + 2.0*(dxy*dxy + dxz*dxz + dyz*dyz)).sqrt();
                }
            }
        }
        
        error
    }
    
    /// Combined error estimator using multiple indicators
    pub fn combined_error(&self, solution: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        // Wavelet-based error
        let wavelet_error = self.estimate_error(solution)?;
        
        // Gradient-based error
        let gradient_error = self.gradient_error(solution);
        
        // Hessian-based error
        let hessian_error = self.hessian_error(solution);
        
        // Combine with weights
        let combined = &wavelet_error * 0.5 + &gradient_error * 0.3 + &hessian_error * 0.2;
        
        Ok(combined)
    }
    
    /// Smooth error field to avoid isolated refinements
    fn smooth_error_field(&self, error: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = error.dim();
        let mut smoothed = Array3::zeros((nx, ny, nz));
        let r = self.smoothing_radius as i32;
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mut sum = 0.0;
                    let mut count = 0.0;
                    
                    // Average over neighborhood
                    for di in -r..=r {
                        for dj in -r..=r {
                            for dk in -r..=r {
                                let ni = (i as i32 + di) as usize;
                                let nj = (j as i32 + dj) as usize;
                                let nk = (k as i32 + dk) as usize;
                                
                                if ni < nx && nj < ny && nk < nz {
                                    let weight = 1.0 / (1.0 + (di*di + dj*dj + dk*dk) as f64);
                                    sum += error[[ni, nj, nk]] * weight;
                                    count += weight;
                                }
                            }
                        }
                    }
                    
                    smoothed[[i, j, k]] = sum / count;
                }
            }
        }
        
        smoothed
    }
    
    /// Normalize error by local solution magnitude
    fn normalize_error(&self, error: &Array3<f64>, solution: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = error.dim();
        let mut normalized = Array3::zeros((nx, ny, nz));
        
        // Compute local solution scale
        let solution_scale = solution.mapv(|x| x.abs()).mean().unwrap_or(1.0);
        
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Normalize by local solution magnitude + small epsilon
                    let local_scale = solution[[i, j, k]].abs() + 0.1 * solution_scale;
                    normalized[[i, j, k]] = error[[i, j, k]] / local_scale;
                }
            }
        }
        
        normalized
    }
    
    /// Check if a cell should be refined
    pub fn should_refine(&self, error: f64) -> bool {
        error > self.refine_threshold
    }
    
    /// Check if a cell can be coarsened
    pub fn can_coarsen(&self, error: f64) -> bool {
        error < self.coarsen_threshold
    }
    
    /// Estimate memory savings from current error field
    pub fn estimate_memory_savings(&self, error_field: &Array3<f64>) -> f64 {
        let total_cells = error_field.len();
        let cells_to_refine = error_field.iter()
            .filter(|&&e| self.should_refine(e))
            .count();
        let cells_to_coarsen = error_field.iter()
            .filter(|&&e| self.can_coarsen(e))
            .count();
        
        // Rough estimate: refinement increases cells by 8x, coarsening reduces by 8x
        let refined_cells = cells_to_refine * 8;
        let coarsened_cells = cells_to_coarsen / 8;
        let adaptive_cells = total_cells - cells_to_refine - cells_to_coarsen 
                           + refined_cells + coarsened_cells;
        
        1.0 - (adaptive_cells as f64 / total_cells as f64)
    }
}

/// Richardson extrapolation error estimator
pub struct RichardsonEstimator {
    /// Order of accuracy of the numerical scheme
    order: usize,
}

impl RichardsonEstimator {
    /// Create a new Richardson extrapolation error estimator
    pub fn new(order: usize) -> Self {
        Self { order }
    }
    
    /// Estimate error using two solutions at different resolutions
    pub fn estimate_error(
        &self,
        coarse_solution: &Array3<f64>,
        fine_solution: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = coarse_solution.dim();
        let mut error = Array3::zeros((nx, ny, nz));
        
        // Restriction factor (assuming refinement ratio of 2)
        let factor = 1.0 / (2.0_f64.powi(self.order as i32) - 1.0);
        
        // Restrict fine solution to coarse grid
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // Average fine grid values
                    let mut fine_avg = 0.0;
                    for di in 0..2 {
                        for dj in 0..2 {
                            for dk in 0..2 {
                                let fi = 2*i + di;
                                let fj = 2*j + dj;
                                let fk = 2*k + dk;
                                if fi < fine_solution.dim().0 && 
                                   fj < fine_solution.dim().1 && 
                                   fk < fine_solution.dim().2 {
                                    fine_avg += fine_solution[[fi, fj, fk]];
                                }
                            }
                        }
                    }
                    fine_avg /= 8.0;
                    
                    // Richardson extrapolation error estimate
                    error[[i, j, k]] = factor * (fine_avg - coarse_solution[[i, j, k]]).abs();
                }
            }
        }
        
        Ok(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gradient_error() {
        let estimator = ErrorEstimator::new(WaveletType::Haar, 1e-3, 1e-4);
        
        // Create test field with sharp gradient
        let field = Array3::from_shape_fn((8, 8, 8), |(i, j, k)| {
            if i < 4 { 0.0 } else { 1.0 }
        });
        
        let error = estimator.gradient_error(&field);
        
        // Error should be high at the interface
        assert!(error[[4, 4, 4]] > error[[1, 4, 4]]);
        assert!(error[[4, 4, 4]] > error[[7, 4, 4]]);
    }
    
    #[test]
    fn test_error_thresholds() {
        let estimator = ErrorEstimator::new(WaveletType::Haar, 0.1, 0.01);
        
        assert!(estimator.should_refine(0.2));
        assert!(!estimator.should_refine(0.05));
        assert!(estimator.can_coarsen(0.005));
        assert!(!estimator.can_coarsen(0.05));
    }
}