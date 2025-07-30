//! Discontinuity detection algorithms
//! 
//! This module implements various methods for detecting discontinuities
//! in the solution field to enable automatic switching between spectral
//! and DG methods.

use crate::grid::Grid;
use crate::KwaversResult;
use super::traits::DiscontinuityDetection;
use ndarray::{Array3, Zip, s};

/// Discontinuity detector using multiple detection strategies
pub struct DiscontinuityDetector {
    threshold: f64,
    detection_method: DetectionMethod,
}

/// Available detection methods
#[derive(Debug, Clone, Copy)]
pub enum DetectionMethod {
    /// Gradient-based detection
    Gradient,
    /// Wavelet coefficient analysis
    Wavelet,
    /// Combined approach
    Combined,
}

impl DiscontinuityDetector {
    /// Create a new discontinuity detector
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            detection_method: DetectionMethod::Combined,
        }
    }
    
    /// Create with specific detection method
    pub fn with_method(threshold: f64, method: DetectionMethod) -> Self {
        Self {
            threshold,
            detection_method: method,
        }
    }
    
    /// Detect discontinuities in the field
    pub fn detect(&self, field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<bool>> {
        match self.detection_method {
            DetectionMethod::Gradient => self.gradient_detection(field, grid),
            DetectionMethod::Wavelet => self.wavelet_detection(field, grid),
            DetectionMethod::Combined => {
                let gradient_mask = self.gradient_detection(field, grid)?;
                let wavelet_mask = self.wavelet_detection(field, grid)?;
                
                // Combine both masks (OR operation)
                let mut combined_mask = Array3::from_elem(field.dim(), false);
                Zip::from(&mut combined_mask)
                    .and(&gradient_mask)
                    .and(&wavelet_mask)
                    .for_each(|c, &g, &w| *c = g || w);
                
                Ok(combined_mask)
            }
        }
    }
    
    /// Update the detection threshold
    pub fn update_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }
    
    /// Gradient-based discontinuity detection
    fn gradient_detection(&self, field: &Array3<f64>, grid: &Grid) -> KwaversResult<Array3<bool>> {
        let (nx, ny, nz) = field.dim();
        let mut discontinuity_mask = Array3::from_elem((nx, ny, nz), false);
        
        // Compute gradients in each direction
        let dx = grid.dx;
        let dy = grid.dy;
        let dz = grid.dz;
        
        // Check x-direction
        for i in 1..nx-1 {
            for j in 0..ny {
                for k in 0..nz {
                    let grad_x = (field[[i+1, j, k]] - field[[i-1, j, k]]) / (2.0 * dx);
                    let second_deriv = (field[[i+1, j, k]] - 2.0 * field[[i, j, k]] + field[[i-1, j, k]]) / (dx * dx);
                    
                    // Detect based on normalized gradient and curvature
                    let field_scale = field[[i, j, k]].abs().max(1.0);
                    let indicator = (grad_x.abs() / field_scale) + (second_deriv.abs() * dx / field_scale);
                    
                    if indicator > self.threshold {
                        discontinuity_mask[[i, j, k]] = true;
                    }
                }
            }
        }
        
        // Check y-direction
        for i in 0..nx {
            for j in 1..ny-1 {
                for k in 0..nz {
                    let grad_y = (field[[i, j+1, k]] - field[[i, j-1, k]]) / (2.0 * dy);
                    let second_deriv = (field[[i, j+1, k]] - 2.0 * field[[i, j, k]] + field[[i, j-1, k]]) / (dy * dy);
                    
                    let field_scale = field[[i, j, k]].abs().max(1.0);
                    let indicator = (grad_y.abs() / field_scale) + (second_deriv.abs() * dy / field_scale);
                    
                    if indicator > self.threshold {
                        discontinuity_mask[[i, j, k]] = true;
                    }
                }
            }
        }
        
        // Check z-direction
        for i in 0..nx {
            for j in 0..ny {
                for k in 1..nz-1 {
                    let grad_z = (field[[i, j, k+1]] - field[[i, j, k-1]]) / (2.0 * dz);
                    let second_deriv = (field[[i, j, k+1]] - 2.0 * field[[i, j, k]] + field[[i, j, k-1]]) / (dz * dz);
                    
                    let field_scale = field[[i, j, k]].abs().max(1.0);
                    let indicator = (grad_z.abs() / field_scale) + (second_deriv.abs() * dz / field_scale);
                    
                    if indicator > self.threshold {
                        discontinuity_mask[[i, j, k]] = true;
                    }
                }
            }
        }
        
        // Apply smoothing to avoid isolated points
        self.smooth_mask(&mut discontinuity_mask);
        
        Ok(discontinuity_mask)
    }
    
    /// Wavelet-based discontinuity detection using Haar wavelets
    fn wavelet_detection(&self, field: &Array3<f64>, _grid: &Grid) -> KwaversResult<Array3<bool>> {
        let (nx, ny, nz) = field.dim();
        let mut discontinuity_mask = Array3::from_elem((nx, ny, nz), false);
        
        // Apply 1D Haar wavelet transform in each direction
        // X-direction
        for j in 0..ny {
            for k in 0..nz {
                let slice = field.slice(s![.., j, k]);
                let coeffs = self.haar_wavelet_1d(slice.to_vec());
                
                // Check high-frequency coefficients
                for (i, &coeff) in coeffs.iter().enumerate().skip(coeffs.len() / 2) {
                    if coeff.abs() > self.threshold {
                        let idx = (i - coeffs.len() / 2) * 2; // Map back to spatial index
                        if idx < nx {
                            discontinuity_mask[[idx, j, k]] = true;
                            if idx + 1 < nx {
                                discontinuity_mask[[idx + 1, j, k]] = true;
                            }
                        }
                    }
                }
            }
        }
        
        // Y-direction
        for i in 0..nx {
            for k in 0..nz {
                let slice = field.slice(s![i, .., k]);
                let coeffs = self.haar_wavelet_1d(slice.to_vec());
                
                for (j, &coeff) in coeffs.iter().enumerate().skip(coeffs.len() / 2) {
                    if coeff.abs() > self.threshold {
                        let idx = (j - coeffs.len() / 2) * 2;
                        if idx < ny {
                            discontinuity_mask[[i, idx, k]] = true;
                            if idx + 1 < ny {
                                discontinuity_mask[[i, idx + 1, k]] = true;
                            }
                        }
                    }
                }
            }
        }
        
        // Z-direction
        for i in 0..nx {
            for j in 0..ny {
                let slice = field.slice(s![i, j, ..]);
                let coeffs = self.haar_wavelet_1d(slice.to_vec());
                
                for (k, &coeff) in coeffs.iter().enumerate().skip(coeffs.len() / 2) {
                    if coeff.abs() > self.threshold {
                        let idx = (k - coeffs.len() / 2) * 2;
                        if idx < nz {
                            discontinuity_mask[[i, j, idx]] = true;
                            if idx + 1 < nz {
                                discontinuity_mask[[i, j, idx + 1]] = true;
                            }
                        }
                    }
                }
            }
        }
        
        self.smooth_mask(&mut discontinuity_mask);
        Ok(discontinuity_mask)
    }
    
    /// Simple 1D Haar wavelet transform
    fn haar_wavelet_1d(&self, mut data: Vec<f64>) -> Vec<f64> {
        let n = data.len();
        if n <= 1 {
            return data;
        }
        
        // Pad to power of 2 if necessary
        let next_pow2 = n.next_power_of_two();
        if n != next_pow2 {
            data.resize(next_pow2, 0.0);
        }
        
        let mut temp = vec![0.0; data.len()];
        let mut h = data.len();
        
        while h > 1 {
            h /= 2;
            for i in 0..h {
                temp[i] = (data[2 * i] + data[2 * i + 1]) / 2.0f64.sqrt();
                temp[i + h] = (data[2 * i] - data[2 * i + 1]) / 2.0f64.sqrt();
            }
            data[..2 * h].copy_from_slice(&temp[..2 * h]);
        }
        
        data.truncate(n); // Remove padding
        data
    }
    
    /// Smooth the discontinuity mask to avoid isolated points
    fn smooth_mask(&self, mask: &mut Array3<bool>) {
        let (nx, ny, nz) = mask.dim();
        let mut smoothed = mask.clone();
        
        // Apply 3x3x3 majority filter
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    let mut count = 0;
                    for di in -1..=1 {
                        for dj in -1..=1 {
                            for dk in -1..=1 {
                                let ii = (i as i32 + di) as usize;
                                let jj = (j as i32 + dj) as usize;
                                let kk = (k as i32 + dk) as usize;
                                if mask[[ii, jj, kk]] {
                                    count += 1;
                                }
                            }
                        }
                    }
                    // If more than half of neighbors are discontinuous, mark as discontinuous
                    smoothed[[i, j, k]] = count > 13; // 13 out of 27
                }
            }
        }
        
        mask.assign(&smoothed);
    }
}

impl DiscontinuityDetection for DiscontinuityDetector {
    fn detect(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<bool>> {
        match self.detection_method {
            DetectionMethod::Gradient => self.gradient_detection(field, grid),
            DetectionMethod::Wavelet => self.wavelet_detection(field, grid),
            DetectionMethod::Combined => {
                // Combine both methods
                let grad_mask = self.gradient_detection(field, grid)?;
                let wave_mask = self.wavelet_detection(field, grid)?;
                
                // Union of both masks
                let mut combined_mask = Array3::from_elem(field.dim(), false);
                Zip::from(&mut combined_mask)
                    .and(&grad_mask)
                    .and(&wave_mask)
                    .for_each(|c, &g, &w| *c = g || w);
                
                Ok(combined_mask)
            }
        }
    }
    
    fn update_threshold(&mut self, threshold: f64) {
        self.threshold = threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use std::f64::consts::PI;
    
    #[test]
    fn test_smooth_field_detection() {
        let grid = Grid::new(32, 32, 32, 1.0, 1.0, 1.0);
        let detector = DiscontinuityDetector::new(0.1);
        
        // Smooth field - should not detect discontinuities
        let mut field = Array3::zeros((32, 32, 32));
        for i in 0..32 {
            for j in 0..32 {
                for k in 0..32 {
                    let x = i as f64 / 32.0;
                    let y = j as f64 / 32.0;
                    let z = k as f64 / 32.0;
                    field[[i, j, k]] = (2.0 * PI * x).sin() * (2.0 * PI * y).cos() * (2.0 * PI * z).sin();
                }
            }
        }
        
        let mask = detector.detect(&field, &grid).unwrap();
        let discontinuity_count = mask.iter().filter(|&&x| x).count();
        assert!(discontinuity_count < field.len() / 10); // Less than 10% marked as discontinuous
    }
    
    #[test]
    fn test_step_function_detection() {
        let grid = Grid::new(32, 32, 32, 1.0, 1.0, 1.0);
        let detector = DiscontinuityDetector::new(0.1);
        
        // Step function - should detect discontinuity
        let mut field = Array3::zeros((32, 32, 32));
        for i in 0..32 {
            for j in 0..32 {
                for k in 0..32 {
                    field[[i, j, k]] = if i < 16 { 0.0 } else { 1.0 };
                }
            }
        }
        
        let mask = detector.detect(&field, &grid).unwrap();
        
        // Check that discontinuity is detected around x=16
        for j in 1..31 {
            for k in 1..31 {
                assert!(mask[[15, j, k]] || mask[[16, j, k]]);
            }
        }
    }
}