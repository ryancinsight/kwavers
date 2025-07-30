//! Coupling module for Hybrid Spectral-DG methods
//! 
//! This module handles the seamless integration between spectral and DG
//! solutions, ensuring conservation properties are maintained at interfaces.

use crate::KwaversResult;
use crate::error::{KwaversError, ValidationError};
use super::traits::SolutionCoupling;
use ndarray::{Array3, Zip};

/// Coupler for hybrid spectral-DG methods
pub struct HybridCoupler {
    conservation_tolerance: f64,
    /// Width of the transition zone between methods (in grid points)
    transition_width: usize,
}

impl HybridCoupler {
    /// Create a new hybrid coupler
    pub fn new(conservation_tolerance: f64) -> Self {
        Self {
            conservation_tolerance,
            transition_width: 3, // Default transition width
        }
    }
    
    /// Create with custom transition width
    pub fn with_transition_width(conservation_tolerance: f64, width: usize) -> Self {
        Self {
            conservation_tolerance,
            transition_width: width.max(1),
        }
    }
    

    
    /// Create smooth transition mask
    fn create_transition_mask(&self, mask: &Array3<bool>) -> Array3<f64> {
        let (nx, ny, nz) = mask.dim();
        let mut smooth_mask = Array3::zeros((nx, ny, nz));
        
        // Convert boolean mask to float
        Zip::from(&mut smooth_mask)
            .and(mask)
            .for_each(|s, &m| *s = if m { 1.0 } else { 0.0 });
        
        // Apply smoothing for transition regions
        for _ in 0..self.transition_width {
            let mut temp_mask = smooth_mask.clone();
            
            // 3D smoothing kernel
            for i in 1..nx-1 {
                for j in 1..ny-1 {
                    for k in 1..nz-1 {
                        let sum = smooth_mask[[i-1, j, k]] + smooth_mask[[i+1, j, k]] +
                                 smooth_mask[[i, j-1, k]] + smooth_mask[[i, j+1, k]] +
                                 smooth_mask[[i, j, k-1]] + smooth_mask[[i, j, k+1]] +
                                 smooth_mask[[i, j, k]] * 6.0;
                        temp_mask[[i, j, k]] = sum / 12.0;
                    }
                }
            }
            
            smooth_mask = temp_mask;
        }
        
        smooth_mask
    }
    
    /// Apply conservation correction
    fn apply_conservation_correction(
        &self,
        coupled: &mut Array3<f64>,
        original: &Array3<f64>,
    ) -> KwaversResult<()> {
        // Compute integrals
        let original_integral: f64 = original.sum();
        let coupled_integral: f64 = coupled.sum();
        
        if original_integral.abs() < 1e-10 {
            // Avoid division by zero for zero fields
            return Ok(());
        }
        
        let conservation_error = (coupled_integral - original_integral).abs() / original_integral.abs();
        
        if conservation_error > self.conservation_tolerance {
            // Apply multiplicative correction to maintain conservation
            let correction_factor = original_integral / coupled_integral;
            coupled.mapv_inplace(|x| x * correction_factor);
            
            log::debug!(
                "Applied conservation correction: factor = {:.6}, error = {:.6e}",
                correction_factor,
                conservation_error
            );
        }
        
        Ok(())
    }
    
    /// Detect and smooth interface regions
    fn smooth_interfaces(&self, field: &mut Array3<f64>, mask: &Array3<f64>) {
        let (nx, ny, nz) = field.dim();
        let mut smoothed = field.clone();
        
        // Apply smoothing at interfaces where mask transitions
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    // Check if we're at an interface (mask gradient is non-zero)
                    let mask_grad_x = (mask[[i+1, j, k]] - mask[[i-1, j, k]]).abs();
                    let mask_grad_y = (mask[[i, j+1, k]] - mask[[i, j-1, k]]).abs();
                    let mask_grad_z = (mask[[i, j, k+1]] - mask[[i, j, k-1]]).abs();
                    
                    if mask_grad_x > 0.1 || mask_grad_y > 0.1 || mask_grad_z > 0.1 {
                        // Apply local smoothing
                        let sum = field[[i-1, j, k]] + field[[i+1, j, k]] +
                                 field[[i, j-1, k]] + field[[i, j+1, k]] +
                                 field[[i, j, k-1]] + field[[i, j, k+1]];
                        smoothed[[i, j, k]] = (field[[i, j, k]] + 0.1 * sum) / 1.6;
                    }
                }
            }
        }
        
        field.assign(&smoothed);
    }
}

impl SolutionCoupling for HybridCoupler {
    fn couple(
        &self,
        solution1: &Array3<f64>, // Spectral solution
        solution2: &Array3<f64>, // DG solution
        mask: &Array3<bool>,     // True where DG should be used
        original: &Array3<f64>,   // Original field for conservation
    ) -> KwaversResult<Array3<f64>> {
        if solution1.dim() != solution2.dim() || solution1.dim() != mask.dim() {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "dimensions".to_string(),
                value: format!("{:?} vs {:?}", solution1.dim(), solution2.dim()),
                constraint: "All arrays must have matching dimensions".to_string(),
            }));
        }
        
        // Create smooth transition mask
        let smooth_mask = self.create_transition_mask(mask);
        
        // Blend solutions using smooth mask
        let mut coupled = Array3::zeros(solution1.dim());
        Zip::from(&mut coupled)
            .and(solution1)
            .and(solution2)
            .and(&smooth_mask)
            .for_each(|c, &s1, &s2, &m| {
                // m = 0: use spectral (solution1)
                // m = 1: use DG (solution2)
                // 0 < m < 1: blend
                *c = (1.0 - m) * s1 + m * s2;
            });
        
        // Smooth interfaces to avoid spurious oscillations
        self.smooth_interfaces(&mut coupled, &smooth_mask);
        
        // Apply conservation correction if needed
        self.apply_conservation_correction(&mut coupled, original)?;
        
        Ok(coupled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    
    #[test]
    fn test_coupler_creation() {
        let coupler = HybridCoupler::new(1e-10);
        assert_eq!(coupler.conservation_tolerance, 1e-10);
        assert_eq!(coupler.transition_width, 3);
    }
    
    #[test]
    fn test_smooth_transition_mask() {
        let coupler = HybridCoupler::new(1e-10);
        
        // Create a sharp mask
        let mut mask = Array3::from_elem((10, 10, 10), false);
        for i in 5..10 {
            for j in 0..10 {
                for k in 0..10 {
                    mask[[i, j, k]] = true;
                }
            }
        }
        
        let smooth_mask = coupler.create_transition_mask(&mask);
        
        // Check that transition is smooth
        assert!(smooth_mask[[0, 5, 5]] < 0.1); // Far from interface
        assert!(smooth_mask[[9, 5, 5]] > 0.9); // Far from interface
        assert!((smooth_mask[[5, 5, 5]] - 0.5).abs() < 0.3); // Near interface
    }
    
    #[test]
    fn test_conservation() {
        let coupler = HybridCoupler::new(1e-10);
        
        let original = Array3::from_elem((10, 10, 10), 1.0);
        let solution1 = Array3::from_elem((10, 10, 10), 0.9);
        let solution2 = Array3::from_elem((10, 10, 10), 1.1);
        let mask = Array3::from_elem((10, 10, 10), false);
        
        let coupled = coupler.couple(&solution1, &solution2, &mask, &original).unwrap();
        
        // Check conservation
        let original_sum: f64 = original.sum();
        let coupled_sum: f64 = coupled.sum();
        assert!((coupled_sum - original_sum).abs() / original_sum < 1e-9);
    }
    
    #[test]
    fn test_blending() {
        let coupler = HybridCoupler::new(1e-10);
        
        let solution1 = Array3::from_elem((5, 5, 5), 1.0);
        let solution2 = Array3::from_elem((5, 5, 5), 2.0);
        let mut mask = Array3::from_elem((5, 5, 5), false);
        
        // Set half to use DG
        for i in 3..5 {
            for j in 0..5 {
                for k in 0..5 {
                    mask[[i, j, k]] = true;
                }
            }
        }
        
        let original = Array3::from_elem((5, 5, 5), 1.5);
        let coupled = coupler.couple(&solution1, &solution2, &mask, &original).unwrap();
        
        // Check that spectral region uses solution1
        assert!((coupled[[0, 2, 2]] - 1.0).abs() < 0.2);
        
        // Check that DG region uses solution2
        assert!((coupled[[4, 2, 2]] - 2.0).abs() < 0.2);
    }
}