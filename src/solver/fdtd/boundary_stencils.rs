//! High-order one-sided finite difference stencils for boundaries
//!
//! This module provides consistent-order boundary stencils to maintain
//! the configured spatial accuracy throughout the entire domain.
//!
//! References:
//! - Fornberg, B. (1988). "Generation of finite difference formulas on arbitrarily spaced grids"
//! - Gustafsson, B., Kreiss, H. O., & Oliger, J. (1995). "Time dependent problems and difference methods"

use std::collections::HashMap;

/// High-order forward difference coefficients
/// These maintain the same order of accuracy as the interior stencils
pub struct BoundaryStencils {
    /// Forward difference coefficients for each order
    forward_coeffs: HashMap<usize, Vec<f64>>,
    /// Backward difference coefficients for each order
    backward_coeffs: HashMap<usize, Vec<f64>>,
}

impl BoundaryStencils {
    pub fn new() -> Self {
        let mut forward_coeffs = HashMap::new();
        let mut backward_coeffs = HashMap::new();
        
        // 2nd-order forward/backward differences (1st derivative, 2nd-order accurate)
        // f'(0) = (-3f(0) + 4f(1) - f(2)) / (2h)
        forward_coeffs.insert(2, vec![-3.0/2.0, 4.0/2.0, -1.0/2.0]);
        backward_coeffs.insert(2, vec![1.0/2.0, -4.0/2.0, 3.0/2.0]);
        
        // 4th-order forward/backward differences (1st derivative, 4th-order accurate)
        // f'(0) = (-25f(0) + 48f(1) - 36f(2) + 16f(3) - 3f(4)) / (12h)
        forward_coeffs.insert(4, vec![
            -25.0/12.0, 48.0/12.0, -36.0/12.0, 16.0/12.0, -3.0/12.0
        ]);
        backward_coeffs.insert(4, vec![
            3.0/12.0, -16.0/12.0, 36.0/12.0, -48.0/12.0, 25.0/12.0
        ]);
        
        // 6th-order forward/backward differences (1st derivative, 6th-order accurate)
        // f'(0) = (-147f(0) + 360f(1) - 450f(2) + 400f(3) - 225f(4) + 72f(5) - 10f(6)) / (60h)
        forward_coeffs.insert(6, vec![
            -147.0/60.0, 360.0/60.0, -450.0/60.0, 400.0/60.0, 
            -225.0/60.0, 72.0/60.0, -10.0/60.0
        ]);
        backward_coeffs.insert(6, vec![
            10.0/60.0, -72.0/60.0, 225.0/60.0, -400.0/60.0,
            450.0/60.0, -360.0/60.0, 147.0/60.0
        ]);
        
        Self {
            forward_coeffs,
            backward_coeffs,
        }
    }
    
    /// Get forward difference coefficients for given order
    pub fn get_forward_coeffs(&self, order: usize) -> &[f64] {
        self.forward_coeffs.get(&order)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
    
    /// Get backward difference coefficients for given order
    pub fn get_backward_coeffs(&self, order: usize) -> &[f64] {
        self.backward_coeffs.get(&order)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
    
    /// Apply high-order boundary stencil at a point
    pub fn apply_forward_stencil(
        &self,
        field: &ndarray::ArrayView3<f64>,
        order: usize,
        i: usize,
        j: usize,
        k: usize,
        axis: usize,
        dx: f64,
    ) -> f64 {
        let coeffs = self.get_forward_coeffs(order);
        if coeffs.is_empty() {
            return 0.0;
        }
        
        let mut val = 0.0;
        let (nx, ny, nz) = field.dim();
        
        for (offset, &coeff) in coeffs.iter().enumerate() {
            match axis {
                0 if i + offset < nx => {
                    val += coeff * field[[i + offset, j, k]];
                }
                1 if j + offset < ny => {
                    val += coeff * field[[i, j + offset, k]];
                }
                2 if k + offset < nz => {
                    val += coeff * field[[i, j, k + offset]];
                }
                _ => {}
            }
        }
        
        val / dx
    }
    
    /// Apply high-order backward stencil at a point
    pub fn apply_backward_stencil(
        &self,
        field: &ndarray::ArrayView3<f64>,
        order: usize,
        i: usize,
        j: usize,
        k: usize,
        axis: usize,
        dx: f64,
    ) -> f64 {
        let coeffs = self.get_backward_coeffs(order);
        if coeffs.is_empty() {
            return 0.0;
        }
        
        let mut val = 0.0;
        
        for (offset, &coeff) in coeffs.iter().enumerate() {
            match axis {
                0 if i >= offset => {
                    val += coeff * field[[i - offset, j, k]];
                }
                1 if j >= offset => {
                    val += coeff * field[[i, j - offset, k]];
                }
                2 if k >= offset => {
                    val += coeff * field[[i, j, k - offset]];
                }
                _ => {}
            }
        }
        
        val / dx
    }
}