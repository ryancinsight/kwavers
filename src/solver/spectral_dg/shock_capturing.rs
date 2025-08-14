//! Shock Capturing for Spectral Discontinuous Galerkin Methods
//!
//! This module implements shock-capturing techniques for handling
//! discontinuities in spectral DG simulations, including:
//! - Artificial viscosity methods
//! - Sub-cell resolution techniques
//! - Hybrid spectral/finite-volume approaches
//!
//! # Theory
//!
//! When discontinuities (shocks) are present, spectral methods suffer from
//! Gibbs oscillations. This module provides several approaches to handle these:

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array3, Array4, Axis};


use crate::constants::numerical::{
    WENO_WEIGHT_0, WENO_WEIGHT_1, WENO_WEIGHT_2,
    VON_NEUMANN_RICHTMYER_COEFF, LINEAR_VISCOSITY_COEFF, 
    QUADRATIC_VISCOSITY_COEFF, MAX_VISCOSITY_LIMIT,
    WENO_EPSILON, STENCIL_COEFF_1_4
};

/// Shock detector with multiple indicators
#[derive(Debug, Clone)]
pub struct ShockDetector {
    /// Base threshold for shock detection
    threshold: f64,
    /// Use modal decay indicator
    use_modal_decay: bool,
    /// Use jump indicator
    use_jump_indicator: bool,
    /// Use entropy residual
    use_entropy_residual: bool,
    /// Smoothness exponent for modal decay
    smoothness_exponent: f64,
}

impl Default for ShockDetector {
    fn default() -> Self {
        Self {
            threshold: 0.01,
            use_modal_decay: true,
            use_jump_indicator: true,
            use_entropy_residual: false,
            smoothness_exponent: 2.0,
        }
    }
}

impl ShockDetector {
    /// Create a new shock detector
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            ..Default::default()
        }
    }
    
    /// Detect shocks in the field using multiple indicators
    pub fn detect_shocks(&self, field: &Array3<f64>, grid: &Grid) -> Array3<bool> {
        let (nx, ny, nz) = field.dim();
        let mut shock_mask = Array3::from_elem((nx, ny, nz), false);
        
        // Modal decay indicator
        if self.use_modal_decay {
            // Compute modal coefficients and check decay rate
            // For now, use gradient-based detection as proxy
            for i in 1..nx-1 {
                for j in 1..ny-1 {
                    for k in 1..nz-1 {
                        let grad_x = (field[[i+1, j, k]] - field[[i-1, j, k]]) / (2.0 * grid.dx);
                        let grad_y = (field[[i, j+1, k]] - field[[i, j-1, k]]) / (2.0 * grid.dy);
                        let grad_z = (field[[i, j, k+1]] - field[[i, j, k-1]]) / (2.0 * grid.dz);
                        
                        let grad_mag = (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).sqrt();
                        let field_mag = field[[i, j, k]].abs() + 1e-10;
                        
                        if grad_mag / field_mag > self.threshold {
                            shock_mask[[i, j, k]] = true;
                        }
                    }
                }
            }
        }
        
        // Jump indicator
        if self.use_jump_indicator {
            for i in 1..nx-1 {
                for j in 1..ny-1 {
                    for k in 1..nz-1 {
                        // Check jumps across cell interfaces
                        let jump_x = (field[[i+1, j, k]] - field[[i, j, k]]).abs();
                        let jump_y = (field[[i, j+1, k]] - field[[i, j, k]]).abs();
                        let jump_z = (field[[i, j, k+1]] - field[[i, j, k]]).abs();
                        
                        let max_jump = jump_x.max(jump_y).max(jump_z);
                        let field_scale = field[[i, j, k]].abs() + 1e-10;
                        
                        if max_jump / field_scale > self.threshold * 10.0 {
                            shock_mask[[i, j, k]] = true;
                        }
                    }
                }
            }
        }
        
        shock_mask
    }
    
    /// Compute entropy-based shock indicator
    fn compute_entropy_indicator(
        &self,
        pressure: &Array3<f64>,
        density: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = pressure.dim();
        let mut indicator = Array3::zeros((nx, ny, nz));
        
        // Compute specific entropy s = p/ρ^γ
        let gamma = 1.4; // Adiabatic index
        
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    // Compute local entropy
                    let s_center = pressure[[i, j, k]] / density[[i, j, k]].powf(gamma);
                    
                    // Check entropy in all directions
                    let mut max_entropy_jump = 0.0;
                    
                    for (di, dj, dk) in &[(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)] {
                        let ni = (i as i32 + di) as usize;
                        let nj = (j as i32 + dj) as usize;
                        let nk = (k as i32 + dk) as usize;
                        
                        let s_neighbor = pressure[[ni, nj, nk]] / density[[ni, nj, nk]].powf(gamma);
                        let entropy_jump = (s_center - s_neighbor).abs() / s_center.abs().max(1e-10);
                        max_entropy_jump = f64::max(max_entropy_jump, entropy_jump);
                    }
                    
                    // Entropy should decrease across shocks
                    indicator[[i, j, k]] = (max_entropy_jump / self.threshold).min(1.0);
                }
            }
        }
        
        Ok(indicator)
    }
    
    /// Compute pressure-based shock indicator
    fn compute_pressure_indicator(
        &self,
        pressure: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = pressure.dim();
        let mut indicator = Array3::zeros((nx, ny, nz));
        
        // Use pressure jumps and gradients
        for i in 2..nx-2 {
            for j in 2..ny-2 {
                for k in 2..nz-2 {
                    // Compute pressure jump indicator (Ducros et al.)
                    let p_center = pressure[[i, j, k]];
                    
                    // Second derivatives for detecting discontinuities
                    let d2p_dx2 = (pressure[[i+1, j, k]] - 2.0 * p_center + pressure[[i-1, j, k]]) / (grid.dx * grid.dx);
                    let d2p_dy2 = (pressure[[i, j+1, k]] - 2.0 * p_center + pressure[[i, j-1, k]]) / (grid.dy * grid.dy);
                    let d2p_dz2 = (pressure[[i, j, k+1]] - 2.0 * p_center + pressure[[i, j, k-1]]) / (grid.dz * grid.dz);
                    
                    // First derivatives
                    let dp_dx = (pressure[[i+1, j, k]] - pressure[[i-1, j, k]]) / (2.0 * grid.dx);
                    let dp_dy = (pressure[[i, j+1, k]] - pressure[[i, j-1, k]]) / (2.0 * grid.dy);
                    let dp_dz = (pressure[[i, j, k+1]] - pressure[[i, j, k-1]]) / (2.0 * grid.dz);
                    
                    // Ducros sensor
                    let laplacian = d2p_dx2 + d2p_dy2 + d2p_dz2;
                    let grad_mag = (dp_dx * dp_dx + dp_dy * dp_dy + dp_dz * dp_dz).sqrt();
                    
                    let sensor = laplacian.abs() / (grad_mag + 1e-10 * p_center.abs());
                    indicator[[i, j, k]] = (sensor / self.threshold).min(1.0);
                }
            }
        }
        
        Ok(indicator)
    }
    
    /// Compute velocity divergence indicator
    fn compute_divergence_indicator(
        &self,
        velocity: &Array4<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let (_, nx, ny, nz) = velocity.dim();
        let mut indicator = Array3::zeros((nx, ny, nz));
        
        // Compute velocity divergence
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    let dvx_dx = (velocity[[0, i+1, j, k]] - velocity[[0, i-1, j, k]]) / (2.0 * grid.dx);
                    let dvy_dy = (velocity[[1, i, j+1, k]] - velocity[[1, i, j-1, k]]) / (2.0 * grid.dy);
                    let dvz_dz = (velocity[[2, i, j, k+1]] - velocity[[2, i, j, k-1]]) / (2.0 * grid.dz);
                    
                    let divergence = dvx_dx + dvy_dy + dvz_dz;
                    
                    // Strong compression indicates shock
                    indicator[[i, j, k]] = (-divergence).max(0.0);
                }
            }
        }
        
        // Normalize
        let max_div = indicator.iter().cloned().fold(0.0_f64, f64::max);
        if max_div > 0.0 {
            indicator /= max_div;
        }
        
        Ok(indicator)
    }
}

/// WENO-based shock limiter
#[derive(Debug, Clone)]
pub struct WENOLimiter {
    /// WENO order (3, 5, or 7)
    order: usize,
    /// Small parameter to avoid division by zero
    epsilon: f64,
    /// Power parameter for smoothness indicators
    p: f64,
    /// Threshold for shock detection
    shock_threshold: f64,
}

impl WENOLimiter {
    pub fn new(order: usize) -> KwaversResult<Self> {
        if order != 3 && order != 5 && order != 7 {
            return Err(crate::error::KwaversError::Config(crate::error::ConfigError::InvalidValue {
                parameter: "weno_order".to_string(),
                value: order.to_string(),
                constraint: "WENO order must be 3, 5, or 7".to_string(),
            }).into());
        }
        
        Ok(Self {
            order,
            epsilon: 1e-6,
            p: 2.0,
            shock_threshold: 0.1, // Default shock threshold
        })
    }
    
    /// Apply WENO limiting to a field
    pub fn limit_field(
        &self,
        field: &Array3<f64>,
        shock_indicator: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let mut limited = field.clone();
        
        match self.order {
            3 => self.weno3_limit(&mut limited, shock_indicator)?,
            5 => self.weno5_limit(&mut limited, shock_indicator)?,
            7 => self.weno7_limit(&mut limited, shock_indicator)?,
            _ => unreachable!(),
        }
        
        Ok(limited)
    }
    
    /// WENO3 limiting (third-order WENO)
    fn weno3_limit(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        let mut limited_field = field.clone();
        
        for i in 2..nx-2 {
            for j in 2..ny-2 {
                for k in 2..nz-2 {
                    if shock_indicator[[i, j, k]] > 0.5 {
                        // Apply WENO3 in each direction
                        let weno_x = self.weno3_stencil(&[
                            field[[i-2, j, k]], field[[i-1, j, k]], 
                            field[[i, j, k]], field[[i+1, j, k]], field[[i+2, j, k]]
                        ]);
                        let weno_y = self.weno3_stencil(&[
                            field[[i, j-2, k]], field[[i, j-1, k]], 
                            field[[i, j, k]], field[[i, j+1, k]], field[[i, j+2, k]]
                        ]);
                        let weno_z = self.weno3_stencil(&[
                            field[[i, j, k-2]], field[[i, j, k-1]], 
                            field[[i, j, k]], field[[i, j, k+1]], field[[i, j, k+2]]
                        ]);
                        
                        // Average the limited values
                        limited_field[[i, j, k]] = (weno_x + weno_y + weno_z) / 3.0;
                    }
                }
            }
        }
        
        field.assign(&limited_field);
        Ok(())
    }
    
    /// WENO3 stencil computation
    fn weno3_stencil(&self, v: &[f64; 5]) -> f64 {
        // Three candidate stencils
        let q0 = v[0] / 3.0 - 7.0 * v[1] / 6.0 + 11.0 * v[2] / 6.0;
        let q1 = -v[1] / 6.0 + 5.0 * v[2] / 6.0 + v[3] / 3.0;
        let q2 = v[2] / 3.0 + 5.0 * v[3] / 6.0 - v[4] / 6.0;
        
        // Smoothness indicators (Jiang-Shu)
        let beta0 = 13.0 / 12.0 * (v[0] - 2.0*v[1] + v[2]).powi(2) + 
                    STENCIL_COEFF_1_4 * (v[0] - 4.0*v[1] + 3.0*v[2]).powi(2);
        let beta1 = 13.0 / 12.0 * (v[1] - 2.0*v[2] + v[3]).powi(2) + 
                    STENCIL_COEFF_1_4 * (v[1] - v[3]).powi(2);
        let beta2 = 13.0 / 12.0 * (v[2] - 2.0*v[3] + v[4]).powi(2) + 
                    STENCIL_COEFF_1_4 * (3.0*v[2] - 4.0*v[3] + v[4]).powi(2);
        
        // Optimal weights
        let d0 = WENO_WEIGHT_0;
        let d1 = WENO_WEIGHT_1;
        let d2 = WENO_WEIGHT_2;
        
        // Non-linear weights
        let alpha0 = d0 / (WENO_EPSILON + beta0).powi(2);
        let alpha1 = d1 / (WENO_EPSILON + beta1).powi(2);
        let alpha2 = d2 / (WENO_EPSILON + beta2).powi(2);
        
        let sum_alpha = alpha0 + alpha1 + alpha2;
        
        let w0 = alpha0 / sum_alpha;
        let w1 = alpha1 / sum_alpha;
        let w2 = alpha2 / sum_alpha;
        
        // Final reconstruction
        w0 * q0 + w1 * q1 + w2 * q2
    }
    
    /// WENO5 limiting implementation
    fn weno5_limit(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        let epsilon = 1e-6;
        
        // Process each direction
        for direction in 0..3 {
            match direction {
                0 => self.weno5_limit_x(field, shock_indicator, nx, ny, nz, epsilon)?,
                1 => self.weno5_limit_y(field, shock_indicator, nx, ny, nz, epsilon)?,
                2 => self.weno5_limit_z(field, shock_indicator, nx, ny, nz, epsilon)?,
                _ => unreachable!(),
            }
        }
        
        Ok(())
    }
    
    /// WENO5 limiting in x-direction
    fn weno5_limit_x(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
        epsilon: f64,
    ) -> KwaversResult<()> {
        // Collect indices where shock indicator exceeds threshold
        let indices: Vec<(usize, usize, usize)> = (0..nx)
            .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
            .filter(|&(i, j, k)| {
                i >= 2 && i < nx - 2 && shock_indicator[[i, j, k]] > self.shock_threshold
            })
            .collect();
        
        // Process each index
        for (i, j, k) in indices {
            // Extract stencil values
            let v = [
                field[[i.saturating_sub(2), j, k]],
                field[[i.saturating_sub(1), j, k]],
                field[[i, j, k]],
                field[[i.min(nx-1).saturating_add(1), j, k]],
                field[[(i+2).min(nx-1), j, k]],
            ];
            
            field[[i, j, k]] = self.compute_weno5_value(&v, epsilon);
        }
        
        Ok(())
    }
    
    /// Compute WENO5 reconstruction value from stencil
    fn compute_weno5_value(&self, v: &[f64; 5], epsilon: f64) -> f64 {
        // Three stencils for WENO5
        let p0 = (2.0 * v[0] - 7.0 * v[1] + 11.0 * v[2]) / 6.0;
        let p1 = (-v[1] + 5.0 * v[2] + 2.0 * v[3]) / 6.0;
        let p2 = (2.0 * v[2] + 5.0 * v[3] - v[4]) / 6.0;
        
        // Smoothness indicators
        let beta0 = 13.0/12.0 * (v[0] - 2.0*v[1] + v[2]).powi(2) + 
                   0.25 * (v[0] - 4.0*v[1] + 3.0*v[2]).powi(2);
        let beta1 = 13.0/12.0 * (v[1] - 2.0*v[2] + v[3]).powi(2) + 
                   0.25 * (v[1] - v[3]).powi(2);
        let beta2 = 13.0/12.0 * (v[2] - 2.0*v[3] + v[4]).powi(2) + 
                   0.25 * (3.0*v[2] - 4.0*v[3] + v[4]).powi(2);
        
        // Optimal weights
        let d0 = 0.1;
        let d1 = 0.6;
        let d2 = 0.3;
        
        // WENO weights
        let alpha0 = d0 / (epsilon + beta0).powi(2);
        let alpha1 = d1 / (epsilon + beta1).powi(2);
        let alpha2 = d2 / (epsilon + beta2).powi(2);
        let sum_alpha = alpha0 + alpha1 + alpha2;
        
        let w0 = alpha0 / sum_alpha;
        let w1 = alpha1 / sum_alpha;
        let w2 = alpha2 / sum_alpha;
        
        // WENO5 reconstruction
        w0 * p0 + w1 * p1 + w2 * p2
    }
    
    /// WENO5 limiting in y-direction
    fn weno5_limit_y(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
        epsilon: f64,
    ) -> KwaversResult<()> {
        // Collect indices where shock indicator exceeds threshold
        let indices: Vec<(usize, usize, usize)> = (0..nx)
            .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
            .filter(|&(i, j, k)| {
                j >= 2 && j < ny - 2 && shock_indicator[[i, j, k]] > self.shock_threshold
            })
            .collect();
        
        // Process each index
        for (i, j, k) in indices {
            // Extract stencil values
            let v = [
                field[[i, j.saturating_sub(2), k]],
                field[[i, j.saturating_sub(1), k]],
                field[[i, j, k]],
                field[[i, j.min(ny-1).saturating_add(1), k]],
                field[[i, (j+2).min(ny-1), k]],
            ];
            
            field[[i, j, k]] = self.compute_weno5_value(&v, epsilon);
        }
        
        Ok(())
    }
    
    /// WENO5 limiting in z-direction
    fn weno5_limit_z(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
        epsilon: f64,
    ) -> KwaversResult<()> {
        // Collect indices where shock indicator exceeds threshold
        let indices: Vec<(usize, usize, usize)> = (0..nx)
            .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
            .filter(|&(i, j, k)| {
                k >= 2 && k < nz - 2 && shock_indicator[[i, j, k]] > self.shock_threshold
            })
            .collect();
        
        // Process each index
        for (i, j, k) in indices {
            // Extract stencil values
            let v = [
                field[[i, j, k.saturating_sub(2)]],
                field[[i, j, k.saturating_sub(1)]],
                field[[i, j, k]],
                field[[i, j, k.min(nz-1).saturating_add(1)]],
                field[[i, j, (k+2).min(nz-1)]],
            ];
            
            field[[i, j, k]] = self.compute_weno5_value(&v, epsilon);
        }
        
        Ok(())
    }
    
    /// WENO7 limiting (seventh-order WENO)
    fn weno7_limit(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();
        let mut limited_field = field.clone();
        
        // WENO7 requires wider stencil (9 points)
        for i in 4..nx-4 {
            for j in 4..ny-4 {
                for k in 4..nz-4 {
                    if shock_indicator[[i, j, k]] > 0.5 {
                        // Apply WENO7 in each direction
                        let weno_x = self.weno7_stencil(&[
                            field[[i-4, j, k]], field[[i-3, j, k]], field[[i-2, j, k]], 
                            field[[i-1, j, k]], field[[i, j, k]], field[[i+1, j, k]], 
                            field[[i+2, j, k]], field[[i+3, j, k]], field[[i+4, j, k]]
                        ]);
                        
                        limited_field[[i, j, k]] = weno_x;
                    }
                }
            }
        }
        
        field.assign(&limited_field);
        Ok(())
    }
    
    /// WENO7 stencil computation
    /// Based on Balsara & Shu (2000), "Monotonicity Preserving WENO Schemes"
    fn weno7_stencil(&self, v: &[f64; 9]) -> f64 {
        // Four candidate stencils for WENO7
        let q0 = -v[0]/4.0 + 13.0*v[1]/12.0 - 23.0*v[2]/12.0 + 25.0*v[3]/12.0;
        let q1 = v[1]/12.0 - 5.0*v[2]/12.0 + 13.0*v[3]/12.0 + v[4]/4.0;
        let q2 = -v[2]/12.0 + 7.0*v[3]/12.0 + 7.0*v[4]/12.0 - v[5]/12.0;
        let q3 = v[3]/4.0 + 13.0*v[4]/12.0 - 5.0*v[5]/12.0 + v[6]/12.0;
        
        // Smoothness indicators (more complex for WENO7)
        let beta0 = self.compute_weno7_smoothness(&v[0..5]);
        let beta1 = self.compute_weno7_smoothness(&v[1..6]);
        let beta2 = self.compute_weno7_smoothness(&v[2..7]);
        let beta3 = self.compute_weno7_smoothness(&v[3..8]);
        
        // Optimal weights for WENO7
        let d0 = 0.05;
        let d1 = 0.45;
        let d2 = 0.45;
        let d3 = 0.05;
        
        // Non-linear weights
        let alpha0 = d0 / (WENO_EPSILON + beta0).powi(2);
        let alpha1 = d1 / (WENO_EPSILON + beta1).powi(2);
        let alpha2 = d2 / (WENO_EPSILON + beta2).powi(2);
        let alpha3 = d3 / (WENO_EPSILON + beta3).powi(2);
        
        let sum_alpha = alpha0 + alpha1 + alpha2 + alpha3;
        
        let w0 = alpha0 / sum_alpha;
        let w1 = alpha1 / sum_alpha;
        let w2 = alpha2 / sum_alpha;
        let w3 = alpha3 / sum_alpha;
        
        // Final reconstruction
        w0 * q0 + w1 * q1 + w2 * q2 + w3 * q3
    }
    
    /// Compute WENO7 smoothness indicator for a 5-point stencil
    fn compute_weno7_smoothness(&self, v: &[f64]) -> f64 {
        // Based on Jiang & Shu (1996) smoothness indicators
        let d1 = v[1] - v[0];
        let d2 = v[2] - v[1];
        let d3 = v[3] - v[2];
        let d4 = v[4] - v[3];
        
        let d11 = d2 - d1;
        let d21 = d3 - d2;
        let d31 = d4 - d3;
        
        let d111 = d21 - d11;
        let d211 = d31 - d21;
        
        let d1111 = d211 - d111;
        
        // Sum of squares of derivatives
        d1*d1 + 13.0/3.0*d11*d11 + 781.0/20.0*d111*d111 + 1421461.0/2275.0*d1111*d1111
    }
}

/// Artificial viscosity for shock stabilization
#[derive(Debug, Clone)]
pub struct ArtificialViscosity {
    /// Von Neumann-Richtmyer coefficient
    c_vnr: f64,
    /// Linear viscosity coefficient
    c_linear: f64,
    /// Quadratic viscosity coefficient
    c_quadratic: f64,
    /// Maximum viscosity limit
    max_viscosity: f64,
}

impl Default for ArtificialViscosity {
    fn default() -> Self {
        Self {
            c_vnr: VON_NEUMANN_RICHTMYER_COEFF,
            c_linear: LINEAR_VISCOSITY_COEFF,
            c_quadratic: QUADRATIC_VISCOSITY_COEFF,
            max_viscosity: MAX_VISCOSITY_LIMIT,
        }
    }
}

impl ArtificialViscosity {
    /// Compute artificial viscosity coefficient
    pub fn compute_viscosity(
        &self,
        velocity: &Array4<f64>,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
        shock_indicator: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let (_, nx, ny, nz) = velocity.dim();
        let mut viscosity = Array3::zeros((nx, ny, nz));
        
        // Extract velocity components
        let vx = velocity.index_axis(Axis(0), 0);
        let vy = velocity.index_axis(Axis(0), 1);
        let vz = velocity.index_axis(Axis(0), 2);
        
        let dx = grid.dx.min(grid.dy).min(grid.dz);
        
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    // Only apply where shocks are detected
                    if shock_indicator[[i, j, k]] > 0.1 {
                        // Compute velocity divergence
                        let div_v = (vx[[i+1, j, k]] - vx[[i-1, j, k]]) / (2.0 * grid.dx) +
                                   (vy[[i, j+1, k]] - vy[[i, j-1, k]]) / (2.0 * grid.dy) +
                                   (vz[[i, j, k+1]] - vz[[i, j, k-1]]) / (2.0 * grid.dz);
                        
                        if div_v < 0.0 { // Compression
                            let c = sound_speed[[i, j, k]];
                            let rho = density[[i, j, k]];
                            
                            // Von Neumann-Richtmyer viscosity
                            let q_vnr = self.c_vnr * rho * dx * dx * div_v.powi(2);
                            
                            // Linear and quadratic terms
                            let q_linear = self.c_linear * rho * c * dx * div_v.abs();
                            let q_quadratic = self.c_quadratic * rho * dx * dx * div_v.powi(2) / c;
                            
                            // Total viscosity
                            let q_total = (q_vnr + q_linear + q_quadratic) * shock_indicator[[i, j, k]];
                            viscosity[[i, j, k]] = q_total.min(self.max_viscosity * rho * c * c);
                        }
                    }
                }
            }
        }
        
        Ok(viscosity)
    }
}

// Shock capturing functionality has been integrated into HybridSpectralDGSolver
// Advanced shock handling features are now available through the standard solver API

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_weno3_reconstruction() {
        let limiter = WENOLimiter::new(3).unwrap();
        
        // Test smooth data - WENO3 needs 5 points for reconstruction
        let smooth_stencil = [1.0, 1.5, 2.0, 2.5, 3.0];
        let result = limiter.weno3_stencil(&smooth_stencil);
        println!("WENO3 smooth result: {}", result);
        assert!((result - 2.0).abs() < 0.5); // Should be close to central value
        
        // Test discontinuous data
        let discontinuous_stencil = [1.0, 1.0, 1.0, 5.0, 10.0];
        let result = limiter.weno3_stencil(&discontinuous_stencil);
        println!("WENO3 discontinuous result: {}", result);
        assert!(result < 5.0); // Should limit the jump
    }
    
    #[test]
    fn test_shock_detector() {
        let detector = ShockDetector::new(0.1);
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0);
        
        // Create test data with a shock
        let mut pressure = Array3::from_elem((10, 10, 10), 1.0);
        let mut density = Array3::from_elem((10, 10, 10), 1.0);
        let velocity = Array4::<f64>::zeros((3, 10, 10, 10));
        
        // Add pressure jump
        for i in 5..10 {
            for j in 0..10 {
                for k in 0..10 {
                    pressure[[i, j, k]] = 10.0;
                    density[[i, j, k]] = 2.0;
                }
            }
        }
        
        let indicator = detector.detect_shocks(&pressure, &grid);
        
        // Should detect shock around x=5
        assert!(indicator[[5, 5, 5]]);
        assert!(!indicator[[0, 5, 5]]);
    }
}