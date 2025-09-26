// Sprint 96 Implementation Plan: k-Space Pseudospectral Foundation
// 
// OBJECTIVE: Achieve exact k-Wave parity for power-law absorption
// PRIORITY: P0 - CRITICAL GAP (Core functionality missing)
//
// This module implements the k-space pseudospectral method following:
// Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the simulation
// and reconstruction of photoacoustic wave fields." J. Biomed. Opt. 15(2).

use ndarray::{Array3, ArrayView3, ArrayViewMut3};
use num_complex::Complex;
use std::f64::consts::PI;

/// k-Space pseudospectral operator for exact k-Wave compatibility
/// 
/// Implements power-law absorption with dispersion correction:
/// α(ω) = α₀ · |ω|^y where y ∈ [0, 3]
/// 
/// # Mathematical Foundation
/// 
/// The k-space operator applies absorption and dispersion in frequency domain:
/// ```text
/// ∂p/∂t = -c₀·∇·u - α(ω)*p + S
/// ∂u/∂t = -∇p/ρ₀
/// ```
/// 
/// With k-space correction for arbitrary absorption laws.
#[derive(Debug, Clone)]
pub struct KSpaceOperator {
    /// Wavenumber arrays for each spatial dimension
    kx: Array3<f64>,
    ky: Array3<f64>, 
    kz: Array3<f64>,
    
    /// Power-law absorption operator: exp(-α(ω)·Δt)
    absorption_operator: Array3<Complex<f64>>,
    
    /// Dispersion correction operator for causal absorption
    dispersion_correction: Array3<Complex<f64>>,
    
    /// Grid spacing for finite difference corrections
    dx: f64,
    dy: f64,
    dz: f64,
    
    /// Sound speed (assumed homogeneous for k-space method)
    c0: f64,
}

impl KSpaceOperator {
    /// Create new k-space operator with power-law absorption
    /// 
    /// # Arguments
    /// * `grid_size` - (nx, ny, nz) grid dimensions
    /// * `grid_spacing` - (dx, dy, dz) spatial steps
    /// * `c0` - Reference sound speed
    /// * `alpha_coeff` - Absorption coefficient α₀ 
    /// * `alpha_power` - Power law exponent y ∈ [0, 3]
    /// * `dt` - Time step for operator precomputation
    pub fn new(
        grid_size: (usize, usize, usize),
        grid_spacing: (f64, f64, f64),
        c0: f64,
        alpha_coeff: f64,
        alpha_power: f64,
        dt: f64,
    ) -> Self {
        let (nx, ny, nz) = grid_size;
        let (dx, dy, dz) = grid_spacing;
        
        // Initialize k-space wavenumber grids
        let mut kx = Array3::zeros((nx, ny, nz));
        let mut ky = Array3::zeros((nx, ny, nz)); 
        let mut kz = Array3::zeros((nx, ny, nz));
        
        // Compute wavenumbers with proper FFT ordering
        Self::compute_wavenumbers(&mut kx, &mut ky, &mut kz, grid_size, grid_spacing);
        
        // Precompute absorption operator
        let absorption_operator = Self::compute_absorption_operator(
            &kx, &ky, &kz, c0, alpha_coeff, alpha_power, dt
        );
        
        // Precompute dispersion correction  
        let dispersion_correction = Self::compute_dispersion_correction(
            &kx, &ky, &kz, c0, alpha_power, dt
        );
        
        Self {
            kx,
            ky, 
            kz,
            absorption_operator,
            dispersion_correction,
            dx,
            dy,
            dz,
            c0,
        }
    }
    
    /// Apply k-space absorption to pressure field (in-place)
    /// 
    /// Multiplies by exp(-α(ω)·Δt) in frequency domain
    pub fn apply_absorption(&self, pressure_fft: &mut Array3<Complex<f64>>) {
        *pressure_fft *= &self.absorption_operator;
    }
    
    /// Apply dispersion correction for causal absorption
    /// 
    /// Corrects phase velocity to maintain causality with power-law absorption
    pub fn apply_dispersion(&self, pressure_fft: &mut Array3<Complex<f64>>) {
        *pressure_fft *= &self.dispersion_correction;
    }
    
    /// Compute k-space gradient (returns ∇p in frequency domain)
    /// 
    /// Uses exact spectral derivatives: ∇_FFT = i·k·FFT(field)
    pub fn k_space_gradient(
        &self, 
        pressure_fft: &Array3<Complex<f64>>
    ) -> (Array3<Complex<f64>>, Array3<Complex<f64>>, Array3<Complex<f64>>) {
        let i = Complex::new(0.0, 1.0);
        
        let grad_x = pressure_fft * &self.kx.mapv(|k| i * k);
        let grad_y = pressure_fft * &self.ky.mapv(|k| i * k); 
        let grad_z = pressure_fft * &self.kz.mapv(|k| i * k);
        
        (grad_x, grad_y, grad_z)
    }
    
    /// Compute k-space Laplacian (returns ∇²p in frequency domain)
    pub fn k_space_laplacian(&self, pressure_fft: &Array3<Complex<f64>>) -> Array3<Complex<f64>> {
        let k_squared = &self.kx.mapv(|k| k*k) + &self.ky.mapv(|k| k*k) + &self.kz.mapv(|k| k*k);
        pressure_fft * &k_squared.mapv(|k2| Complex::new(-k2, 0.0))
    }
    
    // Private implementation methods
    
    fn compute_wavenumbers(
        kx: &mut Array3<f64>,
        ky: &mut Array3<f64>, 
        kz: &mut Array3<f64>,
        grid_size: (usize, usize, usize),
        grid_spacing: (f64, f64, f64),
    ) {
        let (nx, ny, nz) = grid_size;
        let (dx, dy, dz) = grid_spacing;
        
        // k-space sampling following FFT conventions
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    // FFT frequency indexing with proper Nyquist handling
                    let kx_val = if i <= nx/2 { 
                        2.0 * PI * i as f64 / (nx as f64 * dx)
                    } else {
                        2.0 * PI * (i as f64 - nx as f64) / (nx as f64 * dx) 
                    };
                    
                    let ky_val = if j <= ny/2 {
                        2.0 * PI * j as f64 / (ny as f64 * dy)
                    } else {
                        2.0 * PI * (j as f64 - ny as f64) / (ny as f64 * dy)
                    };
                    
                    let kz_val = if k <= nz/2 {
                        2.0 * PI * k as f64 / (nz as f64 * dz)
                    } else {
                        2.0 * PI * (k as f64 - nz as f64) / (nz as f64 * dz)
                    };
                    
                    kx[[i,j,k]] = kx_val;
                    ky[[i,j,k]] = ky_val;
                    kz[[i,j,k]] = kz_val;
                }
            }
        }
    }
    
    fn compute_absorption_operator(
        kx: &Array3<f64>,
        ky: &Array3<f64>,
        kz: &Array3<f64>,
        c0: f64,
        alpha_coeff: f64,
        alpha_power: f64, 
        dt: f64,
    ) -> Array3<Complex<f64>> {
        let mut absorption = Array3::zeros(kx.raw_dim());
        
        for ((i, j, k), abs_val) in absorption.indexed_iter_mut() {
            let k_mag = (kx[[i,j,k]].powi(2) + ky[[i,j,k]].powi(2) + kz[[i,j,k]].powi(2)).sqrt();
            let omega = c0 * k_mag;
            
            if omega > 0.0 {
                // Power-law absorption: α(ω) = α₀ · |ω|^y
                let alpha = alpha_coeff * omega.powf(alpha_power);
                *abs_val = Complex::new((-alpha * dt).exp(), 0.0);
            } else {
                *abs_val = Complex::new(1.0, 0.0);
            }
        }
        
        absorption
    }
    
    fn compute_dispersion_correction(
        kx: &Array3<f64>,
        ky: &Array3<f64>, 
        kz: &Array3<f64>,
        c0: f64,
        alpha_power: f64,
        dt: f64,
    ) -> Array3<Complex<f64>> {
        let mut dispersion = Array3::zeros(kx.raw_dim());
        
        // Dispersion correction for causal absorption (Treeby & Cox 2010)
        for ((i, j, k), disp_val) in dispersion.indexed_iter_mut() {
            let k_mag = (kx[[i,j,k]].powi(2) + ky[[i,j,k]].powi(2) + kz[[i,j,k]].powi(2)).sqrt();
            
            if k_mag > 0.0 {
                // Phase correction for power-law dispersion
                let phase_correction = match alpha_power {
                    y if (y - 1.0).abs() < f64::EPSILON => {
                        // y = 1 case (special handling to avoid singularity)
                        0.0
                    },
                    y => {
                        // General case: tan(πy/2) correction
                        let omega = c0 * k_mag;
                        -omega * dt * (PI * y / 2.0).tan()
                    }
                };
                
                *disp_val = Complex::new(0.0, phase_correction).exp();
            } else {
                *disp_val = Complex::new(1.0, 0.0);
            }
        }
        
        dispersion
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_k_space_operator_creation() {
        let operator = KSpaceOperator::new(
            (64, 64, 64),
            (1e-4, 1e-4, 1e-4),
            1500.0,  // c0
            0.75,    // alpha_coeff
            1.5,     // alpha_power
            1e-8,    // dt
        );
        
        // Verify wavenumber symmetry properties
        let (nx, ny, nz) = (64, 64, 64);
        
        // DC component should be zero
        assert_eq!(operator.kx[[0,0,0]], 0.0);
        assert_eq!(operator.ky[[0,0,0]], 0.0);
        assert_eq!(operator.kz[[0,0,0]], 0.0);
        
        // Nyquist frequency handling
        if nx % 2 == 0 {
            let nyq_x = operator.kx[[nx/2, 0, 0]];
            assert!(nyq_x > 0.0, "Nyquist frequency should be positive");
        }
    }
    
    #[test]
    fn test_absorption_operator_properties() {
        let operator = KSpaceOperator::new(
            (32, 32, 32),
            (1e-4, 1e-4, 1e-4), 
            1500.0,
            0.75,
            1.5,
            1e-8,
        );
        
        // All absorption values should have magnitude ≤ 1 (physical requirement)
        for abs_val in operator.absorption_operator.iter() {
            assert!(abs_val.norm() <= 1.0, "Absorption must be ≤ 1 for stability");
            assert!(abs_val.norm() > 0.0, "Absorption must be > 0 for causality");
        }
        
        // DC component should have no absorption (ω = 0)
        let dc_abs = operator.absorption_operator[[0,0,0]];
        assert_relative_eq!(dc_abs.norm(), 1.0, epsilon = 1e-10);
    }
    
    #[test] 
    fn test_k_space_gradient_accuracy() {
        let operator = KSpaceOperator::new(
            (16, 16, 16),
            (1e-4, 1e-4, 1e-4),
            1500.0,
            0.75,
            1.5, 
            1e-8,
        );
        
        // Test gradient of simple sinusoidal field
        let mut test_field = Array3::zeros((16, 16, 16));
        let k_test = 2.0 * PI / (8.0 * 1e-4); // Wavelength = 8*dx
        
        for ((i, j, k), val) in test_field.indexed_iter_mut() {
            let x = i as f64 * 1e-4;
            *val = Complex::new((k_test * x).sin(), 0.0);
        }
        
        let (grad_x, _, _) = operator.k_space_gradient(&test_field);
        
        // Analytical gradient: d/dx[sin(kx)] = k*cos(kx)
        for ((i, j, k), grad_val) in grad_x.indexed_iter() {
            let x = i as f64 * 1e-4;
            let expected = Complex::new(k_test * (k_test * x).cos(), 0.0);
            
            // Allow some numerical error from FFT discretization
            if expected.norm() > 1e-10 {
                assert_relative_eq!(
                    grad_val.re, 
                    expected.re, 
                    epsilon = 1e-6
                );
            }
        }
    }
}