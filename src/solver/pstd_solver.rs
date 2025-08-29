//! Proper PSTD (Pseudospectral Time-Domain) solver implementation
//!
//! This implements a correct k-space propagation method for acoustic waves
//! based on the second-order wave equation.
//!
//! References:
//! - Mast et al., "A k-space method for large-scale models of wave propagation in tissue"
//!   IEEE Trans. Ultrason. Ferroelectr. Freq. Control 48, 341-354 (2001)
//! - Tabei et al., "A k-space method for coupled first-order acoustic propagation equations"
//!   J. Acoust. Soc. Am. 111, 53-63 (2002)

use crate::{
    error::KwaversResult,
    fft::{Fft3d, Ifft3d},
    grid::Grid,
    medium::Medium,
};
use ndarray::Array3;
use num_complex::Complex;
use std::f64::consts::PI;

/// PSTD solver for acoustic wave propagation
pub struct PstdSolver {
    /// Computational grid
    grid: Grid,
    /// Current pressure field (at time t)
    p_curr: Array3<Complex<f64>>,
    /// Previous pressure field (at time t-dt)
    p_prev: Array3<Complex<f64>>,
    /// FFT plan (cached for performance)
    fft_plan: Fft3d,
    /// Inverse FFT plan (cached for performance)
    ifft_plan: Ifft3d,
    /// Wavenumber arrays
    kx: Array3<f64>,
    ky: Array3<f64>,
    kz: Array3<f64>,
    /// k-space correction filter (optional)
    k_filter: Option<Array3<f64>>,
    /// Pre-cached sound speed map
    sound_speed_map: Array3<f64>,
    /// CFL safety factor
    cfl_safety_factor: f64,
}

impl PstdSolver {
    /// Create a new PSTD solver
    pub fn new(grid: Grid, medium: &dyn Medium, k_space_order: Option<usize>) -> KwaversResult<Self> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        
        // Initialize pressure fields
        let p_curr = Array3::zeros((nx, ny, nz));
        let p_prev = Array3::zeros((nx, ny, nz));
        
        // Create FFT plans once (expensive operation)
        let fft_plan = Fft3d::new(nx, ny, nz);
        let ifft_plan = Ifft3d::new(nx, ny, nz);
        
        // Pre-compute wavenumber arrays
        let (kx, ky, kz) = Self::create_wavenumber_arrays(&grid);
        
        // Create k-space correction filter if requested
        let k_filter = if let Some(order) = k_space_order {
            Some(Self::create_k_space_filter(&grid, &kx, &ky, &kz, order)?)
        } else {
            None
        };
        
        // Pre-cache sound speed map to avoid dynamic dispatch in hot loop
        let sound_speed_map = Self::cache_sound_speed(medium, &grid);
        
        Ok(Self {
            grid,
            p_curr,
            p_prev,
            fft_plan,
            ifft_plan,
            kx,
            ky,
            kz,
            k_filter,
            sound_speed_map,
            cfl_safety_factor: 0.3, // Conservative default for PSTD
        })
    }
    
    /// Pre-cache sound speed from medium to avoid dynamic dispatch
    fn cache_sound_speed(medium: &dyn Medium, grid: &Grid) -> Array3<f64> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut sound_speed = Array3::zeros((nx, ny, nz));
        
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    sound_speed[[i, j, k]] = medium.sound_speed(x, y, z, grid);
                }
            }
        }
        
        sound_speed
    }
    
    /// Create wavenumber arrays for k-space operations
    fn create_wavenumber_arrays(grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut kx = Array3::zeros((nx, ny, nz));
        let mut ky = Array3::zeros((nx, ny, nz));
        let mut kz = Array3::zeros((nx, ny, nz));
        
        let kx_max = PI / grid.dx;
        let ky_max = PI / grid.dy;
        let kz_max = PI / grid.dz;
        
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    // Proper k-space indexing for FFT
                    kx[[i, j, k]] = if i <= nx / 2 {
                        2.0 * PI * i as f64 / (nx as f64 * grid.dx)
                    } else {
                        2.0 * PI * (i as i32 - nx as i32) as f64 / (nx as f64 * grid.dx)
                    };
                    
                    ky[[i, j, k]] = if j <= ny / 2 {
                        2.0 * PI * j as f64 / (ny as f64 * grid.dy)
                    } else {
                        2.0 * PI * (j as i32 - ny as i32) as f64 / (ny as f64 * grid.dy)
                    };
                    
                    kz[[i, j, k]] = if k <= nz / 2 {
                        2.0 * PI * k as f64 / (nz as f64 * grid.dz)
                    } else {
                        2.0 * PI * (k as i32 - nz as i32) as f64 / (nz as f64 * grid.dz)
                    };
                }
            }
        }
        
        (kx, ky, kz)
    }
    
    /// Create k-space correction filter for numerical dispersion
    fn create_k_space_filter(
        grid: &Grid,
        kx: &Array3<f64>,
        ky: &Array3<f64>,
        kz: &Array3<f64>,
        order: usize,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut filter = Array3::ones((nx, ny, nz));
        
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let kx_val = kx[[i, j, k]];
                    let ky_val = ky[[i, j, k]];
                    let kz_val = kz[[i, j, k]];
                    
                    // Sinc correction for finite difference approximation
                    let sinc_x = sinc(kx_val * grid.dx / 2.0);
                    let sinc_y = sinc(ky_val * grid.dy / 2.0);
                    let sinc_z = sinc(kz_val * grid.dz / 2.0);
                    
                    filter[[i, j, k]] = match order {
                        1 => sinc_x * sinc_y * sinc_z,
                        2 => (sinc_x * sinc_y * sinc_z).powi(2),
                        _ => 1.0,
                    };
                }
            }
        }
        
        Ok(filter)
    }
    
    /// Update the pressure field using proper PSTD time-stepping
    pub fn update(
        &mut self,
        source_term: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        
        // Transform current and previous pressure to k-space
        let mut p_curr_k = self.p_curr.clone();
        let mut p_prev_k = self.p_prev.clone();
        
        self.fft_plan.process(&mut p_curr_k, &self.grid);
        self.fft_plan.process(&mut p_prev_k, &self.grid);
        
        // Prepare next pressure field in k-space
        let mut p_next_k = Array3::zeros((nx, ny, nz));
        
        // Apply the second-order time-stepping in k-space
        // p_next = 2*cos(c*|k|*dt)*p_curr - p_prev + dt^2*source_term
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    // Get local sound speed (using cached value)
                    let c = self.sound_speed_map[[i, j, k]];
                    
                    // Calculate wavenumber magnitude
                    let kx_val = self.kx[[i, j, k]];
                    let ky_val = self.ky[[i, j, k]];
                    let kz_val = self.kz[[i, j, k]];
                    let k_mag = (kx_val * kx_val + ky_val * ky_val + kz_val * kz_val).sqrt();
                    
                    // Time evolution operator
                    let propagator = 2.0 * (c * k_mag * dt).cos();
                    
                    // Apply propagation
                    p_next_k[[i, j, k]] = propagator * p_curr_k[[i, j, k]] - p_prev_k[[i, j, k]];
                    
                    // Apply k-space filter if present
                    if let Some(ref filter) = self.k_filter {
                        p_next_k[[i, j, k]] *= filter[[i, j, k]];
                    }
                }
            }
        }
        
        // Add source term in k-space (transform source first)
        if source_term.iter().any(|&x| x.abs() > 1e-10) {
            let mut source_k = source_term.mapv(|x| Complex::new(x * dt * dt, 0.0));
            self.fft_plan.process(&mut source_k, &self.grid);
            p_next_k = p_next_k + source_k;
        }
        
        // Transform back to spatial domain
        self.ifft_plan.process(&mut p_next_k, &self.grid);
        
        // Update pressure fields for next iteration
        self.p_prev = self.p_curr.clone();
        self.p_curr = p_next_k;
        
        Ok(())
    }
    
    /// Get the current pressure field (real part)
    pub fn get_pressure(&self) -> Array3<f64> {
        self.p_curr.mapv(|c| c.re)
    }
    
    /// Set initial pressure field
    pub fn set_pressure(&mut self, pressure: &Array3<f64>) {
        self.p_curr = pressure.mapv(|x| Complex::new(x, 0.0));
    }
    
    /// Check CFL stability condition
    pub fn check_stability(&self, dt: f64) -> KwaversResult<()> {
        let dx_min = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let max_sound_speed = self.sound_speed_map.iter()
            .cloned()
            .fold(0.0, f64::max);
        
        let cfl = max_sound_speed * dt / dx_min;
        
        if cfl > self.cfl_safety_factor {
            return Err(crate::error::ValidationError::RangeValidation {
                field: "CFL".to_string(),
                value: cfl.to_string(),
                min: "0".to_string(),
                max: self.cfl_safety_factor.to_string(),
            }
            .into());
        }
        
        Ok(())
    }
}

/// Sinc function for k-space correction
fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-10 {
        1.0
    } else {
        x.sin() / x
    }
}