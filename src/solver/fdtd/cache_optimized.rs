//! Cache-optimized derivative computations for FDTD
//!
//! This module provides single-pass derivative calculations that maximize
//! cache reuse and minimize memory bandwidth requirements.

use ndarray::{Array3, Zip};
use crate::error::KwaversResult;
use crate::medium::Medium;

/// Cache-optimized FDTD updates
pub struct CacheOptimizedFDTD {
    /// Finite difference coefficients
    fd_coeffs: Vec<f64>,
    /// Grid spacings
    dx: f64,
    dy: f64,
    dz: f64,
}

impl CacheOptimizedFDTD {
    pub fn new(fd_coeffs: Vec<f64>, dx: f64, dy: f64, dz: f64) -> Self {
        Self { fd_coeffs, dx, dy, dz }
    }
    
    /// Compute divergence and update pressure in a single pass
    /// This minimizes cache misses by computing all derivatives at once
    pub fn update_pressure_optimized(
        &self,
        pressure: &mut Array3<f64>,
        vx: &Array3<f64>,
        vy: &Array3<f64>,
        vz: &Array3<f64>,
        medium: &dyn Medium,
        dt: f64,
        cpml_corrections: Option<(&Array3<f64>, &Array3<f64>, &Array3<f64>)>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = pressure.dim();
        let half_stencil = self.fd_coeffs.len() / 2;
        
        // Get medium properties
        let rho_array = medium.density_array();
        let c_array = medium.sound_speed_array();
        
        // Process interior points with full stencil
        // This is the hot loop - all derivatives computed in one pass
        for i in half_stencil..(nx - half_stencil) {
            for j in half_stencil..(ny - half_stencil) {
                for k in half_stencil..(nz - half_stencil) {
                    // Compute dvx/dx using stencil
                    let mut dvx_dx = 0.0;
                    for (idx, &coeff) in self.fd_coeffs.iter().enumerate() {
                        let offset = idx as i32 - half_stencil as i32;
                        if offset != 0 {
                            let ii = (i as i32 + offset) as usize;
                            dvx_dx += coeff * vx[[ii, j, k]];
                        }
                    }
                    dvx_dx /= self.dx;
                    
                    // Compute dvy/dy using stencil
                    let mut dvy_dy = 0.0;
                    for (idx, &coeff) in self.fd_coeffs.iter().enumerate() {
                        let offset = idx as i32 - half_stencil as i32;
                        if offset != 0 {
                            let jj = (j as i32 + offset) as usize;
                            dvy_dy += coeff * vy[[i, jj, k]];
                        }
                    }
                    dvy_dy /= self.dy;
                    
                    // Compute dvz/dz using stencil
                    let mut dvz_dz = 0.0;
                    for (idx, &coeff) in self.fd_coeffs.iter().enumerate() {
                        let offset = idx as i32 - half_stencil as i32;
                        if offset != 0 {
                            let kk = (k as i32 + offset) as usize;
                            dvz_dz += coeff * vz[[i, j, kk]];
                        }
                    }
                    dvz_dz /= self.dz;
                    
                    // Apply CPML corrections if provided
                    if let Some((cpml_x, cpml_y, cpml_z)) = cpml_corrections {
                        dvx_dx += cpml_x[[i, j, k]];
                        dvy_dy += cpml_y[[i, j, k]];
                        dvz_dz += cpml_z[[i, j, k]];
                    }
                    
                    // Compute divergence
                    let div_v = dvx_dx + dvy_dy + dvz_dz;
                    
                    // Update pressure using bulk modulus
                    let bulk_modulus = rho_array[[i, j, k]] * c_array[[i, j, k]].powi(2);
                    pressure[[i, j, k]] -= dt * bulk_modulus * div_v;
                }
            }
        }
        
        // Handle boundaries separately with appropriate stencils
        // (Implementation would use boundary_stencils module)
        
        Ok(())
    }
    
    /// Compute gradients and update velocity in a single pass
    pub fn update_velocity_optimized(
        &self,
        vx: &mut Array3<f64>,
        vy: &mut Array3<f64>,
        vz: &mut Array3<f64>,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        dt: f64,
        cpml_corrections: Option<(&Array3<f64>, &Array3<f64>, &Array3<f64>)>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = pressure.dim();
        let half_stencil = self.fd_coeffs.len() / 2;
        
        // Get density array
        let rho_array = medium.density_array();
        
        // Process interior points with full stencil
        for i in half_stencil..(nx - half_stencil) {
            for j in half_stencil..(ny - half_stencil) {
                for k in half_stencil..(nz - half_stencil) {
                    // Compute dp/dx
                    let mut dp_dx = 0.0;
                    for (idx, &coeff) in self.fd_coeffs.iter().enumerate() {
                        let offset = idx as i32 - half_stencil as i32;
                        if offset != 0 {
                            let ii = (i as i32 + offset) as usize;
                            dp_dx += coeff * pressure[[ii, j, k]];
                        }
                    }
                    dp_dx /= self.dx;
                    
                    // Compute dp/dy
                    let mut dp_dy = 0.0;
                    for (idx, &coeff) in self.fd_coeffs.iter().enumerate() {
                        let offset = idx as i32 - half_stencil as i32;
                        if offset != 0 {
                            let jj = (j as i32 + offset) as usize;
                            dp_dy += coeff * pressure[[i, jj, k]];
                        }
                    }
                    dp_dy /= self.dy;
                    
                    // Compute dp/dz
                    let mut dp_dz = 0.0;
                    for (idx, &coeff) in self.fd_coeffs.iter().enumerate() {
                        let offset = idx as i32 - half_stencil as i32;
                        if offset != 0 {
                            let kk = (k as i32 + offset) as usize;
                            dp_dz += coeff * pressure[[i, j, kk]];
                        }
                    }
                    dp_dz /= self.dz;
                    
                    // Apply CPML corrections if provided
                    if let Some((cpml_x, cpml_y, cpml_z)) = cpml_corrections {
                        dp_dx += cpml_x[[i, j, k]];
                        dp_dy += cpml_y[[i, j, k]];
                        dp_dz += cpml_z[[i, j, k]];
                    }
                    
                    // Update velocity components
                    let inv_rho = 1.0 / rho_array[[i, j, k]];
                    vx[[i, j, k]] -= dt * dp_dx * inv_rho;
                    vy[[i, j, k]] -= dt * dp_dy * inv_rho;
                    vz[[i, j, k]] -= dt * dp_dz * inv_rho;
                }
            }
        }
        
        Ok(())
    }
    
    /// Helper to compute stencil derivative at a single point
    /// Used for boundary handling
    pub fn compute_stencil_deriv_at_point(
        &self,
        field: &Array3<f64>,
        axis: usize,
        i: usize,
        j: usize,
        k: usize,
    ) -> f64 {
        let half_stencil = self.fd_coeffs.len() / 2;
        let mut val = 0.0;
        
        for (idx, &coeff) in self.fd_coeffs.iter().enumerate() {
            let offset = idx as i32 - half_stencil as i32;
            if offset != 0 {
                match axis {
                    0 => {
                        let ii = (i as i32 + offset) as usize;
                        if ii < field.dim().0 {
                            val += coeff * field[[ii, j, k]];
                        }
                    }
                    1 => {
                        let jj = (j as i32 + offset) as usize;
                        if jj < field.dim().1 {
                            val += coeff * field[[i, jj, k]];
                        }
                    }
                    2 => {
                        let kk = (k as i32 + offset) as usize;
                        if kk < field.dim().2 {
                            val += coeff * field[[i, j, kk]];
                        }
                    }
                    _ => {}
                }
            }
        }
        
        let spacing = match axis {
            0 => self.dx,
            1 => self.dy,
            2 => self.dz,
            _ => 1.0,
        };
        
        val / spacing
    }
}