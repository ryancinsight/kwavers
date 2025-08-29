//! FDTD plugin for the plugin-based solver architecture
//!
//! This implements a proper velocity-pressure leapfrog FDTD scheme
//! with support for heterogeneous media and proper boundary conditions.
//!
//! References:
//! - Taflove & Hagness, "Computational Electrodynamics: The Finite-Difference Time-Domain Method"
//! - Virieux, "P-SV wave propagation in heterogeneous media: Velocity-stress finite-difference method"
//!   Geophysics 51, 889-901 (1986)

use crate::{
    error::KwaversResult,
    grid::Grid,
    medium::Medium,
    physics::plugin::{PhysicsPlugin, PluginMetadata, PluginContext},
};
use ndarray::{Array3, Array4};
use std::collections::HashMap;

/// FDTD plugin implementing velocity-pressure leapfrog scheme
pub struct FdtdPlugin {
    /// Plugin metadata
    metadata: PluginMetadata,
    /// Pre-cached density map
    density_map: Array3<f64>,
    /// Pre-cached sound speed map  
    sound_speed_map: Array3<f64>,
    /// Velocity components (staggered grid)
    velocity: Array4<f64>,
    /// CFL safety factor
    cfl_safety_factor: f64,
    /// Enable PML boundaries
    use_pml: bool,
}

impl FdtdPlugin {
    /// Create a new FDTD plugin with pre-cached medium properties
    pub fn new(grid: &Grid, medium: &dyn Medium, use_pml: bool) -> KwaversResult<Self> {
        // Pre-cache medium properties to avoid dynamic dispatch
        let (density_map, sound_speed_map) = Self::cache_medium_properties(medium, grid);
        
        // Initialize velocity field (3 components)
        let velocity = Array4::zeros((3, grid.nx, grid.ny, grid.nz));
        
        let metadata = PluginMetadata {
            name: "FDTD Velocity-Pressure Plugin".to_string(),
            version: "1.0.0".to_string(),
            author: "Kwavers Team".to_string(),
            description: "Proper velocity-pressure leapfrog FDTD implementation".to_string(),
        };
        
        Ok(Self {
            metadata,
            density_map,
            sound_speed_map,
            velocity,
            cfl_safety_factor: 0.5, // Standard for FDTD
            use_pml,
        })
    }
    
    /// Cache medium properties to avoid dynamic dispatch in hot loop
    fn cache_medium_properties(
        medium: &dyn Medium,
        grid: &Grid,
    ) -> (Array3<f64>, Array3<f64>) {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let mut density = Array3::zeros((nx, ny, nz));
        let mut sound_speed = Array3::zeros((nx, ny, nz));
        
        // Pre-sample all medium properties
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;
                    
                    density[[i, j, k]] = medium.density(x, y, z, grid);
                    sound_speed[[i, j, k]] = medium.sound_speed(x, y, z, grid);
                }
            }
        }
        
        (density, sound_speed)
    }
    
    /// Update velocity from pressure gradient (leapfrog step 1)
    fn update_velocity(
        &mut self,
        pressure: &Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        
        // Update vx: v_x^{n+1/2} = v_x^{n-1/2} - dt/rho * dp/dx
        for k in 0..nz {
            for j in 0..ny {
                for i in 1..nx {
                    let rho = 0.5 * (self.density_map[[i, j, k]] + self.density_map[[i-1, j, k]]);
                    let dp_dx = (pressure[[i, j, k]] - pressure[[i-1, j, k]]) / grid.dx;
                    self.velocity[[0, i, j, k]] -= dt / rho * dp_dx;
                }
            }
        }
        
        // Update vy: v_y^{n+1/2} = v_y^{n-1/2} - dt/rho * dp/dy
        for k in 0..nz {
            for j in 1..ny {
                for i in 0..nx {
                    let rho = 0.5 * (self.density_map[[i, j, k]] + self.density_map[[i, j-1, k]]);
                    let dp_dy = (pressure[[i, j, k]] - pressure[[i, j-1, k]]) / grid.dy;
                    self.velocity[[1, i, j, k]] -= dt / rho * dp_dy;
                }
            }
        }
        
        // Update vz: v_z^{n+1/2} = v_z^{n-1/2} - dt/rho * dp/dz
        for k in 1..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let rho = 0.5 * (self.density_map[[i, j, k]] + self.density_map[[i, j, k-1]]);
                    let dp_dz = (pressure[[i, j, k]] - pressure[[i, j, k-1]]) / grid.dz;
                    self.velocity[[2, i, j, k]] -= dt / rho * dp_dz;
                }
            }
        }
    }
    
    /// Update pressure from velocity divergence (leapfrog step 2)
    fn update_pressure(
        &self,
        pressure: &mut Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        
        // Update pressure: p^{n+1} = p^n - dt * rho * c^2 * div(v)
        for k in 0..nz-1 {
            for j in 0..ny-1 {
                for i in 0..nx-1 {
                    let rho = self.density_map[[i, j, k]];
                    let c = self.sound_speed_map[[i, j, k]];
                    let bulk_modulus = rho * c * c;
                    
                    // Calculate velocity divergence (careful with staggered grid)
                    let div_v = (self.velocity[[0, i+1, j, k]] - self.velocity[[0, i, j, k]]) / grid.dx
                              + (self.velocity[[1, i, j+1, k]] - self.velocity[[1, i, j, k]]) / grid.dy
                              + (self.velocity[[2, i, j, k+1]] - self.velocity[[2, i, j, k]]) / grid.dz;
                    
                    pressure[[i, j, k]] -= dt * bulk_modulus * div_v;
                }
            }
        }
    }
    
    /// Apply boundary conditions (simple absorbing for now)
    fn apply_boundary_conditions(&mut self, pressure: &mut Array3<f64>, grid: &Grid) {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        
        if !self.use_pml {
            // Simple absorbing boundary (zeroth order)
            // Set pressure to zero at boundaries
            for k in 0..nz {
                for j in 0..ny {
                    pressure[[0, j, k]] = 0.0;
                    pressure[[nx-1, j, k]] = 0.0;
                }
            }
            
            for k in 0..nz {
                for i in 0..nx {
                    pressure[[i, 0, k]] = 0.0;
                    pressure[[i, ny-1, k]] = 0.0;
                }
            }
            
            for j in 0..ny {
                for i in 0..nx {
                    pressure[[i, j, 0]] = 0.0;
                    pressure[[i, j, nz-1]] = 0.0;
                }
            }
        }
        // PML boundaries would be handled by a separate PML plugin
    }
}

impl PhysicsPlugin for FdtdPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn execute(
        &mut self,
        fields: &mut HashMap<String, Array4<f64>>,
        _medium: &dyn Medium,
        grid: &Grid,
        _time: f64,
        dt: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        // Extract pressure field from HashMap
        let pressure_field = fields.get_mut("pressure")
            .ok_or_else(|| crate::error::KwaversError::InvalidInput(
                "Pressure field not found".to_string()
            ))?;
        
        // Get the pressure array (assuming it's the first component)
        let mut pressure = pressure_field.index_axis(ndarray::Axis(0), 0).to_owned();
        
        // Velocity-pressure leapfrog scheme
        // Step 1: Update velocity from pressure gradient
        self.update_velocity(&pressure, grid, dt);
        
        // Step 2: Update pressure from velocity divergence
        self.update_pressure(&mut pressure, grid, dt);
        
        // Step 3: Apply boundary conditions
        self.apply_boundary_conditions(&mut pressure, grid);
        
        // Write back pressure to fields HashMap
        pressure_field.index_axis_mut(ndarray::Axis(0), 0).assign(&pressure);
        
        Ok(())
    }
    
    fn stability_constraints(&self) -> f64 {
        self.cfl_safety_factor
    }
    
    fn validate(&self, grid: &Grid, dt: f64, _medium: &dyn Medium) -> KwaversResult<()> {
        // Check CFL condition
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
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