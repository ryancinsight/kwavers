//! Properly integrated PSTD solver plugin
//!
//! Implements correct k-space propagation with:
//! - Cached FFT plans (no repeated initialization)
//! - Pre-allocated complex arrays
//! - Proper second-order time stepping
//! - K-space correction filters
//!
//! References:
//! - Mast et al. (2001): "A k-space method for large-scale models"
//! - Tabei et al. (2002): "A k-space method for coupled first-order equations"

use crate::{
    error::KwaversResult,
    fft::{Fft3d, Ifft3d},
    grid::Grid,
    medium::Medium,
    physics::{
        field_mapping::UnifiedFieldType,
        plugin::{PhysicsPlugin, PluginContext, PluginMetadata, PluginState},
    },
};
use ndarray::{Array3, Array4};
use num_complex::Complex;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::Debug;

/// PSTD solver configuration
#[derive(Debug, Clone)]
pub struct PstdConfig {
    /// Enable k-space correction
    pub k_space_correction: bool,
    /// Order of k-space correction (1 or 2)
    pub k_space_order: usize,
    /// CFL safety factor
    pub cfl_safety_factor: f64,
}

impl Default for PstdConfig {
    fn default() -> Self {
        Self {
            k_space_correction: true,
            k_space_order: 2,
            cfl_safety_factor: 0.3,
        }
    }
}

/// Properly integrated PSTD solver plugin
#[derive(Debug)]
pub struct ProperPstdPlugin {
    /// Plugin metadata
    metadata: PluginMetadata,
    /// Current state
    state: PluginState,
    /// Configuration
    config: PstdConfig,
    
    // Grid parameters
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    
    // Pressure fields for second-order time stepping
    /// Current pressure (at time t)
    p_curr: Array3<Complex<f64>>,
    /// Previous pressure (at time t-dt)
    p_prev: Array3<Complex<f64>>,
    /// Work array for FFT operations
    p_work: Array3<Complex<f64>>,
    
    // Pre-cached FFT plans (expensive to create)
    /// Forward FFT plan
    fft_plan: Option<Fft3d>,
    /// Inverse FFT plan
    ifft_plan: Option<Ifft3d>,
    
    // K-space arrays
    /// Wavenumber x-component
    kx: Array3<f64>,
    /// Wavenumber y-component
    ky: Array3<f64>,
    /// Wavenumber z-component
    kz: Array3<f64>,
    /// K-space correction filter
    k_filter: Option<Array3<f64>>,
    
    // Pre-cached medium properties
    /// Sound speed map
    sound_speed_map: Array3<f64>,
    /// Maximum sound speed for CFL
    max_sound_speed: f64,
}

impl ProperPstdPlugin {
    /// Create a new PSTD plugin
    pub fn new(config: PstdConfig) -> Self {
        let metadata = PluginMetadata {
            id: "pstd_proper".to_string(),
            name: "Proper PSTD Solver".to_string(),
            version: "2.0.0".to_string(),
            description: "K-space pseudospectral solver with correct time stepping".to_string(),
            author: "Kwavers Team".to_string(),
            license: "MIT".to_string(),
        };
        
        Self {
            metadata,
            state: PluginState::Created,
            config,
            nx: 1,
            ny: 1,
            nz: 1,
            dx: 1.0,
            dy: 1.0,
            dz: 1.0,
            p_curr: Array3::zeros((1, 1, 1)),
            p_prev: Array3::zeros((1, 1, 1)),
            p_work: Array3::zeros((1, 1, 1)),
            fft_plan: None,
            ifft_plan: None,
            kx: Array3::zeros((1, 1, 1)),
            ky: Array3::zeros((1, 1, 1)),
            kz: Array3::zeros((1, 1, 1)),
            k_filter: None,
            sound_speed_map: Array3::zeros((1, 1, 1)),
            max_sound_speed: 1500.0,
        }
    }
    
    /// Initialize wavenumber arrays
    fn initialize_wavenumbers(&mut self) {
        self.kx = Array3::zeros((self.nx, self.ny, self.nz));
        self.ky = Array3::zeros((self.nx, self.ny, self.nz));
        self.kz = Array3::zeros((self.nx, self.ny, self.nz));
        
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    // Proper k-space indexing for FFT
                    self.kx[[i, j, k]] = if i <= self.nx / 2 {
                        2.0 * PI * i as f64 / (self.nx as f64 * self.dx)
                    } else {
                        2.0 * PI * (i as i32 - self.nx as i32) as f64 / (self.nx as f64 * self.dx)
                    };
                    
                    self.ky[[i, j, k]] = if j <= self.ny / 2 {
                        2.0 * PI * j as f64 / (self.ny as f64 * self.dy)
                    } else {
                        2.0 * PI * (j as i32 - self.ny as i32) as f64 / (self.ny as f64 * self.dy)
                    };
                    
                    self.kz[[i, j, k]] = if k <= self.nz / 2 {
                        2.0 * PI * k as f64 / (self.nz as f64 * self.dz)
                    } else {
                        2.0 * PI * (k as i32 - self.nz as i32) as f64 / (self.nz as f64 * self.dz)
                    };
                }
            }
        }
    }
    
    /// Create k-space correction filter
    fn create_k_filter(&mut self) {
        if !self.config.k_space_correction {
            return;
        }
        
        let mut filter = Array3::ones((self.nx, self.ny, self.nz));
        
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let kx_val = self.kx[[i, j, k]];
                    let ky_val = self.ky[[i, j, k]];
                    let kz_val = self.kz[[i, j, k]];
                    
                    // Sinc correction for finite difference approximation
                    let sinc_x = sinc(kx_val * self.dx / 2.0);
                    let sinc_y = sinc(ky_val * self.dy / 2.0);
                    let sinc_z = sinc(kz_val * self.dz / 2.0);
                    
                    filter[[i, j, k]] = match self.config.k_space_order {
                        1 => sinc_x * sinc_y * sinc_z,
                        2 => (sinc_x * sinc_y * sinc_z).powi(2),
                        _ => 1.0,
                    };
                }
            }
        }
        
        self.k_filter = Some(filter);
    }
    
    /// Cache sound speed from medium
    fn cache_sound_speed(&mut self, medium: &dyn Medium, grid: &Grid) {
        self.sound_speed_map = Array3::zeros((self.nx, self.ny, self.nz));
        self.max_sound_speed = 0.0;
        
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let x = i as f64 * self.dx;
                    let y = j as f64 * self.dy;
                    let z = k as f64 * self.dz;
                    
                    let c = medium.sound_speed(x, y, z, grid);
                    self.sound_speed_map[[i, j, k]] = c;
                    self.max_sound_speed = self.max_sound_speed.max(c);
                }
            }
        }
    }
    
    /// Perform PSTD time step
    fn pstd_step(&mut self, source: &Array3<f64>, dt: f64) -> KwaversResult<()> {
        // Transform current and previous pressure to k-space
        self.p_work.assign(&self.p_curr);
        if let Some(ref mut fft) = self.fft_plan {
            let grid = Grid::new(self.nx, self.ny, self.nz, self.dx, self.dy, self.dz);
            fft.process(&mut self.p_work, &grid);
        }
        let p_curr_k = self.p_work.clone();
        
        self.p_work.assign(&self.p_prev);
        if let Some(ref mut fft) = self.fft_plan {
            let grid = Grid::new(self.nx, self.ny, self.nz, self.dx, self.dy, self.dz);
            fft.process(&mut self.p_work, &grid);
        }
        let p_prev_k = self.p_work.clone();
        
        // Apply second-order time stepping in k-space
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let c = self.sound_speed_map[[i, j, k]];
                    
                    // Wavenumber magnitude
                    let kx_val = self.kx[[i, j, k]];
                    let ky_val = self.ky[[i, j, k]];
                    let kz_val = self.kz[[i, j, k]];
                    let k_mag = (kx_val * kx_val + ky_val * ky_val + kz_val * kz_val).sqrt();
                    
                    // Time evolution operator
                    let propagator = 2.0 * (c * k_mag * dt).cos();
                    
                    // Update pressure in k-space
                    self.p_work[[i, j, k]] = propagator * p_curr_k[[i, j, k]] - p_prev_k[[i, j, k]];
                    
                    // Apply k-space filter
                    if let Some(ref filter) = self.k_filter {
                        self.p_work[[i, j, k]] *= filter[[i, j, k]];
                    }
                    
                    // Add source term
                    if source[[i, j, k]].abs() > 1e-10 {
                        self.p_work[[i, j, k]] += Complex::new(source[[i, j, k]] * dt * dt, 0.0);
                    }
                }
            }
        }
        
        // Transform back to spatial domain
        if let Some(ref mut ifft) = self.ifft_plan {
            let grid = Grid::new(self.nx, self.ny, self.nz, self.dx, self.dy, self.dz);
            ifft.process(&mut self.p_work, &grid);
        }
        
        // Update pressure fields
        self.p_prev = self.p_curr.clone();
        self.p_curr = self.p_work.clone();
        
        Ok(())
    }
}

impl PhysicsPlugin for ProperPstdPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> PluginState {
        self.state
    }
    
    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }
    
    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }
    
    fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()> {
        // Store grid parameters
        self.nx = grid.nx;
        self.ny = grid.ny;
        self.nz = grid.nz;
        self.dx = grid.dx;
        self.dy = grid.dy;
        self.dz = grid.dz;
        
        // Initialize arrays
        let shape = (self.nx, self.ny, self.nz);
        self.p_curr = Array3::zeros(shape);
        self.p_prev = Array3::zeros(shape);
        self.p_work = Array3::zeros(shape);
        
        // Create FFT plans once (expensive operation)
        self.fft_plan = Some(Fft3d::new(self.nx, self.ny, self.nz));
        self.ifft_plan = Some(Ifft3d::new(self.nx, self.ny, self.nz));
        
        // Initialize wavenumbers and filter
        self.initialize_wavenumbers();
        self.create_k_filter();
        
        // Cache sound speed
        self.cache_sound_speed(medium, grid);
        
        self.state = PluginState::Initialized;
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        _medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        // Extract pressure field
        let pressure = fields.index_axis(ndarray::Axis(0), 0);
        
        // Convert to complex for current pressure
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    self.p_curr[[i, j, k]] = Complex::new(pressure[[i, j, k]], 0.0);
                }
            }
        }
        
        // Create source array (could be passed in)
        let source = Array3::zeros((self.nx, self.ny, self.nz));
        
        // Perform PSTD step
        self.pstd_step(&source, dt)?;
        
        // Copy back real part to fields
        let mut pressure_mut = fields.index_axis_mut(ndarray::Axis(0), 0);
        for k in 0..self.nz {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    pressure_mut[[i, j, k]] = self.p_curr[[i, j, k]].re;
                }
            }
        }
        
        self.state = PluginState::Running;
        Ok(())
    }
    
    fn stability_constraints(&self) -> f64 {
        self.config.cfl_safety_factor
    }
    
    fn diagnostics(&self) -> HashMap<String, f64> {
        let mut diag = HashMap::new();
        
        // Calculate CFL number
        let dx_min = self.dx.min(self.dy).min(self.dz);
        let cfl_max = self.max_sound_speed * self.config.cfl_safety_factor / dx_min;
        
        diag.insert("max_sound_speed".to_string(), self.max_sound_speed);
        diag.insert("cfl_max".to_string(), cfl_max);
        
        diag
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