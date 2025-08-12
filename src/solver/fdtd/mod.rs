//! Finite-Difference Time Domain (FDTD) solver
//! 
//! This module implements the FDTD method using Yee's staggered grid scheme
//! for solving Maxwell's equations and acoustic wave equations.
//! 
//! # Theory
//! 
//! The FDTD method discretizes both space and time using finite differences.
//! Key features include:
//! 
//! - **Explicit time stepping**: Simple, efficient updates
//! - **Staggered grid (Yee cell)**: Natural enforcement of divergence conditions
//! - **Second-order accuracy**: In both space and time
//! - **Conditional stability**: CFL condition limits time step
//! 
//! # Algorithm
//! 
//! For acoustic waves, the update equations on a staggered grid are:
//! ```text
//! p^{n+1} = p^n - Δt·ρc²·(∇·v)^{n+1/2}
//! v^{n+3/2} = v^{n+1/2} - Δt/ρ·∇p^{n+1}
//! ```
//! 
//! # Literature References
//! 
//! 1. **Yee, K. S. (1966)**. "Numerical solution of initial boundary value 
//!    problems involving Maxwell's equations in isotropic media." *IEEE 
//!    Transactions on Antennas and Propagation*, 14(3), 302-307. 
//!    DOI: 10.1109/TAP.1966.1138693
//!    - Original Yee algorithm for electromagnetic waves
//!    - Introduction of the staggered grid concept
//! 
//! 2. **Virieux, J. (1986)**. "P-SV wave propagation in heterogeneous media: 
//!    Velocity-stress finite-difference method." *Geophysics*, 51(4), 889-901. 
//!    DOI: 10.1190/1.1442147
//!    - Extension to elastic wave propagation
//!    - Velocity-stress formulation
//! 
//! 3. **Graves, R. W. (1996)**. "Simulating seismic wave propagation in 3D 
//!    elastic media using staggered-grid finite differences." *Bulletin of the 
//!    Seismological Society of America*, 86(4), 1091-1106.
//!    - 3D implementation details
//!    - Higher-order accuracy schemes
//! 
//! 4. **Moczo, P., Kristek, J., & Gális, M. (2014)**. "The finite-difference 
//!    modelling of earthquake motions: Waves and ruptures." *Cambridge University 
//!    Press*. ISBN: 978-1107028814
//!    - Comprehensive treatment of FDTD for wave propagation
//!    - Stability analysis and optimization techniques
//! 
//! 5. **Taflove, A., & Hagness, S. C. (2005)**. "Computational electrodynamics: 
//!    The finite-difference time-domain method" (3rd ed.). *Artech House*. 
//!    ISBN: 978-1580538329
//!    - Definitive reference for FDTD methods
//!    - Advanced topics including subgridding and PML
//! 
//! # Implementation Details
//! 
//! ## Spatial Derivatives
//! 
//! We support 2nd, 4th, and 6th order accurate finite differences:
//! - 2nd order: [-1, 0, 1] / (2Δx)
//! - 4th order: [1/12, -2/3, 0, 2/3, -1/12] / Δx
//! - 6th order: [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60] / Δx
//! 
//! ## Stability Condition
//! 
//! The CFL condition for FDTD is:
//! ```text
//! Δt ≤ CFL / (c√(1/Δx² + 1/Δy² + 1/Δz²))
//! ```
//! where CFL ≈ 0.95 for stability margin.
//! 
//! ## Subgridding
//! 
//! Local mesh refinement following:
//! - Berenger, J. P. (2002). "Application of the CFS PML to the absorption of 
//!   evanescent waves in waveguides." *IEEE Microwave and Wireless Components 
//!   Letters*, 12(6), 218-220.
//! 
//! # Design Principles
//! - SOLID: Single responsibility for finite-difference wave propagation
//! - CUPID: Composable with other solvers via plugin architecture
//! - KISS: Simple, explicit time-stepping algorithm
//! - DRY: Reuses grid utilities and boundary conditions
//! - YAGNI: Implements only necessary features for acoustic simulation

use crate::grid::Grid;
use crate::medium::Medium;
use crate::error::{KwaversResult, KwaversError, ValidationError, ConfigError};
use crate::physics::plugin::{PhysicsPlugin, PluginMetadata, PluginContext, PluginState};
use ndarray::{Array3, Array4, Zip};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use log::{debug, info};

/// FDTD solver configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FdtdConfig {
    /// Spatial derivative order (2, 4, or 6)
    pub spatial_order: usize,
    /// Use staggered grid (Yee cell)
    pub staggered_grid: bool,
    /// CFL safety factor (typically 0.95 for FDTD)
    pub cfl_factor: f64,
    /// Enable subgridding for local refinement
    pub subgridding: bool,
    /// Subgridding refinement factor
    pub subgrid_factor: usize,
}

impl Default for FdtdConfig {
    fn default() -> Self {
        Self {
            spatial_order: 4,
            staggered_grid: true,
            cfl_factor: 0.95,
            subgridding: false,
            subgrid_factor: 2,
        }
    }
}

/// Staggered grid positions for Yee cell
#[derive(Debug, Clone)]
struct StaggeredGrid {
    /// Pressure at cell centers
    pressure_pos: (f64, f64, f64),
    /// Velocity components at face centers
    vx_pos: (f64, f64, f64),
    vy_pos: (f64, f64, f64),
    vz_pos: (f64, f64, f64),
}

impl Default for StaggeredGrid {
    fn default() -> Self {
        Self {
            pressure_pos: (0.0, 0.0, 0.0),      // Cell center
            vx_pos: (0.5, 0.0, 0.0),           // x-face center
            vy_pos: (0.0, 0.5, 0.0),           // y-face center
            vz_pos: (0.0, 0.0, 0.5),           // z-face center
        }
    }
}

/// FDTD solver for acoustic wave propagation
#[derive(Clone, Debug)]
pub struct FdtdSolver {
    /// Configuration
    config: FdtdConfig,
    /// Grid reference
    grid: Grid,
    /// Staggered grid positions
    staggered: StaggeredGrid,
    /// Finite difference coefficients
    fd_coeffs: HashMap<usize, Vec<f64>>,
    /// Performance metrics
    metrics: HashMap<String, f64>,
    /// Subgrid regions (if enabled)
    subgrids: Vec<SubgridRegion>,
    /// C-PML boundary (if enabled)
    cpml_boundary: Option<crate::boundary::cpml::CPMLBoundary>,
}

/// Subgrid region for local refinement
#[derive(Debug, Clone)]
struct SubgridRegion {
    /// Start indices in coarse grid
    start: (usize, usize, usize),
    /// End indices in coarse grid
    end: (usize, usize, usize),
    /// Fine grid data
    fine_pressure: Array3<f64>,
    fine_vx: Array3<f64>,
    fine_vy: Array3<f64>,
    fine_vz: Array3<f64>,
}

impl FdtdSolver {
    /// Create a new FDTD solver
    pub fn new(config: FdtdConfig, grid: &Grid) -> KwaversResult<Self> {
        info!("Initializing FDTD solver with config: {:?}", config);
        
        // Validate configuration
        if ![2, 4, 6].contains(&config.spatial_order) {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "spatial_order".to_string(),
                value: config.spatial_order.to_string(),
                constraint: "must be 2, 4, or 6".to_string(),
            }));
        }
        
        // Initialize finite difference coefficients for central differences
        // Fix: Correct coefficients for higher-order schemes
        let mut fd_coeffs = HashMap::new();
        fd_coeffs.insert(2, vec![0.5]); // 2nd order: (f(x+h) - f(x-h))/(2h)
        fd_coeffs.insert(4, vec![2.0/3.0, -1.0/12.0]); // 4th order: corrected coefficients
        fd_coeffs.insert(6, vec![3.0/4.0, -3.0/20.0, 1.0/60.0]); // 6th order: corrected coefficients
        
        Ok(Self {
            config,
            grid: grid.clone(),
            staggered: StaggeredGrid::default(),
            fd_coeffs,
            metrics: HashMap::new(),
            subgrids: Vec::new(),
            cpml_boundary: None,
        })
    }
    
    /// Enable C-PML boundary conditions
    pub fn enable_cpml(&mut self, config: crate::boundary::cpml::CPMLConfig) -> KwaversResult<()> {
        info!("Enabling C-PML boundary conditions");
        self.cpml_boundary = Some(crate::boundary::cpml::CPMLBoundary::new(
            config,
            &self.grid,
        )?);
        Ok(())
    }
    
    /// Compute spatial derivative using finite differences
    fn compute_derivative(
        &self,
        field: &Array3<f64>,
        axis: usize,
        stagger_offset: f64,
    ) -> Array3<f64> {
        let (nx, ny, nz) = field.dim();
        let mut deriv = Array3::zeros((nx, ny, nz));
        let coeffs = &self.fd_coeffs[&self.config.spatial_order];
        let n_coeffs = coeffs.len();
        
        // Determine grid spacing
        let dx = match axis {
            0 => self.grid.dx,
            1 => self.grid.dy,
            2 => self.grid.dz,
            _ => panic!("Invalid axis"),
        };
        
        // Determine bounds based on stencil size
        let half_stencil = n_coeffs;
        let start = half_stencil;
        let (end_x, end_y, end_z) = (nx - half_stencil, ny - half_stencil, nz - half_stencil);
        
        // Apply finite differences generically using coefficients
        for i in start..end_x {
            for j in start..end_y {
                for k in start..end_z {
                    let mut val = 0.0;
                    
                    // Apply stencil coefficients
                    for (idx, &coeff) in coeffs.iter().enumerate() {
                        let offset = idx + 1;
                        match axis {
                            0 => {
                                val += coeff * (field[[i + offset, j, k]] - field[[i - offset, j, k]]);
                            }
                            1 => {
                                val += coeff * (field[[i, j + offset, k]] - field[[i, j - offset, k]]);
                            }
                            2 => {
                                val += coeff * (field[[i, j, k + offset]] - field[[i, j, k - offset]]);
                            }
                            _ => {}
                        }
                    }
                    
                    deriv[[i, j, k]] = val / dx;
                }
            }
        }
        
        // Handle boundaries with lower-order schemes
        self.apply_boundary_derivatives(&mut deriv, field, axis, dx);
        
        // Apply stagger offset if using staggered grid
        if self.config.staggered_grid && stagger_offset != 0.0 {
            // Interpolate to staggered positions
            self.interpolate_to_staggered(&deriv, axis, stagger_offset)
        } else {
            deriv
        }
    }
    
    /// Apply lower-order derivatives at boundaries
    fn apply_boundary_derivatives(
        &self,
        deriv: &mut Array3<f64>,
        field: &Array3<f64>,
        axis: usize,
        dx: f64,
    ) {
        let (nx, ny, nz) = field.dim();
        let half_stencil = self.fd_coeffs[&self.config.spatial_order].len();
        
        // Use 2nd order at boundaries
        match axis {
            0 => {
                for j in 0..ny {
                    for k in 0..nz {
                        // Left boundary
                        for i in 0..half_stencil.min(nx) {
                            if i > 0 && i < nx - 1 {
                                deriv[[i, j, k]] = (field[[i+1, j, k]] - field[[i-1, j, k]]) / (2.0 * dx);
                            }
                        }
                        // Right boundary
                        for i in (nx - half_stencil).max(0)..nx {
                            if i > 0 && i < nx - 1 {
                                deriv[[i, j, k]] = (field[[i+1, j, k]] - field[[i-1, j, k]]) / (2.0 * dx);
                            }
                        }
                    }
                }
            }
            1 => {
                for i in 0..nx {
                    for k in 0..nz {
                        // Bottom boundary
                        for j in 0..half_stencil.min(ny) {
                            if j > 0 && j < ny - 1 {
                                deriv[[i, j, k]] = (field[[i, j+1, k]] - field[[i, j-1, k]]) / (2.0 * dx);
                            }
                        }
                        // Top boundary
                        for j in (ny - half_stencil).max(0)..ny {
                            if j > 0 && j < ny - 1 {
                                deriv[[i, j, k]] = (field[[i, j+1, k]] - field[[i, j-1, k]]) / (2.0 * dx);
                            }
                        }
                    }
                }
            }
            2 => {
                for i in 0..nx {
                    for j in 0..ny {
                        // Front boundary
                        for k in 0..half_stencil.min(nz) {
                            if k > 0 && k < nz - 1 {
                                deriv[[i, j, k]] = (field[[i, j, k+1]] - field[[i, j, k-1]]) / (2.0 * dx);
                            }
                        }
                        // Back boundary
                        for k in (nz - half_stencil).max(0)..nz {
                            if k > 0 && k < nz - 1 {
                                deriv[[i, j, k]] = (field[[i, j, k+1]] - field[[i, j, k-1]]) / (2.0 * dx);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    
    /// Interpolate field to staggered grid positions
    fn interpolate_to_staggered(
        &self,
        field: &Array3<f64>,
        axis: usize,
        offset: f64,
    ) -> Array3<f64> {
        let (nx, ny, nz) = field.dim();
        
        // Simple linear interpolation for staggered grid
        if offset == 0.5 {
            match axis {
                0 => {
                    // Interpolate to x-face centers
                    Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                        if i < nx - 1 {
                            0.5 * (field[[i, j, k]] + field[[i + 1, j, k]])
                        } else {
                            field[[i, j, k]]
                        }
                    })
                }
                1 => {
                    // Interpolate to y-face centers
                    Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                        if j < ny - 1 {
                            0.5 * (field[[i, j, k]] + field[[i, j + 1, k]])
                        } else {
                            field[[i, j, k]]
                        }
                    })
                }
                2 => {
                    // Interpolate to z-face centers
                    Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
                        if k < nz - 1 {
                            0.5 * (field[[i, j, k]] + field[[i, j, k + 1]])
                        } else {
                            field[[i, j, k]]
                        }
                    })
                }
                _ => panic!("Invalid axis"),
            }
        } else {
            // For non-0.5 offsets, just copy (could implement higher-order interpolation)
            field.to_owned()
        }
    }
    
    /// Update pressure field using velocity divergence
    pub fn update_pressure(
        &mut self,
        pressure: &mut Array3<f64>,
        vx: &Array3<f64>,
        vy: &Array3<f64>,
        vz: &Array3<f64>,
        medium: &dyn crate::medium::Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        // Compute bulk modulus from density and sound speed
        let rho_array = medium.density_array();
        let c_array = medium.sound_speed_array();
        let bulk_modulus = &rho_array * &c_array * &c_array;
        
        // Compute velocity divergence
        let mut dvx_dx = self.compute_derivative(vx, 0, self.staggered.vx_pos.0);
        let mut dvy_dy = self.compute_derivative(vy, 1, self.staggered.vy_pos.1);
        let mut dvz_dz = self.compute_derivative(vz, 2, self.staggered.vz_pos.2);
        
        // Apply C-PML if enabled
        if let Some(ref mut cpml) = self.cpml_boundary {
            // Update C-PML memory variables and apply to divergence components
            cpml.update_acoustic_memory(&dvx_dx, 0)?;
            cpml.apply_cpml_gradient(&mut dvx_dx, 0)?;
            
            cpml.update_acoustic_memory(&dvy_dy, 1)?;
            cpml.apply_cpml_gradient(&mut dvy_dy, 1)?;
            
            cpml.update_acoustic_memory(&dvz_dz, 2)?;
            cpml.apply_cpml_gradient(&mut dvz_dz, 2)?;
        }
        
        // Compute divergence
        let div_v = &dvx_dx + &dvy_dy + &dvz_dz;
        
        // Update pressure: ∂p/∂t = -K·∇·v
        Zip::from(pressure)
            .and(&div_v)
            .and(&bulk_modulus)
            .for_each(|p, &div, &bulk| {
                *p -= dt * bulk * div;
            });
        
        Ok(())
    }
    
    /// Update velocity field using pressure gradient
    pub fn update_velocity(
        &mut self,
        vx: &mut Array3<f64>,
        vy: &mut Array3<f64>,
        vz: &mut Array3<f64>,
        pressure: &Array3<f64>,
        medium: &dyn crate::medium::Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        let rho_array = medium.density_array();
        
        // Compute pressure gradients
        let mut dp_dx = self.compute_derivative(pressure, 0, -self.staggered.vx_pos.0);
        let mut dp_dy = self.compute_derivative(pressure, 1, -self.staggered.vy_pos.1);
        let mut dp_dz = self.compute_derivative(pressure, 2, -self.staggered.vz_pos.2);
        
        // Apply C-PML if enabled
        if let Some(ref mut cpml) = self.cpml_boundary {
            // Update C-PML memory variables and apply to gradients
            cpml.update_acoustic_memory(&dp_dx, 0)?;
            cpml.apply_cpml_gradient(&mut dp_dx, 0)?;
            
            cpml.update_acoustic_memory(&dp_dy, 1)?;
            cpml.apply_cpml_gradient(&mut dp_dy, 1)?;
            
            cpml.update_acoustic_memory(&dp_dz, 2)?;
            cpml.apply_cpml_gradient(&mut dp_dz, 2)?;
        }
        
        // Update velocities: ∂v/∂t = -∇p/ρ
        Zip::from(vx)
            .and(&dp_dx)
            .and(&rho_array)
            .for_each(|v, &grad, &rho| {
                *v -= dt * grad / rho;
            });
            
        Zip::from(vy)
            .and(&dp_dy)
            .and(&rho_array)
            .for_each(|v, &grad, &rho| {
                *v -= dt * grad / rho;
            });
            
        Zip::from(vz)
            .and(&dp_dz)
            .and(&rho_array)
            .for_each(|v, &grad, &rho| {
                *v -= dt * grad / rho;
            });
        
        Ok(())
    }
    
    /// Add a subgrid region for local refinement
    pub fn add_subgrid(
        &mut self,
        start: (usize, usize, usize),
        end: (usize, usize, usize),
    ) -> KwaversResult<()> {
        if !self.config.subgridding {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "subgridding".to_string(),
                value: "false".to_string(),
                constraint: "must be enabled to add subgrids".to_string(),
            }));
        }
        
        let factor = self.config.subgrid_factor;
        let (nx, ny, nz) = (
            (end.0 - start.0) * factor,
            (end.1 - start.1) * factor,
            (end.2 - start.2) * factor,
        );
        
        let subgrid = SubgridRegion {
            start,
            end,
            fine_pressure: Array3::zeros((nx, ny, nz)),
            fine_vx: Array3::zeros((nx, ny, nz)),
            fine_vy: Array3::zeros((nx, ny, nz)),
            fine_vz: Array3::zeros((nx, ny, nz)),
        };
        
        self.subgrids.push(subgrid);
        info!("Added subgrid region from {:?} to {:?}", start, end);
        
        Ok(())
    }
    
    /// Interpolate from coarse to fine grid
    fn interpolate_to_fine(
        &self,
        coarse: &Array3<f64>,
        fine: &mut Array3<f64>,
        region: &SubgridRegion,
    ) {
        let factor = self.config.subgrid_factor;
        
        // Simple linear interpolation
        for i in 0..fine.shape()[0] {
            for j in 0..fine.shape()[1] {
                for k in 0..fine.shape()[2] {
                    let ci = region.start.0 + i / factor;
                    let cj = region.start.1 + j / factor;
                    let ck = region.start.2 + k / factor;
                    
                    if ci < coarse.shape()[0] && cj < coarse.shape()[1] && ck < coarse.shape()[2] {
                        fine[[i, j, k]] = coarse[[ci, cj, ck]];
                    }
                }
            }
        }
    }
    
    /// Restrict from fine to coarse grid
    fn restrict_to_coarse(
        &self,
        fine: &Array3<f64>,
        coarse: &mut Array3<f64>,
        region: &SubgridRegion,
    ) {
        let factor = self.config.subgrid_factor;
        
        // Average fine grid values
        for i in region.start.0..region.end.0 {
            for j in region.start.1..region.end.1 {
                for k in region.start.2..region.end.2 {
                    let mut sum = 0.0;
                    let mut count = 0;
                    
                    for fi in 0..factor {
                        for fj in 0..factor {
                            for fk in 0..factor {
                                let idx_i = (i - region.start.0) * factor + fi;
                                let idx_j = (j - region.start.1) * factor + fj;
                                let idx_k = (k - region.start.2) * factor + fk;
                                
                                if idx_i < fine.shape()[0] && idx_j < fine.shape()[1] && idx_k < fine.shape()[2] {
                                    sum += fine[[idx_i, idx_j, idx_k]];
                                    count += 1;
                                }
                            }
                        }
                    }
                    
                    if count > 0 {
                        coarse[[i, j, k]] = sum / count as f64;
                    }
                }
            }
        }
    }
    
    /// Get maximum stable time step for this FDTD configuration
    pub fn max_stable_dt(&self, max_sound_speed: f64) -> f64 {
        let min_dx = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        
        // Fix: Correct CFL factors for different spatial orders
        let cfl_limit = match self.config.spatial_order {
            2 => 0.58,   // For 2nd-order in 3D: sqrt(1/3) ≈ 0.577
            4 => 0.50,   // More restrictive for 4th-order
            6 => 0.40,   // Even more restrictive for 6th-order
            _ => 0.58,
        };
        
        self.config.cfl_factor * cfl_limit * min_dx / max_sound_speed
    }
    
    /// Check if given timestep satisfies CFL condition
    pub fn check_cfl_stability(&self, dt: f64, max_sound_speed: f64) -> bool {
        let max_dt = self.max_stable_dt(max_sound_speed);
        dt <= max_dt
    }
    
    /// Get performance metrics
    pub fn get_metrics(&self) -> &HashMap<String, f64> {
        &self.metrics
    }
    
    /// Merge metrics from another solver instance
    pub fn merge_metrics(&mut self, other_metrics: &HashMap<String, f64>) {
        for (key, value) in other_metrics {
            // For most metrics, we'll take the maximum value
            // This can be customized based on the metric type
            if key.contains("time") || key.contains("elapsed") {
                // For time-based metrics, accumulate
                let current = self.metrics.get(key).copied().unwrap_or(0.0);
                self.metrics.insert(key.clone(), current + value);
            } else if key.contains("count") || key.contains("calls") {
                // For counters, accumulate
                let current = self.metrics.get(key).copied().unwrap_or(0.0);
                self.metrics.insert(key.clone(), current + value);
            } else {
                // For other metrics (like errors, norms), take the maximum
                let current = self.metrics.get(key).copied().unwrap_or(0.0);
                self.metrics.insert(key.clone(), current.max(*value));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    
    #[test]
    fn test_fdtd_creation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let config = FdtdConfig::default();
        let solver = FdtdSolver::new(config, &grid);
        assert!(solver.is_ok());
    }
    
    #[test]
    fn test_finite_difference_coefficients() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0);
        let config = FdtdConfig::default();
        let solver = FdtdSolver::new(config, &grid).unwrap();
        
        // Check that coefficients are loaded
        assert!(solver.fd_coeffs.contains_key(&2));
        assert!(solver.fd_coeffs.contains_key(&4));
        assert!(solver.fd_coeffs.contains_key(&6));
    }
    
    #[test]
    fn test_derivative_computation() {
        let grid = Grid::new(10, 10, 10, 1.0, 1.0, 1.0);
        let config = FdtdConfig {
            spatial_order: 2,
            staggered_grid: false,
            ..Default::default()
        };
        let solver = FdtdSolver::new(config, &grid).unwrap();
        
        // Create a linear field (derivative should be constant)
        let mut field = Array3::zeros((10, 10, 10));
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    field[[i, j, k]] = i as f64; // Linear in x
                }
            }
        }
        
        let deriv = solver.compute_derivative(&field, 0, 0.0);
        
        // Check that derivative is approximately 1.0 in the interior
        for i in 1..9 {
            for j in 1..9 {
                for k in 1..9 {
                    assert!((deriv[[i, j, k]] - 1.0).abs() < 1e-10, 
                           "Expected derivative 1.0, got {}", deriv[[i, j, k]]);
                }
            }
        }
    }
    
    #[test]
    fn test_cfl_condition() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let config = FdtdConfig::default();
        let solver = FdtdSolver::new(config, &grid).unwrap();
        
        let c_max = 1500.0; // Speed of sound in water
        let dt = solver.max_stable_dt(c_max);
        
        // Check that time step is reasonable
        assert!(dt > 0.0);
        assert!(dt < 1e-3); // Should be smaller than spatial step
    }
    
    #[test]
    fn test_subgrid_creation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
        let config = FdtdConfig {
            subgridding: true,
            subgrid_factor: 2,
            ..Default::default()
        };
        let mut solver = FdtdSolver::new(config, &grid).unwrap();
        
        // Add a subgrid region
        let result = solver.add_subgrid((10, 10, 10), (20, 20, 20));
        assert!(result.is_ok());
        assert_eq!(solver.subgrids.len(), 1);
    }
}

#[cfg(test)]
mod validation_tests;

// Plugin implementation for FDTD solver

/// FDTD solver plugin for integration with the physics pipeline
#[derive(Debug)]
pub struct FdtdPlugin {
    solver: FdtdSolver,
    metadata: PluginMetadata,
}

impl FdtdPlugin {
    /// Create a new FDTD plugin
    pub fn new(config: FdtdConfig, grid: &Grid) -> KwaversResult<Self> {
        let solver = FdtdSolver::new(config, &grid)?;
        let metadata = PluginMetadata {
            id: "fdtd_solver".to_string(),
            name: "FDTD Solver".to_string(),
            version: "1.0.0".to_string(),
            author: "Kwavers Team".to_string(),
            description: "Finite-Difference Time Domain solver with staggered grid".to_string(),
            license: "MIT".to_string(),
        };
        
        Ok(Self { solver, metadata })
    }
}

impl PhysicsPlugin for FdtdPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> PluginState {
        PluginState::Initialized
    }
    
    fn initialize(&mut self, _grid: &Grid, _medium: &dyn crate::medium::Medium) -> KwaversResult<()> {
        // Solver is already initialized in new()
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn crate::medium::Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()> {
        use ndarray::Axis;
        
        // Extract fields from the Array4 as owned arrays
        let mut pressure = fields.index_axis(Axis(0), 0).to_owned();
        let mut velocity_x = fields.index_axis(Axis(0), 4).to_owned();
        let mut velocity_y = fields.index_axis(Axis(0), 5).to_owned();
        let mut velocity_z = fields.index_axis(Axis(0), 6).to_owned();
        
        // FDTD uses leapfrog scheme: update velocity first, then pressure
        
        // Update velocities using current pressure
        self.solver.update_velocity(&mut velocity_x, &mut velocity_y, &mut velocity_z, &pressure, medium, dt)?;
        
        // Update pressure using new velocities
        self.solver.update_pressure(&mut pressure, &velocity_x, &velocity_y, &velocity_z, medium, dt)?;
        
        // Copy back to fields
        fields.index_axis_mut(Axis(0), 0).assign(&pressure);
        fields.index_axis_mut(Axis(0), 4).assign(&velocity_x);
        fields.index_axis_mut(Axis(0), 5).assign(&velocity_y);
        fields.index_axis_mut(Axis(0), 6).assign(&velocity_z);
        
        Ok(())
    }
    
    fn required_fields(&self) -> Vec<crate::physics::composable::FieldType> {
        use crate::physics::composable::FieldType;
        vec![
            FieldType::Pressure,
            FieldType::Velocity,
        ]
    }
    
    fn provided_fields(&self) -> Vec<crate::physics::composable::FieldType> {
        // FDTD updates the same fields it requires
        self.required_fields()
    }
    
    fn clone_plugin(&self) -> Box<dyn PhysicsPlugin> {
        Box::new(FdtdPlugin {
            solver: FdtdSolver::new(self.solver.config.clone(), &self.solver.grid).unwrap(),
            metadata: self.metadata.clone(),
        })
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}