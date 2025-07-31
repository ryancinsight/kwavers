//! Finite-Difference Time Domain (FDTD) solver
//! 
//! This module implements the FDTD method using Yee's staggered grid scheme
//! for solving Maxwell's equations and acoustic wave equations.
//! 
//! # Design Principles
//! - SOLID: Single responsibility for finite-difference wave propagation
//! - CUPID: Composable with other solvers via plugin architecture
//! - KISS: Simple, explicit time-stepping algorithm
//! - DRY: Reuses grid utilities and boundary conditions
//! - YAGNI: Implements only necessary features for acoustic simulation

use crate::grid::Grid;
use crate::medium::Medium;
use crate::boundary::Boundary;
use crate::error::{KwaversResult, KwaversError, ValidationError, ConfigError};
use crate::physics::plugin::{PhysicsPlugin, PluginMetadata, PluginConfig, PluginContext};
use crate::physics::composable::{FieldType, ValidationResult};
use ndarray::{Array3, Array4, Axis, Zip, s};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use log::{debug, info, warn};

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
        // Each coefficient is multiplied by (f(x+ih) - f(x-ih)) for i=1,2,3...
        let mut fd_coeffs = HashMap::new();
        fd_coeffs.insert(2, vec![1.0]); // 2nd order: (f(x+h) - f(x-h))
        fd_coeffs.insert(4, vec![8.0/12.0, -1.0/12.0]); // 4th order
        fd_coeffs.insert(6, vec![45.0/60.0, -9.0/60.0, 1.0/60.0]); // 6th order
        
        Ok(Self {
            config,
            grid: grid.clone(),
            staggered: StaggeredGrid::default(),
            fd_coeffs,
            metrics: HashMap::new(),
            subgrids: Vec::new(),
        })
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
        let mut interpolated = Array3::zeros((nx, ny, nz));
        
        // Simple linear interpolation for staggered grid
        if offset == 0.5 {
            match axis {
                0 => {
                    // Interpolate to x-face centers
                    for i in 0..nx-1 {
                        for j in 0..ny {
                            for k in 0..nz {
                                interpolated[[i, j, k]] = 0.5 * (field[[i, j, k]] + field[[i+1, j, k]]);
                            }
                        }
                    }
                }
                1 => {
                    // Interpolate to y-face centers
                    for i in 0..nx {
                        for j in 0..ny-1 {
                            for k in 0..nz {
                                interpolated[[i, j, k]] = 0.5 * (field[[i, j, k]] + field[[i, j+1, k]]);
                            }
                        }
                    }
                }
                2 => {
                    // Interpolate to z-face centers
                    for i in 0..nx {
                        for j in 0..ny {
                            for k in 0..nz-1 {
                                interpolated[[i, j, k]] = 0.5 * (field[[i, j, k]] + field[[i, j, k+1]]);
                            }
                        }
                    }
                }
                _ => panic!("Invalid axis"),
            }
        } else {
            // For non-0.5 offsets, just copy (could implement higher-order interpolation)
            interpolated.assign(field);
        }
        
        interpolated
    }
    
    /// Update pressure field using FDTD
    pub fn update_pressure(
        &mut self,
        pressure: &mut Array3<f64>,
        velocity_x: &Array3<f64>,
        velocity_y: &Array3<f64>,
        velocity_z: &Array3<f64>,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        debug!("FDTD pressure update, dt={}", dt);
        let start = std::time::Instant::now();
        
        // Get medium properties
        let rho_array = medium.density_array();
        let c_array = medium.sound_speed_array();
        
        // Compute velocity divergence
        let dvx_dx = self.compute_derivative(velocity_x, 0, self.staggered.vx_pos.0);
        let dvy_dy = self.compute_derivative(velocity_y, 1, self.staggered.vy_pos.1);
        let dvz_dz = self.compute_derivative(velocity_z, 2, self.staggered.vz_pos.2);
        
        // Update pressure: ∂p/∂t = -ρc²∇·v
        Zip::from(pressure)
            .and(&dvx_dx)
            .and(&dvy_dy)
            .and(&dvz_dz)
            .and(&rho_array)
            .and(&c_array)
            .for_each(|p, &dx, &dy, &dz, &rho, &c| {
                let divergence = dx + dy + dz;
                *p -= dt * rho * c * c * divergence;
            });
        
        // Update metrics
        let elapsed = start.elapsed().as_secs_f64();
        self.metrics.insert("pressure_update_time".to_string(), elapsed);
        
        Ok(())
    }
    
    /// Update velocity field using FDTD
    pub fn update_velocity(
        &mut self,
        velocity_x: &mut Array3<f64>,
        velocity_y: &mut Array3<f64>,
        velocity_z: &mut Array3<f64>,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        dt: f64,
    ) -> KwaversResult<()> {
        debug!("FDTD velocity update, dt={}", dt);
        let start = std::time::Instant::now();
        
        // Get density array
        let rho_array = medium.density_array();
        
        // Compute pressure gradients
        let dp_dx = self.compute_derivative(pressure, 0, -self.staggered.vx_pos.0);
        let dp_dy = self.compute_derivative(pressure, 1, -self.staggered.vy_pos.1);
        let dp_dz = self.compute_derivative(pressure, 2, -self.staggered.vz_pos.2);
        
        // Update velocities: ∂v/∂t = -∇p/ρ
        Zip::from(velocity_x)
            .and(&dp_dx)
            .and(&rho_array)
            .for_each(|v, &grad, &rho| {
                *v -= dt * grad / rho;
            });
        
        Zip::from(velocity_y)
            .and(&dp_dy)
            .and(&rho_array)
            .for_each(|v, &grad, &rho| {
                *v -= dt * grad / rho;
            });
        
        Zip::from(velocity_z)
            .and(&dp_dz)
            .and(&rho_array)
            .for_each(|v, &grad, &rho| {
                *v -= dt * grad / rho;
            });
        
        // Update metrics
        let elapsed = start.elapsed().as_secs_f64();
        self.metrics.insert("velocity_update_time".to_string(), elapsed);
        
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
    
    /// Get maximum stable time step
    pub fn max_stable_dt(&self, c_max: f64) -> f64 {
        let dx_min = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        
        // CFL condition depends on spatial order
        let cfl_limit = match self.config.spatial_order {
            2 => 1.0 / 3.0_f64.sqrt(),  // ~0.577
            4 => 0.5,                    // More restrictive for higher order
            6 => 0.4,                    // Even more restrictive
            _ => 0.5,
        };
        
        self.config.cfl_factor * cfl_limit * dx_min / c_max
    }
    
    /// Get performance metrics
    pub fn get_metrics(&self) -> &HashMap<String, f64> {
        &self.metrics
    }
}

/// FDTD solver as a physics plugin
pub struct FdtdPlugin {
    solver: FdtdSolver,
    metadata: PluginMetadata,
}

impl FdtdPlugin {
    pub fn new(config: FdtdConfig, grid: &Grid) -> KwaversResult<Self> {
        let solver = FdtdSolver::new(config, grid)?;
        let metadata = PluginMetadata {
            id: "fdtd_solver".to_string(),
            name: "FDTD Solver".to_string(),
            version: "1.0.0".to_string(),
            description: "Finite-Difference Time Domain solver with staggered grid support".to_string(),
            author: "Kwavers Team".to_string(),
            license: "MIT".to_string(),
        };
        Ok(Self {
            solver,
            metadata,
        })
    }
}

impl std::fmt::Debug for FdtdPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FdtdPlugin")
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl PhysicsPlugin for FdtdPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn required_fields(&self) -> Vec<FieldType> {
        vec![
            FieldType::Pressure,  // Needs pressure for velocity update
            FieldType::Velocity,  // Needs velocity for pressure update
        ]
    }
    
    fn provided_fields(&self) -> Vec<FieldType> {
        vec![
            FieldType::Pressure,  // Updates pressure field
            FieldType::Velocity,  // Updates velocity fields
        ]
    }
    
    fn initialize(
        &mut self,
        _config: Option<Box<dyn PluginConfig>>,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        // Check grid size compatibility with spatial order
        let min_size = match self.solver.config.spatial_order {
            2 => 3,
            4 => 5,
            6 => 7,
            _ => 3,
        };
        
        if grid.nx < min_size || grid.ny < min_size || grid.nz < min_size {
            return Err(KwaversError::Validation(ValidationError::FieldValidation {
                field: "grid_size".to_string(),
                value: format!("{}x{}x{}", grid.nx, grid.ny, grid.nz),
                constraint: format!("minimum {} points for order {}", min_size, self.solver.config.spatial_order),
            }));
        }
        
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        // Work directly with field views to avoid cloning
        
        // Update velocities first (leapfrog scheme)
        // Need to clone pressure for velocity update
        let pressure = fields.index_axis(Axis(0), 0).to_owned();
        {
            let mut velocity_x = fields.index_axis_mut(Axis(0), 4);
            let mut velocity_y = fields.index_axis_mut(Axis(0), 5);
            let mut velocity_z = fields.index_axis_mut(Axis(0), 6);
            
            self.solver.update_velocity(&mut velocity_x, &mut velocity_y, &mut velocity_z, &pressure, medium, dt)?;
        }
        
        // Then update pressure using the updated velocities
        {
            let velocity_x = fields.index_axis(Axis(0), 4);
            let velocity_y = fields.index_axis(Axis(0), 5);
            let velocity_z = fields.index_axis(Axis(0), 6);
            let mut pressure = fields.index_axis_mut(Axis(0), 0);
            
            self.solver.update_pressure(&mut pressure, &velocity_x, &velocity_y, &velocity_z, medium, dt)?;
        }
        
        // Handle subgrids if enabled
        if self.solver.config.subgridding {
            let subgrid_factor = self.solver.config.subgrid_factor;
            let fine_dt = dt / subgrid_factor as f64;
            
            for i in 0..self.solver.subgrids.len() {
                // Interpolate to fine grid
                let subgrid = &self.solver.subgrids[i];
                let mut fine_pressure = subgrid.fine_pressure.clone();
                let mut fine_vx = subgrid.fine_vx.clone();
                let mut fine_vy = subgrid.fine_vy.clone();
                let mut fine_vz = subgrid.fine_vz.clone();
                
                self.solver.interpolate_to_fine(&pressure, &mut fine_pressure, subgrid);
                self.solver.interpolate_to_fine(&velocity_x, &mut fine_vx, subgrid);
                self.solver.interpolate_to_fine(&velocity_y, &mut fine_vy, subgrid);
                self.solver.interpolate_to_fine(&velocity_z, &mut fine_vz, subgrid);
                
                // Update fine grid with smaller time steps
                for _ in 0..subgrid_factor {
                    // Update fine grid (simplified - would need proper boundary handling)
                    // This is a placeholder for actual subgrid updates
                }
                
                // Restrict back to coarse grid
                self.solver.restrict_to_coarse(&fine_pressure, &mut pressure, subgrid);
                self.solver.restrict_to_coarse(&fine_vx, &mut velocity_x, subgrid);
                self.solver.restrict_to_coarse(&fine_vy, &mut velocity_y, subgrid);
                self.solver.restrict_to_coarse(&fine_vz, &mut velocity_z, subgrid);
                
                // Update the subgrid
                let subgrid_mut = &mut self.solver.subgrids[i];
                subgrid_mut.fine_pressure = fine_pressure;
                subgrid_mut.fine_vx = fine_vx;
                subgrid_mut.fine_vy = fine_vy;
                subgrid_mut.fine_vz = fine_vz;
            }
        }
        
        // Write back to fields array
        fields.index_axis_mut(Axis(0), 0).assign(&pressure);
        fields.index_axis_mut(Axis(0), 4).assign(&velocity_x);
        fields.index_axis_mut(Axis(0), 5).assign(&velocity_y);
        fields.index_axis_mut(Axis(0), 6).assign(&velocity_z);
        
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.solver.get_metrics().clone()
    }
    
    fn reset(&mut self) -> KwaversResult<()> {
        self.solver.metrics.clear();
        self.solver.subgrids.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::HomogeneousMedium;
    
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