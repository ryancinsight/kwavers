//! Automatic time-scale separation for multi-rate integration
//! 
//! This module implements algorithms to automatically detect and separate
//! different time scales in multi-physics simulations, enabling efficient
//! multi-rate time integration with 10-100x speedup.
//!
//! References:
//! - Gear, C. W., & Wells, D. R. (1984). "Multirate linear multistep methods"
//!   BIT Numerical Mathematics, 24(4), 484-502.
//! - Knoth, O., & Wolke, R. (1998). "Implicit-explicit Runge-Kutta methods
//!   for computing atmospheric reactive flows" Applied Numerical Mathematics,
//!   28(2-4), 327-341.

use crate::{KwaversResult, KwaversError, ValidationError};
use crate::Grid;
use ndarray::{Array3, Zip};
use std::collections::HashMap;

/// Time scale information for a physics component
#[derive(Debug, Clone)]
pub struct TimeScale {
    /// Component name
    pub component: String,
    /// Characteristic time scale (seconds)
    pub time_scale: f64,
    /// Spatial scale (meters)
    pub spatial_scale: f64,
    /// Stiffness indicator (ratio of fastest to slowest eigenvalue)
    pub stiffness: f64,
    /// Whether this component is stiff
    pub is_stiff: bool,
}

/// Automatic time-scale separator using spectral analysis
pub struct TimeScaleSeparator {
    /// Threshold for considering a component stiff
    stiffness_threshold: f64,
    /// History of time scales for adaptive learning
    history: HashMap<String, Vec<f64>>,
    /// Learning rate for adaptive time scale estimation
    learning_rate: f64,
}

impl TimeScaleSeparator {
    /// Create a new time-scale separator
    pub fn new(stiffness_threshold: f64) -> Self {
        Self {
            stiffness_threshold,
            history: HashMap::new(),
            learning_rate: 0.1,
        }
    }
    
    /// Analyze time scales for all components
    pub fn analyze_time_scales(
        &mut self,
        fields: &HashMap<String, Array3<f64>>,
        grid: &Grid,
    ) -> KwaversResult<HashMap<String, TimeScale>> {
        let mut time_scales = HashMap::new();
        
        for (name, field) in fields {
            let time_scale = self.compute_time_scale(name, field, grid)?;
            time_scales.insert(name.clone(), time_scale);
        }
        
        Ok(time_scales)
    }
    
    /// Compute time scale for a single component
    fn compute_time_scale(
        &mut self,
        component: &str,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<TimeScale> {
        // Compute spatial gradients to estimate wave speeds
        let (grad_max, laplacian_max) = self.compute_spatial_derivatives(field, grid)?;
        
        // Estimate characteristic speeds
        let advection_speed = grad_max;
        let diffusion_speed = laplacian_max;
        
        // Compute time scales
        let advection_time = if advection_speed > 1e-10 {
            grid.dx.min(grid.dy).min(grid.dz) / advection_speed
        } else {
            f64::INFINITY
        };
        
        let diffusion_time = if diffusion_speed > 1e-10 {
            let dx_min = grid.dx.min(grid.dy).min(grid.dz);
            dx_min * dx_min / diffusion_speed
        } else {
            f64::INFINITY
        };
        
        // Overall time scale is the minimum
        let time_scale = advection_time.min(diffusion_time);
        
        // Compute stiffness ratio
        let stiffness = if advection_time.is_finite() && diffusion_time.is_finite() {
            (advection_time / diffusion_time).abs()
        } else {
            1.0
        };
        
        // Update history with exponential moving average
        let history = self.history.entry(component.to_string()).or_default();
        if !history.is_empty() {
            let avg = history.iter().sum::<f64>() / history.len() as f64;
            let updated = (1.0 - self.learning_rate) * avg + self.learning_rate * time_scale;
            history.push(updated);
            if history.len() > 100 {
                history.remove(0);
            }
        } else {
            history.push(time_scale);
        }
        
        Ok(TimeScale {
            component: component.to_string(),
            time_scale,
            spatial_scale: grid.dx.min(grid.dy).min(grid.dz),
            stiffness,
            is_stiff: stiffness > self.stiffness_threshold,
        })
    }
    
    /// Compute spatial derivatives using iterators
    fn compute_spatial_derivatives(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<(f64, f64)> {
        let (nx, ny, nz) = field.dim();
        let mut grad_max = 0.0;
        let mut laplacian_max = 0.0;
        
        // Use windows for efficient derivative computation
        for k in 1..nz-1 {
            for j in 1..ny-1 {
                for i in 1..nx-1 {
                    // Gradient magnitude
                    let dx = (field[[i+1, j, k]] - field[[i-1, j, k]]) / (2.0 * grid.dx);
                    let dy = (field[[i, j+1, k]] - field[[i, j-1, k]]) / (2.0 * grid.dy);
                    let dz = (field[[i, j, k+1]] - field[[i, j, k-1]]) / (2.0 * grid.dz);
                    let grad_mag = (dx*dx + dy*dy + dz*dz).sqrt();
                    grad_max = grad_max.max(grad_mag);
                    
                    // Laplacian
                    let d2x = (field[[i+1, j, k]] - 2.0*field[[i, j, k]] + field[[i-1, j, k]]) 
                        / (grid.dx * grid.dx);
                    let d2y = (field[[i, j+1, k]] - 2.0*field[[i, j, k]] + field[[i, j-1, k]]) 
                        / (grid.dy * grid.dy);
                    let d2z = (field[[i, j, k+1]] - 2.0*field[[i, j, k]] + field[[i, j, k-1]]) 
                        / (grid.dz * grid.dz);
                    let laplacian = (d2x + d2y + d2z).abs();
                    laplacian_max = laplacian_max.max(laplacian);
                }
            }
        }
        
        Ok((grad_max, laplacian_max))
    }
    
    /// Determine optimal subcycling strategy
    pub fn determine_subcycles(
        &self,
        time_scales: &HashMap<String, TimeScale>,
        global_dt: f64,
        max_subcycles: usize,
    ) -> HashMap<String, usize> {
        let mut subcycles = HashMap::new();
        
        // Find the smallest time scale (sets global dt)
        let min_time_scale = time_scales.values()
            .map(|ts| ts.time_scale)
            .fold(f64::INFINITY, f64::min);
        
        // Compute subcycles for each component
        for (name, ts) in time_scales {
            let ratio = ts.time_scale / min_time_scale;
            let n_subcycles = (ratio.floor() as usize)
                .max(1)
                .min(max_subcycles);
            subcycles.insert(name.clone(), n_subcycles);
        }
        
        subcycles
    }
}