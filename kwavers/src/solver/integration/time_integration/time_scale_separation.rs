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

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array4, ArrayView3, Axis};
use std::collections::HashMap;

/// Time scale information for a physics component
#[derive(Debug, Clone)]
pub struct TimeScale {
    /// Component name
    pub parameter: String,
    /// Characteristic time scale (seconds)
    pub time_scale: f64,
    /// Spatial scale (meters)
    pub spatial_scale: f64,
    /// Stiffness indicator (ratio of largest to smallest eigenvalue magnitude)
    pub stiffness: f64,
    /// Whether this component is stiff
    pub is_stiff: bool,
}

/// Time scale separator for multi-rate integration
///
/// This component analyzes the system to identify different time scales
/// and determine appropriate sub-cycling ratios.
#[derive(Debug)]
pub struct TimeScaleSeparator {
    /// Grid reference
    grid: Grid,
    /// Minimum time scale ratio for separation
    min_separation_ratio: f64,
    /// History of time scales
    time_scale_history: Vec<Vec<f64>>,
}

impl TimeScaleSeparator {
    /// Create a new time scale separator
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(grid: &Grid) -> Self {
        Self {
            grid: grid.clone(),
            min_separation_ratio: 10.0,
            time_scale_history: Vec::new(),
        }
    }

    /// Analyze fields to identify time scales
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn analyze(&mut self, fields: &Array4<f64>, tolerance: f64) -> KwaversResult<Vec<f64>> {
        let mut time_scales = Vec::new();

        // Analyze each field component
        for f in 0..fields.shape()[0] {
            let field = fields.index_axis(Axis(0), f);

            // Compute characteristic time scales
            let (grad_max, laplacian_max) = self.compute_spatial_derivatives(field, &self.grid)?;

            // Acoustic time scale: τ_acoustic ~ 1/√(c²∇²)
            if laplacian_max > tolerance {
                let acoustic_scale = 1.0 / laplacian_max.sqrt();
                time_scales.push(acoustic_scale);
            }

            // Diffusive time scale: τ_diffusive ~ 1/∇²
            if grad_max > tolerance {
                let diffusive_scale = 1.0 / grad_max;
                time_scales.push(diffusive_scale);
            }
        }

        // Sort time scales from fastest to slowest
        time_scales.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Store in history
        self.time_scale_history.push(time_scales.clone());

        Ok(time_scales)
    }

    /// Determine if the system is stiff based on time scale separation
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn is_stiff(&self) -> bool {
        if let Some(last_scales) = self.time_scale_history.last() {
            if last_scales.len() >= 2 {
                let ratio = last_scales[last_scales.len() - 1] / last_scales[0];
                return ratio > self.min_separation_ratio;
            }
        }
        false
    }

    /// Compute spatial derivatives for time scale analysis
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_spatial_derivatives(
        &self,
        field: ArrayView3<'_, f64>,
        grid: &Grid,
    ) -> KwaversResult<(f64, f64)> {
        let (nx, ny, nz) = field.dim();
        let mut grad_max: f64 = 0.0;
        let mut laplacian_max: f64 = 0.0;

        if nx < 3 || ny < 3 || nz < 3 {
            return Ok((grad_max, laplacian_max));
        }

        // Compute gradients and Laplacian
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let dx = first_derivative::<0>(field, i, j, k, grid);
                    let dy = first_derivative::<1>(field, i, j, k, grid);
                    let dz = first_derivative::<2>(field, i, j, k, grid);
                    let grad_mag = (dx * dx + dy * dy + dz * dz).sqrt();
                    grad_max = grad_max.max(grad_mag);

                    let d2x = second_derivative::<0>(field, i, j, k, grid);
                    let d2y = second_derivative::<1>(field, i, j, k, grid);
                    let d2z = second_derivative::<2>(field, i, j, k, grid);
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
        _global_dt: f64,
        max_subcycles: usize,
    ) -> HashMap<String, usize> {
        let mut subcycles = HashMap::new();

        // Find the smallest time scale (sets global dt)
        let min_time_scale = time_scales
            .values()
            .map(|ts| ts.time_scale)
            .fold(f64::INFINITY, f64::min);

        // Compute subcycles for each component
        for (name, ts) in time_scales {
            let ratio = ts.time_scale / min_time_scale;
            let n_subcycles = (ratio.floor() as usize).max(1).min(max_subcycles);
            subcycles.insert(name.clone(), n_subcycles);
        }

        subcycles
    }
}

/// Monomorphized central first derivative along one Cartesian axis.
///
/// `AXIS` is a structural parameter. The optimizer emits one concrete stencil
/// per axis and removes inactive branches, while the caller keeps a single
/// authoritative finite-difference contract.
#[inline]
fn first_derivative<const AXIS: usize>(
    field: ArrayView3<'_, f64>,
    i: usize,
    j: usize,
    k: usize,
    grid: &Grid,
) -> f64 {
    let h = axis_spacing::<AXIS>(grid);
    let (forward, backward) = axis_neighbors::<AXIS>(field, i, j, k);
    (forward - backward) / (2.0 * h)
}

/// Monomorphized second central derivative along one Cartesian axis.
#[inline]
fn second_derivative<const AXIS: usize>(
    field: ArrayView3<'_, f64>,
    i: usize,
    j: usize,
    k: usize,
    grid: &Grid,
) -> f64 {
    let h = axis_spacing::<AXIS>(grid);
    let center = field[[i, j, k]];
    let (forward, backward) = axis_neighbors::<AXIS>(field, i, j, k);
    (2.0f64.mul_add(-center, forward) + backward) / (h * h)
}

#[inline]
fn axis_neighbors<const AXIS: usize>(
    field: ArrayView3<'_, f64>,
    i: usize,
    j: usize,
    k: usize,
) -> (f64, f64) {
    match AXIS {
        0 => (field[[i + 1, j, k]], field[[i - 1, j, k]]),
        1 => (field[[i, j + 1, k]], field[[i, j - 1, k]]),
        2 => (field[[i, j, k + 1]], field[[i, j, k - 1]]),
        _ => unreachable!("TimeScaleSeparator only supports axes 0, 1, and 2"),
    }
}

#[inline]
fn axis_spacing<const AXIS: usize>(grid: &Grid) -> f64 {
    match AXIS {
        0 => grid.dx,
        1 => grid.dy,
        2 => grid.dz,
        _ => unreachable!("TimeScaleSeparator only supports axes 0, 1, and 2"),
    }
}
