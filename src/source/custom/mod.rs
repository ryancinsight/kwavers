//! Custom source types
//!
//! This module provides interfaces for creating custom source implementations
//! that can be defined by users for specialized applications.

use crate::grid::Grid;
use crate::signal::Signal;
use crate::source::{Source, SourceField};
use ndarray::Array3;
use std::fmt::Debug;
use std::sync::Arc;

/// Custom source builder trait
/// Allows users to create custom sources by implementing this trait
pub trait CustomSourceBuilder: Debug + Sync + Send {
    /// Build the custom source
    fn build(&self, signal: Arc<dyn Signal>) -> Box<dyn Source>;
}

/// Simple custom source implementation
/// Users can extend this for their specific needs
#[derive(Debug)]
pub struct SimpleCustomSource {
    positions: Vec<(f64, f64, f64)>,
    weights: Vec<f64>,
    signal: Arc<dyn Signal>,
    source_type: SourceField,
}

impl SimpleCustomSource {
    /// Create a new simple custom source
    pub fn new(
        positions: Vec<(f64, f64, f64)>,
        weights: Vec<f64>,
        signal: Arc<dyn Signal>,
        source_type: SourceField,
    ) -> Self {
        assert_eq!(
            positions.len(),
            weights.len(),
            "Positions and weights must have the same length"
        );
        Self {
            positions,
            weights,
            signal,
            source_type,
        }
    }

    /// Create a custom source builder
    pub fn builder() -> SimpleCustomSourceBuilder {
        SimpleCustomSourceBuilder::new()
    }
}

impl Source for SimpleCustomSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for (pos_idx, (x, y, z)) in self.positions.iter().enumerate() {
            if let Some((i, j, k)) = grid.position_to_indices(*x, *y, *z) {
                mask[(i, j, k)] = self.weights[pos_idx];
            }
        }

        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        self.positions.clone()
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }

    fn source_type(&self) -> SourceField {
        self.source_type
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Find the closest position and use its weight
        let mut closest_weight = 0.0;
        let mut min_distance = f64::INFINITY;

        for (pos_idx, (px, py, pz)) in self.positions.iter().enumerate() {
            let distance = ((x - px).powi(2) + (y - py).powi(2) + (z - pz).powi(2)).sqrt();
            if distance < min_distance {
                min_distance = distance;
                closest_weight = self.weights[pos_idx];
            }
        }

        closest_weight * self.signal.amplitude(t)
    }
}

/// Builder for simple custom sources
#[derive(Debug, Default)]
pub struct SimpleCustomSourceBuilder {
    positions: Vec<(f64, f64, f64)>,
    weights: Vec<f64>,
    source_type: SourceField,
}

impl SimpleCustomSourceBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_position(mut self, position: (f64, f64, f64), weight: f64) -> Self {
        self.positions.push(position);
        self.weights.push(weight);
        self
    }

    pub fn source_type(mut self, source_type: SourceField) -> Self {
        self.source_type = source_type;
        self
    }

    pub fn build(self, signal: Arc<dyn Signal>) -> SimpleCustomSource {
        SimpleCustomSource::new(self.positions, self.weights, signal, self.source_type)
    }
}

impl CustomSourceBuilder for SimpleCustomSourceBuilder {
    fn build(&self, signal: Arc<dyn Signal>) -> Box<dyn Source> {
        Box::new(SimpleCustomSource::new(
            self.positions.clone(),
            self.weights.clone(),
            signal,
            self.source_type,
        ))
    }
}

/// Function-based custom source
/// Allows users to define sources using arbitrary functions
pub struct FunctionSource {
    function: Box<dyn Fn(f64, f64, f64, f64) -> f64 + Sync + Send>,
    signal: Arc<dyn Signal>,
    source_type: SourceField,
}

impl std::fmt::Debug for FunctionSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionSource")
            .field("source_type", &self.source_type)
            .field("signal", &"<function>")
            .field("function", &"<closure>")
            .finish()
    }
}

impl FunctionSource {
    /// Create a new function-based source
    pub fn new<F: Fn(f64, f64, f64, f64) -> f64 + Sync + Send + 'static>(
        function: F,
        signal: Arc<dyn Signal>,
        source_type: SourceField,
    ) -> Self {
        Self {
            function: Box::new(function),
            signal,
            source_type,
        }
    }
}

impl Source for FunctionSource {
    fn create_mask(&self, grid: &Grid) -> Array3<f64> {
        let mut mask = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for ((i, j, k), val) in mask.indexed_iter_mut() {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let z = k as f64 * grid.dz;

            // Evaluate the function at time 0 to get the spatial distribution
            *val = (self.function)(x, y, z, 0.0);
        }

        mask
    }

    fn amplitude(&self, t: f64) -> f64 {
        self.signal.amplitude(t)
    }

    fn positions(&self) -> Vec<(f64, f64, f64)> {
        // For function sources, we can't easily determine positions
        // Return empty vector
        vec![]
    }

    fn signal(&self) -> &dyn Signal {
        self.signal.as_ref()
    }

    fn source_type(&self) -> SourceField {
        self.source_type
    }

    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, _grid: &Grid) -> f64 {
        (self.function)(x, y, z, t) * self.signal.amplitude(t)
    }
}
