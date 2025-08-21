//! Buffer zone management for domain interfaces

use serde::{Deserialize, Serialize};

/// Buffer zones for smooth transitions between domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferZones {
    /// Width of buffer zone in grid points
    pub width: usize,
    /// Blending function type
    pub blend_type: BlendType,
}

/// Type of blending function for buffer zones
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BlendType {
    /// Linear blending
    Linear,
    /// Cosine blending for smoother transition
    Cosine,
    /// Exponential blending
    Exponential,
}

impl Default for BufferZones {
    fn default() -> Self {
        Self {
            width: 4,
            blend_type: BlendType::Cosine,
        }
    }
}

/// Overlap region between adjacent domains
#[derive(Debug, Clone)]
pub struct OverlapRegion {
    /// First domain index
    pub domain1: usize,
    /// Second domain index
    pub domain2: usize,
    /// Overlap region bounds
    pub bounds: ((usize, usize, usize), (usize, usize, usize)),
    /// Blending weights for domain1
    pub weights1: Vec<f64>,
    /// Blending weights for domain2
    pub weights2: Vec<f64>,
}

impl OverlapRegion {
    /// Create a new overlap region
    pub fn new(
        domain1: usize,
        domain2: usize,
        bounds: ((usize, usize, usize), (usize, usize, usize)),
    ) -> Self {
        let size =
            (bounds.1 .0 - bounds.0 .0) * (bounds.1 .1 - bounds.0 .1) * (bounds.1 .2 - bounds.0 .2);

        Self {
            domain1,
            domain2,
            bounds,
            weights1: vec![0.5; size],
            weights2: vec![0.5; size],
        }
    }

    /// Compute blending weights based on position
    pub fn compute_weights(&mut self, buffer: &BufferZones) {
        let (start, end) = self.bounds;
        let mut idx = 0;

        for i in start.0..end.0 {
            for j in start.1..end.1 {
                for k in start.2..end.2 {
                    let weight =
                        self.compute_blend_weight(i - start.0, end.0 - start.0, buffer.blend_type);

                    self.weights1[idx] = weight;
                    self.weights2[idx] = 1.0 - weight;
                    idx += 1;
                }
            }
        }
    }

    /// Compute blend weight for a position
    fn compute_blend_weight(&self, pos: usize, width: usize, blend_type: BlendType) -> f64 {
        let x = pos as f64 / width as f64;

        match blend_type {
            BlendType::Linear => x,
            BlendType::Cosine => 0.5 * (1.0 - (x * std::f64::consts::PI).cos()),
            BlendType::Exponential => 1.0 - (-5.0 * x).exp(),
        }
    }
}
