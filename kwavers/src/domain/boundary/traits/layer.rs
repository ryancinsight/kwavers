//! `BoundaryLayer` and `BoundaryLayerManager` for multi-sided boundary geometry.

use crate::domain::grid::GridTopology;

use super::types::BoundaryDirections;

/// Geometry descriptor for a single absorbing boundary layer.
#[derive(Debug, Clone)]
pub struct BoundaryLayer {
    /// First grid index in the layer (inclusive).
    pub start: usize,
    /// Last grid index in the layer (exclusive).
    pub end: usize,
    /// Thickness in grid points (`end - start`).
    pub thickness: usize,
    /// Axis: 0 = x, 1 = y, 2 = z.
    pub direction: usize,
    /// `false` = min-side, `true` = max-side.
    pub is_max_side: bool,
}

impl BoundaryLayer {
    /// Construct a boundary layer.
    ///
    /// # Panics
    /// Panics if `end <= start`.
    pub fn new(start: usize, end: usize, direction: usize, is_max_side: bool) -> Self {
        assert!(end > start, "Invalid boundary layer: end <= start");
        Self {
            start,
            end,
            thickness: end - start,
            direction,
            is_max_side,
        }
    }

    /// Return `true` if `index` falls inside this layer.
    #[inline]
    pub fn contains(&self, index: usize) -> bool {
        index >= self.start && index < self.end
    }

    /// Return the normalized distance into the layer for `index`, in `[0, 1]`.
    ///
    /// 0 = at the outer (domain) edge (maximum absorption).
    /// 1 = at the inner (domain interior) edge (no absorption).
    #[inline]
    pub fn normalized_distance(&self, index: usize) -> f64 {
        if !self.contains(index) {
            return 0.0;
        }

        let pos = if self.is_max_side {
            (self.end - 1 - index) as f64
        } else {
            (index - self.start) as f64
        };

        (pos / (self.thickness - 1) as f64).min(1.0)
    }

    /// Compute a polynomial absorption profile at `index`.
    ///
    /// `σ(d) = σ_max · (1 − d)^order` where `d ∈ [0, 1]` is the normalized
    /// distance (1 = interior, 0 = edge).
    pub fn polynomial_profile(&self, index: usize, order: u32, sigma_max: f64) -> f64 {
        let d = self.normalized_distance(index);
        sigma_max * (1.0 - d).powi(order as i32)
    }
}

/// Manager for all boundary layers of a simulation domain.
#[derive(Debug, Clone)]
pub struct BoundaryLayerManager {
    layers: Vec<BoundaryLayer>,
}

impl BoundaryLayerManager {
    /// Create boundary layers from grid dimensions, layer thickness, and direction flags.
    pub fn new(grid: &dyn GridTopology, thickness: usize, directions: BoundaryDirections) -> Self {
        let dims = grid.dimensions();
        let mut layers = Vec::new();

        if directions.x_min && dims[0] > thickness {
            layers.push(BoundaryLayer::new(0, thickness, 0, false));
        }
        if directions.x_max && dims[0] > thickness {
            layers.push(BoundaryLayer::new(dims[0] - thickness, dims[0], 0, true));
        }
        if directions.y_min && dims[1] > thickness {
            layers.push(BoundaryLayer::new(0, thickness, 1, false));
        }
        if directions.y_max && dims[1] > thickness {
            layers.push(BoundaryLayer::new(dims[1] - thickness, dims[1], 1, true));
        }
        if directions.z_min && dims[2] > thickness {
            layers.push(BoundaryLayer::new(0, thickness, 2, false));
        }
        if directions.z_max && dims[2] > thickness {
            layers.push(BoundaryLayer::new(dims[2] - thickness, dims[2], 2, true));
        }

        Self { layers }
    }

    /// Iterate over layers that act on the given axis.
    pub fn layers_for_direction(&self, direction: usize) -> impl Iterator<Item = &BoundaryLayer> {
        self.layers
            .iter()
            .filter(move |layer| layer.direction == direction)
    }

    /// Return `true` if any layer covers any of the given `indices`.
    pub fn contains(&self, indices: [usize; 3]) -> bool {
        self.layers.iter().any(|layer| {
            layer.direction == 0 && layer.contains(indices[0])
                || layer.direction == 1 && layer.contains(indices[1])
                || layer.direction == 2 && layer.contains(indices[2])
        })
    }

    /// Compute the combined absorption at `indices` by summing all active layers.
    pub fn combined_absorption<F>(&self, indices: [usize; 3], profile_fn: F) -> f64
    where
        F: Fn(&BoundaryLayer, usize) -> f64,
    {
        let mut total = 0.0;
        for layer in &self.layers {
            let idx = indices[layer.direction];
            if layer.contains(idx) {
                total += profile_fn(layer, idx);
            }
        }
        total
    }
}
