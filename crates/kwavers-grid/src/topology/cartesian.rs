use super::{GridTopology, TopologyDimension};
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use std::f64::consts::PI;

/// Cartesian grid topology (standard rectilinear grid)
///
/// Maps indices directly to (x, y, z) coordinates with uniform spacing.
#[derive(Debug, Clone)]
pub struct CartesianTopology {
    /// Dimensions [nx, ny, nz]
    pub dimensions: [usize; 3],
    /// Spacing [dx, dy, dz] in meters
    pub spacing: [f64; 3],
    /// Origin offset [x0, y0, z0]
    pub origin: [f64; 3],
}

impl CartesianTopology {
    /// Create a new Cartesian topology
    /// # Errors
    /// - Returns [`KwaversError::Config`] if the precondition for a Config-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(dimensions: [usize; 3], spacing: [f64; 3], origin: [f64; 3]) -> KwaversResult<Self> {
        if dimensions[0] == 0 || dimensions[1] == 0 || dimensions[2] == 0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "grid dimensions".to_owned(),
                value: format!("{:?}", dimensions),
                constraint: "All dimensions must be positive".to_owned(),
            }));
        }

        if spacing[0] <= 0.0 || spacing[1] <= 0.0 || spacing[2] <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "grid spacing".to_owned(),
                value: format!("{:?}", spacing),
                constraint: "All spacing values must be positive".to_owned(),
            }));
        }

        if !spacing[0].is_finite() || !spacing[1].is_finite() || !spacing[2].is_finite() {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "grid spacing".to_owned(),
                value: format!("{:?}", spacing),
                constraint: "All spacing values must be finite".to_owned(),
            }));
        }

        Ok(Self {
            dimensions,
            spacing,
            origin,
        })
    }

    /// Create a uniform Cartesian grid (same spacing in all directions)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn uniform(n: usize, spacing: f64, origin: [f64; 3]) -> KwaversResult<Self> {
        Self::new([n, n, n], [spacing, spacing, spacing], origin)
    }
}

impl GridTopology for CartesianTopology {
    fn dimensionality(&self) -> TopologyDimension {
        let active = (self.dimensions[0] > 1) as usize
            + (self.dimensions[1] > 1) as usize
            + (self.dimensions[2] > 1) as usize;

        match active {
            0 | 1 => TopologyDimension::One,
            2 => TopologyDimension::Two,
            _ => TopologyDimension::Three,
        }
    }

    fn size(&self) -> usize {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    }

    fn dimensions(&self) -> [usize; 3] {
        self.dimensions
    }

    fn spacing(&self) -> [f64; 3] {
        self.spacing
    }

    fn extents(&self) -> [f64; 3] {
        [
            (self.dimensions[0] - 1) as f64 * self.spacing[0],
            (self.dimensions[1] - 1) as f64 * self.spacing[1],
            (self.dimensions[2] - 1) as f64 * self.spacing[2],
        ]
    }

    fn indices_to_coordinates(&self, indices: [usize; 3]) -> [f64; 3] {
        [
            (indices[0] as f64).mul_add(self.spacing[0], self.origin[0]),
            (indices[1] as f64).mul_add(self.spacing[1], self.origin[1]),
            (indices[2] as f64).mul_add(self.spacing[2], self.origin[2]),
        ]
    }

    fn coordinates_to_indices(&self, coords: [f64; 3]) -> Option<[usize; 3]> {
        let extents = self.extents();

        for i in 0..3 {
            let rel = coords[i] - self.origin[i];
            if rel < 0.0 || rel > extents[i] {
                return None;
            }
        }

        let i = ((coords[0] - self.origin[0]) / self.spacing[0])
            .floor()
            .min((self.dimensions[0] - 1) as f64) as usize;
        let j = ((coords[1] - self.origin[1]) / self.spacing[1])
            .floor()
            .min((self.dimensions[1] - 1) as f64) as usize;
        let k = ((coords[2] - self.origin[2]) / self.spacing[2])
            .floor()
            .min((self.dimensions[2] - 1) as f64) as usize;

        Some([i, j, k])
    }

    fn metric_coefficient(&self, _indices: [usize; 3]) -> f64 {
        self.spacing[0] * self.spacing[1] * self.spacing[2]
    }

    fn is_uniform(&self) -> bool {
        use crate::structure::GRID_SPACING_EQUALITY_EPSILON as EPSILON;
        (self.spacing[0] - self.spacing[1]).abs() < EPSILON
            && (self.spacing[1] - self.spacing[2]).abs() < EPSILON
    }

    fn k_max(&self) -> f64 {
        let min_spacing = self.spacing[0].min(self.spacing[1]).min(self.spacing[2]);
        PI / min_spacing
    }
}
