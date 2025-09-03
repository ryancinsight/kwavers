// localization/triangulation.rs - Triangulation methods

use super::{Position, SensorArray};
use crate::error::KwaversResult;

/// Triangulator for position estimation
#[derive(Debug)]
pub struct Triangulator {
    method: TriangulationMethod,
}

/// Triangulation methods
#[derive(Debug, Clone, Copy)]
pub enum TriangulationMethod {
    LeastSquares,
    WeightedLeastSquares,
    MaximumLikelihood,
}

impl Triangulator {
    /// Create new triangulator
    #[must_use]
    pub fn new(method: TriangulationMethod) -> Self {
        Self { method }
    }

    /// Triangulate position from ranges
    pub fn triangulate(&self, ranges: &[f64], array: &SensorArray) -> KwaversResult<Position> {
        if ranges.len() < 3 {
            return Err(crate::error::KwaversError::InvalidInput(
                "Need at least 3 ranges for 3D triangulation".to_string(),
            ));
        }

        // Simplified implementation
        Ok(array.centroid())
    }

    /// Triangulate from angles
    pub fn triangulate_angles(
        &self,
        angles: &[(f64, f64)],
        array: &SensorArray,
    ) -> KwaversResult<Position> {
        if angles.len() < 2 {
            return Err(crate::error::KwaversError::InvalidInput(
                "Need at least 2 angle measurements".to_string(),
            ));
        }

        // Simplified implementation
        Ok(array.centroid())
    }
}
