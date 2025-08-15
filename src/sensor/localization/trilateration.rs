//! Trilateration solver for 3D localization

use super::SensorArray;

/// Trilateration solver
pub struct TrilaterationSolver<'a> {
    array: &'a SensorArray,
}

impl<'a> TrilaterationSolver<'a> {
    /// Create a new trilateration solver
    pub fn new(array: &'a SensorArray) -> Self {
        Self { array }
    }
}

/// Trilateration result
#[derive(Debug, Clone)]
pub struct TrilaterationResult {
    /// Estimated position [x, y, z]
    pub position: [f64; 3],
    /// Uncertainty estimate
    pub uncertainty: f64,
}