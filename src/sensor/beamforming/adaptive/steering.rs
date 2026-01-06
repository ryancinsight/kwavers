// adaptive_beamforming/steering.rs - Steering vector computation

use ndarray::{Array1, Array2};
use num_complex::Complex64;

/// Steering vector for a given direction
#[derive(Debug)]
pub struct SteeringVector {
    pub direction: (f64, f64), // (azimuth, elevation)
    pub vector: Array1<Complex64>,
}

/// Collection of steering vectors
#[derive(Debug)]
pub struct SteeringMatrix {
    pub vectors: Array2<Complex64>,
    pub directions: Vec<(f64, f64)>,
}

impl SteeringMatrix {
    #[must_use]
    pub fn new(num_elements: usize, directions: Vec<(f64, f64)>) -> Self {
        let num_dirs = directions.len();
        let vectors = Array2::from_elem((num_elements, num_dirs), Complex64::new(1.0, 0.0));

        Self {
            vectors,
            directions,
        }
    }
}
