//! Transducer geometry definition

use ndarray::Array2;

/// Transducer geometry definition
#[derive(Debug, Clone)]
pub struct TransducerGeometry {
    /// Element positions [x, y, z] in meters
    pub element_positions: Array2<f64>,
    /// Element sizes [width, height] in meters
    pub element_sizes: Array2<f64>,
    /// Element orientations (normal vectors)
    pub element_normals: Array2<f64>,
    /// Element apodization weights
    pub apodization: Option<Vec<f64>>,
    /// Element delays in seconds
    pub delays: Option<Vec<f64>>,
}
