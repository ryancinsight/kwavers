//! Bubble field definitions
//!
//! Defines the state fields for bubble dynamics.

use ndarray::Array3;

/// Bubble state fields for interfacing with physics modules
#[derive(Debug)]
pub struct BubbleStateFields {
    pub radius: Array3<f64>,
    pub temperature: Array3<f64>,
    pub pressure: Array3<f64>,
    pub velocity: Array3<f64>,
    pub is_collapsing: Array3<f64>,
    pub compression_ratio: Array3<f64>,
}

impl BubbleStateFields {
    #[must_use]
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            radius: Array3::zeros(shape),
            temperature: Array3::from_elem(shape, 293.15),
            pressure: Array3::from_elem(shape, 101325.0),
            velocity: Array3::zeros(shape),
            is_collapsing: Array3::zeros(shape),
            compression_ratio: Array3::from_elem(shape, 1.0),
        }
    }
}
