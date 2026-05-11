//! `FrequencyOperator`: frequency-domain absorption and diffraction operators for KZK.

use ndarray::Array3;

/// Frequency domain operator for the KZK equation.
#[derive(Debug, Clone)]
pub struct FrequencyOperator {
    /// Frequency grid points.
    pub frequencies: Vec<f64>,
    /// Absorption operator in frequency domain.
    pub absorption_operator: Array3<f64>,
    /// Diffraction operator in frequency domain.
    pub diffraction_operator: Array3<f64>,
}
