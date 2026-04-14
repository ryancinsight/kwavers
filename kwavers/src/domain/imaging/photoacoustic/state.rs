use ndarray::Array3;

/// Canonical mutable state bundle for the retained photoacoustic vertical.
#[derive(Debug, Clone, Default)]
pub struct PhotoacousticState {
    pub optical_fluence: Option<Array3<f64>>,
    pub initial_pressure: Option<Array3<f64>>,
    pub reconstruction: Option<Array3<f64>>,
}
