use ndarray::Array3;

/// Container for common initialization material properties
#[derive(Debug, Clone)]
pub struct MaterialFields {
    /// Ambient density map (rho0)
    pub rho0: Array3<f64>,
    /// Ambient sound speed map (c0)
    pub c0: Array3<f64>,
}

impl MaterialFields {
    /// Create new zero-initialized material fields with given shape
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            rho0: Array3::zeros(shape),
            c0: Array3::zeros(shape),
        }
    }
}
