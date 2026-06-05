use kwavers_grid::GridDimensions;
use ndarray::Array3;

/// Reusable optical stage workspace.
#[derive(Debug, Clone)]
pub struct OpticalWorkspace {
    pub source: Array3<f64>,
    pub fluence: Array3<f64>,
}

impl OpticalWorkspace {
    #[must_use]
    pub fn new(dimensions: GridDimensions) -> Self {
        let shape = (dimensions.nx, dimensions.ny, dimensions.nz);
        Self {
            source: Array3::zeros(shape),
            fluence: Array3::zeros(shape),
        }
    }
}
