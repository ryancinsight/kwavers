use kwavers_grid::GridDimensions;
use ndarray::Array3;

/// Reusable source-generation workspace.
#[derive(Debug, Clone)]
pub struct SourceWorkspace {
    pub pressure: Array3<f64>,
}

impl SourceWorkspace {
    #[must_use]
    pub fn new(dimensions: GridDimensions) -> Self {
        Self {
            pressure: Array3::zeros((dimensions.nx, dimensions.ny, dimensions.nz)),
        }
    }
}
