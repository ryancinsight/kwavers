use ndarray::Array3;

/// Reusable CPU acoustic propagation workspace.
#[derive(Debug, Clone)]
pub struct AcousticWorkspace {
    pub previous: Array3<f64>,
    pub current: Array3<f64>,
    pub next: Array3<f64>,
}

impl AcousticWorkspace {
    #[must_use]
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            previous: Array3::zeros(shape),
            current: Array3::zeros(shape),
            next: Array3::zeros(shape),
        }
    }
}
