use ndarray::{Array2, Array3};

/// Workspace for retained photoacoustic reconstruction stages.
#[derive(Debug, Clone)]
pub struct ReconstructionWorkspace {
    pub sensor_data: Array2<f64>,
    pub image: Array3<f64>,
}

impl ReconstructionWorkspace {
    #[must_use]
    pub fn new(
        num_time_samples: usize,
        num_sensors: usize,
        image_shape: (usize, usize, usize),
    ) -> Self {
        Self {
            sensor_data: Array2::zeros((num_time_samples, num_sensors)),
            image: Array3::zeros(image_shape),
        }
    }
}
