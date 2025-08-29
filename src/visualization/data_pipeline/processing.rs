//! Data processing stages for visualization

use super::ProcessingOperation;
use ndarray::{Array3, ArrayView3, ArrayViewMut3, Zip};

/// Processing configuration
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    pub normalize_range: (f32, f32),
    pub log_epsilon: f32,
    pub gaussian_sigma: f32,
    pub gradient_threshold: f32,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            normalize_range: (0.0, 1.0),
            log_epsilon: 1e-6,
            gaussian_sigma: 1.0,
            gradient_threshold: 0.1,
        }
    }
}

/// Processing stage for data transformation
pub struct ProcessingStage {
    config: ProcessingConfig,
}

impl ProcessingStage {
    /// Create new processing stage
    pub fn new(config: ProcessingConfig) -> Self {
        Self { config }
    }

    /// Apply processing operation to data
    pub fn apply(&self, operation: ProcessingOperation, data: &mut Array3<f64>) {
        match operation {
            ProcessingOperation::None => {}
            ProcessingOperation::Normalize => self.normalize(data),
            ProcessingOperation::LogScale => self.log_scale(data),
            ProcessingOperation::GradientMagnitude => self.gradient_magnitude(data),
            ProcessingOperation::GaussianSmooth => self.gaussian_smooth(data),
            ProcessingOperation::IsosurfaceExtraction => {
                // Isosurface extraction would modify data for marching cubes
            }
        }
    }

    /// Normalize data to specified range
    fn normalize(&self, data: &mut Array3<f64>) {
        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max - min).abs() < f64::EPSILON {
            return;
        }

        let range = max - min;
        let (target_min, target_max) = self.config.normalize_range;
        let target_range = (target_max - target_min) as f64;

        data.mapv_inplace(|v| target_min as f64 + (v - min) * target_range / range);
    }

    /// Apply logarithmic scaling
    fn log_scale(&self, data: &mut Array3<f64>) {
        let epsilon = self.config.log_epsilon as f64;
        data.mapv_inplace(|v| if v > epsilon { v.ln() } else { epsilon.ln() });
    }

    /// Compute gradient magnitude
    fn gradient_magnitude(&self, data: &mut Array3<f64>) {
        let shape = data.dim();
        let mut gradient = Array3::zeros(shape);

        // Compute central differences for gradient
        for i in 1..shape.0 - 1 {
            for j in 1..shape.1 - 1 {
                for k in 1..shape.2 - 1 {
                    let dx = (data[[i + 1, j, k]] - data[[i - 1, j, k]]) / 2.0;
                    let dy = (data[[i, j + 1, k]] - data[[i, j - 1, k]]) / 2.0;
                    let dz = (data[[i, j, k + 1]] - data[[i, j, k - 1]]) / 2.0;
                    gradient[[i, j, k]] = (dx * dx + dy * dy + dz * dz).sqrt();
                }
            }
        }

        *data = gradient;
    }

    /// Apply Gaussian smoothing
    fn gaussian_smooth(&self, data: &mut Array3<f64>) {
        // Simplified box filter as approximation
        let shape = data.dim();
        let mut smoothed = data.clone();

        for i in 1..shape.0 - 1 {
            for j in 1..shape.1 - 1 {
                for k in 1..shape.2 - 1 {
                    let mut sum = 0.0;
                    let mut count = 0;

                    for di in -1i32..=1 {
                        for dj in -1i32..=1 {
                            for dk in -1i32..=1 {
                                let ii = (i as i32 + di) as usize;
                                let jj = (j as i32 + dj) as usize;
                                let kk = (k as i32 + dk) as usize;
                                sum += data[[ii, jj, kk]];
                                count += 1;
                            }
                        }
                    }

                    smoothed[[i, j, k]] = sum / count as f64;
                }
            }
        }

        *data = smoothed;
    }
}
