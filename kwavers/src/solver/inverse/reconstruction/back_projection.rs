//! Universal back-projection reconstruction

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{Array2, Array3};

use super::config::{ReconstructionConfig, Reconstructor};
use super::filters::apply_reconstruction_filter;
use crate::core::constants::numerical::FOUR_PI;

/// Universal back-projection for arbitrary geometries
/// Based on Xu & Wang (2005)
#[derive(Debug)]
pub struct UniversalBackProjection {
    /// Weight function for back-projection
    weight_function: WeightFunction,
}

#[derive(Debug, Clone)]
pub enum WeightFunction {
    /// Uniform weighting
    Uniform,
    /// Solid angle weighting
    SolidAngle,
    /// Distance-based weighting
    Distance { power: f64 },
}

impl UniversalBackProjection {
    #[must_use]
    pub fn new(weight_function: WeightFunction) -> Self {
        Self { weight_function }
    }

    /// Compute back-projection for a single voxel
    fn compute_voxel_value(
        &self,
        voxel_pos: [f64; 3],
        sensor_data: &Array2<f64>,
        sensor_positions: &[[f64; 3]],
        config: &ReconstructionConfig,
    ) -> f64 {
        let mut value = 0.0;
        let dt = 1.0 / config.sampling_frequency;

        for (sensor_idx, sensor_pos) in sensor_positions.iter().enumerate() {
            // Calculate distance from voxel to sensor
            let dx = voxel_pos[0] - sensor_pos[0];
            let dy = voxel_pos[1] - sensor_pos[1];
            let dz = voxel_pos[2] - sensor_pos[2];
            let distance = dz.mul_add(dz, dx.mul_add(dx, dy * dy)).sqrt();

            // Calculate time delay
            let time_delay = distance / config.sound_speed;
            let sample_idx = (time_delay / dt).round() as usize;

            if sample_idx < sensor_data.dim().1 {
                let sensor_value = sensor_data[[sensor_idx, sample_idx]];

                // Apply weighting
                let weight = match self.weight_function {
                    WeightFunction::Uniform => 1.0,
                    WeightFunction::SolidAngle => 1.0 / (FOUR_PI * distance * distance),
                    WeightFunction::Distance { power } => 1.0 / distance.powf(power),
                };

                value += sensor_value * weight;
            }
        }

        value / sensor_positions.len() as f64
    }
}

impl Reconstructor for UniversalBackProjection {
    fn reconstruct(
        &self,
        sensor_data: &Array2<f64>,
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
        config: &ReconstructionConfig,
    ) -> KwaversResult<Array3<f64>> {
        let mut reconstructed = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Apply filter if specified
        let filtered_data =
            apply_reconstruction_filter(sensor_data, &config.filter, config.sampling_frequency)?;

        // Perform back-projection
        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    reconstructed[[i, j, k]] = self.compute_voxel_value(
                        [x, y, z],
                        &filtered_data,
                        sensor_positions,
                        config,
                    );
                }
            }
        }

        Ok(reconstructed)
    }

    fn name(&self) -> &str {
        "Universal Back-Projection"
    }
}
