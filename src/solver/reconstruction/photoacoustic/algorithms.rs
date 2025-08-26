//! Core photoacoustic reconstruction algorithms
//!
//! This module implements the main reconstruction algorithms for
//! photoacoustic imaging.

use crate::error::KwaversResult;
use ndarray::{Array3, ArrayView2, ArrayView3};

use super::config::PhotoacousticConfig;
use super::filters::Filters;
use super::iterative::{IterativeAlgorithm, IterativeMethods};
use super::utils::Utils;

/// Photoacoustic reconstruction algorithms
#[derive(Debug, Clone)]
pub enum PhotoacousticAlgorithm {
    /// Universal back-projection
    UniversalBackProjection,
    /// Filtered back-projection with Hilbert transform
    FilteredBackProjection,
    /// Time reversal reconstruction
    TimeReversal,
    /// Fourier domain reconstruction
    FourierDomain,
    /// Iterative reconstruction (SIRT/ART)
    Iterative {
        algorithm: IterativeAlgorithm,
        iterations: usize,
        relaxation_factor: f64,
    },
    /// Model-based reconstruction
    ModelBased,
}

/// Main photoacoustic reconstructor
pub struct PhotoacousticReconstructor {
    pub(crate) config: PhotoacousticConfig,
    filters: Filters,
    iterative: IterativeMethods,
    utils: Utils,
}

impl PhotoacousticReconstructor {
    /// Create a new photoacoustic reconstructor
    pub fn new(config: PhotoacousticConfig) -> Self {
        Self {
            filters: Filters::new(&config),
            iterative: IterativeMethods::new(&config),
            utils: Utils::new(),
            config,
        }
    }

    /// Universal back-projection algorithm
    ///
    /// Reference: Xu & Wang (2005) Physical Review E 71, 016706
    pub fn universal_back_projection(
        &self,
        sensor_data: ArrayView2<f64>,
        sensor_positions: &[[f64; 3]],
        grid_size: [usize; 3],
        sound_speed: f64,
        sampling_frequency: f64,
    ) -> KwaversResult<Array3<f64>> {
        let (n_samples, n_sensors) = sensor_data.dim();
        let dt = 1.0 / sampling_frequency;

        // Initialize reconstruction volume
        let mut reconstruction = Array3::zeros((grid_size[0], grid_size[1], grid_size[2]));

        // Apply preprocessing if configured
        let processed_data = if let Some(bandpass) = self.config.bandpass_filter {
            self.filters.apply_bandpass_filter(
                &sensor_data.to_owned(),
                bandpass,
                sampling_frequency,
            )?
        } else {
            sensor_data.to_owned()
        };

        // Apply envelope detection if configured
        let processed_data = if self.config.envelope_detection {
            self.filters.apply_envelope_detection(&processed_data)?
        } else {
            processed_data
        };

        // Grid spacing (assume uniform)
        let dx = 1.0 / grid_size[0] as f64;
        let dy = 1.0 / grid_size[1] as f64;
        let dz = 1.0 / grid_size[2] as f64;

        // Perform back-projection
        for i in 0..grid_size[0] {
            for j in 0..grid_size[1] {
                for k in 0..grid_size[2] {
                    // Grid point position
                    let grid_pos = [i as f64 * dx, j as f64 * dy, k as f64 * dz];

                    // Sum contributions from all sensors
                    let mut value = 0.0;
                    for (sensor_idx, sensor_pos) in sensor_positions.iter().enumerate() {
                        // Calculate distance from grid point to sensor
                        let distance = self.utils.euclidean_distance(&grid_pos, sensor_pos);

                        // Calculate time of flight
                        let time_idx = (distance / sound_speed / dt) as usize;

                        if time_idx < n_samples {
                            // Get sensor data at this time
                            let sensor_value = processed_data[[time_idx, sensor_idx]];

                            // Apply spherical spreading compensation
                            let weight = self.utils.calculate_back_projection_weight(
                                distance,
                                sound_speed,
                                dt,
                            );

                            value += sensor_value * weight;
                        }
                    }

                    reconstruction[[i, j, k]] = value / n_sensors as f64;
                }
            }
        }

        // Apply reconstruction filter if configured
        if self.config.regularization_parameter > 0.0 {
            self.filters.apply_reconstruction_filter(&reconstruction)
        } else {
            Ok(reconstruction)
        }
    }

    /// Filtered back-projection algorithm
    pub fn filtered_back_projection(
        &self,
        sensor_data: ArrayView2<f64>,
        sensor_positions: &[[f64; 3]],
    ) -> KwaversResult<Array3<f64>> {
        // Apply FBP filter
        let filtered_data = self.filters.apply_fbp_filter(&sensor_data.to_owned())?;

        // Perform back-projection with filtered data
        self.universal_back_projection(
            filtered_data.view(),
            sensor_positions,
            self.config.grid_size,
            self.config.sound_speed,
            self.config.sampling_frequency,
        )
    }

    /// Time reversal reconstruction
    pub fn time_reversal_reconstruction(
        &self,
        sensor_data: ArrayView3<f64>,
        sound_speed: f64,
    ) -> KwaversResult<Array3<f64>> {
        // Time reversal reconstruction
        // This is a placeholder - actual implementation would involve
        // running the wave equation backward in time
        let mut reconstruction = sensor_data.to_owned();

        // Apply time reversal processing
        self.utils
            .propagate_time_reversed_signals(&mut reconstruction, sound_speed)?;

        Ok(reconstruction)
    }

    /// Iterative reconstruction using SIRT/ART/OSEM
    pub fn iterative_reconstruction(
        &self,
        sensor_data: ArrayView2<f64>,
        sensor_positions: &[[f64; 3]],
        grid_size: [usize; 3],
    ) -> KwaversResult<Array3<f64>> {
        self.iterative
            .reconstruct(sensor_data, sensor_positions, grid_size)
    }

    /// Model-based reconstruction
    pub fn model_based_reconstruction(
        &self,
        sensor_data: ArrayView2<f64>,
        sensor_positions: &[[f64; 3]],
    ) -> KwaversResult<Array3<f64>> {
        // Build forward model
        let forward_model = self.utils.build_forward_model(
            sensor_positions,
            self.config.grid_size,
            self.config.sound_speed,
            self.config.sampling_frequency,
        )?;

        // Solve regularized least squares problem
        self.utils.solve_regularized_least_squares(
            &forward_model,
            sensor_data,
            self.config.regularization_parameter,
        )
    }
}
