//! Inherent impl methods for ReverseTimeMigration: migration, propagation, imaging conditions.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::{s, Array2, Array3, Array4, Zip};

use super::super::config::RtmImagingCondition;
use super::super::constants::{
    DEFAULT_RICKER_FREQUENCY, DEFAULT_TIME_STEP, RTM_AMPLITUDE_THRESHOLD, RTM_LAPLACIAN_SCALING,
    RTM_STORAGE_DECIMATION,
};
use super::super::fd_coeffs::{FD_COEFF_0, FD_COEFF_1, FD_COEFF_2};
use super::super::wavelet::RickerWavelet;
use super::types::ReverseTimeMigration;

impl ReverseTimeMigration {
    /// Perform RTM for a single shot gather
    pub fn migrate_shot(
        &mut self,
        shot_data: &Array2<f64>,
        source_position: (usize, usize, usize),
        receiver_positions: &[(usize, usize, usize)],
        grid: &Grid,
    ) -> KwaversResult<()> {
        let n_time_steps = shot_data.shape()[1];

        // Step 1: Forward propagation with source wavefield storage
        let source_wavefield = self.forward_propagation(source_position, grid, n_time_steps)?;

        // Step 2: Backward propagation with receiver data
        let receiver_wavefield =
            self.backward_propagation(shot_data, receiver_positions, grid, n_time_steps)?;

        // Step 3: Apply imaging condition
        self.apply_imaging_condition(&source_wavefield, &receiver_wavefield)?;

        // Step 4: Update source illumination
        self.update_source_illumination(&source_wavefield)?;

        Ok(())
    }

    /// Forward propagation of source wavefield
    fn forward_propagation(
        &self,
        source_position: (usize, usize, usize),
        grid: &Grid,
        n_time_steps: usize,
    ) -> KwaversResult<Array4<f64>> {
        // Storage strategy: decimated wavefield storage
        let storage_size = n_time_steps.div_ceil(RTM_STORAGE_DECIMATION);
        let mut stored_wavefield = Array4::zeros((storage_size, grid.nx, grid.ny, grid.nz));

        // Initialize wavefields
        let mut pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut pressure_previous = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Create source wavelet
        let wavelet = RickerWavelet::new(DEFAULT_RICKER_FREQUENCY);
        let source_time_function = wavelet.generate_time_series(DEFAULT_TIME_STEP, n_time_steps);

        // Time stepping
        for (t, &source_value) in source_time_function.iter().enumerate() {
            // Apply source
            pressure[source_position] += source_value;

            // Update wavefield
            self.update_wavefield(&mut pressure, &pressure_previous, grid)?;

            // Store decimated wavefield
            if t % RTM_STORAGE_DECIMATION == 0 {
                let storage_idx = t / RTM_STORAGE_DECIMATION;
                stored_wavefield
                    .slice_mut(s![storage_idx, .., .., ..])
                    .assign(&pressure);
            }

            // Swap time levels
            std::mem::swap(&mut pressure, &mut pressure_previous);
        }

        // Reconstruct full wavefield if needed (using interpolation)
        if RTM_STORAGE_DECIMATION > 1 {
            self.reconstruct_full_wavefield(stored_wavefield, n_time_steps, grid)
        } else {
            Ok(stored_wavefield)
        }
    }

    /// Backward propagation of receiver wavefield
    fn backward_propagation(
        &self,
        shot_data: &Array2<f64>,
        receiver_positions: &[(usize, usize, usize)],
        grid: &Grid,
        n_time_steps: usize,
    ) -> KwaversResult<Array4<f64>> {
        let mut backward_wavefield = Array4::zeros((n_time_steps, grid.nx, grid.ny, grid.nz));

        // Initialize wavefields
        let mut pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut pressure_previous = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Time-reversed loop
        for t in (0..n_time_steps).rev() {
            // Apply receiver data as sources
            for (rec_idx, &rec_pos) in receiver_positions.iter().enumerate() {
                pressure[rec_pos] += shot_data[[rec_idx, t]];
            }

            // Update wavefield
            self.update_wavefield(&mut pressure, &pressure_previous, grid)?;

            // Store wavefield
            backward_wavefield
                .slice_mut(s![t, .., .., ..])
                .assign(&pressure);

            // Swap time levels
            std::mem::swap(&mut pressure, &mut pressure_previous);
        }

        Ok(backward_wavefield)
    }

    /// Update wavefield using finite differences
    fn update_wavefield(
        &self,
        pressure: &mut Array3<f64>,
        pressure_previous: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<()> {
        let dt = DEFAULT_TIME_STEP;
        let (nx, ny, nz) = pressure.dim();

        // Compute Laplacian
        let mut laplacian = Array3::zeros((nx, ny, nz));

        for i in 2..(nx - 2) {
            for j in 2..(ny - 2) {
                for k in 2..(nz - 2) {
                    // 4th-order finite differences
                    let d2p_dx2 = (FD_COEFF_2 * pressure[[i - 2, j, k]]
                        + FD_COEFF_1 * pressure[[i - 1, j, k]]
                        + FD_COEFF_0 * pressure[[i, j, k]]
                        + FD_COEFF_1 * pressure[[i + 1, j, k]]
                        + FD_COEFF_2 * pressure[[i + 2, j, k]])
                        / (grid.dx * grid.dx);

                    let d2p_dy2 = (FD_COEFF_2 * pressure[[i, j - 2, k]]
                        + FD_COEFF_1 * pressure[[i, j - 1, k]]
                        + FD_COEFF_0 * pressure[[i, j, k]]
                        + FD_COEFF_1 * pressure[[i, j + 1, k]]
                        + FD_COEFF_2 * pressure[[i, j + 2, k]])
                        / (grid.dy * grid.dy);

                    let d2p_dz2 = (FD_COEFF_2 * pressure[[i, j, k - 2]]
                        + FD_COEFF_1 * pressure[[i, j, k - 1]]
                        + FD_COEFF_0 * pressure[[i, j, k]]
                        + FD_COEFF_1 * pressure[[i, j, k + 1]]
                        + FD_COEFF_2 * pressure[[i, j, k + 2]])
                        / (grid.dz * grid.dz);

                    laplacian[[i, j, k]] = d2p_dx2 + d2p_dy2 + d2p_dz2;
                }
            }
        }

        // Update pressure
        Zip::from(pressure)
            .and(pressure_previous)
            .and(&laplacian)
            .and(&self.velocity_model)
            .for_each(|p, &p_prev, &lap, &vel| {
                let vel2_dt2 = vel * vel * dt * dt;
                *p = 2.0 * *p - p_prev + vel2_dt2 * lap;
            });

        Ok(())
    }

    /// Apply imaging condition to correlate wavefields
    fn apply_imaging_condition(
        &mut self,
        source_wavefield: &Array4<f64>,
        receiver_wavefield: &Array4<f64>,
    ) -> KwaversResult<()> {
        let n_time_steps = source_wavefield.shape()[0];

        match self.config.rtm_imaging_condition {
            RtmImagingCondition::ZeroLag => {
                // Zero-lag cross-correlation: I(x) = ∫ S(x,t) * R(x,t) dt
                for t in 0..n_time_steps {
                    let source_slice = source_wavefield.slice(s![t, .., .., ..]);
                    let receiver_slice = receiver_wavefield.slice(s![t, .., .., ..]);

                    Zip::from(&mut self.image)
                        .and(&source_slice)
                        .and(&receiver_slice)
                        .for_each(|img, &s, &r| {
                            if s.abs() > RTM_AMPLITUDE_THRESHOLD {
                                *img += s * r;
                            }
                        });
                }
            }

            RtmImagingCondition::Normalized => {
                // Normalized cross-correlation
                let mut source_energy = Array3::<f64>::zeros(self.image.dim());
                let mut receiver_energy = Array3::<f64>::zeros(self.image.dim());

                for t in 0..n_time_steps {
                    let source_slice = source_wavefield.slice(s![t, .., .., ..]);
                    let receiver_slice = receiver_wavefield.slice(s![t, .., .., ..]);

                    Zip::from(&mut self.image)
                        .and(&source_slice)
                        .and(&receiver_slice)
                        .for_each(|img, &s, &r| {
                            *img += s * r;
                        });

                    Zip::from(&mut source_energy)
                        .and(&source_slice)
                        .for_each(|e, &s| {
                            *e += s * s;
                        });

                    Zip::from(&mut receiver_energy)
                        .and(&receiver_slice)
                        .for_each(|e, &r| {
                            *e += r * r;
                        });
                }

                // Normalize
                Zip::from(&mut self.image)
                    .and(&source_energy)
                    .and(&receiver_energy)
                    .for_each(|img, &se, &re| {
                        let product: f64 = se * re;
                        let norm = product.sqrt();
                        if norm > RTM_AMPLITUDE_THRESHOLD {
                            *img /= norm;
                        }
                    });
            }

            RtmImagingCondition::Laplacian => {
                // Laplacian imaging condition: I(x) = ∫ ∇²S(x,t) * R(x,t) dt
                for t in 0..n_time_steps {
                    let source_slice = source_wavefield.slice(s![t, .., .., ..]).to_owned();
                    let receiver_slice = receiver_wavefield.slice(s![t, .., .., ..]);

                    // Compute Laplacian of source wavefield
                    let source_laplacian = self.compute_laplacian(&source_slice)?;

                    Zip::from(&mut self.image)
                        .and(&source_laplacian)
                        .and(&receiver_slice)
                        .for_each(|img, &lap, &r| {
                            *img += RTM_LAPLACIAN_SCALING * lap * r;
                        });
                }
            }

            RtmImagingCondition::EnergyNormalized => {
                // Energy-normalized condition: I(x) = ∫ S(x,t)R(x,t)dt / ∫ S(x,t)^2 dt.
                let mut source_energy = Array3::<f64>::zeros(self.image.dim());

                for t in 0..n_time_steps {
                    let source_slice = source_wavefield.slice(s![t, .., .., ..]);
                    let receiver_slice = receiver_wavefield.slice(s![t, .., .., ..]);

                    Zip::from(&mut self.image)
                        .and(&source_slice)
                        .and(&receiver_slice)
                        .for_each(|img, &source, &receiver| {
                            *img += source * receiver;
                        });

                    Zip::from(&mut source_energy)
                        .and(&source_slice)
                        .for_each(|energy, &source| {
                            *energy += source * source;
                        });
                }

                Zip::from(&mut self.image)
                    .and(&source_energy)
                    .for_each(|img, &energy| {
                        if energy > RTM_AMPLITUDE_THRESHOLD {
                            *img /= energy;
                        }
                    });
            }

            RtmImagingCondition::SourceNormalized => {
                // Source-normalized derivative condition: I(x) = ∫ ∂S/∂t R(x,t) dt.
                for t in 0..n_time_steps {
                    let receiver_slice = receiver_wavefield.slice(s![t, .., .., ..]);
                    let source_derivative = if t == 0 && n_time_steps > 1 {
                        &source_wavefield.slice(s![1, .., .., ..])
                            - &source_wavefield.slice(s![0, .., .., ..])
                    } else if t + 1 == n_time_steps && n_time_steps > 1 {
                        &source_wavefield.slice(s![t, .., .., ..])
                            - &source_wavefield.slice(s![t - 1, .., .., ..])
                    } else if n_time_steps > 1 {
                        0.5 * (&source_wavefield.slice(s![t + 1, .., .., ..])
                            - &source_wavefield.slice(s![t - 1, .., .., ..]))
                    } else {
                        Array3::<f64>::zeros(self.image.dim())
                    };

                    Zip::from(&mut self.image)
                        .and(&source_derivative)
                        .and(&receiver_slice)
                        .for_each(|img, &source_dt, &receiver| {
                            *img += source_dt * receiver;
                        });
                }
            }

            RtmImagingCondition::Poynting => {
                // Discrete Poynting condition: I(x) = ∫ ∇S(x,t) · ∇R(x,t) dt.
                let (_, nx, ny, nz) = source_wavefield.dim();
                for t in 0..n_time_steps {
                    let source = source_wavefield.slice(s![t, .., .., ..]);
                    let receiver = receiver_wavefield.slice(s![t, .., .., ..]);

                    for i in 1..(nx - 1) {
                        for j in 1..(ny - 1) {
                            for k in 1..(nz - 1) {
                                let source_dx =
                                    0.5 * (source[[i + 1, j, k]] - source[[i - 1, j, k]]);
                                let source_dy =
                                    0.5 * (source[[i, j + 1, k]] - source[[i, j - 1, k]]);
                                let source_dz =
                                    0.5 * (source[[i, j, k + 1]] - source[[i, j, k - 1]]);
                                let receiver_dx =
                                    0.5 * (receiver[[i + 1, j, k]] - receiver[[i - 1, j, k]]);
                                let receiver_dy =
                                    0.5 * (receiver[[i, j + 1, k]] - receiver[[i, j - 1, k]]);
                                let receiver_dz =
                                    0.5 * (receiver[[i, j, k + 1]] - receiver[[i, j, k - 1]]);

                                self.image[[i, j, k]] += source_dx * receiver_dx
                                    + source_dy * receiver_dy
                                    + source_dz * receiver_dz;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute Laplacian of a 3D field
    fn compute_laplacian(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut laplacian = Array3::zeros((nx, ny, nz));

        for i in 1..(nx - 1) {
            for j in 1..(ny - 1) {
                for k in 1..(nz - 1) {
                    laplacian[[i, j, k]] = field[[i + 1, j, k]]
                        + field[[i - 1, j, k]]
                        + field[[i, j + 1, k]]
                        + field[[i, j - 1, k]]
                        + field[[i, j, k + 1]]
                        + field[[i, j, k - 1]]
                        - 6.0 * field[[i, j, k]];
                }
            }
        }

        Ok(laplacian)
    }

    /// Update source illumination for normalization
    fn update_source_illumination(&mut self, source_wavefield: &Array4<f64>) -> KwaversResult<()> {
        let n_time_steps = source_wavefield.shape()[0];

        for t in 0..n_time_steps {
            let source_slice = source_wavefield.slice(s![t, .., .., ..]);

            Zip::from(&mut self.source_illumination)
                .and(&source_slice)
                .for_each(|illum, &s| {
                    *illum += s * s;
                });
        }

        Ok(())
    }

    /// Reconstruct full wavefield from decimated storage
    fn reconstruct_full_wavefield(
        &self,
        decimated: Array4<f64>,
        n_time_steps: usize,
        grid: &Grid,
    ) -> KwaversResult<Array4<f64>> {
        let mut full_wavefield = Array4::zeros((n_time_steps, grid.nx, grid.ny, grid.nz));

        // Linear interpolation between stored snapshots
        for t in 0..n_time_steps {
            let t_decimated = t / RTM_STORAGE_DECIMATION;
            let t_remainder = t % RTM_STORAGE_DECIMATION;

            if t_remainder == 0 {
                // Exact snapshot
                full_wavefield
                    .slice_mut(s![t, .., .., ..])
                    .assign(&decimated.slice(s![t_decimated, .., .., ..]));
            } else if t_decimated + 1 < decimated.shape()[0] {
                // Interpolate between snapshots
                let weight = t_remainder as f64 / RTM_STORAGE_DECIMATION as f64;
                let snapshot1 = decimated.slice(s![t_decimated, .., .., ..]);
                let snapshot2 = decimated.slice(s![t_decimated + 1, .., .., ..]);

                Zip::from(full_wavefield.slice_mut(s![t, .., .., ..]))
                    .and(&snapshot1)
                    .and(&snapshot2)
                    .for_each(|f, &s1, &s2| {
                        *f = (1.0 - weight) * s1 + weight * s2;
                    });
            }
        }

        Ok(full_wavefield)
    }

    /// Apply post-processing to the migrated image
    pub fn post_process_image(&mut self) -> KwaversResult<()> {
        // Apply source illumination normalization
        Zip::from(&mut self.image)
            .and(&self.source_illumination)
            .for_each(|img, &illum| {
                if illum > RTM_AMPLITUDE_THRESHOLD {
                    *img /= illum.sqrt();
                }
            });

        // Apply Laplacian filter for artifact removal (optional)
        if self.config.base_config.filter != crate::solver::reconstruction::FilterType::None {
            let filtered = self.apply_laplacian_filter(&self.image)?;
            self.image.assign(&filtered);
        }

        Ok(())
    }

    /// Apply Laplacian filter for artifact removal
    fn apply_laplacian_filter(&self, image: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let laplacian = self.compute_laplacian(image)?;
        Ok(image - &(0.1 * laplacian)) // Mild Laplacian filtering
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::inverse::reconstruction::seismic::config::SeismicImagingConfig;

    fn rtm_with_condition(condition: RtmImagingCondition) -> ReverseTimeMigration {
        let mut config = SeismicImagingConfig::default();
        config.rtm_imaging_condition = condition;
        ReverseTimeMigration::new(config, Array3::from_elem((3, 3, 3), 1500.0))
    }

    #[test]
    fn energy_normalized_condition_matches_cross_correlation_over_source_energy() {
        let mut rtm = rtm_with_condition(RtmImagingCondition::EnergyNormalized);
        let source = Array4::from_elem((2, 3, 3, 3), 2.0);
        let receiver = Array4::from_elem((2, 3, 3, 3), 3.0);

        rtm.apply_imaging_condition(&source, &receiver).unwrap();

        assert_eq!(rtm.image[[1, 1, 1]], 1.5);
        assert!(rtm.image.iter().all(|value| (*value - 1.5).abs() < 1e-12));
    }

    #[test]
    fn source_normalized_condition_uses_temporal_source_derivative() {
        let mut rtm = rtm_with_condition(RtmImagingCondition::SourceNormalized);
        let mut source = Array4::zeros((3, 3, 3, 3));
        for t in 0..3 {
            source.slice_mut(s![t, .., .., ..]).fill(t as f64);
        }
        let receiver = Array4::from_elem((3, 3, 3, 3), 3.0);

        rtm.apply_imaging_condition(&source, &receiver).unwrap();

        assert_eq!(rtm.image[[1, 1, 1]], 9.0);
        assert!(rtm.image.iter().all(|value| (*value - 9.0).abs() < 1e-12));
    }

    #[test]
    fn poynting_condition_accumulates_spatial_gradient_dot_product() {
        let mut rtm = rtm_with_condition(RtmImagingCondition::Poynting);
        let mut source = Array4::zeros((2, 3, 3, 3));
        let mut receiver = Array4::zeros((2, 3, 3, 3));
        for t in 0..2 {
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        let value = (i + j + k) as f64;
                        source[[t, i, j, k]] = value;
                        receiver[[t, i, j, k]] = 2.0 * value;
                    }
                }
            }
        }

        rtm.apply_imaging_condition(&source, &receiver).unwrap();

        assert_eq!(rtm.image[[1, 1, 1]], 12.0);
        assert_eq!(rtm.image[[0, 1, 1]], 0.0);
    }
}
