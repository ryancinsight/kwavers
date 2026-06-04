//! Forward and backward wavefield propagation for RTM.
//!
//! # Forward propagation
//! Drives a Ricker wavelet from `source_position` forward in time, storing
//! every `RTM_STORAGE_DECIMATION`-th snapshot.  Linear interpolation
//! reconstructs the full wavefield when `RTM_STORAGE_DECIMATION > 1`.
//!
//! # Backward propagation
//! Injects receiver data at `receiver_positions` as sources in time-reversed
//! order (t = T → 0), building the adjoint wavefield.
//!
//! # Wavefield update
//! Each step delegates to [`super::wavefield::update_wavefield`], which uses
//! parallel 4th-order finite-difference stencils.
//!
//! Reference: Claerbout (1985), *Imaging the Earth's Interior*, Ch. 3.

use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use ndarray::{s, Array3, Array4, Zip};

use super::super::super::constants::RTM_STORAGE_DECIMATION;
use super::super::super::wavelet::SeismicRickerWavelet;
use super::super::types::ReverseTimeMigration;

impl ReverseTimeMigration {
    /// Forward-propagate the source wavelet; return the (possibly
    /// decimated-then-reconstructed) `Array4<f64>` of shape
    /// `(n_time_steps, nx, ny, nz)`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn forward_propagation(
        &self,
        source_position: (usize, usize, usize),
        grid: &Grid,
        n_time_steps: usize,
    ) -> KwaversResult<Array4<f64>> {
        let storage_size = n_time_steps.div_ceil(RTM_STORAGE_DECIMATION);
        let mut stored = Array4::zeros((storage_size, grid.nx, grid.ny, grid.nz));

        let mut pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut pressure_prev = Array3::zeros((grid.nx, grid.ny, grid.nz));

        let wavelet = SeismicRickerWavelet::new(self.config.source_frequency_hz);
        let src_signal = wavelet.generate_time_series(self.config.dt, n_time_steps);

        for (t, &src_val) in src_signal.iter().enumerate() {
            pressure[source_position] += src_val;
            self.update_wavefield(&mut pressure, &pressure_prev, grid)?;

            if t % RTM_STORAGE_DECIMATION == 0 {
                stored
                    .slice_mut(s![t / RTM_STORAGE_DECIMATION, .., .., ..])
                    .assign(&pressure);
            }

            std::mem::swap(&mut pressure, &mut pressure_prev);
        }

        if RTM_STORAGE_DECIMATION > 1 {
            self.reconstruct_full_wavefield(stored, n_time_steps, grid)
        } else {
            Ok(stored)
        }
    }

    /// Backward-propagate receiver data (time-reversed injection).
    ///
    /// Returns `Array4<f64>` of shape `(n_time_steps, nx, ny, nz)`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn backward_propagation(
        &self,
        shot_data: &ndarray::Array2<f64>,
        receiver_positions: &[(usize, usize, usize)],
        grid: &Grid,
        n_time_steps: usize,
    ) -> KwaversResult<Array4<f64>> {
        let mut backward = Array4::zeros((n_time_steps, grid.nx, grid.ny, grid.nz));

        let mut pressure = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut pressure_prev = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for t in (0..n_time_steps).rev() {
            for (rec_idx, &rec_pos) in receiver_positions.iter().enumerate() {
                pressure[rec_pos] += shot_data[[rec_idx, t]];
            }

            self.update_wavefield(&mut pressure, &pressure_prev, grid)?;

            backward.slice_mut(s![t, .., .., ..]).assign(&pressure);

            std::mem::swap(&mut pressure, &mut pressure_prev);
        }

        Ok(backward)
    }

    /// Reconstruct a full `n_time_steps`-frame wavefield from a decimated
    /// snapshot array via linear interpolation.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn reconstruct_full_wavefield(
        &self,
        decimated: Array4<f64>,
        n_time_steps: usize,
        grid: &Grid,
    ) -> KwaversResult<Array4<f64>> {
        let mut full = Array4::zeros((n_time_steps, grid.nx, grid.ny, grid.nz));

        for t in 0..n_time_steps {
            let t_dec = t / RTM_STORAGE_DECIMATION;
            let t_rem = t % RTM_STORAGE_DECIMATION;

            if t_rem == 0 {
                full.slice_mut(s![t, .., .., ..])
                    .assign(&decimated.slice(s![t_dec, .., .., ..]));
            } else if t_dec + 1 < decimated.shape()[0] {
                let weight = t_rem as f64 / RTM_STORAGE_DECIMATION as f64;
                let snap1 = decimated.slice(s![t_dec, .., .., ..]);
                let snap2 = decimated.slice(s![t_dec + 1, .., .., ..]);

                Zip::from(full.slice_mut(s![t, .., .., ..]))
                    .and(&snap1)
                    .and(&snap2)
                    .par_for_each(|f, &s1, &s2| {
                        *f = (1.0 - weight).mul_add(s1, weight * s2);
                    });
            }
        }

        Ok(full)
    }
}
