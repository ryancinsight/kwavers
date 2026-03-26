use crate::core::constants::numerical;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::math::fft::{fft_3d_array, ifft_3d_array};
use ndarray::{Array3, Zip};
use rustfft::num_complex::Complex;

use super::super::wave_model::NonlinearWave;

impl NonlinearWave {
    /// Applies k-space correction for the linear wave propagation.
    ///
    /// This implements the k-space pseudo-spectral method for accurate
    /// wave propagation without numerical dispersion.
    ///
    /// # Arguments
    ///
    /// * `pressure` - Current pressure field
    /// * `medium` - Medium properties
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// The k-space corrected pressure field
    pub(crate) fn apply_k_space_correction(
        &self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // Transform to k-space
        let pressure_k = fft_3d_array(pressure);

        // Get k-space grid
        let kx = grid.compute_kx();
        let ky = grid.compute_ky();
        let kz = grid.compute_kz();

        // Get spatially-varying sound speed
        let c_array = medium.sound_speed_array();
        let c = c_array.mean().unwrap_or(self.max_sound_speed);
        let mut result_k = Array3::<Complex<f64>>::zeros(pressure_k.raw_dim());

        // Use pre-computed k_squared if available
        if let Some(ref k_squared) = self.k_squared {
            Zip::from(&mut result_k)
                .and(&pressure_k)
                .and(k_squared)
                .for_each(|r, &p, &k2| {
                    let k = k2.sqrt();
                    let sinc_factor = if k > numerical::EPSILON {
                        (c * k * self.dt / 2.0).sin() / (c * k * self.dt / 2.0)
                    } else {
                        1.0
                    };
                    *r = p * Complex::new(sinc_factor * (-c.powi(2) * k2 * self.dt.powi(2)), 0.0)
                        .exp();
                });
        } else {
            // Compute k-squared on the fly
            result_k.indexed_iter_mut().for_each(|((i, j, k), val)| {
                let k_mag_sq = kx[i].powi(2) + ky[j].powi(2) + kz[k].powi(2);
                let k_mag = k_mag_sq.sqrt();

                let sinc_factor = if k_mag > numerical::EPSILON {
                    (c * k_mag * self.dt / 2.0).sin() / (c * k_mag * self.dt / 2.0)
                } else {
                    1.0
                };

                *val = pressure_k[(i, j, k)]
                    * Complex::new(sinc_factor * (-c.powi(2) * k_mag_sq * self.dt.powi(2)), 0.0)
                        .exp();
            });
        }

        // Transform back to spatial domain
        Ok(ifft_3d_array(&result_k))
    }

    /// Computes the spectral gradient of a field.
    ///
    /// # Arguments
    ///
    /// * `field` - Input field
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// Tuple of (`grad_x`, `grad_y`, `grad_z`)
    pub(crate) fn compute_spectral_gradient(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
        // Transform to k-space
        let field_k = fft_3d_array(field);

        // Get k-space grid
        let kx = grid.compute_kx();
        let ky = grid.compute_ky();
        let kz = grid.compute_kz();

        // Compute gradients in k-space
        let mut grad_x_k = Array3::<Complex<f64>>::zeros(field_k.raw_dim());
        let mut grad_y_k = Array3::<Complex<f64>>::zeros(field_k.raw_dim());
        let mut grad_z_k = Array3::<Complex<f64>>::zeros(field_k.raw_dim());

        grad_x_k.indexed_iter_mut().for_each(|((i, j, k), val)| {
            *val = field_k[(i, j, k)] * Complex::new(0.0, kx[i]);
        });

        grad_y_k.indexed_iter_mut().for_each(|((i, j, k), val)| {
            *val = field_k[(i, j, k)] * Complex::new(0.0, ky[j]);
        });

        grad_z_k.indexed_iter_mut().for_each(|((i, j, k), val)| {
            *val = field_k[(i, j, k)] * Complex::new(0.0, kz[k]);
        });

        // Transform back to spatial domain
        Ok((
            ifft_3d_array(&grad_x_k),
            ifft_3d_array(&grad_y_k),
            ifft_3d_array(&grad_z_k),
        ))
    }

    /// Computes the spectral Laplacian of a field.
    ///
    /// # Arguments
    ///
    /// * `field` - Input field
    /// * `grid` - Computational grid
    ///
    /// # Returns
    ///
    /// The Laplacian of the field
    pub(crate) fn compute_spectral_laplacian(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // Transform to k-space
        let field_k = fft_3d_array(field);

        // Apply Laplacian operator in k-space
        let mut laplacian_k = Array3::<Complex<f64>>::zeros(field_k.raw_dim());

        if let Some(ref k_squared) = self.k_squared {
            // Use pre-computed k-squared
            Zip::from(&mut laplacian_k)
                .and(&field_k)
                .and(k_squared)
                .for_each(|l, &f, &k2| {
                    *l = f * Complex::new(-k2, 0.0);
                });
        } else {
            // Compute k-squared on the fly
            let kx = grid.compute_kx();
            let ky = grid.compute_ky();
            let kz = grid.compute_kz();

            laplacian_k.indexed_iter_mut().for_each(|((i, j, k), val)| {
                let k_mag_sq = kx[i].powi(2) + ky[j].powi(2) + kz[k].powi(2);
                *val = field_k[(i, j, k)] * Complex::new(-k_mag_sq, 0.0);
            });
        }

        // Transform back to spatial domain
        Ok(ifft_3d_array(&laplacian_k))
    }
}
