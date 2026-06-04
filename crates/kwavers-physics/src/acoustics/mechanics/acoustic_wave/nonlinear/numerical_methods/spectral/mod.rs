use kwavers_core::constants::numerical;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_domain::medium::Medium;
use kwavers_math::fft::Complex64 as Complex;
use kwavers_math::fft::{fft_3d_array, ifft_3d_array};
use ndarray::{Array3, Zip};

use super::super::wave_model::NonlinearWave;

impl NonlinearWave {
    /// Applies the 2/3-rule anti-aliasing filter to a 3-D spectral field in-place.
    ///
    /// Physical-space products of band-limited fields (p·∇²p, |∇p|²) generate
    /// wavenumber content up to twice the input bandwidth. Zeroing the top 1/3 of
    /// bins along each axis before those products are formed prevents energy above
    /// the 2/3 Nyquist from aliasing back into the resolved band.
    ///
    /// ## Cutoff rule
    ///
    /// For axis length n, cutoff index `cx = n / 3` (integer division). Bins with
    /// absolute frequency index ≤ cx are retained. DFT layout: positive frequencies
    /// occupy [0, n/2], negative frequencies [n/2+1, n-1]. Zeroed range along each
    /// axis: (cx, n − cx), exclusive.
    ///
    /// ## Reference
    ///
    /// Canuto, Hussaini, Quarteroni & Zang (2006) *Spectral Methods in Fluid
    /// Dynamics*, §3.2.5; Kreiss & Oliger (1972).
    pub(crate) fn apply_dealiasing_filter(
        field_k: &mut Array3<Complex>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        let cx = nx / 3;
        let cy = ny / 3;
        let cz = nz / 3;
        Zip::indexed(field_k).for_each(|(i, j, k), val| {
            if (i > cx && i < nx - cx) || (j > cy && j < ny - cy) || (k > cz && k < nz - cz) {
                *val = Complex::new(0.0, 0.0);
            }
        });
    }
}

impl NonlinearWave {
    /// Applies k-space correction for the linear wave propagation.
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if `kx contiguous`.
    /// - Panics if `ky contiguous`.
    /// - Panics if `kz contiguous`.
    ///
    pub(crate) fn apply_k_space_correction(
        &self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let pressure_k = fft_3d_array(pressure);

        let kx = grid.compute_kx();
        let ky = grid.compute_ky();
        let kz = grid.compute_kz();

        let c_array = medium.sound_speed_array();
        let c = c_array.mean().unwrap_or(self.max_sound_speed);
        let mut result_k = Array3::<Complex>::zeros(pressure_k.raw_dim());
        let dt = self.dt;

        if let Some(ref k_squared) = self.k_squared {
            Zip::from(&mut result_k)
                .and(&pressure_k)
                .and(k_squared)
                .par_for_each(|r, &p, &k2| {
                    let k = k2.sqrt();
                    let sinc_factor = if k > numerical::EPSILON {
                        (c * k * dt / 2.0).sin() / (c * k * dt / 2.0)
                    } else {
                        1.0
                    };
                    *r = p * Complex::new(sinc_factor * (c * k * dt).cos(), 0.0);
                });
        } else {
            let kx_s = kx.as_slice().expect("kx contiguous");
            let ky_s = ky.as_slice().expect("ky contiguous");
            let kz_s = kz.as_slice().expect("kz contiguous");
            Zip::indexed(&mut result_k)
                .and(&pressure_k)
                .par_for_each(|(i, j, k), val, &pk| {
                    let k_mag_sq =
                        kz_s[k].mul_add(kz_s[k], kx_s[i].mul_add(kx_s[i], ky_s[j] * ky_s[j]));
                    let k_mag = k_mag_sq.sqrt();
                    let sinc_factor = if k_mag > numerical::EPSILON {
                        (c * k_mag * dt / 2.0).sin() / (c * k_mag * dt / 2.0)
                    } else {
                        1.0
                    };
                    *val = pk * Complex::new(sinc_factor * (c * k_mag * dt).cos(), 0.0);
                });
        }

        Ok(ifft_3d_array(&result_k))
    }
}

/// Spectral derivative utilities used only by tests.
///
/// `compute_spectral_gradient` and `compute_spectral_laplacian` verify the
/// underlying spectral differentiation formulas in isolation. Production code
/// uses the inlined, dealiased path in `compute_nonlinear_term`.
#[cfg(test)]
impl NonlinearWave {
    pub(crate) fn compute_spectral_gradient(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
        let field_k = fft_3d_array(field);

        let kx = grid.compute_kx();
        let ky = grid.compute_ky();
        let kz = grid.compute_kz();

        let mut grad_x_k = Array3::<Complex>::zeros(field_k.raw_dim());
        let mut grad_y_k = Array3::<Complex>::zeros(field_k.raw_dim());
        let mut grad_z_k = Array3::<Complex>::zeros(field_k.raw_dim());

        let kx_s = kx.as_slice().expect("kx contiguous");
        let ky_s = ky.as_slice().expect("ky contiguous");
        let kz_s = kz.as_slice().expect("kz contiguous");

        Zip::indexed(&mut grad_x_k)
            .and(&field_k)
            .par_for_each(|(i, _j, _k), val, &fk| {
                *val = fk * Complex::new(0.0, kx_s[i]);
            });

        Zip::indexed(&mut grad_y_k)
            .and(&field_k)
            .par_for_each(|(_i, j, _k), val, &fk| {
                *val = fk * Complex::new(0.0, ky_s[j]);
            });

        Zip::indexed(&mut grad_z_k)
            .and(&field_k)
            .par_for_each(|(_i, _j, k), val, &fk| {
                *val = fk * Complex::new(0.0, kz_s[k]);
            });

        Ok((
            ifft_3d_array(&grad_x_k),
            ifft_3d_array(&grad_y_k),
            ifft_3d_array(&grad_z_k),
        ))
    }

    /// Computes the spectral Laplacian of a field.
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if `kx contiguous`.
    /// - Panics if `ky contiguous`.
    /// - Panics if `kz contiguous`.
    ///
    pub(crate) fn compute_spectral_laplacian(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let field_k = fft_3d_array(field);

        let mut laplacian_k = Array3::<Complex>::zeros(field_k.raw_dim());

        if let Some(ref k_squared) = self.k_squared {
            Zip::from(&mut laplacian_k)
                .and(&field_k)
                .and(k_squared)
                .par_for_each(|l, &f, &k2| {
                    *l = f * (-k2);
                });
        } else {
            let kx = grid.compute_kx();
            let ky = grid.compute_ky();
            let kz = grid.compute_kz();
            let kx_s = kx.as_slice().expect("kx contiguous");
            let ky_s = ky.as_slice().expect("ky contiguous");
            let kz_s = kz.as_slice().expect("kz contiguous");

            Zip::indexed(&mut laplacian_k)
                .and(&field_k)
                .par_for_each(|(i, j, k), val, &fk| {
                    let k_mag_sq =
                        kz_s[k].mul_add(kz_s[k], kx_s[i].mul_add(kx_s[i], ky_s[j] * ky_s[j]));
                    *val = fk * (-k_mag_sq);
                });
        }

        Ok(ifft_3d_array(&laplacian_k))
    }
}

#[cfg(test)]
mod tests;
