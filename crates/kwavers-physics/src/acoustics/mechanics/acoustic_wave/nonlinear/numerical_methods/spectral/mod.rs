use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_math::fft::Complex64 as Complex;
use kwavers_math::fft::{fft_3d_array_into, ifft_3d_array_into};
use kwavers_medium::Medium;
use leto::Array3 as LetoArray3;
use leto::Array3;

use crate::parallel::for_each_indexed_mut;
use super::super::wave_model::NonlinearWave;

/// Spectral derivative utilities used only by tests.
///
/// `compute_spectral_gradient` and `compute_spectral_laplacian` verify the
/// underlying spectral differentiation formulas in isolation. Production code
/// uses the inlined, dealiased path in `compute_nonlinear_term`.
#[cfg(test)]
use kwavers_math::fft::{fft_3d_array, ifft_3d_array};

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
        field_k: &mut LetoArray3<Complex>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        let cx = nx / 3;
        let cy = ny / 3;
        let cz = nz / 3;
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if (i > cx && i < nx - cx) || (j > cy && j < ny - cy) || (k > cz && k < nz - cz)
                    {
                        field_k[[i, j, k]] = Complex::new(0.0, 0.0);
                    }
                }
            }
        }
    }
}

impl NonlinearWave {
    /// Applies k-space correction for the linear wave propagation.
    ///
    /// Precomputes FFT(pressure), applies the correction factor as a pointwise
    /// complex multiply (parallelised), and inverse-FFTs into a persistent output
    /// buffer — zero heap allocations per timestep.
    ///
    /// # Panics
    /// Panics if `k_space_correction`, `k_buf`, or `k_out` are not initialised
    /// (call `precompute_k_space_correction` first).
    ///
    pub(crate) fn apply_k_space_correction(
        &mut self,
        pressure: &Array3<f64>,
        _medium: &dyn Medium,
        _grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let correction = self
            .k_space_correction
            .as_ref()
            .expect("k_space_correction must be precomputed");
        let k_buf = self
            .k_buf
            .as_mut()
            .expect("k_buf must be initialised");
        let k_out = self
            .k_out
            .as_mut()
            .expect("k_out must be initialised");

        // FFT pressure into persistent buffer: zero-alloc.
        fft_3d_array_into(pressure, k_buf);

        // In-place pointwise complex multiply (parallelised).
        for_each_indexed_mut(k_buf.view_mut(), |(i, j, k), val| {
            *val = *val * correction[[i, j, k]];
        });

        // IFFT into persistent output buffer (uses k_buf as scratch): zero-alloc.
        ifft_3d_array_into(k_buf, k_out);

        Ok(k_out.clone())
    }
}

#[cfg(test)]
impl NonlinearWave {
    pub(crate) fn apply_k_space_correction_test(
        &self,
        pressure: &Array3<f64>,
        medium: &dyn Medium,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        use kwavers_core::constants::numerical;
        use kwavers_math::fft::{fft_3d_array, ifft_3d_array};

        let pressure_k = fft_3d_array(pressure);
        let [nx, ny, nz] = pressure_k.shape();
        let c = medium
            .sound_speed_array()
            .iter()
            .fold(0.0f64, |acc, &v| acc.max(v));
        let mut result_k = LetoArray3::<Complex>::zeros([nx, ny, nz]);
        let dt = self.dt;

        if let Some(ref k_squared) = self.k_squared {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let p = pressure_k[[i, j, k]];
                        let k_mag = k_squared[[i, j, k]].sqrt();
                        let sinc_factor = if k_mag > numerical::EPSILON {
                            (c * k_mag * dt / 2.0).sin() / (c * k_mag * dt / 2.0)
                        } else {
                            1.0
                        };
                        result_k[[i, j, k]] =
                            p * Complex::new(sinc_factor * (c * k_mag * dt).cos(), 0.0);
                    }
                }
            }
        } else {
            let kx = grid.compute_kx();
            let ky = grid.compute_ky();
            let kz = grid.compute_kz();
            let kx_s = kx.as_slice().expect("kx contiguous");
            let ky_s = ky.as_slice().expect("ky contiguous");
            let kz_s = kz.as_slice().expect("kz contiguous");
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let pk = pressure_k[[i, j, k]];
                        let k_mag_sq = kz_s[k].mul_add(
                            kz_s[k],
                            kx_s[i].mul_add(kx_s[i], ky_s[j] * ky_s[j]),
                        );
                        let k_mag = k_mag_sq.sqrt();
                        let sinc_factor = if k_mag > numerical::EPSILON {
                            (c * k_mag * dt / 2.0).sin() / (c * k_mag * dt / 2.0)
                        } else {
                            1.0
                        };
                        result_k[[i, j, k]] =
                            pk * Complex::new(sinc_factor * (c * k_mag * dt).cos(), 0.0);
                    }
                }
            }
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
        let [nx, ny, nz] = field_k.shape();

        let kx = grid.compute_kx();
        let ky = grid.compute_ky();
        let kz = grid.compute_kz();

        let mut grad_x_k = LetoArray3::<Complex>::zeros([nx, ny, nz]);
        let mut grad_y_k = LetoArray3::<Complex>::zeros([nx, ny, nz]);
        let mut grad_z_k = LetoArray3::<Complex>::zeros([nx, ny, nz]);

        let kx_s = kx.as_slice().expect("kx contiguous");
        let ky_s = ky.as_slice().expect("ky contiguous");
        let kz_s = kz.as_slice().expect("kz contiguous");

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let fk = field_k[[i, j, k]];
                    grad_x_k[[i, j, k]] = fk * Complex::new(0.0, kx_s[i]);
                    grad_y_k[[i, j, k]] = fk * Complex::new(0.0, ky_s[j]);
                    grad_z_k[[i, j, k]] = fk * Complex::new(0.0, kz_s[k]);
                }
            }
        }

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
        let [nx, ny, nz] = field_k.shape();

        let mut laplacian_k = LetoArray3::<Complex>::zeros([nx, ny, nz]);

        if let Some(ref k_squared) = self.k_squared {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        laplacian_k[[i, j, k]] = field_k[[i, j, k]] * (-k_squared[[i, j, k]]);
                    }
                }
            }
        } else {
            let kx = grid.compute_kx();
            let ky = grid.compute_ky();
            let kz = grid.compute_kz();
            let kx_s = kx.as_slice().expect("kx contiguous");
            let ky_s = ky.as_slice().expect("ky contiguous");
            let kz_s = kz.as_slice().expect("kz contiguous");

            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let k_mag_sq =
                            kz_s[k].mul_add(kz_s[k], kx_s[i].mul_add(kx_s[i], ky_s[j] * ky_s[j]));
                        laplacian_k[[i, j, k]] = field_k[[i, j, k]] * (-k_mag_sq);
                    }
                }
            }
        }

        Ok(ifft_3d_array(&laplacian_k))
    }
}

#[cfg(test)]
mod tests;
