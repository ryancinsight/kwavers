use crate::core::constants::numerical;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::math::fft::Complex64 as Complex;
use crate::math::fft::{fft_3d_array, ifft_3d_array};
use ndarray::{Array3, Zip};

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
        // Transform to k-space
        let pressure_k = fft_3d_array(pressure);

        // Get k-space grid
        let kx = grid.compute_kx();
        let ky = grid.compute_ky();
        let kz = grid.compute_kz();

        // Get spatially-varying sound speed
        let c_array = medium.sound_speed_array();
        let c = c_array.mean().unwrap_or(self.max_sound_speed);
        let mut result_k = Array3::<Complex>::zeros(pressure_k.raw_dim());
        let dt = self.dt;

        // Use pre-computed k_squared if available
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
                    *r = p * Complex::new(sinc_factor * (-c * c * k2 * dt * dt), 0.0).exp();
                });
        } else {
            // Compute k-squared on the fly
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
                    *val =
                        pk * Complex::new(sinc_factor * (-c * c * k_mag_sq * dt * dt), 0.0)
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if `kx contiguous`.
    /// - Panics if `ky contiguous`.
    /// - Panics if `kz contiguous`.
    ///
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
        let mut grad_x_k = Array3::<Complex>::zeros(field_k.raw_dim());
        let mut grad_y_k = Array3::<Complex>::zeros(field_k.raw_dim());
        let mut grad_z_k = Array3::<Complex>::zeros(field_k.raw_dim());

        let kx_s = kx.as_slice().expect("kx contiguous");
        let ky_s = ky.as_slice().expect("ky contiguous");
        let kz_s = kz.as_slice().expect("kz contiguous");

        Zip::indexed(&mut grad_x_k).and(&field_k).par_for_each(|(i, _j, _k), val, &fk| {
            *val = fk * Complex::new(0.0, kx_s[i]);
        });

        Zip::indexed(&mut grad_y_k).and(&field_k).par_for_each(|(_i, j, _k), val, &fk| {
            *val = fk * Complex::new(0.0, ky_s[j]);
        });

        Zip::indexed(&mut grad_z_k).and(&field_k).par_for_each(|(_i, _j, k), val, &fk| {
            *val = fk * Complex::new(0.0, kz_s[k]);
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
        // Transform to k-space
        let field_k = fft_3d_array(field);

        // Apply Laplacian operator in k-space
        let mut laplacian_k = Array3::<Complex>::zeros(field_k.raw_dim());

        if let Some(ref k_squared) = self.k_squared {
            // Use pre-computed k-squared
            Zip::from(&mut laplacian_k)
                .and(&field_k)
                .and(k_squared)
                .par_for_each(|l, &f, &k2| {
                    *l = f * (-k2);
                });
        } else {
            // Compute k-squared on the fly
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

        // Transform back to spatial domain
        Ok(ifft_3d_array(&laplacian_k))
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::wave_model::NonlinearWave;
    use crate::domain::grid::Grid;
    use crate::domain::medium::HomogeneousMedium;
    use ndarray::Array3;
    use std::f64::consts::PI;

    /// A spatially uniform field has zero spectral gradient in every direction.
    ///
    /// Physics: DC-only spectrum; i·kx·F(k) = 0 for all non-zero k.
    /// Tolerance: N·ε_mach·10 = 512·2.2e-16·10 ≈ 1.1e-12.
    #[test]
    fn compute_spectral_gradient_zero_for_constant_field() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let w = NonlinearWave::new(&grid, 1e-7);
        let field = Array3::<f64>::from_elem((8, 8, 8), 42.0);

        let (gx, gy, gz) = w.compute_spectral_gradient(&field, &grid).unwrap();

        let tol = 512.0 * f64::EPSILON * 10.0;
        for &v in gx.iter().chain(gy.iter()).chain(gz.iter()) {
            assert!(
                v.abs() < tol,
                "gradient of constant field must be zero (got {v:.3e})"
            );
        }
    }

    /// A spatially uniform field has zero spectral Laplacian.
    ///
    /// Physics: −k²·F(k) = 0 for the DC bin (k=0); all other modes are zero.
    #[test]
    fn compute_spectral_laplacian_zero_for_constant_field() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.precompute_k_squared(&grid);
        let field = Array3::<f64>::from_elem((8, 8, 8), 100.0);

        let lap = w.compute_spectral_laplacian(&field, &grid).unwrap();

        let tol = 512.0 * f64::EPSILON * 10.0;
        for &v in lap.iter() {
            assert!(
                v.abs() < tol,
                "Laplacian of constant field must be zero (got {v:.3e})"
            );
        }
    }

    /// For f[i,j,k] = sin(2π·i/N), the x-gradient is (2π/(N·dx))·cos(2π·i/N).
    ///
    /// Mathematical proof:
    ///   F(k₁,0,0) = -(iN/2), F(-k₁,0,0) = (iN/2)  (DFT of sin)
    ///   df/dx |_spectral = IFFT(i·kx·F) = (2π/(N·dx))·cos(2πi/N)
    ///   where kx[1] = 2π/(N·dx).
    #[test]
    fn compute_spectral_gradient_x_analytical_for_single_mode_sinusoid() {
        let n = 8_usize;
        let dx = 0.001_f64;
        let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
        let w = NonlinearWave::new(&grid, 1e-7);

        // f[i,j,k] = sin(2π·i/n)
        let mut field = Array3::<f64>::zeros((n, n, n));
        for i in 0..n {
            let v = (2.0 * PI * i as f64 / n as f64).sin();
            for j in 0..n {
                for k in 0..n {
                    field[[i, j, k]] = v;
                }
            }
        }

        let (grad_x, _gy, _gz) = w.compute_spectral_gradient(&field, &grid).unwrap();

        // Analytical: df/dx = k1x · cos(2π·i/n), k1x = 2π/(n·dx)
        let k1x = 2.0 * PI / (n as f64 * dx);
        let tol = 1e-9 * k1x; // relative to wavenumber magnitude
        for i in 0..n {
            let expected = k1x * (2.0 * PI * i as f64 / n as f64).cos();
            let got = grad_x[[i, 0, 0]];
            assert!(
                (got - expected).abs() < tol,
                "grad_x at i={i}: got {got:.6e} expected {expected:.6e} (tol {tol:.3e})"
            );
        }
    }

    /// For f[i,j,k] = sin(2π·i/N), the spectral Laplacian equals −k1x²·f.
    ///
    /// Mathematical proof:
    ///   ∇²f|_spectral = IFFT(−k²·F) = −k1x²·sin(2π·i/n)
    ///   where k1x = 2π/(N·dx).
    #[test]
    fn compute_spectral_laplacian_negative_definite_for_single_mode_sinusoid() {
        let n = 8_usize;
        let dx = 0.001_f64;
        let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.precompute_k_squared(&grid);

        let mut field = Array3::<f64>::zeros((n, n, n));
        for i in 0..n {
            let v = (2.0 * PI * i as f64 / n as f64).sin();
            for j in 0..n {
                for k in 0..n {
                    field[[i, j, k]] = v;
                }
            }
        }

        let lap = w.compute_spectral_laplacian(&field, &grid).unwrap();

        // ∇²f = −k1x² · f
        let k1x = 2.0 * PI / (n as f64 * dx);
        let factor = -(k1x * k1x);
        let tol = 1e-9 * k1x * k1x;
        for i in 0..n {
            let expected = factor * field[[i, 0, 0]];
            let got = lap[[i, 0, 0]];
            assert!(
                (got - expected).abs() < tol,
                "Laplacian at i={i}: got {got:.6e} expected {expected:.6e}"
            );
        }
    }

    /// y-gradient of a y-only sinusoid satisfies the same analytical formula.
    #[test]
    fn compute_spectral_gradient_y_analytical_for_single_mode_sinusoid() {
        let n = 8_usize;
        let dx = 0.001_f64;
        let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
        let w = NonlinearWave::new(&grid, 1e-7);

        let mut field = Array3::<f64>::zeros((n, n, n));
        for j in 0..n {
            let v = (2.0 * PI * j as f64 / n as f64).sin();
            for i in 0..n {
                for k in 0..n {
                    field[[i, j, k]] = v;
                }
            }
        }

        let (_gx, grad_y, _gz) = w.compute_spectral_gradient(&field, &grid).unwrap();

        let k1 = 2.0 * PI / (n as f64 * dx);
        let tol = 1e-9 * k1;
        for j in 0..n {
            let expected = k1 * (2.0 * PI * j as f64 / n as f64).cos();
            let got = grad_y[[0, j, 0]];
            assert!(
                (got - expected).abs() < tol,
                "grad_y at j={j}: got {got:.6e} expected {expected:.6e}"
            );
        }
    }

    /// `apply_k_space_correction` on a zero-pressure field returns a zero field.
    ///
    /// Trivial null case: FFT(0) = 0; any linear operator on 0 = 0; IFFT(0) = 0.
    #[test]
    fn apply_k_space_correction_zero_field_returns_zero() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::water(&grid);
        let mut w = NonlinearWave::new(&grid, 1e-7);
        w.precompute_k_squared(&grid);

        let pressure = Array3::<f64>::zeros((8, 8, 8));
        let corrected = w.apply_k_space_correction(&pressure, &medium, &grid).unwrap();

        let tol = 512.0 * f64::EPSILON * 10.0;
        for &v in corrected.iter() {
            assert!(
                v.abs() < tol,
                "k-space correction of zero field must be zero (got {v:.3e})"
            );
        }
    }
}
