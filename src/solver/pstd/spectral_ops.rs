//! Spectral operations for PSTD solver

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::utils::{fft_3d_array, ifft_3d_array};
use ndarray::{Array3, Zip};
use num_complex::Complex;

/// Spectral operations handler
#[derive(Debug)]
pub struct SpectralOperations {
    pub kx: Array3<f64>,
    pub ky: Array3<f64>,
    pub kz: Array3<f64>,
    pub k_squared: Array3<f64>,
    pub kappa: Option<Array3<f64>>,
}

impl SpectralOperations {
    /// Create new spectral operations
    pub fn new(grid: &Grid) -> Self {
        let (kx, ky, kz) = Self::compute_wavenumbers(grid);
        let k_squared = Self::compute_k_squared(&kx, &ky, &kz);

        Self {
            kx,
            ky,
            kz,
            k_squared,
            kappa: None,
        }
    }

    /// Set k-space correction factors for heterogeneous media
    pub fn set_kspace_correction(&mut self, kappa: Array3<f64>) {
        self.kappa = Some(kappa);
    }

    /// Compute wavenumber arrays
    fn compute_wavenumbers(grid: &Grid) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        let mut kx = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut ky = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut kz = Array3::zeros((grid.nx, grid.ny, grid.nz));

        let kx_1d = grid.compute_kx();
        let ky_1d = grid.compute_ky();
        let kz_1d = grid.compute_kz();

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    kx[[i, j, k]] = kx_1d[i];
                    ky[[i, j, k]] = ky_1d[j];
                    kz[[i, j, k]] = kz_1d[k];
                }
            }
        }

        (kx, ky, kz)
    }

    /// Compute k² for Laplacian
    fn compute_k_squared(kx: &Array3<f64>, ky: &Array3<f64>, kz: &Array3<f64>) -> Array3<f64> {
        let mut k_squared = Array3::zeros(kx.dim());
        Zip::from(&mut k_squared)
            .and(kx)
            .and(ky)
            .and(kz)
            .for_each(|k2, &kx, &ky, &kz| {
                *k2 = kx * kx + ky * ky + kz * kz;
            });
        k_squared
    }

    /// Compute gradient in spectral space
    pub fn compute_gradient(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
        let field_hat = fft_3d_array(field);

        let grad_x_hat = &field_hat * &self.kx.mapv(|k| Complex::new(0.0, k));
        let grad_y_hat = &field_hat * &self.ky.mapv(|k| Complex::new(0.0, k));
        let grad_z_hat = &field_hat * &self.kz.mapv(|k| Complex::new(0.0, k));

        // Apply k-space correction if available
        let (grad_x_hat, grad_y_hat, grad_z_hat) = if let Some(ref kappa) = self.kappa {
            let kappa_complex = kappa.mapv(|k| Complex::new(k, 0.0));
            (
                grad_x_hat * &kappa_complex,
                grad_y_hat * &kappa_complex,
                grad_z_hat * &kappa_complex,
            )
        } else {
            (grad_x_hat, grad_y_hat, grad_z_hat)
        };

        Ok((
            ifft_3d_array(&grad_x_hat),
            ifft_3d_array(&grad_y_hat),
            ifft_3d_array(&grad_z_hat),
        ))
    }

    /// Compute Laplacian in spectral space
    pub fn compute_laplacian(
        &self,
        field: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let field_hat = fft_3d_array(field);

        let mut laplacian_hat = field_hat.clone();
        Zip::from(&mut laplacian_hat)
            .and(&self.k_squared)
            .for_each(|l, &k2| *l *= -k2);

        if let Some(ref kappa) = self.kappa {
            Zip::from(&mut laplacian_hat)
                .and(kappa)
                .for_each(|l, &k| *l *= k);
        }

        Ok(ifft_3d_array(&laplacian_hat))
    }

    /// Apply anti-aliasing (2/3 rule)
    /// Compute divergence of a vector field
    #[must_use]
    pub fn compute_divergence(
        &self,
        vx: &Array3<f64>,
        vy: &Array3<f64>,
        vz: &Array3<f64>,
    ) -> Array3<f64> {
        use crate::utils::{fft_3d_array, ifft_3d_array};
        use num_complex::Complex;

        // Transform to k-space
        let vx_hat = fft_3d_array(vx);
        let vy_hat = fft_3d_array(vy);
        let vz_hat = fft_3d_array(vz);

        // Compute divergence in k-space: div(v) = ikx*vx + iky*vy + ikz*vz
        let mut div_hat = Array3::zeros(vx_hat.raw_dim());
        let i = Complex::new(0.0, 1.0);
        for ((idx, d), &vx) in div_hat.indexed_iter_mut().zip(vx_hat.iter()) {
            let kx = self.kx[idx];
            let ky = self.ky[idx];
            let kz = self.kz[idx];
            let vy = vy_hat[idx];
            let vz = vz_hat[idx];
            *d = i * (kx * vx + ky * vy + kz * vz);
        }

        // Transform back to real space
        ifft_3d_array(&div_hat)
    }

    pub fn apply_antialiasing(&self, field_hat: &mut Array3<Complex<f64>>, grid: &Grid) {
        let kx_max = 2.0 * std::f64::consts::PI / grid.dx / 3.0;
        let ky_max = 2.0 * std::f64::consts::PI / grid.dy / 3.0;
        let kz_max = 2.0 * std::f64::consts::PI / grid.dz / 3.0;

        Zip::from(field_hat)
            .and(&self.kx)
            .and(&self.ky)
            .and(&self.kz)
            .for_each(|f, &kx, &ky, &kz| {
                if kx.abs() > kx_max || ky.abs() > ky_max || kz.abs() > kz_max {
                    *f = Complex::new(0.0, 0.0);
                }
            });
    }
}
