//! Spectral Laplacian for the Kuznetsov right-hand side.
//!
//! `KuznetsovSpectralOperator` holds the grid's wavenumber tables and one
//! half-spectrum buffer, so an evaluation allocates nothing.

use crate::forward::lanes::for_each_z_lane;
use kwavers_grid::Grid;
use kwavers_math::fft::{get_fft_for_grid, Complex64, Fft3d, Fft3dInOutExt};
use leto::{Array1, Array3};
use std::f64::consts::PI;
use std::sync::Arc;

/// Spectral Laplacian of a real field on a periodic grid.
#[derive(Debug)]
pub struct KuznetsovSpectralOperator {
    kx_vec: Array1<f64>,
    ky_vec: Array1<f64>,
    /// The z wavenumbers of the half spectrum's `nz/2+1` bins, all
    /// non-negative.
    kz_vec: Array1<f64>,
    fft: Arc<Fft3d>,
    /// Half spectrum `(nx, ny, nz/2+1)` of the field: the transform writes it,
    /// the symbol scales it in place and the inverse consumes it.
    field_hat: Array3<Complex64>,
}

/// The first `len` discrete wavenumbers of an `n`-point DFT:
/// `k[i] = 2·k_Nyquist·i/n` for `i ≤ n/2` and `2·k_Nyquist·(i−n)/n` above.
///
/// The factor is 2, not 2π: `k_Nyquist = π/d` already carries π, and an extra
/// π would make the spectral Laplacian, and so the effective wave speed,
/// π² ≈ 9.87× too large.
fn wavenumbers(n: usize, nyquist: f64, len: usize) -> Array1<f64> {
    (0..len)
        .map(|i| {
            let bin = if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            2.0 * nyquist * bin / n as f64
        })
        .collect()
}

impl KuznetsovSpectralOperator {
    /// Create a new spectral operator for the given grid
    pub fn new(grid: &Grid) -> Self {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let nz_half = nz / 2 + 1;
        Self {
            kx_vec: wavenumbers(nx, PI / grid.dx, nx),
            ky_vec: wavenumbers(ny, PI / grid.dy, ny),
            kz_vec: wavenumbers(nz, PI / grid.dz, nz_half),
            fft: get_fft_for_grid(nx, ny, nz),
            field_hat: Array3::from_elem([nx, ny, nz_half], Complex64::default()),
        }
    }

    /// Writes the spectral Laplacian `∇²f = F⁻¹[−(kx² + ky² + kz²)·F[f]]` of
    /// `field` into `laplacian_out`.
    ///
    /// A real field's spectrum is Hermitian and `−|k|²` is even in `k`, so the
    /// symbol scales the half spectrum and the inverse completes it.
    ///
    /// # Panics
    ///
    /// Panics if `field` or `laplacian_out` does not have the grid shape the
    /// operator was built for.
    pub fn compute_laplacian_workspace(
        &mut self,
        field: &Array3<f64>,
        laplacian_out: &mut Array3<f64>,
    ) {
        self.fft.forward_r2c_into(field, &mut self.field_hat);

        let [_, ny, nz_half] = self.field_hat.shape();
        let kx_values = self
            .kx_vec
            .as_slice()
            .expect("invariant: wavenumber tables are collected contiguous");
        let ky_values = self
            .ky_vec
            .as_slice()
            .expect("invariant: wavenumber tables are collected contiguous");
        let kz_values = self
            .kz_vec
            .as_slice()
            .expect("invariant: wavenumber tables are collected contiguous");
        let spectrum = self
            .field_hat
            .as_slice_mut()
            .expect("invariant: the half spectrum is allocated contiguous");
        for_each_z_lane(
            spectrum,
            [ny, nz_half],
            size_of::<Complex64>(),
            |_, i, j, lane| {
                let (kx, ky) = (kx_values[i], ky_values[j]);
                for (value, &kz) in lane.iter_mut().zip(kz_values) {
                    let k_sq = kz.mul_add(kz, kx.mul_add(kx, ky * ky));
                    *value *= -k_sq;
                }
            },
        );

        self.fft
            .inverse_c2r_into(&mut self.field_hat, laplacian_out);
    }
}
