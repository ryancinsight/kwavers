//! `KspaceShiftGpu` — GPU dispatcher for k-space staggered-grid phase shift.
//!
//! ## Algorithm (Treeby & Cox 2010, Eq. 12)
//!
//! For a 3D complex spectrum `F(kx,ky,kz)` and shift vector `(sx,sy,sz)` (m):
//!
//! ```text
//! F'(kx,ky,kz) = F(kx,ky,kz) · exp(−i·(kx·sx + ky·sy + kz·sz))
//! ```
//!
//! The CPU fallback applies this per element:
//!
//! ```text
//! phase = −(kx[ix]·sx + ky[jy]·sy + kz[kz]·sz)
//! Re' = Re·cos(phase) − Im·sin(phase)
//! Im' = Re·sin(phase) + Im·cos(phase)
//! ```
//!
//! # References
//!
//! - Treeby BE, Cox BT (2010). J. Biomed. Opt. 15(2):021314. (Eq. 12–13)
//! - Liu Q-H (1998). Geophysics 63(6):2082–2089. (PSTD staggered shift)

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// GPU dispatcher for the k-space staggered-grid phase shift.
#[derive(Debug)]
pub struct KspaceShiftGpu {
    nx: usize,
    ny: usize,
    nz: usize,
}

impl KspaceShiftGpu {
    /// Create a new k-space shift dispatcher for a grid of size `nx × ny × nz`.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn new(nx: usize, ny: usize, nz: usize) -> KwaversResult<Self> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(KwaversError::InvalidInput(
                "KspaceShiftGpu: grid dimensions must be > 0".to_string(),
            ));
        }
        Ok(Self { nx, ny, nz })
    }

    /// Apply complex phase shift to spectrum in-place (CPU fallback).
    ///
    /// ## Parameters
    ///
    /// - `real_in` / `imag_in`  — input 3D complex spectrum (shape `[nx, ny, nz]`)
    /// - `kx_vec` / `ky_vec` / `kz_vec` — wavenumber vectors [rad/m]
    /// - `shift` — `[sx, sy, sz]` half-cell shift vector (m)
    /// - `real_out` / `imag_out` — output buffers; fully overwritten
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn apply_shift_into(
        &self,
        real_in: &Array3<f64>,
        imag_in: &Array3<f64>,
        kx_vec: &[f64],
        ky_vec: &[f64],
        kz_vec: &[f64],
        shift: [f64; 3],
        real_out: &mut Array3<f64>,
        imag_out: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let expected = (self.nx, self.ny, self.nz);
        for (name, arr) in [
            ("real_in", real_in),
            ("imag_in", imag_in),
            ("real_out", &*real_out),
            ("imag_out", &*imag_out),
        ] {
            if arr.dim() != expected {
                return Err(KwaversError::InvalidInput(format!(
                    "KspaceShiftGpu: {name} shape {:?} ≠ expected {expected:?}",
                    arr.dim()
                )));
            }
        }
        if kx_vec.len() != self.nx || ky_vec.len() != self.ny || kz_vec.len() != self.nz {
            return Err(KwaversError::InvalidInput(
                "KspaceShiftGpu: wavenumber vector length mismatch".to_string(),
            ));
        }

        let [sx, sy, sz] = shift;
        for kz in 0..self.nz {
            for jy in 0..self.ny {
                for ix in 0..self.nx {
                    let phase = -(kx_vec[ix] * sx + ky_vec[jy] * sy + kz_vec[kz] * sz);
                    let c = phase.cos();
                    let s = phase.sin();
                    let re = real_in[[ix, jy, kz]];
                    let im = imag_in[[ix, jy, kz]];
                    real_out[[ix, jy, kz]] = re * c - im * s;
                    imag_out[[ix, jy, kz]] = re * s + im * c;
                }
            }
        }

        Ok(())
    }

    /// Convenience wrapper that allocates output buffers.
    ///
    /// Returns `(real_out, imag_out)`. Prefer [`Self::apply_shift_into`] in time-step loops.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn apply_shift(
        &self,
        real_in: &Array3<f64>,
        imag_in: &Array3<f64>,
        kx_vec: &[f64],
        ky_vec: &[f64],
        kz_vec: &[f64],
        shift: [f64; 3],
    ) -> KwaversResult<(Array3<f64>, Array3<f64>)> {
        let mut real_out = Array3::zeros((self.nx, self.ny, self.nz));
        let mut imag_out = Array3::zeros((self.nx, self.ny, self.nz));
        self.apply_shift_into(
            real_in,
            imag_in,
            kx_vec,
            ky_vec,
            kz_vec,
            shift,
            &mut real_out,
            &mut imag_out,
        )?;
        Ok((real_out, imag_out))
    }

    /// Grid dimensions `(nx, ny, nz)`.
    pub fn grid_dims(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}
