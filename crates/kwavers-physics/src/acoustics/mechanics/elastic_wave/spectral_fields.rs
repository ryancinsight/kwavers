//! Spectral Field Representations for Elastic Wave Propagation
//!
//! Provides complex-valued field containers for FFT-based (pseudo-spectral)
//! methods.  Physical fields are real-valued; their Fourier transforms are
//! complex and used to compute spatial derivatives in the wavenumber domain.
//!
//! ## Mathematical Foundation
//!
//! For a real field f(x) on a periodic domain of length L with N grid points,
//! the DFT pair is:
//!
//! ```text
//! F[k] = Σ_{n=0}^{N-1} f[n] · exp(−2πi k n / N)
//! f[n] = (1/N) Σ_{k=0}^{N-1} F[k] · exp(+2πi k n / N)
//! ```
//!
//! Spatial derivatives in the wavenumber domain become multiplications:
//! `∂f/∂x ↔ ik_x · F[k_x]` where `k_x = 2πn/(N·dx)`.
//!
//! ## Theorem (Plancherel – Round-Trip Identity)
//!
//! **Statement.** For a real-valued array f ∈ ℝ^N stored as f64:
//!
//! ```text
//! IFFT(FFT(f)) = f    (up to floating-point rounding)
//! ```
//!
//! with round-trip error bounded by `|f_recovered - f| ≤ N · ε_mach · ‖F‖_∞`
//! where ε_mach ≈ 2.2 × 10⁻¹⁶ for f64.
//!
//! **Proof sketch.** The DFT is a unitary transformation (Parseval's theorem
//! applies); the numerical FFT implements it with O(N log N) floating-point
//! operations each introducing at most O(ε_mach) relative error.  The accumulated
//! error is O(log N · ε_mach) per element, bounded by N · ε_mach for the
//! worst-case sum.
//!
//! **Consequence.** `from_real` followed by `to_real` recovers the original
//! physical field to within round-trip error.  The tests below verify this
//! bound for 8×8×8 grids (N = 512): tolerance = 512 · ε_mach · 10.
//!
//! ## Theorem (Spectral Derivative Exactness)
//!
//! For a bandlimited signal (no aliasing), multiplication by `i·k` in the
//! wavenumber domain is *exact* — there is no truncation error.  For an
//! N-point DFT, the Nyquist bin (k = N/2) must be zeroed to preserve
//! real-valued output after IFFT.  This is the standard spectral differentiation
//! algorithm used in k-space and PSTD methods.
//!
//! ## References
//!
//! - Fornberg B. (1998). *A Practical Guide to Pseudospectral Methods*.
//!   Cambridge University Press, §1.4 (differentiation matrices).
//! - Kreiss H.O., Oliger J. (1972). SIAM J. Numer. Anal. 9(1), 112–128.
//!   (spectral accuracy for periodic problems)

use leto::Array3 as LetoArray3;
use kwavers_math::fft::Complex64;

/// Complex-valued stress field components for spectral methods
#[derive(Debug, Clone)]
pub struct SpectralStressFields {
    /// Normal stress components in frequency domain
    pub txx: LetoArray3<Complex64>,
    pub tyy: LetoArray3<Complex64>,
    pub tzz: LetoArray3<Complex64>,
    /// Shear stress components in frequency domain
    pub txy: LetoArray3<Complex64>,
    pub txz: LetoArray3<Complex64>,
    pub tyz: LetoArray3<Complex64>,
}

impl SpectralStressFields {
    /// Create new spectral stress fields
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            txx: LetoArray3::from_elem([nx, ny, nz], Complex64::default()),
            tyy: LetoArray3::from_elem([nx, ny, nz], Complex64::default()),
            tzz: LetoArray3::from_elem([nx, ny, nz], Complex64::default()),
            txy: LetoArray3::from_elem([nx, ny, nz], Complex64::default()),
            txz: LetoArray3::from_elem([nx, ny, nz], Complex64::default()),
            tyz: LetoArray3::from_elem([nx, ny, nz], Complex64::default()),
        }
    }

    /// Initialize from real-valued stress fields via FFT
    #[must_use]
    pub fn from_real(real_fields: &super::fields::StressFields) -> Self {
        use kwavers_math::fft::fft_3d_array;

        let [nx, ny, nz] = real_fields.txx.shape();
        let mut spectral = Self::new(nx, ny, nz);

        spectral.txx = fft_3d_array(&real_fields.txx);
        spectral.tyy = fft_3d_array(&real_fields.tyy);
        spectral.tzz = fft_3d_array(&real_fields.tzz);
        spectral.txy = fft_3d_array(&real_fields.txy);
        spectral.txz = fft_3d_array(&real_fields.txz);
        spectral.tyz = fft_3d_array(&real_fields.tyz);

        spectral
    }

    /// Convert back to real-valued fields via inverse FFT
    #[must_use]
    pub fn to_real(&self) -> super::fields::StressFields {
        use kwavers_math::fft::ifft_3d_array;

        let [nx, ny, nz] = self.txx.shape();
        let mut real_fields = super::fields::StressFields::new(nx, ny, nz);

        real_fields.txx = ifft_3d_array(&self.txx);
        real_fields.tyy = ifft_3d_array(&self.tyy);
        real_fields.tzz = ifft_3d_array(&self.tzz);
        real_fields.txy = ifft_3d_array(&self.txy);
        real_fields.txz = ifft_3d_array(&self.txz);
        real_fields.tyz = ifft_3d_array(&self.tyz);

        real_fields
    }
}

/// Complex-valued velocity field components for spectral methods
#[derive(Debug, Clone)]
pub struct SpectralVelocityFields {
    /// Velocity components in frequency domain
    pub vx: LetoArray3<Complex64>,
    pub vy: LetoArray3<Complex64>,
    pub vz: LetoArray3<Complex64>,
}

impl SpectralVelocityFields {
    /// Create new spectral velocity fields
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            vx: LetoArray3::from_elem([nx, ny, nz], Complex64::default()),
            vy: LetoArray3::from_elem([nx, ny, nz], Complex64::default()),
            vz: LetoArray3::from_elem([nx, ny, nz], Complex64::default()),
        }
    }

    /// Initialize from real-valued velocity fields via FFT
    #[must_use]
    pub fn from_real(real_fields: &super::fields::VelocityFields) -> Self {
        use kwavers_math::fft::fft_3d_array;

        let [nx, ny, nz] = real_fields.vx.shape();
        let mut spectral = Self::new(nx, ny, nz);

        spectral.vx = fft_3d_array(&real_fields.vx);
        spectral.vy = fft_3d_array(&real_fields.vy);
        spectral.vz = fft_3d_array(&real_fields.vz);

        spectral
    }

    /// Convert back to real-valued fields via inverse FFT
    #[must_use]
    pub fn to_real(&self) -> super::fields::VelocityFields {
        use kwavers_math::fft::ifft_3d_array;

        let [nx, ny, nz] = self.vx.shape();
        let mut real_fields = super::fields::VelocityFields::new(nx, ny, nz);

        real_fields.vx = ifft_3d_array(&self.vx);
        real_fields.vy = ifft_3d_array(&self.vy);
        real_fields.vz = ifft_3d_array(&self.vz);

        real_fields
    }
}

#[cfg(test)]
mod tests {
    use super::super::fields::{StressFields, VelocityFields};
    use super::*;

    /// FFT → IFFT round-trip for stress fields must recover the real-valued input
    /// to floating-point precision (tolerance = N · ε_mach · 10 where N = 512).
    #[test]
    fn spectral_stress_from_real_to_real_round_trip_is_identity() {
        let (nx, ny, nz) = (8, 8, 8);
        let mut real = StressFields::new(nx, ny, nz);
        // Fill txx with (i + 2j + 3k) pattern
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    *real.txx.get_mut([i, j, k]).unwrap() = (i + 2 * j + 3 * k) as f64;
                    *real.txy.get_mut([i, j, k]).unwrap() = (i * j + k + 1) as f64;
                }
            }
        }

        let spectral = SpectralStressFields::from_real(&real);
        let recovered = spectral.to_real();

        let tol = 512.0 * f64::EPSILON * 10.0;
        for (orig, rec) in real.txx.iter().zip(recovered.txx.iter()) {
            assert!(
                (orig - rec).abs() < tol,
                "txx round-trip error {:.3e} > tol {:.3e}",
                (orig - rec).abs(),
                tol
            );
        }
        for (orig, rec) in real.txy.iter().zip(recovered.txy.iter()) {
            assert!(
                (orig - rec).abs() < tol,
                "txy round-trip error {:.3e} > tol {:.3e}",
                (orig - rec).abs(),
                tol
            );
        }
    }

    /// FFT → IFFT round-trip for velocity fields must recover the real-valued
    /// input to floating-point precision.
    #[test]
    fn spectral_velocity_from_real_to_real_round_trip_is_identity() {
        let (nx, ny, nz) = (8, 8, 8);
        let mut real = VelocityFields::new(nx, ny, nz);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    *real.vx.get_mut([i, j, k]).unwrap() = (i + j * 3 + k * 7) as f64 * 0.1;
                    *real.vy.get_mut([i, j, k]).unwrap() = (i * 2 + j + k * 5) as f64 * 0.05;
                }
            }
        }

        let spectral = SpectralVelocityFields::from_real(&real);
        let recovered = spectral.to_real();

        let tol = 512.0 * f64::EPSILON * 10.0;
        for (orig, rec) in real.vx.iter().zip(recovered.vx.iter()) {
            assert!(
                (orig - rec).abs() < tol,
                "vx round-trip error {:.3e} > tol {:.3e}",
                (orig - rec).abs(),
                tol
            );
        }
        for (orig, rec) in real.vy.iter().zip(recovered.vy.iter()) {
            assert!(
                (orig - rec).abs() < tol,
                "vy round-trip error {:.3e} > tol {:.3e}",
                (orig - rec).abs(),
                tol
            );
        }
    }
}
