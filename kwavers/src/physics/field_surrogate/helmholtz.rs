//! Finite-difference Helmholtz residual for the focal-envelope field.
//!
//! In a homogeneous medium with sound speed `c0`, a time-harmonic
//! pressure phasor `p(x, y, z)` at angular frequency `ω = 2π·f0`
//! satisfies the homogeneous Helmholtz equation in source-free regions:
//!
//!   ∇²p + k²p = 0,   k = ω / c0.
//!
//! For the focal-envelope statistics produced by a focused-bowl PSTD
//! pulse (`p_max`, `p_min`, `p_rms`) the *envelope* approximately
//! satisfies the same equation in the source-free interior of the
//! grid — exactly in the linear regime, approximately under weak
//! nonlinearity. This module computes the finite-difference Helmholtz
//! residual `R(x, y, z) = ∇²p + k²p` so it can be:
//!
//! 1. used as a soft physics-residual loss term during training of
//!    the parameterised field-surrogate PINN
//!    (`solver::inverse::pinn::ml::field_surrogate`), and
//! 2. measured *on cached PSTD kernels* as a sanity check that the
//!    cube data is Helmholtz-consistent — small residual in the
//!    interior validates both the wave solver and the residual
//!    formulation.
//!
//! ## Numerical method
//!
//! Second-order central differences:
//!
//!   ∂²p/∂x² ≈ (p(x+dx) − 2·p(x) + p(x−dx)) / dx².
//!
//! Boundary voxels (one-cell shell) are filled with `0.0` since the
//! one-sided stencil is biased and uninformative for residual
//! analysis; callers should mask them out before computing
//! summary statistics.

use ndarray::Array3;

use super::kernel::FocalKernel;

/// Default sound speed for water at body temperature (m/s). The
/// kernel cube is generated on a homogeneous water-equivalent medium,
/// so `c0 = 1500` is the canonical reference; callers can override
/// for tissue or layered media.
pub const HELMHOLTZ_C0_WATER: f64 = 1500.0;

/// Compute the Helmholtz residual `R = ∇²p + k²p` at every voxel.
///
/// `p` is the envelope field (e.g. the per-voxel peak rarefactional
/// pressure stored in `FocalKernel::field`). `dx_m` is the isotropic
/// grid spacing. `f0` is the source centre frequency in Hz; `c0` is
/// the medium sound speed in m/s.
///
/// The output array has the same shape as `p`. Boundary voxels (the
/// one-cell shell on every face) are zero.
#[must_use]
pub fn helmholtz_residual_field(p: &Array3<f64>, dx_m: f64, f0: f64, c0: f64) -> Array3<f64> {
    debug_assert!(dx_m > 0.0);
    debug_assert!(f0 > 0.0);
    debug_assert!(c0 > 0.0);
    let (nx, ny, nz) = p.dim();
    let mut r = Array3::<f64>::zeros((nx, ny, nz));
    if nx < 3 || ny < 3 || nz < 3 {
        return r;
    }
    let inv_dx2 = 1.0 / (dx_m * dx_m);
    let k = 2.0 * std::f64::consts::PI * f0 / c0;
    let k2 = k * k;
    for i in 1..(nx - 1) {
        for j in 1..(ny - 1) {
            for kk in 1..(nz - 1) {
                let pc = p[[i, j, kk]];
                let lap = (p[[i + 1, j, kk]] - 2.0 * pc + p[[i - 1, j, kk]] + p[[i, j + 1, kk]]
                    - 2.0 * pc
                    + p[[i, j - 1, kk]]
                    + p[[i, j, kk + 1]]
                    - 2.0 * pc
                    + p[[i, j, kk - 1]])
                    * inv_dx2;
                r[[i, j, kk]] = lap + k2 * pc;
            }
        }
    }
    r
}

/// Convenience wrapper: residual on a [`FocalKernel`]'s field array.
#[must_use]
pub fn helmholtz_residual_kernel(kernel: &FocalKernel, c0: f64) -> Array3<f64> {
    helmholtz_residual_field(&kernel.field, kernel.dx_m, kernel.f0, c0)
}

/// Summary statistics for a Helmholtz-residual analysis.
#[derive(Debug, Clone, Copy)]
pub struct HelmholtzResidualStats {
    /// Maximum absolute residual over the interior (Pa/m²).
    pub max_abs: f64,
    /// RMS residual over the interior (Pa/m²).
    pub rms: f64,
    /// Maximum absolute envelope value (Pa) — useful for normalisation.
    pub p_max_abs: f64,
    /// Dimensionless residual ratio `rms / (k² · p_max_abs)`. Values
    /// `≪ 1` indicate the envelope is Helmholtz-consistent.
    pub normalised_ratio: f64,
}

/// Compute summary statistics over the interior (boundary shell
/// excluded) of a residual field.
#[must_use]
pub fn helmholtz_residual_stats(
    residual: &Array3<f64>,
    p: &Array3<f64>,
    f0: f64,
    c0: f64,
) -> HelmholtzResidualStats {
    let (nx, ny, nz) = residual.dim();
    let mut max_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut count = 0usize;
    for i in 1..nx.saturating_sub(1) {
        for j in 1..ny.saturating_sub(1) {
            for kk in 1..nz.saturating_sub(1) {
                let r = residual[[i, j, kk]];
                if r.abs() > max_abs {
                    max_abs = r.abs();
                }
                sum_sq += r * r;
                count += 1;
            }
        }
    }
    let rms = if count > 0 {
        (sum_sq / count as f64).sqrt()
    } else {
        0.0
    };
    let p_max_abs = p.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    let k = 2.0 * std::f64::consts::PI * f0 / c0;
    let scale = k * k * p_max_abs;
    let normalised_ratio = if scale > 0.0 { rms / scale } else { 0.0 };
    HelmholtzResidualStats {
        max_abs,
        rms,
        p_max_abs,
        normalised_ratio,
    }
}
