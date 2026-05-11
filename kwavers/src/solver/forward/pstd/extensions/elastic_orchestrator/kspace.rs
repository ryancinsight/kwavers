//! k-space correction for the elastic PSTD orchestrator.
//!
//! # Theorem (Tabei et al. 2002 / Treeby–Cox 2010 Eq. 18)
//!
//! For an elastic leapfrog scheme with time step `dt` and reference P-wave
//! speed `c_ref`, the spectral correction factor
//!
//! ```text
//!   κ(k) = sinc(c_ref · dt · |k| / 2)
//!          = sin(c_ref · dt · |k| / 2) / (c_ref · dt · |k| / 2)
//! ```
//!
//! applied to every spectral spatial derivative replaces the leapfrog
//! temporal update with the exact analytical wave-equation solution for
//! plane waves traveling at `c = c_ref`. In the acoustic (μ = 0) limit this
//! recovers the canonical acoustic k-space correction. The elastic extension
//! is due to Tabei et al. (2002) and applied to both the stress
//! (strain-rate) and velocity (divergence-of-stress) spectral passes.
//!
//! # Stability
//!
//! `κ(k) = sinc(arg) ∈ (0, 1]` for `arg = c_ref·dt·|k|/2 ∈ [0, π/2)`.
//! At the elastic CFL limit (CFL = c_ref·dt/dx = 1.0 in 1D, `arg_max =
//! π/2`) κ_min = 2/π ≈ 0.637. Multiplying the spectral derivative by κ < 1
//! scales all wavenumber modes downward, which is unconditionally
//! dissipation-free (κ is real and positive) and preserves the leapfrog
//! stability region.
//!
//! # References
//!
//! - Tabei M., Mast T. D. & Waag R. C. (2002). "A k-space method for
//!   coupled first-order acoustic propagation equations." J. Acoust. Soc.
//!   Am. 111(1), 53–63.
//! - Treeby B. E. & Cox B. T. (2010). "k-Wave: MATLAB toolbox for the
//!   simulation and reconstruction of photoacoustic wave fields." J. Biomed.
//!   Opt. 15(2), 021314.

use super::types::ElasticPstdMedium;
use ndarray::Array3;

/// Maximum P-wave speed `c_p = sqrt((λ + 2μ)/ρ)` across the medium.
///
/// Used as `c_ref` for the k-space correction. For a homogeneous medium this
/// equals the P-wave speed exactly. For a heterogeneous medium it is the
/// conservative upper bound, keeping `kappa ≤ 1` everywhere so the
/// correction never amplifies any mode. Returns 0.0 (no correction) when the
/// medium is degenerate (ρ ≤ 0 everywhere).
pub(super) fn max_p_wave_speed(medium: &ElasticPstdMedium) -> f64 {
    medium
        .lame_lambda
        .iter()
        .zip(medium.lame_mu.iter())
        .zip(medium.density.iter())
        .map(|((&l, &m), &r)| {
            if r > 0.0 { ((l + 2.0 * m) / r).sqrt() } else { 0.0 }
        })
        .fold(0.0_f64, f64::max)
}

/// Pre-compute the k-space correction `kappa[i,j,k] = sinc(c_ref·dt·|k|/2)`.
///
/// `kx`, `ky`, `kz` are the raw wavenumber arrays shaped `(N_α, 1, 1)` built
/// by `wavenumber_axis`. The full wavenumber magnitude at voxel `(i,j,k)` is
/// `|k|² = kx[i]² + ky[j]² + kz[k]²`.
///
/// At `|k| = 0` (DC mode) the sinc function returns 1.0 by L'Hôpital's
/// rule, enforced by the `arg < 1e-12` guard.
pub(super) fn build_kappa(
    kx: &Array3<f64>,
    ky: &Array3<f64>,
    kz: &Array3<f64>,
    (nx, ny, nz): (usize, usize, usize),
    c_ref: f64,
    dt: f64,
) -> Array3<f64> {
    Array3::from_shape_fn((nx, ny, nz), |(i, j, k)| {
        let k_sq = kx[[i, 0, 0]] * kx[[i, 0, 0]]
            + ky[[j, 0, 0]] * ky[[j, 0, 0]]
            + kz[[k, 0, 0]] * kz[[k, 0, 0]];
        let arg = 0.5 * c_ref * dt * k_sq.sqrt();
        if arg < 1e-12 { 1.0 } else { arg.sin() / arg }
    })
}
