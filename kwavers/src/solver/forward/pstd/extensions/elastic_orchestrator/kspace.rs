//! k-space correction and spectral derivative helpers for the elastic PSTD
//! orchestrator.
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
use ndarray::{Array3, Zip};
use num_complex::Complex;

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
/// Build the standard FFTW-convention wavenumber axis for `n` points with
/// grid spacing `dx`.
///
/// Returns an `(n, 1, 1)` array (broadcast-ready for spectral derivative
/// operators) with:
/// - `k[i] = i · dk` for `i < n/2` (positive frequencies)
/// - `k[i] = (i − n) · dk` for `i ≥ n/2` (negative frequencies / Nyquist)
///
/// where `dk = 2π / (n · dx)`. The DC mode (`i = 0`) has `k = 0`.
/// For `n ≤ 1` the array is all zeros (only the DC mode exists).
pub(super) fn wavenumber_axis(n: usize, dx: f64) -> Array3<f64> {
    let mut k = Array3::<f64>::zeros((n, 1, 1));
    if n <= 1 {
        return k;
    }
    let dk = 2.0 * std::f64::consts::PI / (n as f64 * dx);
    for i in 0..n / 2 {
        k[[i, 0, 0]] = i as f64 * dk;
    }
    for i in n / 2..n {
        k[[i, 0, 0]] = (i as f64 - n as f64) * dk;
    }
    k
}

/// Recover the grid spacing `dx` from a precomputed complex spectral
/// derivative operator axis shaped `(n, 1, 1)`.
///
/// The axis carries `D[i] = i·k[i]·exp(±i·k[i]·dx/2)` where
/// `k[1] = dk = 2π / (n·dx)`. Hence `|D[1]| = dk` and
/// `dx = 2π / (n·|D[1]|)`. Returns `1.0` for degenerate axes (`n < 2` or
/// `|D[1]| = 0`).
pub(super) fn grid_spacing_from_wavenumber(
    d_op: &Array3<Complex<f64>>,
    n: usize,
) -> f64 {
    if n < 2 {
        return 1.0;
    }
    let dk = d_op[[1, 0, 0]].norm();
    if dk == 0.0 {
        return 1.0;
    }
    2.0 * std::f64::consts::PI / (n as f64 * dk)
}

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

// ─── Spectral derivative helpers ─────────────────────────────────────────────
//
// These three functions apply a staggered-grid spectral derivative operator
// (shaped `(n_axis, 1, 1)`, passed as a contiguous slice) combined with the
// k-space correction `kappa` to a complex spectral field. They are shared by
// both the standard leapfrog path (`orchestrator.rs`) and the split-field PML
// path (`split_field_step.rs`).

/// Compute `output[i,j,k] = input[i,j,k] · op_x[i] · kappa[i,j,k]`.
///
/// `op_x` is a contiguous slice of length `nx` from a `(nx, 1, 1)` operator
/// array. The x-axis index `i` selects the per-wavenumber multiplier.
/// Applied to both the stress-update (neg-shift) and velocity-update
/// (pos-shift) derivative operators; the caller chooses the correct slice.
#[inline]
pub(super) fn spectral_mul_x(
    input: &Array3<Complex<f64>>,
    op_x: &[Complex<f64>],
    kappa: &Array3<f64>,
    output: &mut Array3<Complex<f64>>,
) {
    Zip::indexed(output.view_mut())
        .and(input.view())
        .and(kappa.view())
        .for_each(|(i, _, _), out, inp, kap| {
            *out = *inp * op_x[i] * kap;
        });
}

/// Compute `output[i,j,k] = input[i,j,k] · op_y[j] · kappa[i,j,k]`.
///
/// `op_y` is a contiguous slice of length `ny` from a `(ny, 1, 1)` operator
/// array indexed by the y-axis position `j`.
#[inline]
pub(super) fn spectral_mul_y(
    input: &Array3<Complex<f64>>,
    op_y: &[Complex<f64>],
    kappa: &Array3<f64>,
    output: &mut Array3<Complex<f64>>,
) {
    Zip::indexed(output.view_mut())
        .and(input.view())
        .and(kappa.view())
        .for_each(|(_, j, _), out, inp, kap| {
            *out = *inp * op_y[j] * kap;
        });
}

/// Compute `output[i,j,k] = input[i,j,k] · op_z[k] · kappa[i,j,k]`.
///
/// `op_z` is a contiguous slice of length `nz` from a `(nz, 1, 1)` operator
/// array indexed by the z-axis position `k`.
#[inline]
pub(super) fn spectral_mul_z(
    input: &Array3<Complex<f64>>,
    op_z: &[Complex<f64>],
    kappa: &Array3<f64>,
    output: &mut Array3<Complex<f64>>,
) {
    Zip::indexed(output.view_mut())
        .and(input.view())
        .and(kappa.view())
        .for_each(|(_, _, k), out, inp, kap| {
            *out = *inp * op_z[k] * kap;
        });
}
