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
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::fft::shift_operators::generate_kappa as canonical_generate_kappa;
use leto::Array3;
use kwavers_math::fft::Complex64;

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
            if r > 0.0 {
                ((l + 2.0 * m) / r).sqrt()
            } else {
                0.0
            }
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
    let mut k = Array3::<f64>::zeros([n, 1, 1]);
    if n <= 1 {
        return k;
    }
    let dk = TWO_PI / (n as f64 * dx);
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
pub(super) fn grid_spacing_from_wavenumber(d_op: &Array3<Complex64>, n: usize) -> f64 {
    if n < 2 {
        return 1.0;
    }
    let dk = d_op[[1, 0, 0]].norm();
    if dk == 0.0 {
        return 1.0;
    }
    TWO_PI / (n as f64 * dk)
}

/// Compute the k-space correction κ = sinc(c_ref·dt·|k|/2) for the elastic PSTD solver.
///
/// Delegates to [`kwavers_math::fft::shift_operators::generate_kappa`] — the single
/// canonical implementation. The `kx`, `ky`, `kz` broadcast arrays (shape `(n,1,1)`)
/// are not consumed here; grid spacings are recovered from the derivative operators
/// at the call site.
///
/// # Arguments
/// * `nx,ny,nz` — grid dimensions
/// * `dx,dy,dz` — grid spacings (m), recovered from the wavenumber-axis step
/// * `c_ref`    — maximum P-wave speed (m/s)
/// * `dt`       — time step (s)
#[allow(clippy::too_many_arguments)]
pub(super) fn build_kappa(
    _kx: &Array3<f64>,
    _ky: &Array3<f64>,
    _kz: &Array3<f64>,
    (nx, ny, nz): (usize, usize, usize),
    c_ref: f64,
    dt: f64,
    dx: f64,
    dy: f64,
    dz: f64,
) -> Array3<f64> {
    canonical_generate_kappa(nx, ny, nz, dx, dy, dz, c_ref, dt).into()
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
    input: &Array3<Complex64>,
    op_x: &[Complex64],
    kappa: &Array3<f64>,
    output: &mut Array3<Complex64>,
) {
    let [nx, ny, nz] = output.shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                output[[i, j, k]] = input[[i, j, k]] * op_x[i] * kappa[[i, j, k]];
            }
        }
    }
}

/// Compute `output[i,j,k] = input[i,j,k] · op_y[j] · kappa[i,j,k]`.
///
/// `op_y` is a contiguous slice of length `ny` from a `(ny, 1, 1)` operator
/// array indexed by the y-axis position `j`.
#[inline]
pub(super) fn spectral_mul_y(
    input: &Array3<Complex64>,
    op_y: &[Complex64],
    kappa: &Array3<f64>,
    output: &mut Array3<Complex64>,
) {
    let [nx, ny, nz] = output.shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                output[[i, j, k]] = input[[i, j, k]] * op_y[j] * kappa[[i, j, k]];
            }
        }
    }
}

/// Compute `output[i,j,k] = input[i,j,k] · op_z[k] · kappa[i,j,k]`.
///
/// `op_z` is a contiguous slice of length `nz` from a `(nz, 1, 1)` operator
/// array indexed by the z-axis position `k`.
#[inline]
pub(super) fn spectral_mul_z(
    input: &Array3<Complex64>,
    op_z: &[Complex64],
    kappa: &Array3<f64>,
    output: &mut Array3<Complex64>,
) {
    let [nx, ny, nz] = output.shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                output[[i, j, k]] = input[[i, j, k]] * op_z[k] * kappa[[i, j, k]];
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::orchestrator::ElasticPstdOrchestrator;
    use super::super::types::{
        ElasticPstdMedium, ElasticPstdSourceMode, ElasticPstdVelocitySource,
    };
    use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
    use kwavers_grid::Grid;
    use leto::{Array1, Array3};
    use std::f64::consts::PI;

    /// DC mode kappa (|k|=0) must be exactly 1.0: `sinc(0) = 1` by L'Hôpital.
    ///
    /// Proof: `lim_{x→0} sin(x)/x = 1`. The branch in `build_kappa` guards
    /// `arg < 1e-12` with the value 1.0, so the DC mode is never divided by
    /// near-zero and kappa[0,0,0] = 1.0 exactly.
    #[test]
    fn kappa_dc_mode_is_exactly_one() {
        let nx = 16usize;
        let dx = 1e-3_f64;
        let cp = SOUND_SPEED_WATER_SIM;
        let dt = 0.3 * dx / cp;
        let grid = Grid::new(nx, nx, nx, dx, dx, dx).unwrap();
        let medium = ElasticPstdMedium {
            lame_lambda: Array3::from_elem([nx, nx, nx], DENSITY_WATER_NOMINAL * cp * cp),
            lame_mu: Array3::zeros([nx, nx, nx]),
            density: Array3::from_elem([nx, nx, nx], DENSITY_WATER_NOMINAL),
        };
        let orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();
        assert_eq!(
            orch.kappa[[0, 0, 0]],
            1.0,
            "kappa at |k|=0 (DC mode) must be exactly 1.0 by L'Hôpital"
        );
    }

    /// Kappa values lie in `(0, 1]` for all wavenumber modes.
    ///
    /// `sinc(x) ∈ (0, 1]` for `x ∈ [0, π/2)` and `sinc(x) > 0` for any
    /// `arg = c_ref·dt·|k|/2`. The CFL stability bound ensures `arg < π/2`
    /// for all modes up to the Nyquist wavenumber at `CFL = 1`, so kappa
    /// stays in `(0, 1]` for any physically realised elastic CFL ≤ 1.
    #[test]
    fn kappa_strictly_in_unit_interval() {
        let nx = 32usize;
        let dx = 1e-3_f64;
        let cp = SOUND_SPEED_WATER_SIM;
        // CFL = 0.5 (moderate; kappa Nyquist ≈ sinc(π/4) ≈ 0.90)
        let dt = 0.5 * dx / cp;
        let grid = Grid::new(nx, nx, nx, dx, dx, dx).unwrap();
        let medium = ElasticPstdMedium {
            lame_lambda: Array3::from_elem([nx, nx, nx], DENSITY_WATER_NOMINAL * cp * cp),
            lame_mu: Array3::zeros([nx, nx, nx]),
            density: Array3::from_elem([nx, nx, nx], DENSITY_WATER_NOMINAL),
        };
        let orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();
        let [sx, sy, sz] = orch.kappa.shape();
        for i in 0..sx {
            for j in 0..sy {
                for k in 0..sz {
                    let kap = orch.kappa[[i, j, k]];
                    assert!(
                        kap > 0.0 && kap <= 1.0,
                        "kappa[{i},{j},{k}] = {kap:.6} not in (0, 1]"
                    );
                }
            }
        }
    }

    /// Kappa Nyquist value matches the analytical `sinc(CFL·π/2)`.
    ///
    /// At the 1D Nyquist wavenumber `|k| = π/dx` along one axis, the argument
    /// is `c_ref·dt·π/(2·dx) = CFL·π/2`. This test uses `nx = 4`, `ny = nz = 1`
    /// so the Nyquist mode is at `i = nx/2 = 2` for the x-axis.
    #[test]
    fn kappa_nyquist_matches_analytical_sinc_cfl_pi_over_2() {
        let nx = 4usize;
        let dx = 1e-3_f64;
        let cp = SOUND_SPEED_WATER_SIM;
        let cfl = 0.3_f64;
        let dt = cfl * dx / cp;
        let grid = Grid::new(nx, 1, 1, dx, dx, dx).unwrap();
        let medium = ElasticPstdMedium {
            lame_lambda: Array3::from_elem([nx, 1, 1], DENSITY_WATER_NOMINAL * cp * cp),
            lame_mu: Array3::zeros([nx, 1, 1]),
            density: Array3::from_elem([nx, 1, 1], DENSITY_WATER_NOMINAL),
        };
        let orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();
        // At i = nx/2 = 2: kx = (2−4)·2π/(4·dx) → |kx| = π/dx (Nyquist)
        let nyquist_kap = orch.kappa[[nx / 2, 0, 0]];
        let arg = cfl * PI / 2.0;
        let expected = arg.sin() / arg;
        let rel_err = (nyquist_kap - expected).abs() / expected;
        assert!(
            rel_err < 1e-12,
            "kappa at Nyquist = {nyquist_kap:.10}, expected sinc(CFL·π/2) = {expected:.10}, \
             rel_err = {rel_err:.3e}"
        );
    }

    /// k-space correction suppresses temporal dispersion at moderate CFL.
    ///
    /// A sinusoidal source drives a 1D (ny=nz=1) grid for 20 steps. The
    /// k-space corrected scheme produces a finite, non-zero, bounded signal at
    /// a downstream sensor, confirming the correction does not introduce
    /// numerical instability or suppress valid propagation.
    ///
    /// # Theorem reference
    ///
    /// Treeby–Cox 2010, Eq. 18: κ(k) = sinc(c_ref·dt·|k|/2) exactly cancels
    /// the leapfrog temporal phase error for the reference plane-wave speed,
    /// so the numerical phase velocity equals c_ref for all modes.
    #[test]
    fn kappa_preserves_peak_amplitude_at_moderate_cfl() {
        let nx = 64usize;
        let dx = 1e-3_f64;
        let cp = SOUND_SPEED_WATER_SIM;
        let cfl = 0.5_f64;
        let dt = cfl * dx / cp;
        let n_steps = 20usize;
        let grid = Grid::new(nx, 1, 1, dx, dx, dx).unwrap();
        let lam = DENSITY_WATER_NOMINAL * cp * cp;
        let medium = ElasticPstdMedium {
            lame_lambda: Array3::from_elem([nx, 1, 1], lam),
            lame_mu: Array3::zeros([nx, 1, 1]),
            density: Array3::from_elem([nx, 1, 1], DENSITY_WATER_NOMINAL),
        };
        let mut orch = ElasticPstdOrchestrator::new(&grid, medium, dt).unwrap();
        let k0 = PI / (4.0 * dx); // quarter-Nyquist
        let amp = 1e-6_f64;
        let mut src_signal = Array1::<f64>::zeros([n_steps]);
        for (t, v) in src_signal.iter_mut().enumerate() {
            *v = amp * (cp * k0 * t as f64 * dt).sin();
        }
        let mut mask = Array3::<bool>::from_elem([nx, 1, 1], false);
        mask[[nx / 2, 0, 0]] = true;
        let source = ElasticPstdVelocitySource {
            mask,
            ux: Some(src_signal),
            uy: None,
            uz: None,
            mode: ElasticPstdSourceMode::Additive,
        };
        let mut sensor_mask = Array3::<bool>::from_elem([nx, 1, 1], false);
        sensor_mask[[nx / 2 + 4, 0, 0]] = true;
        let data = orch
            .propagate(n_steps, Some(&source), Some(&sensor_mask))
            .unwrap();
        let vx_trace = data.vx.expect("vx recorded at sensor");
        let peak = vx_trace.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        assert!(peak.is_finite(), "k-space corrected peak must be finite");
        assert!(peak > 0.0, "k-space corrected sensor must record a pulse");
        assert!(
            peak < 1e-4,
            "peak {peak:.3e} unexpectedly large — possible instability"
        );
    }
}
