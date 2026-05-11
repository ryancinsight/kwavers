//! Staggered-grid shift operator functions and k-space correction arrays.
//!
//! # Theory: Staggered-Grid Spectral Derivative
//!
//! For a uniformly-spaced staggered grid with spacing `ds`, the spectral
//! derivative of a field `f` at the half-cell-shifted position `x + Δx/2` is:
//! ```text
//!   ∂f/∂x |_{x + Δx/2} = IFFT( i·kx · exp(+i·kx·Δx/2) · κ(k) · FFT(f) )
//! ```
//! The operator `i·kx · exp(±i·kx·Δx/2)` is precomputed for each axis as a
//! 1-D complex array (one entry per wavenumber bin) and stored as
//! `ddx_k_shift_pos` (positive half-shift, used for p→u gradients) and
//! `ddx_k_shift_neg` (negative half-shift, used for u→ρ divergences).
//!
//! # Critical Note on the Nyquist Bin
//!
//! For even `n`, the Nyquist bin at `idx = n/2` **must NOT be zeroed**. k-Wave
//! C++ includes this bin in propagation; zeroing it removes ~18 % of k-space
//! energy and causes a 1.64× amplitude error (confirmed 2026-03-27, commit
//! fix in orchestrator.rs). The operator value `i·k·exp(±i·k·ds/2)` at
//! `k = ±π/ds` evaluates to the same real number regardless of the sign of k,
//! so no sign ambiguity arises at Nyquist.
//!
//! # K-Space Temporal Correction Factor κ
//!
//! The k-space correction factor eliminates the temporal phase error of the
//! leapfrog scheme for all spatial frequencies simultaneously:
//! ```text
//!   κ[i,j,k] = sinc( 0.5 · c_ref · dt · |k| )
//!            = sin( 0.5 · c_ref · dt · |k| ) / ( 0.5 · c_ref · dt · |k| )
//! ```
//! where `|k| = sqrt(kx² + ky² + kz²)` and `sinc` is the unnormalized sinc.
//! Without κ, temporal phase error is O(dt²); with κ, the dispersion relation
//! becomes exactly ω = c_ref·|k| for all spatial frequencies.
//!
//! # References
//!
//! - Treeby, B.E. & Cox, B.T. (2010). k-Wave: MATLAB toolbox for the simulation
//!   and reconstruction of photoacoustic wave fields. J. Biomed. Opt. 15(2),
//!   021314. doi:10.1117/1.3360308  (Eqs. 14–17; §II.A κ factor)
//! - Liu, Q.-H. (1998). The PSTD algorithm: A time-domain method requiring only
//!   two cells per wavelength. Microwave Opt. Technol. Lett. 15(3), 158–165.
//!   (staggered-shift derivation)

use crate::math::fft::Complex64;
use ndarray::{Array1, Array3};
use std::f64::consts::PI;

/// Generate the 1-D staggered-shift operator for one axis.
///
/// Returns `(shift_pos, shift_neg)` where for each wavenumber bin `idx`:
/// ```text
///   shift_pos[idx] = i·k · exp(+i·k·ds/2)   (pressure → velocity direction)
///   shift_neg[idx] = i·k · exp(−i·k·ds/2)   (velocity → density direction)
/// ```
///
/// Wavenumbers are laid out in FFT order:
/// `k[idx] = (idx  ) · dk`  for `idx ≤ n/2`
/// `k[idx] = (idx−n) · dk`  for `idx >  n/2`
/// where `dk = 2π / (n · ds)`.
///
/// # Arguments
/// * `n`  — number of grid points in this dimension
/// * `dk` — wavenumber spacing = 2π / (n · ds)
/// * `ds` — grid spacing (Δx, Δy, or Δz)
///
/// # Panics
/// Never panics; all arithmetic is well-defined for all `n ≥ 1`.
#[must_use] 
pub fn generate_shift_1d(n: usize, dk: f64, ds: f64) -> (Array1<Complex64>, Array1<Complex64>) {
    let i_unit = Complex64::new(0.0, 1.0);
    let mut shift_pos = Array1::zeros(n);
    let mut shift_neg = Array1::zeros(n);

    for idx in 0..n {
        // Signed wavenumber in FFT order.
        // Nyquist bin (idx = n/2 for even n) uses k = +n/2 · dk = +π/ds.
        // Do NOT zero this bin — removing it eliminates ~18% of k-space energy.
        let signed = if idx <= n / 2 {
            idx as isize
        } else {
            idx as isize - n as isize
        };
        let k_val = dk * signed as f64;
        let exponent = k_val * ds * 0.5; // k · ds/2

        // exp(+i·k·ds/2) = cos(k·ds/2) + i·sin(k·ds/2)
        // exp(−i·k·ds/2) = cos(k·ds/2) − i·sin(k·ds/2)
        let cos_exp = exponent.cos();
        let sin_exp = exponent.sin();

        // i·k · exp(±i·k·ds/2)
        shift_pos[idx] = i_unit * Complex64::new(k_val, 0.0) * Complex64::new(cos_exp, sin_exp);
        shift_neg[idx] = i_unit * Complex64::new(k_val, 0.0) * Complex64::new(cos_exp, -sin_exp);
    }

    (shift_pos, shift_neg)
}

/// Generate the 3-D temporal k-space correction array κ.
///
/// ```text
///   κ[i,j,k] = sinc( 0.5 · c_ref · dt · |k| )
///            = sin( 0.5 · c_ref · dt · |k| ) / ( 0.5 · c_ref · dt · |k| )
/// ```
/// where `|k| = sqrt(kx² + ky² + kz²)` and `sinc` is the **unnormalized** sinc.
///
/// Applied in k-space before IFFT to both velocity and density updates.
/// Eliminates temporal phase error of the leapfrog scheme to machine precision
/// for all spatial frequencies (compared with O(dt²) without the correction):
/// the dispersion relation becomes exactly ω = c_ref·|k| for all |k|.
///
/// # Derivation
///
/// For the staggered-time leapfrog scheme on a spectral grid, substituting a
/// plane wave p ∝ exp(i(k·x − ω·t)) gives the numerical dispersion relation:
///
/// ```text
///   sin(ω·dt/2) = c_ref · dt/2 · |k| · κ
/// ```
///
/// Setting κ = sin(c_ref·dt/2·|k|) / (c_ref·dt/2·|k|) (unnormalized sinc)
/// forces sin(ω·dt/2) = sin(c_ref·dt/2·|k|), so ω = c_ref·|k| exactly.
///
/// **Note on sinc conventions**:
/// - k-Wave MATLAB uses `sinc(c_ref·k·dt/(2π))` where MATLAB sinc is
///   normalized (sin(πx)/(πx)), which is algebraically equal to the
///   unnormalized sinc `sin(x)/x` at `x = c_ref·k·dt/2`. ✓
/// - k-Wave Python `kspaceFirstOrder3D.py` uses `np.sinc(c_ref·k·dt/2)`
///   (normalized sinc of a different argument) but this is pre-processing
///   code only — the C++ binary computes κ internally and does NOT read
///   the Python-computed array from the HDF5 input file. The C++ binary
///   matches the MATLAB formula.
/// - For additive source injection, k-Wave uses `source_kappa = cos(x)`,
///   implemented separately in `generate_source_kappa`.
///
/// # Arguments
/// * `nx,ny,nz` — grid dimensions
/// * `dx,dy,dz` — grid spacings (m)
/// * `c_ref`    — reference sound speed used for the κ correction (m/s)
/// * `dt`       — time step (s)
///
/// # References
/// - Treeby & Cox (2010), §II.A, Eq. 13.
/// - k-Wave MATLAB kspaceFirstOrder3D.m: `ifftshift(sinc(c_ref*k*dt/(2*pi)))`
#[must_use] 
pub fn generate_kappa(
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    c_ref: f64,
    dt: f64,
) -> Array3<f64> {
    let dk_x = 2.0 * PI / (nx as f64 * dx);
    let dk_y = 2.0 * PI / (ny as f64 * dy);
    let dk_z = 2.0 * PI / (nz as f64 * dz);

    let mut kappa = Array3::zeros((nx, ny, nz));

    for i in 0..nx {
        let si = if i <= nx / 2 {
            i as isize
        } else {
            i as isize - nx as isize
        };
        let kx = dk_x * si as f64;
        for j in 0..ny {
            let sj = if j <= ny / 2 {
                j as isize
            } else {
                j as isize - ny as isize
            };
            let ky = dk_y * sj as f64;
            for k in 0..nz {
                let sk = if k <= nz / 2 {
                    k as isize
                } else {
                    k as isize - nz as isize
                };
                let kz = dk_z * sk as f64;
                let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();
                // Unnormalized sinc: sin(x)/x, with limit 1 as x→0
                let x = 0.5 * c_ref * dt * k_mag;
                kappa[[i, j, k]] = if x < 1e-10 { 1.0 } else { x.sin() / x };
            }
        }
    }

    kappa
}

/// Returns the k-space source-injection κ_src filter for additive sources.
///
/// ```text
///   κ_src[i,j,k] = cos( 0.5 · c_ref · dt · |k| )
/// ```
///
/// This is **different** from the propagation κ (`generate_kappa` uses
/// unnormalized sinc). The cos correction is derived from the requirement
/// that an additive source injected at time step n produces a wave whose
/// amplitude matches the analytic source signal — the cos factor accounts
/// for the half-step offset between the source-injection time and the
/// wave propagation evaluation time in the staggered leapfrog scheme.
///
/// Matches:
/// - k-Wave MATLAB: `source_kappa = ifftshift(cos(c_ref*k*dt/2))`
/// - k-Wave Python 3D: `source_kappa = np.fft.ifftshift(np.cos(c_ref*k*dt/2))`
///
/// # References
/// - Treeby & Cox (2010), §II.B (source term k-space correction).
#[must_use] 
pub fn generate_source_kappa(
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    c_ref: f64,
    dt: f64,
) -> Array3<f64> {
    let dk_x = 2.0 * PI / (nx as f64 * dx);
    let dk_y = 2.0 * PI / (ny as f64 * dy);
    let dk_z = 2.0 * PI / (nz as f64 * dz);

    let mut kappa = Array3::zeros((nx, ny, nz));

    for i in 0..nx {
        let si = if i <= nx / 2 {
            i as isize
        } else {
            i as isize - nx as isize
        };
        let kx = dk_x * si as f64;
        for j in 0..ny {
            let sj = if j <= ny / 2 {
                j as isize
            } else {
                j as isize - ny as isize
            };
            let ky = dk_y * sj as f64;
            for k in 0..nz {
                let sk = if k <= nz / 2 {
                    k as isize
                } else {
                    k as isize - nz as isize
                };
                let kz = dk_z * sk as f64;
                let k_mag = (kx * kx + ky * ky + kz * kz).sqrt();
                kappa[[i, j, k]] = (0.5 * c_ref * dt * k_mag).cos();
            }
        }
    }

    kappa
}
