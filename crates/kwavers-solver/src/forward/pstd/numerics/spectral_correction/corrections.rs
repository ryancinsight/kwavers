use kwavers_grid::Grid;
use ndarray::{Array3, Zip};
use std::f64::consts::PI;

use super::SpectralCorrectionMethod;
use kwavers_core::constants::numerical::TWO_PI;

/// Dispatch to the selected spectral correction method.
///
/// All five methods compute kappa as a 3D array of correction factors applied
/// to spectral derivatives.  Each element is independent of its neighbours, so
/// every function uses `Zip::indexed(...).par_for_each` for full Rayon
/// parallelism over the grid volume.
pub fn compute_spectral_correction_dispatch(
    grid: &Grid,
    method: SpectralCorrectionMethod,
    dt: f64,
    c_ref: f64,
    cfl_number: f64,
    max_correction: f64,
) -> Array3<f64> {
    match method {
        SpectralCorrectionMethod::ExactDispersion => {
            compute_exact_dispersion_correction(grid, dt, c_ref, max_correction)
        }
        SpectralCorrectionMethod::Treeby2010 => {
            compute_treeby2010_correction(grid, dt, c_ref, cfl_number, max_correction)
        }
        SpectralCorrectionMethod::LiuPSTD => {
            compute_liu_pstd_correction(grid, dt, c_ref, max_correction)
        }
        SpectralCorrectionMethod::LowDispersionPSTD => {
            compute_low_dispersion_pstd_correction(grid, dt, c_ref, max_correction)
        }
        SpectralCorrectionMethod::SincSpatial => compute_sinc_spatial_correction(grid),
    }
}

/// Exact dispersion correction: compensates the phase-velocity error of the
/// modified-wavenumber k_mod relative to the physical wavenumber k_phys.
///
/// ## Algorithm
///
/// For each spectral bin (i,j,k):
///
/// ```text
/// k_phys = √(kx² + ky² + kz²)
/// k_mod  = √(kx_mod² + ky_mod² + kz_mod²)   (modified wavenumber after FD stencil)
///
/// kx_mod = 2 sin(kx·dx/2) / dx
/// ω_phys = c_ref · k_phys
/// ω_num  = 2 asin(c_ref · dt · k_mod / 2) / dt
///
/// κ = ω_phys / ω_num    (clipped to [1/max, max])
/// ```
///
/// ## Theorem (race-freedom)
///
/// Each element `kappa[(i,j,k)]` depends only on `(i,j,k)` via
/// `compute_wavenumber_component` (pure function of the grid constants).
/// No two parallel tasks share a write target → `par_for_each` is race-free.
fn compute_exact_dispersion_correction(
    grid: &Grid,
    dt: f64,
    c_ref: f64,
    max_correction: f64,
) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);

    Zip::indexed(kappa.view_mut()).par_for_each(|(i, j, k), val| {
        let kx = compute_wavenumber_component(i, nx, dx);
        let ky = compute_wavenumber_component(j, ny, dy);
        let kz = compute_wavenumber_component(k, nz, dz);

        let k_phys = kz.mul_add(kz, kx.mul_add(kx, ky * ky)).sqrt();
        if k_phys <= 0.0 {
            return; // DC component: kappa = 1 (already initialised)
        }

        let kx_mod = 2.0 * (kx * dx / 2.0).sin() / dx;
        let ky_mod = 2.0 * (ky * dy / 2.0).sin() / dy;
        let kz_mod = 2.0 * (kz * dz / 2.0).sin() / dz;
        let k_mod = (kx_mod * kx_mod + ky_mod * ky_mod + kz_mod * kz_mod).sqrt();

        if k_mod <= 0.0 {
            return;
        }

        let omega_phys = c_ref * k_phys;
        let arg = c_ref * dt * k_mod / 2.0;

        if arg < 1.0 {
            let omega_num = 2.0 * arg.asin() / dt;
            let correction = omega_phys / omega_num;
            *val = correction.min(max_correction).max(1.0 / max_correction);
        } else {
            *val = 1.0 / max_correction;
        }
    });

    kappa
}

/// Treeby & Cox 2010 k-space correction κ = sinc(c_ref·dt·|k|/2).
///
/// This is the canonical kappa derived in Treeby B. E. & Cox B. T. (2010),
/// "Modeling power law absorption and dispersion for acoustic propagation
/// using the fractional Laplacian," J. Acoust. Soc. Am. 127(5) 2741-2748,
/// Eq. 18. With this kappa applied to the spatial derivative `i·k`, the
/// leapfrog k-space scheme reproduces the EXACT analytical wave-equation
/// update step for plane waves at `c = c_ref`:
///
/// ```text
/// p^{n+1} - 2*p^n + p^{n-1} = -4*sin^2(c*dt*|k|/2) * p^n   (in k-space)
/// ```
///
/// which matches the exact solution `p(t±dt) = exp(±i·c·|k|·dt) · p(t)`.
///
/// **Reference implementation**: k-wave-python's `kspace_solver.py` line 389:
///     `self.kappa = xp.sinc((self.c_ref * k_mag * self.dt / 2) / np.pi)`
///
/// where numpy's `sinc(x) = sin(πx)/(πx)`, so passing `(arg)/π` recovers the
/// unnormalised `sin(arg)/arg`. Equivalent to `sin(c·k·dt/2)/(c·k·dt/2)`.
fn compute_treeby2010_correction(
    grid: &Grid,
    dt: f64,
    c_ref: f64,
    _cfl: f64,
    max_correction: f64,
) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);

    Zip::indexed(kappa.view_mut()).par_for_each(|(i, j, k), val| {
        let kx = compute_wavenumber_component(i, nx, dx);
        let ky = compute_wavenumber_component(j, ny, dy);
        let kz = compute_wavenumber_component(k, nz, dz);

        let k_phys = kz.mul_add(kz, kx.mul_add(kx, ky * ky)).sqrt();
        if k_phys <= 0.0 {
            return; // DC: kappa = 1 (already initialised)
        }

        // arg = c_ref · dt · |k_phys| / 2
        // kappa = sinc(arg) = sin(arg) / arg  (unnormalised)
        let arg = 0.5 * c_ref * dt * k_phys;
        let correction = if arg.abs() < 1e-12 {
            1.0 // L'Hôpital: lim sin(x)/x = 1
        } else {
            arg.sin() / arg
        };
        *val = correction.min(max_correction).max(1.0 / max_correction);
    });

    kappa
}

/// Liu (1995) PSTD k-space correction.
///
/// Combines a Taylor-expansion low-k correction with a sinc-based high-k
/// correction to improve dispersion across the full wavenumber range of the
/// pseudospectral method.
///
/// ## Formula
///
/// ```text
/// For k·dx < π:   κ = 1 + (c·dt/dx)² · (k·dx)² / 24
/// Otherwise:       κ = (k·dx/2) / sin(k·dx/2)        [inverse sinc]
/// ```
///
/// Reference: Liu Y (1995). Geophysics 60(4), 1038–1042. DOI: 10.1190/1.1443854
fn compute_liu_pstd_correction(
    grid: &Grid,
    dt: f64,
    c_ref: f64,
    max_correction: f64,
) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);
    let dx_min = dx.min(dy).min(dz);
    let stability_factor = c_ref * dt / dx_min;

    Zip::indexed(kappa.view_mut()).par_for_each(|(i, j, k), val| {
        let kx = compute_wavenumber_component(i, nx, dx);
        let ky = compute_wavenumber_component(j, ny, dy);
        let kz = compute_wavenumber_component(k, nz, dz);

        let k_mag = kz.mul_add(kz, kx.mul_add(kx, ky * ky)).sqrt();
        if k_mag <= 0.0 {
            return; // DC: kappa = 1
        }

        let k_dx = k_mag * dx_min;
        let correction = if k_dx < PI {
            1.0 + stability_factor * stability_factor * k_dx * k_dx / 24.0
        } else {
            let sinc = (k_dx / 2.0).sin() / (k_dx / 2.0);
            1.0 / sinc
        };

        *val = correction.min(max_correction).max(1.0 / max_correction);
    });

    kappa
}

/// Low-dispersion PSTD correction: κ = c·dt·|k| / (2·sin(c·dt·|k|/2)).
///
/// This is the exact inverse of the temporal phase error introduced by the
/// leapfrog time-integration scheme.  It cancels the sin(ωΔt/2)/(ωΔt/2)
/// frequency-dependent amplitude attenuation, restoring dispersion-free
/// propagation for all resolved wavenumbers.
///
/// ## Theorem (cancellation of leapfrog dispersion)
///
/// The leapfrog scheme advances as exp(i·ω_num·dt) where sin(ω_num·dt/2) =
/// sin(c·|k|·dt/2).  Multiplying the spectral derivative by κ = c·dt·|k| /
/// (2·sin(c·|k|·dt/2)) restores the exact phase velocity c for all |k| ≤ π/dx.
fn compute_low_dispersion_pstd_correction(
    grid: &Grid,
    dt: f64,
    c_ref: f64,
    max_correction: f64,
) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);

    Zip::indexed(kappa.view_mut()).par_for_each(|(i, j, k), val| {
        let kx = compute_wavenumber_component(i, nx, dx);
        let ky = compute_wavenumber_component(j, ny, dy);
        let kz = compute_wavenumber_component(k, nz, dz);

        let k_mag = kz.mul_add(kz, kx.mul_add(kx, ky * ky)).sqrt();
        if k_mag <= 0.0 {
            return; // DC: kappa = 1
        }

        let omega_dt = c_ref * dt * k_mag;
        let correction = if omega_dt > 1e-12 {
            omega_dt / (2.0 * (omega_dt / 2.0).sin())
        } else {
            1.0 // L'Hôpital limit as ω·dt → 0
        };

        *val = correction.min(max_correction).max(1.0 / max_correction);
    });

    kappa
}

/// Sinc spatial correction: 1 / sinc(kx·dx/2) × 1/sinc(ky·dy/2) × 1/sinc(kz·dz/2).
///
/// Corrects for the sinc-like spectral response of a nearest-neighbour grid
/// interpolation or zero-order-hold source injection.  Each axis contributes
/// an independent sinc factor; the correction is their product.
///
/// ## Theorem (axis-independence)
///
/// Because the correction factorises over axes, the product-form computation
/// `1 / (sinc_x · sinc_y · sinc_z)` is mathematically equivalent to
/// three independent 1D sinc corrections applied sequentially.  The parallel
/// `Zip::indexed` evaluation is therefore correct by independence of axes.
fn compute_sinc_spatial_correction(grid: &Grid) -> Array3<f64> {
    let mut kappa = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0);
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);

    Zip::indexed(kappa.view_mut()).par_for_each(|(i, j, k), val| {
        let kx = compute_wavenumber_component(i, nx, dx);
        let ky = compute_wavenumber_component(j, ny, dy);
        let kz = compute_wavenumber_component(k, nz, dz);

        let sinc_x = if kx.abs() > 1e-12 {
            let arg = kx * dx / 2.0;
            arg.sin() / arg
        } else {
            1.0
        };
        let sinc_y = if ky.abs() > 1e-12 {
            let arg = ky * dy / 2.0;
            arg.sin() / arg
        } else {
            1.0
        };
        let sinc_z = if kz.abs() > 1e-12 {
            let arg = kz * dz / 2.0;
            arg.sin() / arg
        } else {
            1.0
        };

        *val = 1.0 / (sinc_x * sinc_y * sinc_z);
    });

    kappa
}

/// Compute the signed wavenumber component for FFT bin `index` in a domain of
/// `n` points with grid spacing `dx`.
///
/// ## Formula (standard DFT wavenumber layout)
///
/// ```text
/// k[n]  = 2π·n / (N·dx)          for n = 0, 1, …, N/2
/// k[n]  = 2π·(n − N) / (N·dx)    for n = N/2+1, …, N−1
/// ```
///
/// This is the standard FFT wavenumber ordering where bins 0..N/2 are
/// non-negative frequencies and bins N/2+1..N−1 are negative frequencies.
#[inline]
pub(super) fn compute_wavenumber_component(index: usize, n: usize, dx: f64) -> f64 {
    if index <= n / 2 {
        TWO_PI * index as f64 / (n as f64 * dx)
    } else {
        TWO_PI * (index as f64 - n as f64) / (n as f64 * dx)
    }
}

/// Apply the pre-computed kappa correction to a complex spectral field in-place.
///
/// ## Theorem (race-freedom)
///
/// Each element of `field_k` is multiplied by the corresponding scalar in
/// `kappa`; no two Rayon tasks share a memory location.
pub fn apply_correction(field_k: &mut Array3<num_complex::Complex<f64>>, kappa: &Array3<f64>) {
    Zip::from(field_k).and(kappa).par_for_each(|f, &k| {
        *f *= k;
    });
}
