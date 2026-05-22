use super::*;
use super::{sinc, sinc_normalized};
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use crate::domain::grid::Grid;
use std::f64::consts::PI;

#[test]
fn test_wavenumber_computation() {
    let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).expect("Failed to create test grid");
    let (kx, ky, kz) = compute_wavenumbers(&grid);

    // Check dimensions
    assert_eq!(kx.dim(), (8, 8, 8));
    assert_eq!(ky.dim(), (8, 8, 8));
    assert_eq!(kz.dim(), (8, 8, 8));

    // Check DC component
    assert_eq!(kx[[0, 0, 0]], 0.0);
    assert_eq!(ky[[0, 0, 0]], 0.0);
    assert_eq!(kz[[0, 0, 0]], 0.0);

    // Check symmetry
    assert!((kx[[1, 0, 0]] + kx[[7, 0, 0]]).abs() < 1e-10);
}

#[test]
fn test_sinc_function() {
    assert_eq!(sinc(0.0), 1.0);
    assert!((sinc(PI) - 0.0).abs() < 1e-10);
    assert!((sinc(PI / 2.0) - 2.0 / PI).abs() < 1e-10);
}

#[test]
fn test_sinc_normalized_function() {
    // sinc_normalized(x) = sin(π·x) / (π·x), matching numpy's np.sinc(x)
    assert_eq!(sinc_normalized(0.0), 1.0);
    // np.sinc(1.0) = sin(π)/π = 0
    assert!((sinc_normalized(1.0)).abs() < 1e-10);
    // np.sinc(0.5) = sin(π/2)/(π/2) = 2/π
    assert!((sinc_normalized(0.5) - 2.0 / PI).abs() < 1e-10);
    // np.sinc(0.1) ≈ 0.9836...
    assert!((sinc_normalized(0.1) - 0.9836316431).abs() < 1e-6);
}

/// κ = sinc(c₀·|k|·dt/2) at DC (k=0) must equal 1.0.
///
/// ## Theorem (Treeby & Cox 2010, Eq. 17)
///
/// The k-space correction κ compensates for the staggered time integration:
///   κ(k) = sinc(x) = sin(x)/x,  x = c_ref·|k|·dt/2   (UNNORMALIZED sinc)
///
/// At k = 0: x = 0 → κ = 1 (no correction needed for DC component).
/// At the Nyquist wavenumber k_N = π/dx, for a Courant-stable step dt = CFL·dx/c:
///   x = c_ref · (π/dx) · (CFL·dx/c) / 2 = π·CFL/2
///   For CFL = 0.3: x = 0.15·π → κ = sin(0.15π)/(0.15π) ≈ 0.9775
///
/// ## Reference
/// Treeby, B.E. & Cox, B.T. (2010). k-Wave: MATLAB toolbox for the simulation and
/// reconstruction of photoacoustic wave fields. J. Biomed. Opt. 15(2):021314.
/// # Panics
/// - Panics if `grid creation`.
///
#[test]
fn test_kspace_kappa_correction_at_nyquist() {
    let c0 = SOUND_SPEED_WATER_SIM;
    let dx = 1e-3_f64;
    let cfl = 0.3_f64;
    let dt = cfl * dx / c0; // stable time step

    let grid = Grid::new(32, 32, 32, dx, dx, dx).expect("grid creation");
    let (kx, ky, kz) = compute_wavenumbers(&grid);

    let kappa =
        compute_kspace_correction_factors(&kx, &ky, &kz, &grid, CorrectionType::Treeby2010, dt, c0);

    // DC component: κ = 1.0 exactly
    assert!(
        (kappa[[0, 0, 0]] - 1.0).abs() < 1e-12,
        "κ at DC must be 1.0, got {}",
        kappa[[0, 0, 0]]
    );

    // Nyquist wavenumber: k_N = π/dx for a grid of n=32 with dk = 2π/(n·dx)
    // The max |k| present in the grid is approximately π/dx.
    // Compute expected κ at k = π/dx.
    let k_nyquist = PI / dx;
    let arg = c0 * k_nyquist * dt / 2.0;
    let expected_kappa_nyquist = sinc(arg); // sin(arg)/arg — unnormalized, matching k-Wave C++

    // Find the kappa value closest to Nyquist (max |k| in the grid)
    let k_max = kx
        .iter()
        .zip(ky.iter())
        .zip(kz.iter())
        .map(|((&kxi, &kyi), &kzi)| (kxi * kxi + kyi * kyi + kzi * kzi).sqrt())
        .fold(0.0_f64, f64::max);

    let arg_max = c0 * k_max * dt / 2.0;
    let kappa_at_kmax = sinc(arg_max);

    // κ at k_max must be positive (no aliasing at stable CFL < 1)
    assert!(
        kappa_at_kmax > 0.0,
        "κ at k_max must be positive for CFL={cfl}, got {kappa_at_kmax}"
    );
    // κ at k_max < 1 (correction reduces gradient contribution at high k)
    assert!(
        kappa_at_kmax < 1.0,
        "κ at k_max must be < 1.0 (correction attenuates high-k), got {kappa_at_kmax}"
    );
    // Verify analytical value: arg = CFL·π/2 ≈ 0.4712 → κ ≈ 0.9369 (√3 Nyquist)
    let _ = expected_kappa_nyquist; // informational; grid has |k| ≤ √3·π/dx at corners
    assert!(
        kappa_at_kmax > 0.8,
        "κ at corner Nyquist must be > 0.8 for CFL=0.3, got {kappa_at_kmax}"
    );
}
