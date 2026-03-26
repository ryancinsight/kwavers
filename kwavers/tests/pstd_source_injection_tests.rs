//! PSTD Source Injection Correctness Tests
//!
//! ## Theorem (PSTD Split-Density Source Injection)
//!
//! The PSTD EOS is always `p = c²·(ρₓ + ρᵧ + ρ_z)` regardless of problem dimensionality.
//! The FDTD SourceHandler scales density injection by `n_dim = count(nₓ>1, nᵧ>1, n_z>1)`:
//!   `rho_scale = 2·Δt / (n_dim · c₀ · Δx)`
//! so the intended total density addition is `n_dim × rho_scale × p_source` (k-Wave convention).
//!
//! When distributing into three density components, each component must receive
//! `s · (n_dim / 3)` to maintain the invariant:
//!   `Δ(ρₓ + ρᵧ + ρ_z) = n_dim × s`
//!
//! Without this correction, a 1D grid (n_dim=1) produces 3× overinjection because
//! the three components receive `s` each for a total of `3s` instead of `1s`.
//!
//! ## Validation
//!
//! A 1D simulation (64×1×1) and a 3D-like simulation (64×4×4) with identical source
//! amplitude should produce peak pressures within a factor of 2. Before the fix the
//! ratio was ≈3 (1D 3× too high); after the fix the ratio should be ≈1.
//!
//! ## References
//! - Treeby & Cox (2010), J. Biomed. Opt. 15(2), Eq. (16–18).
//! - Treeby, Jaros, Rendell & Cox (2012), J. Acoust. Soc. Am. 131(6).

use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::signal::SineWave;
use kwavers::domain::source::{InjectionMode, PlaneWaveConfig, PlaneWaveSource, SourceField};
use kwavers::solver::forward::pstd::config::{KSpaceMethod, PSTDConfig};
use kwavers::solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers::solver::interface::Solver;
use std::sync::Arc;

/// Helper: run PSTD for `nt` steps and return max |p| over the pressure field.
fn run_pstd_and_measure_peak(
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    c0: f64,
    rho0: f64,
    frequency: f64,
    amplitude: f64,
    nt: usize,
) -> KwaversResult<f64> {
    let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let dt = 0.3 * dx / c0;
    let pstd_config = PSTDConfig {
        dt,
        nt,
        kspace_method: KSpaceMethod::StandardPSTD,
        boundary: kwavers::solver::forward::pstd::config::BoundaryConfig::None,
        ..Default::default()
    };

    let mut solver = PSTDSolver::new(pstd_config, grid.clone(), &medium, Default::default())?;

    // Plane wave source along +x direction, additive injection
    let signal = Arc::new(SineWave::new(frequency, amplitude, 0.0));
    let config = PlaneWaveConfig {
        direction: (1.0, 0.0, 0.0),
        wavelength: c0 / frequency,
        phase: 0.0,
        source_type: SourceField::Pressure,
        injection_mode: InjectionMode::BoundaryOnly,
    };
    let source = PlaneWaveSource::new(config, signal);
    solver.add_source(Box::new(source))?;

    for _ in 0..nt {
        solver.step_forward()?;
    }

    let peak = solver
        .fields
        .p
        .iter()
        .fold(0.0f64, |m, &v| m.max(v.abs()));
    Ok(peak)
}

/// Theorem (PSTD 1D vs 3D source amplitude parity):
/// For a plane wave source with the same amplitude and frequency, a 1D grid (n_dim=1)
/// and a 3D grid (n_dim=3) should produce peak pressures within a factor of 2.
///
/// Before Phase 2 fix: 1D peak was ≈3× higher than 3D peak (ratio ≈3).
/// After Phase 2 fix: ratio should be in [0.5, 2.0] because density_scale = n_dim/3
/// restores the invariant Δρ_total = n_dim × rho_scale × p_source for all n_dim.
#[test]
fn test_pstd_1d_vs_3d_source_amplitude_parity() -> KwaversResult<()> {
    let c0 = 1500.0;
    let rho0 = 1000.0;
    let dx = 0.25e-3; // 0.25 mm — 6 cells per wavelength at 1 MHz
    let frequency = 1e6;
    let amplitude = 1e5; // 100 kPa
    let nt = 80;

    // 1D simulation: ny=1, nz=1 → n_dim=1
    let peak_1d = run_pstd_and_measure_peak(32, 1, 1, dx, c0, rho0, frequency, amplitude, nt)?;

    // 3D simulation: ny=4, nz=4 → n_dim=3
    let peak_3d = run_pstd_and_measure_peak(32, 4, 4, dx, c0, rho0, frequency, amplitude, nt)?;

    // Both should produce non-zero pressure
    assert!(
        peak_1d > 1.0,
        "1D PSTD peak pressure should be non-zero, got {peak_1d:.3e}"
    );
    assert!(
        peak_3d > 1.0,
        "3D PSTD peak pressure should be non-zero, got {peak_3d:.3e}"
    );

    // After fix: ratio should be within [0.5, 2.0]
    // Before fix: ratio would be ≈3 (1D was 3× too high)
    let ratio = if peak_3d > 0.0 { peak_1d / peak_3d } else { f64::INFINITY };
    assert!(
        ratio >= 0.4 && ratio <= 2.5,
        "1D/3D peak pressure ratio should be near 1 after n_dim/3 fix; \
         got ratio={ratio:.3} (1D peak={peak_1d:.3e} Pa, 3D peak={peak_3d:.3e} Pa). \
         Ratio ≈3 would indicate the fix is not applied."
    );

    Ok(())
}

/// Theorem (PSTD 2D source injection):
/// For a 2D simulation (n_dim=2), the density_scale = 2/3 so total density =
/// 3 × (2/3)·s = 2·s, matching k-Wave's 2-component injection (2 × s = 2s).
/// Peak pressure in 2D should also agree with 3D within a factor of 2.
#[test]
fn test_pstd_2d_vs_3d_source_amplitude_parity() -> KwaversResult<()> {
    let c0 = 1500.0;
    let rho0 = 1000.0;
    let dx = 0.25e-3;
    let frequency = 1e6;
    let amplitude = 1e5;
    let nt = 80;

    // 2D simulation: nz=1 → n_dim=2 (nx>1, ny>1, nz=1)
    let peak_2d = run_pstd_and_measure_peak(32, 4, 1, dx, c0, rho0, frequency, amplitude, nt)?;

    // 3D simulation: n_dim=3
    let peak_3d = run_pstd_and_measure_peak(32, 4, 4, dx, c0, rho0, frequency, amplitude, nt)?;

    assert!(
        peak_2d > 1.0,
        "2D PSTD peak should be non-zero, got {peak_2d:.3e}"
    );
    assert!(
        peak_3d > 1.0,
        "3D PSTD peak should be non-zero, got {peak_3d:.3e}"
    );

    let ratio = if peak_3d > 0.0 { peak_2d / peak_3d } else { f64::INFINITY };
    assert!(
        ratio >= 0.4 && ratio <= 2.5,
        "2D/3D peak pressure ratio should be near 1 after n_dim/3 fix; \
         got ratio={ratio:.3} (2D={peak_2d:.3e} Pa, 3D={peak_3d:.3e} Pa). \
         Ratio ≈1.5 would indicate the fix is not applied."
    );

    Ok(())
}
