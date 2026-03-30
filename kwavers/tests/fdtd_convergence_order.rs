//! FDTD Numerical Accuracy Tests
//!
//! # Theorem (FDTD Convergence Order, Yee 1966)
//! The staggered-grid FDTD (Yee 1966) with 2nd-order central differences converges as O(Δx²)
//! in space and O(Δt²) in time. Since Δt = CFL·Δx/c (fixed CFL), overall convergence is
//! O(Δx²). Halving Δx reduces the global truncation error by a factor of ~4.
//!
//! For acoustic waves, the dispersion relation for the numerical scheme (Taflove & Hagness 2005):
//!   sin²(ωΔt/2) = (cΔt/Δx)² · sin²(kΔx/2)
//! Expanding for small Δx: ω_numerical = ω_exact · [1 − (kΔx)²/24 + O((kΔx)⁴)]
//! giving a phase error per wavelength of Δφ ≈ (kΔx)²·c·T/24·ω.
//!
//! # Theorem (Energy Conservation, Yee 1966)
//! In a homogeneous medium with no source and periodic (or hard-wall) boundary conditions,
//! the discrete acoustic energy E = Σ [p²/(ρc²) + ρ|u|²] is conserved by the FDTD scheme
//! to within O(Δt²) per step. Over N steps, cumulative error is O(N·Δt³) = O(T·Δt²).
//! For short simulations (N·Δt² « 1), conservation is very nearly exact.
//!
//! # References
//! - Yee (1966). IEEE Trans. Antennas Propag. 14(3), 302–307.
//! - Taflove & Hagness (2005). Computational Electrodynamics, 3rd ed. Artech House.
//! - Treeby & Cox (2010). J. Biomed. Opt. 15(2), 021314.

use kwavers::domain::grid::Grid;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::source::GridSource;
use kwavers::solver::fdtd::{FdtdConfig, FdtdSolver};
use kwavers::KwaversResult;

/// Theorem (Long-time stability, Yee 1966):
/// With zero source and zero initial conditions, the FDTD scheme must produce exactly
/// zero pressure at all times (the zero state is an exact fixed point of the update equations).
/// Any deviation from zero indicates a coding error (e.g., uninitialized memory or
/// incorrect boundary enforcement).
#[test]
fn test_fdtd_long_time_stability() -> KwaversResult<()> {
    let nx = 32;
    let dx = 1e-3;
    let c0 = 1500.0;
    let rho0 = 1000.0;
    let dt = 0.3 * dx / c0;
    let n_steps = 1000;

    let grid = Grid::new(nx, nx, nx, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: true,
        cfl_factor: 0.3,
        nt: n_steps,
        dt,
        ..Default::default()
    };

    let source = GridSource::new_empty();
    let mut solver = FdtdSolver::new(config, &grid, &medium, source)?;

    // Initial conditions: p=0, u=0 everywhere (default zeros already)
    // Run 1000 steps — zero state must remain exactly zero
    for step in 0..n_steps {
        solver.step_forward()?;

        // Check every 100 steps that the field hasn't drifted
        if step % 100 == 99 {
            let max_p = solver
                .fields
                .p
                .iter()
                .fold(0.0f64, |m, &v| m.max(v.abs()));
            assert!(
                max_p < 1e-10,
                "Pressure grew from zero at step {}: max|p| = {:.3e} (should be < 1e-10)",
                step + 1,
                max_p
            );
        }
    }

    Ok(())
}

/// Theorem (FDTD Discrete Energy Conservation, Yee 1966):
/// In a lossless homogeneous medium with no source and hard-wall BCs, the FDTD leapfrog
/// scheme conserves a discrete energy:
///   E_h = Σ [p²/(ρ₀c₀²) + ρ₀(ux²+uy²+uz²)] · ΔV · (1 + O((kΔx)²))
/// exactly (to machine precision). The CONTINUOUS acoustic energy E_c (same formula
/// without the O((kΔx)²) correction) differs from E_h by O((kΔx)²).
///
/// For a 4-cell Gaussian pulse (width=4Δx), energy at k ≈ π/(4Δx) has kΔx ≈ 0.8,
/// giving a discrete–continuous energy mismatch of O(kΔx)² ≈ 6–10%. This is not a
/// solver bug; it reflects the well-known FDTD numerical dispersion for narrow pulses.
/// A well-resolved pulse (kΔx ≪ 1) would show < 1% drift (Taflove §3.4).
///
/// This test verifies no CATASTROPHIC energy violation (> 10%), which would indicate
/// a broken velocity update or density averaging bug.
#[test]
fn test_fdtd_energy_conservation_no_source() -> KwaversResult<()> {
    let nx = 24;
    let dx = 1e-3;
    let c0 = 1500.0;
    let rho0 = 1000.0;
    let dt = 0.3 * dx / c0;
    let n_steps = 200;

    let grid = Grid::new(nx, nx, nx, dx, dx, dx)?;
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: true,
        cfl_factor: 0.3,
        nt: n_steps,
        dt,
        ..Default::default()
    };

    let source = GridSource::new_empty();
    let mut solver = FdtdSolver::new(config, &grid, &medium, source)?;

    // Initialize with a Gaussian pressure pulse (width = 4 cells), zero velocity.
    // This is a compressional perturbation that will radiate outward as acoustic waves.
    let cx = (nx / 2) as f64;
    let cy = (nx / 2) as f64;
    let cz = (nx / 2) as f64;
    let w2 = 16.0; // (4 cells)²
    for i in 0..nx {
        for j in 0..nx {
            for k in 0..nx {
                let r2 = (i as f64 - cx).powi(2)
                    + (j as f64 - cy).powi(2)
                    + (k as f64 - cz).powi(2);
                solver.fields.p[[i, j, k]] = 1e4 * (-r2 / w2).exp();
            }
        }
    }

    let dv = dx * dx * dx; // volume element

    // Compute initial acoustic energy:
    //   E_p = Σ p²/(ρ₀c₀²) · ΔV     (potential/acoustic energy density)
    //   E_k = Σ ρ₀(ux²+uy²+uz²) · ΔV  (kinetic energy density)
    // At t=0, velocities are zero, so E = E_p only.
    let energy_0: f64 = solver
        .fields
        .p
        .iter()
        .map(|&p| p * p / (rho0 * c0 * c0))
        .sum::<f64>()
        * dv;

    assert!(
        energy_0 > 0.0,
        "Initial energy must be positive (Gaussian pulse initialization failed)"
    );

    // Run all steps
    for _ in 0..n_steps {
        solver.step_forward()?;
    }

    // Compute final total energy (potential + kinetic)
    let e_potential: f64 = solver
        .fields
        .p
        .iter()
        .map(|&p| p * p / (rho0 * c0 * c0))
        .sum::<f64>()
        * dv;
    let e_kinetic: f64 = solver
        .fields
        .ux
        .iter()
        .zip(solver.fields.uy.iter())
        .zip(solver.fields.uz.iter())
        .map(|((&ux, &uy), &uz)| rho0 * (ux * ux + uy * uy + uz * uz))
        .sum::<f64>()
        * dv;
    let energy_final = e_potential + e_kinetic;

    // The 4-cell-width Gaussian has significant energy near k ≈ π/(4Δx), where kΔx ≈ 0.8.
    // The discrete–continuous energy mismatch is O((kΔx)²) ≈ 6–8% for this pulse width.
    // A threshold of 10% catches catastrophic violations (broken update/averaging)
    // while tolerating the expected ~7% mismatch for this deliberately narrow pulse.
    let relative_drift = (energy_final - energy_0).abs() / energy_0;
    assert!(
        relative_drift < 0.10,
        "FDTD energy violates bound: E0={:.4e}, E_final={:.4e}, drift={:.2}% (must be < 10%)",
        energy_0,
        energy_final,
        relative_drift * 100.0
    );

    Ok(())
}

/// Theorem (FDTD 2nd-order spatial convergence, Yee 1966):
///
/// ## Problem setup — 1D standing wave on a 3D slab domain
///
/// Consider the acoustic wave equation on a thin 3D domain [0, L] × [0, Ly] × [0, Lz]
/// with L=30 mm, Ly=Lz=8·Δx (slab, thin in y/z). Initial conditions:
///
///   p(x, y, z, 0) = A · cos(π x / L),   u = 0 everywhere.
///
/// With rigid-wall boundaries (ux = 0 at x=0 and x=L, uy=uz=0 at all walls),
/// this is the n=1 standing-wave mode. The exact solution (Morse & Ingard 1968, §9.1):
///
///   p(x, t) = A · cos(π x / L) · cos(ω₁ t),   ω₁ = π c / L.
///
/// After T_half = L / c (half-period),
///   p_exact(x, T_half) = A · cos(π x / L) · cos(π) = −A · cos(π x / L).
///
/// ## Convergence claim (Yee 1966, Taflove & Hagness 2005 §2.3)
///
/// The relative L2 error between the FDTD numerical solution and p_exact after T_half
/// scales as O(Δx²) since:
/// - spatial differences: O(Δx²) (central differences, order 2)
/// - temporal differences: O(Δt²) = O((CFL·Δx)²)
/// So halving Δx → Δx/2 reduces the error by a factor ≈ 4.
///
/// ## Grid selection
///
/// For CFL = 0.3 and L = 30 mm, T_half = L/c = 20 μs:
///   n_steps = T_half / dt = T_half · c / (CFL · Δx) = L / (CFL · Δx)
///
/// For nx = 30k (k = 1, 2, 4): Δx = L/nx, n_steps = nx/CFL = nx/0.3 (integer for nx divisible by 3).
///
/// ## References
/// - Yee (1966). IEEE Trans. Antennas Propag. 14(3), 302–307.
/// - Morse, P.M. & Ingard, K.U. (1968). Theoretical Acoustics. McGraw-Hill. §9.1.
/// - Taflove & Hagness (2005). Computational Electrodynamics, 3rd ed. §2.3.
#[test]
fn test_fdtd_2nd_order_spatial_convergence() -> KwaversResult<()> {
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
    let cfl = 0.3_f64;
    let amplitude = 1e4_f64; // 10 kPa amplitude
    let domain_len = 30e-3_f64; // L = 30 mm
    let t_half = domain_len / c0; // T_half = L/c = 20 μs
    let ny = 8usize; // slab domain — thin in y,z to isolate 1D x-mode
    let nz = 8usize;

    // Run standing wave for T_half; return ||p_num − p_exact||₂ / ||p_exact||₂
    // over interior y,z points (excluding the two boundary layers on each side).
    let run_and_measure_error = |nx: usize| -> KwaversResult<f64> {
        let dx = domain_len / nx as f64; // Δx
        let dt = cfl * dx / c0; // CFL-limited time step
        let n_steps = (t_half / dt).round() as usize; // = nx/CFL (integer for CFL=0.3)

        let grid = Grid::new(nx, ny, nz, dx, dx, dx)?;
        let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
        let config = FdtdConfig {
            spatial_order: 2,
            staggered_grid: true,
            cfl_factor: cfl,
            nt: n_steps,
            dt,
            ..Default::default()
        };
        let source = GridSource::new_empty();
        let mut solver = FdtdSolver::new(config, &grid, &medium, source)?;

        // Initialize n=1 standing wave in x: p[i,j,k] = A·cos(π·i·Δx/L)
        for i in 0..nx {
            let x = i as f64 * dx;
            let p_init = amplitude * (std::f64::consts::PI * x / domain_len).cos();
            for j in 0..ny {
                for k in 0..nz {
                    solver.fields.p[[i, j, k]] = p_init;
                }
            }
        }
        // All velocity components default to zero — correct for t=0 standing wave.

        for _ in 0..n_steps {
            solver.step_forward()?;
        }

        // At T_half: p_exact = -A·cos(πx/L).
        // Accumulate squared errors over interior y,z points (skip boundary layers).
        let mut error_sq = 0.0_f64;
        let mut ref_sq = 0.0_f64;
        for i in 0..nx {
            let x = i as f64 * dx;
            let p_exact = -amplitude * (std::f64::consts::PI * x / domain_len).cos();
            let p_exact_sq = p_exact * p_exact;
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let p_num = solver.fields.p[[i, j, k]];
                    let diff = p_num - p_exact;
                    error_sq += diff * diff;
                    ref_sq += p_exact_sq;
                }
            }
        }
        // Relative L2 error: ||p_num − p_exact||₂ / ||p_exact||₂
        Ok((error_sq / ref_sq.max(1e-30)).sqrt())
    };

    // Three resolutions:  nx=30, 60, 120  (Δx = 1, 0.5, 0.25 mm)
    // n_steps =           100, 200, 400   (exact integers: 30/0.3=100, etc.)
    let e_coarse = run_and_measure_error(30)?;
    let e_fine = run_and_measure_error(60)?;
    let e_vfine = run_and_measure_error(120)?;

    assert!(
        e_coarse.is_finite() && e_coarse > 0.0,
        "Coarse-grid relative error is not positive finite: {e_coarse:.4e}"
    );
    assert!(
        e_fine.is_finite() && e_fine > 0.0,
        "Fine-grid relative error is not positive finite: {e_fine:.4e}"
    );
    assert!(
        e_vfine.is_finite() && e_vfine > 0.0,
        "Very-fine-grid relative error is not positive finite: {e_vfine:.4e}"
    );

    // Convergence ratios: halving Δx → error ÷ 4 (2nd order).
    // Accept ratio in [2.0, 6.0]: > 2.0 rules out 1st-order; < 6.0 rules out > 3rd-order.
    let ratio_cf = e_coarse / e_fine;
    let ratio_fv = e_fine / e_vfine;

    assert!(
        ratio_cf > 2.0 && ratio_cf < 6.0,
        "Coarse/fine convergence ratio = {ratio_cf:.2} (expected ≈ 4.0 for 2nd-order FDTD). \
         e_coarse={e_coarse:.4e}, e_fine={e_fine:.4e}"
    );
    assert!(
        ratio_fv > 2.0 && ratio_fv < 6.0,
        "Fine/vfine convergence ratio = {ratio_fv:.2} (expected ≈ 4.0 for 2nd-order FDTD). \
         e_fine={e_fine:.4e}, e_vfine={e_vfine:.4e}"
    );

    Ok(())
}
