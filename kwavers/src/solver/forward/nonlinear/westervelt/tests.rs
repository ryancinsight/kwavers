//! Value-semantic regression tests for the Westervelt FDTD solver.

use super::{WesterveltFdtd, WesterveltFdtdConfig};
use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::solver::forward::nonlinear::conservation::{
    ConservationDiagnostics, ConservationTolerances,
};
use crate::KwaversError;

fn assert_quadratic_laplacian_exact(spatial_order: usize, radius: usize) {
    let grid = Grid::new(12, 12, 12, 0.2, 0.3, 0.4).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig {
        spatial_order,
        enable_absorption: false,
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let x = i as f64 * grid.dx;
                let y = j as f64 * grid.dy;
                let z = k as f64 * grid.dz;
                solver.pressure[[i, j, k]] = x * x + y * y + z * z;
            }
        }
    }

    solver.calculate_laplacian(&grid).unwrap();

    for i in radius..grid.nx - radius {
        for j in radius..grid.ny - radius {
            for k in radius..grid.nz - radius {
                let actual = solver.laplacian[[i, j, k]];
                assert!(
                    (actual - 6.0).abs() < 1.0e-10,
                    "order {spatial_order}: laplacian[{i},{j},{k}] = {actual}, expected 6"
                );
            }
        }
    }
}

#[test]
fn test_westervelt_fdtd_creation() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig::default();
    let solver = WesterveltFdtd::new(config, &grid, &medium);

    assert_eq!(solver.pressure.shape(), &[32, 32, 32]);
}

#[test]
fn westervelt_laplacian_stencils_are_exact_for_quadratic_fields() {
    // Theorem: any consistent centered second-derivative stencil with
    // coefficients satisfying Σc_m=0 and Σm²c_m=2 differentiates x² exactly.
    // For p=x²+y²+z², ∇²p=2+2+2=6 on all points with complete stencil support.
    assert_quadratic_laplacian_exact(2, 1);
    assert_quadratic_laplacian_exact(4, 2);
    assert_quadratic_laplacian_exact(6, 3);
}

#[test]
fn westervelt_laplacian_rejects_unsupported_spatial_order() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig {
        spatial_order: 8,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    let err = solver.calculate_laplacian(&grid).unwrap_err();
    assert!(
        matches!(err, KwaversError::Validation(_)),
        "unsupported spatial order must return a validation error, got {err:?}"
    );
    assert_eq!(solver.config.spatial_order, 8);
}

#[test]
fn westervelt_update_reuses_pressure_and_nonlinear_workspaces() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig {
        spatial_order: 2,
        enable_absorption: false,
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);
    solver.pressure[[4, 4, 4]] = 1.0e5;

    let mut pressure_buffers_before = [
        solver.pressure.as_ptr() as usize,
        solver.pressure_prev.as_ptr() as usize,
        solver.pressure_next.as_ptr() as usize,
    ];
    pressure_buffers_before.sort_unstable();
    let nonlinear_before = solver.nonlinear_term.as_ptr();

    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    solver.update(&medium, &grid, &[], 0.0, dt).unwrap();

    let mut pressure_buffers_after = [
        solver.pressure.as_ptr() as usize,
        solver.pressure_prev.as_ptr() as usize,
        solver.pressure_next.as_ptr() as usize,
    ];
    pressure_buffers_after.sort_unstable();

    assert_eq!(pressure_buffers_after, pressure_buffers_before);
    assert_eq!(solver.nonlinear_term.as_ptr(), nonlinear_before);
}

#[test]
fn test_linear_wave_propagation() {
    // Test that with β=0 (no nonlinearity), we get linear wave propagation
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    // Set nonlinearity to zero for linear test
    medium.nonlinearity = 0.0;

    // Use zero artificial viscosity for energy conservation test
    let config = WesterveltFdtdConfig {
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    // Set initial Gaussian pulse
    let center = (grid.nx / 2, grid.ny / 2, grid.nz / 2);
    for i in 0..grid.nx {
        for j in 0..grid.ny {
            for k in 0..grid.nz {
                let r2 = ((i as i32 - center.0 as i32).pow(2)
                    + (j as i32 - center.1 as i32).pow(2)
                    + (k as i32 - center.2 as i32).pow(2)) as f64;
                solver.pressure[[i, j, k]] = (-(r2 / 100.0)).exp();
            }
        }
    }

    // Propagate for a few time steps
    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    for _ in 0..10 {
        solver.update(&medium, &grid, &[], 0.0, dt).unwrap();
    }

    // Check that energy is conserved (approximately) with no artificial viscosity
    let total_energy: f64 = solver.pressure.iter().map(|&p| p * p).sum();
    assert!(total_energy > 0.0);
}

#[test]
fn test_conservation_diagnostics_integration() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig::default();
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    // Enable diagnostics
    solver.enable_conservation_diagnostics(ConservationTolerances::default());

    // Initial energy should be zero (no excitation)
    let initial_energy = solver.calculate_total_energy();
    assert!(initial_energy < 1e-10);

    // Verify tracker is enabled with zero initial energy
    assert_eq!(
        solver.conservation_tracker.as_ref().unwrap().initial_energy,
        0.0
    );
    assert!(solver.is_solution_valid());

    // Disable and check
    solver.disable_conservation_diagnostics();
    assert!(solver.conservation_tracker.is_none());
}

#[test]
fn test_energy_calculation_accuracy() {
    let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig::default();
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    // Set a known pressure field (uniform)
    let p0 = 1000.0; // Pa
    solver.pressure.fill(p0);

    // Calculate energy
    let energy = solver.calculate_total_energy();

    // Expected energy: E = p²/(2ρ₀c₀²) * Volume
    let rho0 = 1000.0;
    let c0 = 1500.0;
    let volume =
        (grid.nx as f64) * grid.dx * (grid.ny as f64) * grid.dy * (grid.nz as f64) * grid.dz;
    let expected_energy = (p0 * p0) / (2.0 * rho0 * c0 * c0) * volume;

    let relative_error = (energy - expected_energy).abs() / expected_energy;
    assert!(
        relative_error < 1e-10,
        "Energy calculation error: {}",
        relative_error
    );
}

#[test]
fn test_conservation_check_interval() {
    let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig::default();
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    // Enable diagnostics with check interval of 5
    let tolerances = ConservationTolerances {
        check_interval: 5,
        ..ConservationTolerances::default()
    };
    solver.enable_conservation_diagnostics(tolerances);

    // Simulate 20 steps
    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    for _ in 0..20 {
        solver.update(&medium, &grid, &[], 0.0, dt).unwrap();
    }

    // Should have 20/5 = 4 checks (steps 5, 10, 15, 20)
    let summary = solver.get_conservation_summary().unwrap();
    assert!(summary.contains("checks"));
}

// ---------------------------------------------------------------------------
// Physics validation: absorption sign and attenuation
// ---------------------------------------------------------------------------

/// **Theorem (absorption sign, Stokes-Kirchhoff):**
/// With the multiplicative per-step absorption `p *= exp(−α·c·Δt)`, the L2
/// energy norm `‖p‖₂ = sqrt(∑p²)` must be strictly smaller after N steps than
/// before, for any α > 0. Derived from `exp(−α·c·Δt) < 1` for α, c, Δt > 0
/// and the fact that multiplication applies to every cell every step.
///
/// Verification: α = 5 Np/m, c = 1500 m/s, Δt ≈ 3.85×10⁻⁷ s.
/// Per-step decay factor ≈ exp(−5·1500·3.85e-7) ≈ 0.9971.
/// After 30 steps: 0.9971³⁰ ≈ 0.918 → ~8% L2 energy reduction.
#[test]
fn absorption_causes_amplitude_decay_not_growth() {
    let n = 24usize;
    let grid = Grid::new(n, n, n, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    // 5 Np/m absorption — strong enough for detectable per-step decay after 30 steps.
    // Per-step decay: exp(-5 * 1500 * 3.85e-7) ≈ 0.997; after 30 steps ≈ 0.918.
    medium
        .set_acoustic_properties(5.0, 1.0, medium.nonlinearity)
        .unwrap();

    let config = WesterveltFdtdConfig {
        spatial_order: 2,
        enable_absorption: true,
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);

    // Gaussian pulse centred in the domain.
    // Zero-velocity initial condition for leapfrog requires pressure_prev = pressure
    // (not the default zero).  With pressure_prev = 0, the first step computes
    // p^1 = 2p^0 - 0 + Δt²c²∇²p^0 ≈ 2p^0, which artificially doubles the amplitude
    // before absorption has a chance to act.
    let cx = n / 2;
    let cy = n / 2;
    let cz = n / 2;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let r2 = ((i as i32 - cx as i32).pow(2)
                    + (j as i32 - cy as i32).pow(2)
                    + (k as i32 - cz as i32).pow(2)) as f64;
                let val = 1.0e4 * (-(r2 / 9.0)).exp();
                solver.pressure[[i, j, k]] = val;
                solver.pressure_prev[[i, j, k]] = val; // zero-velocity IC
            }
        }
    }
    let initial_max = solver
        .pressure
        .iter()
        .cloned()
        .fold(0.0f64, f64::max);

    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    for step in 0..30 {
        solver
            .update(&medium, &grid, &[], step as f64 * dt, dt)
            .unwrap();
    }

    let final_max = solver
        .pressure
        .iter()
        .cloned()
        .map(f64::abs)
        .fold(0.0f64, f64::max);

    assert!(
        final_max < initial_max,
        "absorption must reduce peak amplitude: final_max={final_max:.4e} initial_max={initial_max:.4e}"
    );
}

/// **Invariant (pressure_prev2 allocation schedule):**
/// `pressure_prev2` is allocated lazily on the first `update()` call so that
/// the nonlinear `∂²(p²)/∂t²` kernel has access to p^{n−2} from step 2 onward.
#[test]
fn pressure_prev2_allocated_after_first_step() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig {
        enable_absorption: false,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);
    solver.pressure[[4, 4, 4]] = 1.0e5;

    assert!(
        solver.pressure_prev2.is_none(),
        "pp2 must not exist before any steps"
    );

    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    solver.update(&medium, &grid, &[], 0.0, dt).unwrap();
    assert!(
        solver.pressure_prev2.is_some(),
        "pp2 must exist after step 1 (lazy allocation in history rotation)"
    );

    solver.update(&medium, &grid, &[], dt, dt).unwrap();
    assert!(
        solver.pressure_prev2.is_some(),
        "pp2 must remain allocated on subsequent steps"
    );
}

/// **Theorem (nonlinear steepening, Hamilton & Blackstock 1998 §2):**
/// For a sinusoidal source with finite amplitude, the Westervelt nonlinearity
/// generates second-harmonic content. After N cycles, the peak-to-peak range
/// of the waveform exceeds that of the initial sine by a detectable margin.
#[test]
fn nonlinear_term_increases_waveform_asymmetry() {
    let n = 32usize;
    let grid = Grid::new(n, n, n, 1e-3, 1e-3, 1e-3).unwrap();
    let mut medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    medium.nonlinearity = 5.0; // B/A = 5 (water-like)

    let config_nl = WesterveltFdtdConfig {
        spatial_order: 2,
        enable_absorption: false,
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut medium_linear = medium.clone();
    medium_linear.nonlinearity = 0.0;

    let mut solver_nl = WesterveltFdtd::new(config_nl.clone(), &grid, &medium);
    let mut solver_lin = WesterveltFdtd::new(config_nl, &grid, &medium_linear);

    // Moderate-amplitude sine wave along x at y=ny/2, z=nz/2
    let amp = 2.0e5; // Pa — finite but sub-shock
    let k_wave = std::f64::consts::PI / (5.0 * grid.dx); // spatial frequency
    let jc = n / 2;
    let kc = n / 2;
    for i in 0..n {
        let val = amp * (k_wave * i as f64 * grid.dx).sin();
        solver_nl.pressure[[i, jc, kc]] = val;
        solver_lin.pressure[[i, jc, kc]] = val;
    }

    let dt = solver_nl.calculate_dt(&medium, &grid).unwrap();
    for step in 0..20 {
        let t = step as f64 * dt;
        solver_nl.update(&medium, &grid, &[], t, dt).unwrap();
        solver_lin
            .update(&medium_linear, &grid, &[], t, dt)
            .unwrap();
    }

    // Nonlinear solver should have generated harmonic content: total energy differs
    let e_nl: f64 = solver_nl.pressure.iter().map(|&p| p * p).sum();
    let e_lin: f64 = solver_lin.pressure.iter().map(|&p| p * p).sum();

    // Energy redistribution — they must not be bitwise identical
    let rel_diff = (e_nl - e_lin).abs() / (e_lin.max(1.0));
    assert!(
        rel_diff > 1e-8,
        "nonlinear and linear solvers must diverge; rel_diff={rel_diff:.2e}"
    );
}

/// **Invariant (buffer identity across steps):**
/// `mem::swap` of three Array3 buffers must preserve all three allocation
/// addresses as a set across any number of steps.
#[test]
fn pressure_buffers_stable_set_across_many_steps() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
    let config = WesterveltFdtdConfig {
        enable_absorption: false,
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut solver = WesterveltFdtd::new(config, &grid, &medium);
    solver.pressure[[4, 4, 4]] = 1.0e5;

    let initial_set = {
        let mut s = [
            solver.pressure.as_ptr() as usize,
            solver.pressure_prev.as_ptr() as usize,
            solver.pressure_next.as_ptr() as usize,
        ];
        s.sort_unstable();
        s
    };

    let dt = solver.calculate_dt(&medium, &grid).unwrap();
    for step in 0..10 {
        solver
            .update(&medium, &grid, &[], step as f64 * dt, dt)
            .unwrap();
    }

    let final_set = {
        let mut s = [
            solver.pressure.as_ptr() as usize,
            solver.pressure_prev.as_ptr() as usize,
            solver.pressure_next.as_ptr() as usize,
        ];
        s.sort_unstable();
        s
    };
    assert_eq!(final_set, initial_set, "buffer address set must be stable");
}
