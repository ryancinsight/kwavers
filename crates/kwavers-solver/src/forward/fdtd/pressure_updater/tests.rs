use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_domain::grid::Grid;
use kwavers_domain::medium::HomogeneousMedium;
use kwavers_domain::source::GridSource;
use crate::forward::fdtd::config::FdtdConfig;
use crate::forward::fdtd::solver::FdtdSolver;
use ndarray::{Array3, Zip};

/// Westervelt correction must produce non-zero perturbation after two history steps.
///
/// For c=1500 m/s, ρ=1000 kg/m³, B/A=6 (β=4), p=1 MPa initial:
/// the nonlinear correction at step 2 is non-zero.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_westervelt_correction_nonzero_after_history() {
    let n = 4usize;
    let dx = 1e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let dt = 0.3 * dx / c0;

    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 6.0, &grid);

    let config = FdtdConfig {
        enable_nonlinear: true,
        staggered_grid: false,
        spatial_order: 2,
        dt,
        nt: 4,
        cfl_factor: 0.3,
        ..Default::default()
    };

    let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

    solver.fields.p.fill(1e6_f64);
    solver.p_prev = Some(solver.fields.p.clone());
    solver.p_prev2 = Some(Array3::zeros((n, n, n)));

    let p_before = solver.fields.p[[1, 1, 1]];
    solver.apply_westervelt_nonlinear_correction(dt);
    let p_after = solver.fields.p[[1, 1, 1]];

    assert_ne!(
        p_before, p_after,
        "Westervelt correction must change pressure when history is available"
    );
}

/// Scratch-buffer divergence matches analytical reference for a linear velocity field.
///
/// ## Theorem (2nd-order central-difference exactness for linear functions)
///
/// For `ux[i,j,k] = i·Δx`, `uy[i,j,k] = j·Δx`, `uz[i,j,k] = k·Δx`, the 2nd-order
/// centered stencil applied at any interior node satisfies:
/// ```text
///   ∂ux/∂x = (ux[i+1] − ux[i−1]) / (2Δx) = (Δx) / (Δx) = 1.0   (exact for degree-1)
/// ```
/// and identically for `∂uy/∂y` and `∂uz/∂z`. Hence `div(u)|interior = 3.0` exactly.
///
/// Justification: the 2nd-order central-difference stencil has a Taylor truncation
/// error proportional to the 3rd derivative; linear fields have zero 3rd derivative,
/// so the stencil is exact regardless of grid spacing.
///
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_fdtd_pressure_numerical_identity() {
    let n = 16usize;
    let dx = 1e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let dt = 0.3 * dx / c0;

    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let config = FdtdConfig {
        enable_nonlinear: false,
        staggered_grid: false,
        spatial_order: 2,
        dt,
        nt: 10,
        cfl_factor: 0.3,
        ..Default::default()
    };

    let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                solver.fields.ux[[i, j, k]] = (i as f64) * dx;
                solver.fields.uy[[i, j, k]] = (j as f64) * dx;
                solver.fields.uz[[i, j, k]] = (k as f64) * dx;
            }
        }
    }

    // Compute divergence via pre-allocated scratch buffers.
    let mut dvx_s = Array3::<f64>::zeros((n, n, n));
    let mut dvy_s = Array3::<f64>::zeros((n, n, n));
    let mut dvz_s = Array3::<f64>::zeros((n, n, n));
    solver
        .central_operator
        .apply_x_into(solver.fields.ux.view(), &mut dvx_s)
        .unwrap();
    solver
        .central_operator
        .apply_y_into(solver.fields.uy.view(), &mut dvy_s)
        .unwrap();
    solver
        .central_operator
        .apply_z_into(solver.fields.uz.view(), &mut dvz_s)
        .unwrap();

    // Verify analytical reference: interior divergence = 3.0 exactly (linear field, O2 stencil).
    for i in 1..n - 1 {
        for j in 1..n - 1 {
            for k in 1..n - 1 {
                let div = dvx_s[[i, j, k]] + dvy_s[[i, j, k]] + dvz_s[[i, j, k]];
                assert!(
                    (div - 3.0).abs() < 1e-10,
                    "Interior divergence must equal 3.0 at [{i},{j},{k}]: got {div}"
                );
            }
        }
    }

    // Pressure update must produce non-zero output after 10 steps.
    for _ in 0..10 {
        solver.update_pressure_cpu(dt).unwrap();
    }
    let p_max = solver
        .fields
        .p
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .abs();
    assert!(p_max > 0.0, "Pressure must be non-zero after 10 steps");
}

/// Staggered-grid divergence must match explicit linear sum in scratch buffer.
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_staggered_divergence_uses_scratch_buffer() {
    let n = 12usize;
    let dx = 1e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let dt = 0.3 * dx / c0;

    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);

    let config = FdtdConfig {
        enable_nonlinear: false,
        staggered_grid: true,
        spatial_order: 2,
        dt,
        nt: 4,
        cfl_factor: 0.3,
        ..Default::default()
    };

    let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                solver.fields.ux[[i, j, k]] = (i as f64) * 1e-3;
                solver.fields.uy[[i, j, k]] = (j as f64) * 2e-3;
                solver.fields.uz[[i, j, k]] = (k as f64) * 3e-3;
            }
        }
    }

    solver.compute_divergence_staggered().unwrap();

    let dvx = solver
        .staggered_operator
        .apply_backward_x(solver.fields.ux.view())
        .unwrap();
    let dvy = solver
        .staggered_operator
        .apply_backward_y(solver.fields.uy.view())
        .unwrap();
    let dvz = solver
        .staggered_operator
        .apply_backward_z(solver.fields.uz.view())
        .unwrap();

    let mut expected = dvz.clone();
    Zip::from(&mut expected)
        .and(&dvx)
        .and(&dvy)
        .for_each(|d, &dx_v, &dy_v| *d += dx_v + dy_v);

    assert_eq!(solver.divergence_scratch, expected);
}
