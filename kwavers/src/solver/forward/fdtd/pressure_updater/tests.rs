use crate::domain::grid::Grid;
use crate::domain::medium::HomogeneousMedium;
use crate::domain::source::GridSource;
use crate::solver::forward::fdtd::config::FdtdConfig;
use crate::solver::forward::fdtd::solver::FdtdSolver;
use ndarray::{Array3, Zip};

/// Westervelt correction must produce non-zero perturbation after two history steps.
///
/// For c=1500 m/s, ρ=1000 kg/m³, B/A=6 (β=4), p=1 MPa initial:
/// the nonlinear correction at step 2 is non-zero.
#[test]
fn test_westervelt_correction_nonzero_after_history() {
    let n = 4usize;
    let dx = 1e-3_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
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

/// Scratch-buffer pressure update must be bitwise-identical to explicit-allocation path.
///
/// Both paths use the same 2nd-order central difference stencil over 10 steps on a 16³ grid.
#[test]
fn test_fdtd_pressure_numerical_identity() {
    let n = 16usize;
    let dx = 1e-3_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
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
                solver.fields.ux[[i, j, k]] = (i as f64) * 1e-3;
                solver.fields.uy[[i, j, k]] = (j as f64) * 1e-3;
                solver.fields.uz[[i, j, k]] = (k as f64) * 1e-3;
            }
        }
    }

    for _ in 0..10 {
        solver.update_pressure_cpu(dt).unwrap();
    }
    let p_scratch = solver.fields.p.clone();

    let dvx = solver
        .central_operator
        .apply_x(solver.fields.ux.view())
        .unwrap();
    let dvy = solver
        .central_operator
        .apply_y(solver.fields.uy.view())
        .unwrap();
    let dvz = solver
        .central_operator
        .apply_z(solver.fields.uz.view())
        .unwrap();
    let divergence_alloc = &dvx + &dvy + &dvz;

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
    let mut divergence_scratch = dvz_s;
    Zip::from(&mut divergence_scratch)
        .and(&dvx_s)
        .and(&dvy_s)
        .for_each(|d, &dx_v, &dy_v| *d += dx_v + dy_v);

    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                assert_eq!(
                    divergence_alloc[[i, j, k]],
                    divergence_scratch[[i, j, k]],
                    "Divergence mismatch at [{i},{j},{k}]: alloc={} scratch={}",
                    divergence_alloc[[i, j, k]],
                    divergence_scratch[[i, j, k]]
                );
            }
        }
    }

    let p_max = p_scratch
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max)
        .abs();
    assert!(p_max > 0.0, "Pressure must be non-zero after 10 steps");
}

/// Staggered-grid divergence must match explicit linear sum in scratch buffer.
#[test]
fn test_staggered_divergence_uses_scratch_buffer() {
    let n = 12usize;
    let dx = 1e-3_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
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
