//! DG solver convergence tests: SSP-RK3 temporal accuracy and energy stability.
//!
//! ## Scope
//!
//! Tests are limited to polynomial order p=1 (n_nodes=2).  The GLL quadrature
//! implementation requires n_nodes ≥ 2 (p=0 is unsupported), and p≥2 has a
//! pre-existing singularity in `legendre_poly_and_deriv` at the boundary nodes
//! x=±1 where the denominator (1−x²) is zero.
//!
//! ## Tests
//!
//! 1. **SSP-RK3 temporal convergence** — smooth sinusoidal advection; Richardson
//!    extrapolation confirms temporal convergence order ≥ 2.5 (nominal 3.0 for
//!    SSP-RK3; 2.5 threshold accounts for spatial truncation at moderate resolution).
//!
//! 2. **L2 boundedness** — `‖u^n‖` must stay below 10× the initial norm over 100
//!    steps.  The SSP-RK3 TVD property does not guarantee strict L2 non-growth, but
//!    it does prevent blow-up.
//!
//! 3. **SSP-RK3 vs ForwardEuler p=1** — both integrators must produce finite,
//!    bounded solutions over 30 steps at CFL ~ 0.003 (conservative Euler regime).
//!
//! ## References
//!
//! - Shu & Osher (1988). J. Comput. Phys. 77(2):439–471.
//! - Cockburn & Shu (2001). J. Sci. Comput. 16(3):173–261.
//! - Hesthaven & Warburton (2008). *Nodal Discontinuous Galerkin Methods*. §3.

use kwavers_grid::Grid;
use kwavers_solver::forward::pstd::dg::config::{
    DGConfig, DgBoundaryCondition, DgTimeIntegrator, ShockCaptureConfig, WenoDegree,
};
use kwavers_solver::forward::pstd::dg::dg_solver::core::DGSolver;
use kwavers_solver::forward::pstd::dg::{BasisType, FluxType, LimiterType};
use leto::Array3;
use std::f64::consts::PI;
use std::sync::Arc;

// ── helpers ────────────────────────────────────────────────────────────────

/// Build a uniform Grid whose dimensions are all multiples of `n_nodes=2`.
///
/// `project_to_dg` computes element counts as `nx / n_nodes`, so all three
/// dimensions must be divisible by n_nodes.  For a quasi-1D test we use
/// `n_x_elems` elements in x and 1 element in each of y and z.
fn make_grid_p1(n_x_elems: usize) -> Arc<Grid> {
    // p=1 ⟹ n_nodes=2; 1 element in y/z means ny=nz=2
    let n_nodes = 2usize;
    let nx = n_x_elems * n_nodes;
    let ny = n_nodes;
    let nz = n_nodes;
    let dx = 1.0 / nx as f64;
    Arc::new(Grid::new(nx, ny, nz, dx, dx, dx).expect("valid grid"))
}

/// Initialise a smooth sinusoidal field: `u(x) = sin(2π x / L)`.
fn sinusoidal_field(grid: &Grid) -> Array3<f64> {
    let mut field = Array3::zeros((grid.nx, grid.ny, grid.nz));
    let lx = grid.nx as f64 * grid.dx;
    for ix in 0..grid.nx {
        let x = (ix as f64 + 0.5) * grid.dx;
        let val = (2.0 * PI * x / lx).sin();
        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                field[(ix, iy, iz)] = val;
            }
        }
    }
    field
}

/// L2 norm of modal coefficients (all degrees of freedom).
fn l2_modal(coeffs: &Array3<f64>) -> f64 {
    coeffs.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// L2 norm of a grid field.
fn l2_grid(field: &Array3<f64>) -> f64 {
    field.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Construct a `DGConfig` for p=1 with the given time integrator.
fn config_p1(integrator: DgTimeIntegrator) -> DGConfig {
    DGConfig {
        polynomial_order: 1,
        basis_type: BasisType::Legendre,
        flux_type: FluxType::LaxFriedrichs,
        time_integrator: integrator,
        use_limiter: false,
        limiter_type: LimiterType::Minmod,
        shock_threshold: 0.1,
        shock_capture: ShockCaptureConfig {
            enabled: false,
            limiter: WenoDegree::Weno3,
            threshold: 0.1,
            apply_per_stage: false,
        },
        sound_speed: 1500.0,
        boundary_conditions: [DgBoundaryCondition::Periodic; 3],
        cpml: None,
    }
}

// ── Test 1: SSP-RK3 temporal convergence (p=1) ────────────────────────────

/// Richardson extrapolation confirms O(dt³) for SSP-RK3.
///
/// Three step sizes `dt`, `dt/2`, `dt/4` to the same end time; convergence
/// order = `log₂(‖e_coarse‖ / ‖e_fine‖)`.  Acceptance: ≥ 2.5.
#[test]
fn test_ssprk3_temporal_convergence_p1() {
    let n_x_elems = 8usize;
    let grid = make_grid_p1(n_x_elems);
    let dx = grid.dx;
    let wave_speed = 1500.0_f64;
    // CFL_max for DG(p=1) = 1/(2·1+1) = 1/3; use 0.1 × CFL_max
    let dt_base = 0.1 / 3.0 * dx / wave_speed;
    let n_steps = 8;

    let run = |n_refine: u32| -> Array3<f64> {
        let scale = 2u32.pow(n_refine) as f64;
        let dt = dt_base / scale;
        let n = (n_steps as f64 * scale) as usize;
        let config = config_p1(DgTimeIntegrator::SspRk3);
        let mut solver = DGSolver::new(config, Arc::clone(&grid)).expect("solver");
        let mut field = sinusoidal_field(&grid);
        solver.project_to_dg(&field).expect("project");
        for _ in 0..n {
            solver.solve_step(&mut field, dt).expect("step");
        }
        solver.project_to_grid(&mut field).expect("back-project");
        field
    };

    let f0 = run(0);
    let f1 = run(1);
    let f2 = run(2);

    let err_coarse: f64 = (&f0 - &f1).iter().map(|v| v * v).sum::<f64>().sqrt();
    let err_fine: f64 = (&f1 - &f2).iter().map(|v| v * v).sum::<f64>().sqrt();

    if err_coarse > 1e-14 && err_fine > 1e-14 {
        let order = (err_coarse / err_fine).log2();
        assert!(
            order >= 2.5,
            "SSP-RK3 temporal order too low: {:.2} (expected ≥ 2.5)",
            order
        );
    }
    assert!(
        l2_grid(&f0).is_finite(),
        "SSP-RK3 produced non-finite field"
    );
}

// ── Test 2: L2 norm stays bounded over 100 steps (p=1, SSP-RK3) ──────────

/// SSP-RK3 TVD property prevents blow-up.  The L2 norm may grow slightly for
/// smooth waves but must remain < 10× the initial over 100 steps.
#[test]
fn test_ssprk3_l2_bounded_p1() {
    let grid = make_grid_p1(8);
    let dx = grid.dx;
    let wave_speed = 1500.0_f64;
    let dt = 0.1 / 3.0 * dx / wave_speed;

    let config = config_p1(DgTimeIntegrator::SspRk3);
    let mut solver = DGSolver::new(config, Arc::clone(&grid)).expect("solver");
    let mut field = sinusoidal_field(&grid);
    solver.project_to_dg(&field).expect("project");
    let norm_init = l2_modal(solver.modal_coefficients().unwrap());
    assert!(norm_init > 0.0);

    for step in 0..100 {
        solver.solve_step(&mut field, dt).expect("step");
        let norm = l2_modal(solver.modal_coefficients().unwrap());
        assert!(norm.is_finite(), "non-finite norm at step {}", step);
        assert!(
            norm < 10.0 * norm_init,
            "L2 blow-up at step {}: {:.3e} > 10 × {:.3e}",
            step,
            norm,
            norm_init
        );
    }
}

// ── Test 3: ForwardEuler stays bounded at conservative CFL ────────────────

/// Forward Euler (p=1) is unconditionally unstable per Cockburn & Shu 2001 §4,
/// but at very small CFL (≈ 0.003) it should not blow up over 30 steps.
#[test]
fn test_forward_euler_bounded_p1() {
    let grid = make_grid_p1(8);
    let dx = grid.dx;
    let wave_speed = 1500.0_f64;
    // Very small dt: CFL ≈ 0.003
    let dt = 0.01 / 3.0 * dx / wave_speed;

    let config = config_p1(DgTimeIntegrator::ForwardEuler);
    let mut solver = DGSolver::new(config, Arc::clone(&grid)).expect("solver");
    let mut field = sinusoidal_field(&grid);
    solver.project_to_dg(&field).expect("project");
    let norm_init = l2_modal(solver.modal_coefficients().unwrap());

    for step in 0..30 {
        solver.solve_step(&mut field, dt).expect("step");
        let norm = l2_modal(solver.modal_coefficients().unwrap());
        assert!(norm.is_finite(), "ForwardEuler non-finite at step {}", step);
        assert!(
            norm < 100.0 * norm_init,
            "ForwardEuler blow-up at step {}: {:.3e}",
            step,
            norm
        );
    }
}

// ── Test 4: ShockCaptureConfig round-trips through DGSolver ───────────────

/// Constructing a solver with `ShockCaptureConfig { enabled: true, limiter: Weno7,
/// apply_per_stage: true }` must succeed and run without panic.
#[test]
fn test_shock_capture_config_roundtrip() {
    let grid = make_grid_p1(4);
    let dx = grid.dx;
    let wave_speed = 1500.0_f64;
    let dt = 0.05 / 3.0 * dx / wave_speed;

    let config = DGConfig {
        polynomial_order: 1,
        basis_type: BasisType::Legendre,
        flux_type: FluxType::LaxFriedrichs,
        time_integrator: DgTimeIntegrator::SspRk3,
        use_limiter: true,
        limiter_type: LimiterType::Minmod,
        shock_threshold: 0.1,
        shock_capture: ShockCaptureConfig {
            enabled: true,
            limiter: WenoDegree::Weno7,
            threshold: 0.05,
            apply_per_stage: true,
        },
        sound_speed: 1500.0,
        boundary_conditions: [DgBoundaryCondition::Periodic; 3],
        cpml: None,
    };

    let mut solver = DGSolver::new(config, Arc::clone(&grid)).expect("solver");
    let mut field = sinusoidal_field(&grid);
    solver.project_to_dg(&field).expect("project");
    for _ in 0..10 {
        solver.solve_step(&mut field, dt).expect("step");
    }
    assert!(
        l2_modal(solver.modal_coefficients().unwrap()).is_finite(),
        "non-finite norm with shock-capture config"
    );
}

// ── Test 5: initial solver state ─────────────────────────────────────────

/// `has_modal_coefficients()` must be false before the first `project_to_dg` call.
#[test]
fn test_dgsolver_initial_state() {
    let grid = make_grid_p1(4);
    let config = config_p1(DgTimeIntegrator::SspRk3);
    let solver = DGSolver::new(config, Arc::clone(&grid)).expect("solver");
    assert!(
        !solver.has_modal_coefficients(),
        "modal coefficients must be None before first projection"
    );
}
