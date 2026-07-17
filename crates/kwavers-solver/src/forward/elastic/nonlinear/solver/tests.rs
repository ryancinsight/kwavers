use super::super::config::NonlinearSWEConfig;
use super::super::material::HyperelasticModel;
use super::NonlinearElasticWaveSolver;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use leto::Array3;

/// CFL stability factor used internally by [`NonlinearElasticWaveSolver::calculate_time_step_for_amplitude`].
const CFL: f64 = 0.45;
/// Reference displacement scale used in the nonlinear CFL formula \[m\].
const U_REF: f64 = 1e-3;

/// Verify construction invariants for `NonlinearElasticWaveSolver`.
///
/// ## Invariants (contractual at construction)
///
/// 1. Grid dimensions are preserved from the input `Grid`.
/// 2. Attenuation coefficient ≥ 0 (physical requirement: absorption is non-negative).
/// 3. Configuration parameters are stored without modification.
///
/// These invariants are required by all subsequent time-stepping calls.
#[test]
fn test_nonlinear_solver_creation() {
    let nx = 16_usize;
    let ny = 16_usize;
    let nz = 16_usize;
    let dx = 0.001_f64;
    let grid = Grid::new(nx, ny, nz, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.5,
        1.0,
        &grid,
    );
    let material = HyperelasticModel::neo_hookean_soft_tissue();
    let beta = 0.1_f64;
    let config = NonlinearSWEConfig {
        nonlinearity_parameter: beta,
        ..Default::default()
    };

    let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config).unwrap();

    // Invariant 1: grid dimensions match input.
    assert_eq!(solver.grid.nx, nx, "grid.nx must equal {nx}");
    assert_eq!(solver.grid.ny, ny, "grid.ny must equal {ny}");
    assert_eq!(solver.grid.nz, nz, "grid.nz must equal {nz}");

    // Invariant 2: attenuation is non-negative.
    assert!(
        solver.attenuation_np_per_m >= 0.0,
        "attenuation_np_per_m must be >= 0, got {}",
        solver.attenuation_np_per_m
    );

    // Invariant 3: nonlinearity_parameter preserved.
    assert_eq!(
        solver.config.nonlinearity_parameter, beta,
        "config.nonlinearity_parameter must be {beta}"
    );
}

/// CFL condition for nonlinear elastic waves (LeVeque 2002, §11.1):
///
/// `dt = CFL · Δx / c_max`  where  `c_max = c₀ · (1 + β · |u| / u_ref)`
///
/// With default config (β=0.1, max_strain=1.0, Δx=1 mm, max_dt=9.0×10⁻⁷ s):
/// - `max_abs_u = 1.0 × 10⁻³ m = u_ref` → `max_u_over_ref = 1.0`
/// - `c_max = 1500 × 1.1 = 1650 m/s`
/// - `dt_cfl = 0.45 × 0.001 / 1650 ≈ 2.727 × 10⁻⁷ s < max_dt`
///   → `dt = dt_cfl` (cap does not apply)
#[test]
fn cfl_time_step_matches_analytical_formula() {
    let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::soft_tissue(1000.0, 0.49, &grid);
    let material = HyperelasticModel::neo_hookean_soft_tissue();
    let config = NonlinearSWEConfig::default(); // β=0.1, max_strain=1.0, max_dt=9.0e-7

    let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config).unwrap();
    let dt = solver.calculate_time_step();

    const C0: f64 = SOUND_SPEED_WATER_SIM; // config.sound_speed()
    const BETA: f64 = 0.1; // default nonlinearity_parameter
    const MAX_STRAIN: f64 = 1.0; // default max_strain
    const DX: f64 = 0.001; // grid spacing [m]
    const MAX_DT: f64 = 9.0e-7; // default max_dt [s]

    let max_abs_u = MAX_STRAIN * U_REF; // = 1e-3
    let max_u_over_ref = max_abs_u / U_REF; // = 1.0
    let c_max = C0 * BETA.mul_add(max_u_over_ref, 1.0); // 1500 * 1.1 = 1650
    let dt_cfl = CFL * DX / c_max; // 0.45 * 0.001 / 1650
    let expected = dt_cfl.min(MAX_DT); // dt_cfl < MAX_DT, so expected = dt_cfl

    assert!(
        (dt - expected).abs() < 1e-20,
        "CFL time step mismatch: expected={:.10e} s, computed={:.10e} s",
        expected,
        dt
    );
}

/// When Δx is large, `dt_cfl = CFL·Δx/c_max > max_dt`, so the cap enforces `dt = max_dt`.
///
/// With Δx = 0.1 m (100 mm):
/// - `dt_cfl = 0.45 × 0.1 / 1650 ≈ 2.727 × 10⁻⁵ s >> max_dt = 9.0 × 10⁻⁷ s`
///   → `dt = max_dt = 9.0 × 10⁻⁷ s`
#[test]
fn cfl_time_step_capped_by_max_dt_for_coarse_grid() {
    let dx_coarse = 0.1_f64; // 100 mm → dt_cfl >> max_dt
    let grid = Grid::new(16, 16, 16, dx_coarse, dx_coarse, dx_coarse).unwrap();
    let medium = HomogeneousMedium::soft_tissue(1000.0, 0.49, &grid);
    let material = HyperelasticModel::neo_hookean_soft_tissue();
    let config = NonlinearSWEConfig::default(); // max_dt = 9.0e-7

    let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config).unwrap();
    let dt = solver.calculate_time_step();

    // dt_cfl = 0.45 * 0.1 / 1650 ≈ 2.727e-5 >> max_dt; cap applies.
    let expected = 9.0e-7_f64;
    assert!(
        (dt - expected).abs() < 1e-20,
        "max_dt cap not applied: expected={:.6e}, computed={:.6e}",
        expected,
        dt
    );
}

/// Verify the monotone time ordering invariant of `propagate_waves`.
///
/// `history[0].time == 0` (initial snapshot before any stepping).
/// For all i ≥ 1, `history[i].time > history[i-1].time`
/// (each saved frame advances simulation time by at least one step).
/// `history.last().time > 0` (propagation advanced past t=0).
#[test]
fn wave_propagation_history_is_monotone_in_time() {
    let grid = Grid::new(16, 8, 8, 0.001, 0.001, 0.001).unwrap();
    let medium = HomogeneousMedium::soft_tissue(1000.0, 0.49, &grid);
    let material = HyperelasticModel::neo_hookean_soft_tissue();
    let config = NonlinearSWEConfig {
        nonlinearity_parameter: 0.05,
        enable_harmonics: false,
        max_dt: 1e-7,
        ..Default::default()
    };

    let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config).unwrap();
    let mut initial = Array3::zeros((16, 8, 8));
    initial[[8, 4, 4]] = 1e-6;

    let history = solver.propagate_waves(&initial).unwrap();

    assert!(
        (history.len()) >= 2,
        "history must contain at least initial + final frame; got {}",
        (history.len())
    );

    assert_eq!(
        history[0].time, 0.0,
        "initial history frame must be at t=0; got {:.6e}",
        history[0].time
    );

    for i in 1..(history.len()) {
        assert!(
            history[i].time > history[i - 1].time,
            "history time not strictly monotone at index {}: {:.6e} ≤ {:.6e}",
            i,
            history[i].time,
            history[i - 1].time
        );
    }

    assert!(
        history.last().unwrap().time > 0.0,
        "final time must be > 0; got {:.6e}",
        history.last().unwrap().time
    );
}

/// Verify the final simulation time matches the expected integration duration.
///
/// For Δx = 1 mm, NX = 16, β = 0.05, max_dt = 1e-7 s, initial amplitude = 1e-6 m:
///
/// `domain_time = NX · Δx / c₀ = 16 × 0.001 / 1500 ≈ 1.067 × 10⁻⁵ s`
///
/// `dt = min(CFL·Δx/c_max, max_dt)`:
///   - `c_max = 1500 × (1 + 0.05 × (1e-6/1e-3)) ≈ 1500.075 m/s`
///   - `dt_cfl ≈ 0.45 × 0.001 / 1500.075 ≈ 3.0×10⁻⁷ s`
///   - `dt = min(3.0e-7, 1e-7) = 1e-7 s`
///
/// `n_steps = ⌈simulation_time / dt⌉` ≥ 2, `final_time = n_steps × dt`.
/// We verify `final_time ≈ domain_time` to within one step tolerance.
#[test]
fn wave_propagation_final_time_matches_domain_crossing_time() {
    let nx = 16_usize;
    let dx = 0.001_f64;
    let grid = Grid::new(nx, 8, 8, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::soft_tissue(1000.0, 0.49, &grid);
    let material = HyperelasticModel::neo_hookean_soft_tissue();
    let max_dt = 1e-7_f64;
    let beta = 0.05_f64;
    let config = NonlinearSWEConfig {
        nonlinearity_parameter: beta,
        enable_harmonics: false,
        max_dt,
        ..Default::default()
    };

    let solver = NonlinearElasticWaveSolver::new(&grid, &medium, material, config).unwrap();
    let mut initial = Array3::zeros((nx, 8, 8));
    initial[[8, 4, 4]] = 1e-6;

    let history = solver.propagate_waves(&initial).unwrap();
    let final_time = history.last().unwrap().time;

    // Compute expected final_time from the algorithm:
    //   max_abs_u = 1e-6, beta=0.05, c=1500, u_ref=1e-3, dx=0.001
    //   max_u_over_ref = 1e-6/1e-3 = 1e-3
    //   c_max = 1500*(1+0.05*1e-3)
    //   dt_cfl = 0.45 * 0.001 / c_max; dt = min(dt_cfl, 1e-7) = 1e-7
    let c = SOUND_SPEED_WATER_SIM;
    let max_abs_u = 1e-6_f64;
    let max_u_over_ref = (max_abs_u / U_REF).max(0.0);
    let c_max = c * beta.mul_add(max_u_over_ref, 1.0);
    let dt_cfl = CFL * dx / c_max.max(f64::EPSILON);
    let dt = dt_cfl.min(max_dt).max(f64::EPSILON);

    let domain_time = (nx as f64 * dx) / c;
    // simulation_time = domain_time.min(frac * t_shock).max(dt)
    // With max_abs_u < 1e-3 and beta = 0.05: frac = 0.30
    // t_shock: gradient at i=7,9 from single point at [8,4,4]:
    //   grad ~ |initial[9,4,4] - initial[7,4,4]| / (2*dx) = 1e-6 / 0.002 = 5e-4
    //   t_shock = u_ref / (c*beta*grad) = 1e-3 / (1500*0.05*5e-4) ≈ 0.02667 s
    let max_grad = max_abs_u / (2.0 * dx); // central difference across the single nonzero point
    let t_shock = if beta > 0.0 && max_grad > 0.0 {
        (U_REF / (c * beta * max_grad)).max(dt)
    } else {
        f64::INFINITY
    };
    let frac = 0.30; // beta=0.05, max_abs_u < 1e-3
    let simulation_time = domain_time.min(frac * t_shock).max(dt);
    let n_steps = ((simulation_time / dt).ceil() as usize).max(2);
    let expected_final_time = n_steps as f64 * dt;

    assert!(
        (final_time - expected_final_time).abs() <= dt,
        "final time mismatch: expected≈{:.6e} s, got {:.6e} s (tolerance=dt={:.2e})",
        expected_final_time,
        final_time,
        dt
    );
}
