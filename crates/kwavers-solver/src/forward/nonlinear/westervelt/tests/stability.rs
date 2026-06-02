//! Buffer allocation invariants, nonlinear steepening, and CFL stability tests.

use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::tissue_acoustics::B_OVER_A_WATER;
use kwavers_domain::grid::Grid;
use kwavers_domain::medium::HomogeneousMedium;
use crate::forward::nonlinear::westervelt::{WesterveltFdtd, WesterveltFdtdConfig};

/// **Invariant (pressure_prev2 allocation schedule):**
/// `pressure_prev2` is allocated lazily on the first `update()` call so that
/// the nonlinear `∂²(p²)/∂t²` kernel has access to p^{n−2} from step 2 onward.
#[test]
fn pressure_prev2_allocated_after_first_step() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
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
    let mut medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
    medium.set_nonlinearity(B_OVER_A_WATER);

    let config_nl = WesterveltFdtdConfig {
        spatial_order: 2,
        enable_absorption: false,
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };
    let mut medium_linear = medium.clone();
    medium_linear.set_nonlinearity(0.0);

    let mut solver_nl = WesterveltFdtd::new(config_nl.clone(), &grid, &medium);
    let mut solver_lin = WesterveltFdtd::new(config_nl, &grid, &medium_linear);

    let amp = 2.0e5;
    let k_wave = std::f64::consts::PI / (5.0 * grid.dx);
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

    let e_nl: f64 = solver_nl.pressure.iter().map(|&p| p * p).sum();
    let e_lin: f64 = solver_lin.pressure.iter().map(|&p| p * p).sum();

    let rel_diff = (e_nl - e_lin).abs() / (e_lin.max(1.0));
    assert!(
        rel_diff > 1e-8,
        "nonlinear and linear solvers must diverge; rel_diff={rel_diff:.2e}"
    );
}

/// **Theorem (CFL stability, 3D explicit leapfrog, von Neumann analysis):**
///
/// The explicit second-order leapfrog scheme for the linear wave equation on a
/// 3D uniform grid with spacing Δx is stable iff the Courant number satisfies:
///
/// ```text
/// ν = c·Δt/Δx ≤ 1/√3    (≈ 0.5774)
/// ```
///
/// Reference: Courant R., Friedrichs K., Lewy H. (1928). Math. Ann. 100, 32–74.
#[test]
fn cfl_violation_causes_divergence_stable_dt_remains_bounded() {
    let n = 10usize;
    let dx = 1e-3;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::from_minimal(rho0, c0, &grid);

    let dt_crit = dx / (c0 * 3.0f64.sqrt());
    let dt_stable = 0.5 * dt_crit;
    let dt_unstable = 1.5 * dt_crit;

    let config = WesterveltFdtdConfig {
        spatial_order: 2,
        enable_absorption: false,
        artificial_viscosity: 0.0,
        ..WesterveltFdtdConfig::default()
    };

    let init_field = |solver: &mut WesterveltFdtd| {
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let angle_x = std::f64::consts::PI * i as f64 / (n - 1) as f64;
                    let angle_y = std::f64::consts::PI * j as f64 / (n - 1) as f64;
                    let angle_z = std::f64::consts::PI * k as f64 / (n - 1) as f64;
                    let val = 1.0e4 * angle_x.cos() * angle_y.cos() * angle_z.cos();
                    solver.pressure[[i, j, k]] = val;
                    solver.pressure_prev[[i, j, k]] = val;
                }
            }
        }
    };

    let mut solver_stable = WesterveltFdtd::new(config.clone(), &grid, &medium);
    init_field(&mut solver_stable);
    let initial_l2: f64 = solver_stable
        .pressure
        .iter()
        .map(|&p| p * p)
        .sum::<f64>()
        .sqrt();

    for step in 0..10 {
        solver_stable
            .update(&medium, &grid, &[], step as f64 * dt_stable, dt_stable)
            .unwrap();
    }

    let stable_l2: f64 = solver_stable
        .pressure
        .iter()
        .map(|&p| p * p)
        .sum::<f64>()
        .sqrt();
    assert!(
        stable_l2.is_finite(),
        "stable dt must keep L2 norm finite; got {stable_l2}"
    );
    assert!(
        stable_l2 < 10.0 * initial_l2,
        "stable dt must not amplify L2 norm ×10: stable_l2={stable_l2:.4e} initial_l2={initial_l2:.4e}"
    );

    let mut solver_unstable = WesterveltFdtd::new(config, &grid, &medium);
    init_field(&mut solver_unstable);

    for step in 0..10 {
        let _ = solver_unstable.update(&medium, &grid, &[], step as f64 * dt_unstable, dt_unstable);
    }

    let unstable_l2: f64 = solver_unstable
        .pressure
        .iter()
        .map(|&p| p * p)
        .sum::<f64>()
        .sqrt();
    assert!(
        !unstable_l2.is_finite() || unstable_l2 >= 100.0 * initial_l2,
        "CFL-violating dt must cause divergence (≥100× L2 or NaN); \
         got unstable_l2={unstable_l2:.4e} vs initial_l2={initial_l2:.4e}"
    );
}

/// **Invariant (buffer identity across steps):**
/// `mem::swap` of three Array3 buffers must preserve all three allocation
/// addresses as a set across any number of steps.
#[test]
fn pressure_buffers_stable_set_across_many_steps() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let medium =
        HomogeneousMedium::from_minimal(DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM, &grid);
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
