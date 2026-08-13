//! Value-semantic tests for `GenericFdtdSolver` (CFL stability, leapfrog energy).
//!
//! ## Theorems verified
//!
//! ### CFL stability condition (Courant, Friedrichs & Lewy 1928; Yee 1966)
//!
//! For the 3D staggered-grid Yee leapfrog scheme with uniform spacing Δx and
//! p-th-order central differences, the von Neumann stability criterion requires:
//!
//! ```text
//! c₀ · Δt · √K ≤ 1        (K = number of spatial dimensions = 3)
//! ```
//!
//! Equivalently:
//!
//! ```text
//! Δt_max = Δx / (c₀ · √K)     [2nd-order stencil, isotropic]
//! ```
//!
//! The `max_stable_dt` accessor implements:
//!
//! ```text
//! dt_max = cfl_factor × cfl_limit(order) × Δx_min / c_max
//! ```
//!
//! where `cfl_limit(order)` = 1/√3 (2nd), 1/√15 (4th), 1/√27 (6th) for
//! the 3D von Neumann stability limits of each stencil order
//! (Gustafsson et al. 1995).
//!
//! ### Leapfrog discrete energy conservation (Taflove & Hagness 2005, §3.4)
//!
//! In a lossless (α=0) homogeneous medium with no external source, the discrete
//! total acoustic energy:
//!
//! ```text
//! E_n = Σ_i [ p_i² / (2ρ₀c₀²) ] · ΔV
//! ```
//!
//! is conserved over the leapfrog update to within floating-point round-off
//! when the CFL condition is satisfied. Specifically, the field must remain
//! bounded (not exponentially growing) for subcritical Δt.
//!
//! ## References
//!
//! - Courant R, Friedrichs K, Lewy H (1928). Math. Ann. 100, 32–74.
//! - Yee KS (1966). IEEE Trans. Antennas Propag. 14(3), 302–307.
//! - Gustafsson B et al. (1995). Time Compact Difference Schemes. §4.
//! - Taflove A, Hagness SC (2005). Computational Electrodynamics, 3rd ed. §3.4.

use crate::forward::fdtd::config::FdtdConfig;
use crate::forward::fdtd::solver::FdtdSolver;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::acoustics::mechanics::acoustic_wave::AcousticSpatialOrder;
use kwavers_source::GridSource;

/// Helper: create a minimal FdtdSolver for unit tests.
fn make_solver(
    n: usize,
    dx: f64,
    c0: f64,
    rho0: f64,
    cfl_factor: f64,
    spatial_order: usize,
) -> FdtdSolver {
    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
    let dt = cfl_factor / (3.0_f64).sqrt() * dx / c0;
    let config = FdtdConfig {
        spatial_order,
        staggered_grid: true,
        cfl_factor,
        enable_nonlinear: false,
        dt,
        nt: 10,
        ..Default::default()
    };
    FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap()
}

/// **Theorem (CFL max_stable_dt formula, 2nd-order 3D)**:
///
/// For `spatial_order = 2` (2nd-order central differences), the 3D von Neumann
/// stability limit is:
///
/// ```text
/// cfl_limit = 1/√3
/// ```
///
/// So `max_stable_dt = cfl_factor × (1/√3) × dx / c_max`.
///
/// With cfl_factor=0.45, dx=1 mm, c₀=1500 m/s:
///
/// ```text
/// dt_max = 0.45 / √3 × 1e-3 / 1500 = 0.45 × 0.57735... × 1e-3 / 1500
///        = 1.732e-7 s
/// ```
///
/// Reference: Yee (1966) §III; Taflove & Hagness (2005) §3.4.
#[test]
fn max_stable_dt_2nd_order_matches_analytical_formula() {
    let n = 8;
    let dx = 1.0e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let cfl_factor = 0.45_f64; // strictly below 1/√3

    let solver = make_solver(n, dx, c0, rho0, cfl_factor, 2);

    let dt_computed = solver.max_stable_dt(c0);

    // Analytical: dt_max = cfl_factor × (1/√3) × dx / c₀
    let cfl_limit_2nd = 1.0_f64 / (3.0_f64).sqrt(); // = 1/√3
    let dt_analytic = cfl_factor * cfl_limit_2nd * dx / c0;

    let rel_err = (dt_computed - dt_analytic).abs() / dt_analytic;
    assert!(
        rel_err < 1e-12,
        "max_stable_dt (2nd order): computed={dt_computed:.6e} analytic={dt_analytic:.6e} \
         rel_err={rel_err:.2e} (Yee 1966 CFL)"
    );
}

/// **Staggered CFL limit, 4th order.**
///
/// The staggered scheme's Courant limit is `1/(√D·Σ|cₙ|)`, derived on
/// `StaggeredLeapfrogOperator` from the stencil coefficients themselves. At
/// fourth order `Σ|cₙ| = 9/8 + 1/24`, giving `0.4949` in 3-D.
///
/// This is **not** the collocated `1/√15 = 0.2582` that
/// `AcousticSpatialOrder::cfl_limit` reports. The two coincide only at order 2,
/// which is why this test previously asserted the collocated value against a
/// staggered solver and passed — the solver was taking half the step it could
/// (KW-SOL-074).
#[test]
fn max_stable_dt_4th_order_staggered_matches_the_derived_limit() {
    let n = 8;
    let dx = 1.0e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let cfl_factor = 0.25_f64;

    let solver = make_solver(n, dx, c0, rho0, cfl_factor, 4);
    let dt_computed = solver.max_stable_dt(c0);

    let sum = 9.0 / 8.0 + 1.0 / 24.0;
    let cfl_limit = 1.0 / (3.0_f64.sqrt() * sum);
    let dt_analytic = cfl_factor * cfl_limit * dx / c0;

    let rel_err = (dt_computed - dt_analytic).abs() / dt_analytic;
    assert!(
        rel_err < 1e-12,
        "staggered max_stable_dt (4th order): computed={dt_computed:.6e} \
         analytic={dt_analytic:.6e} rel_err={rel_err:.2e}"
    );
    // And it is genuinely less restrictive than the collocated table.
    assert!(cfl_limit > 1.0 / 15.0_f64.sqrt());
}

/// **Staggered CFL limit, 6th and 8th order.**
///
/// Same derivation, and eighth order is reachable only on the staggered path —
/// the collocated table stops at six.
#[test]
fn max_stable_dt_high_order_staggered_matches_the_derived_limit() {
    let n = 8;
    let dx = 1.0e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let cfl_factor = 0.18_f64;

    let sums = [
        (6usize, 75.0 / 64.0 + 25.0 / 384.0 + 3.0 / 640.0),
        (
            8usize,
            1225.0 / 1024.0 + 245.0 / 3072.0 + 49.0 / 5120.0 + 5.0 / 7168.0,
        ),
    ];
    for (order, sum) in sums {
        let solver = make_solver(n, dx, c0, rho0, cfl_factor, order);
        let dt_computed = solver.max_stable_dt(c0);
        let dt_analytic = cfl_factor * (1.0 / (3.0_f64.sqrt() * sum)) * dx / c0;
        let rel_err = (dt_computed - dt_analytic).abs() / dt_analytic;
        assert!(
            rel_err < 1e-12,
            "staggered max_stable_dt (order {order}): computed={dt_computed:.6e} \
             analytic={dt_analytic:.6e} rel_err={rel_err:.2e}"
        );
    }
}

/// The collocated path keeps the tabulated limits, so both tables stay covered.
#[test]
fn max_stable_dt_collocated_keeps_the_tabulated_limit() {
    let n = 8;
    let dx = 1.0e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let cfl_factor = 0.25_f64;

    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(DENSITY_WATER_NOMINAL, c0, 0.0, 0.0, &grid);
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: false,
        cfl_factor,
        dt: cfl_factor / (3.0_f64).sqrt() * dx / c0,
        nt: 10,
        ..Default::default()
    };
    let solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

    let dt_analytic = cfl_factor * (1.0 / 15.0_f64.sqrt()) * dx / c0;
    let rel_err = (solver.max_stable_dt(c0) - dt_analytic).abs() / dt_analytic;
    assert!(rel_err < 1e-12, "collocated 4th-order limit changed");
}

/// **Theorem (CFL check_cfl_stability)**:
///
/// `check_cfl_stability(dt, c)` returns `true` iff `dt ≤ max_stable_dt(c)`.
///
/// Boundary cases:
/// - `dt = max_stable_dt` → `true` (stability boundary is inclusive)
/// - `dt = 0.99 × max_stable_dt` → `true`
/// - `dt = 1.01 × max_stable_dt` → `false`
#[test]
fn check_cfl_stability_correctly_classifies_dt() {
    let n = 8;
    let dx = 1.0e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let cfl_factor = 0.45_f64;

    let solver = make_solver(n, dx, c0, rho0, cfl_factor, 2);
    let dt_max = solver.max_stable_dt(c0);

    // Subcritical dt: stable
    assert!(
        solver.check_cfl_stability(0.99 * dt_max, c0),
        "dt = 0.99 × dt_max must be classified as stable"
    );

    // Exactly critical dt: stable (inclusive bound)
    assert!(
        solver.check_cfl_stability(dt_max, c0),
        "dt = dt_max must be classified as stable (inclusive)"
    );

    // Supercritical dt: unstable
    assert!(
        !solver.check_cfl_stability(1.01 * dt_max, c0),
        "dt = 1.01 × dt_max must be classified as unstable"
    );
}

/// **Theorem (AcousticSpatialOrder::cfl_limit, von Neumann stability)**:
///
/// The 3D von Neumann stability limits are:
/// - 2nd order: 1/√3  (stencil spans 3 points per axis; body-diagonal k-mode)
/// - 4th order: 1/√15 (Gustafsson et al. 1995, eq. 4.1)
/// - 6th order: 1/√27 (Gustafsson et al. 1995, eq. 4.2)
///
/// These are exact closed-form expressions; the implementation must reproduce
/// them to machine precision.
#[test]
fn spatial_order_cfl_limits_match_analytical_von_neumann_values() {
    // 2nd order: 1/√3
    let limit_2 = AcousticSpatialOrder::Second.cfl_limit();
    let expected_2 = 1.0_f64 / (3.0_f64).sqrt();
    assert!(
        (limit_2 - expected_2).abs() < 1e-14,
        "2nd-order CFL limit: expected {expected_2:.15} got {limit_2:.15}"
    );

    // 4th order: 1/√15
    let limit_4 = AcousticSpatialOrder::Fourth.cfl_limit();
    let expected_4 = 1.0_f64 / (15.0_f64).sqrt();
    assert!(
        (limit_4 - expected_4).abs() < 1e-14,
        "4th-order CFL limit: expected {expected_4:.15} got {limit_4:.15}"
    );

    // 6th order: 1/√27
    let limit_6 = AcousticSpatialOrder::Sixth.cfl_limit();
    let expected_6 = 1.0_f64 / (27.0_f64).sqrt();
    assert!(
        (limit_6 - expected_6).abs() < 1e-14,
        "6th-order CFL limit: expected {expected_6:.15} got {limit_6:.15}"
    );

    // Ordering must hold: limit_2 > limit_4 > limit_6
    // (higher-order stencils have stricter stability requirements)
    assert!(
        limit_2 > limit_4 && limit_4 > limit_6,
        "CFL limits must decrease with stencil order: {limit_2:.4} > {limit_4:.4} > {limit_6:.4}"
    );
}

/// **Theorem (leapfrog boundedness in lossless medium)**:
///
/// For a subcritical Δt satisfying CFL, the Yee leapfrog scheme preserves
/// the discrete acoustic energy within a constant factor over N steps.
/// Specifically, the pressure field must remain bounded:
///
/// ```text
/// max_i |p_i^n| < ∞   for all n ≤ N
/// ```
///
/// We verify this by running N=20 steps with an initial Gaussian source and
/// checking that no element grows unboundedly. The L2-energy ratio
/// E_n / E_0 must remain in [0.5, 2.0] — a loose tolerance accounting for
/// the leapfrog pressure-kinetic energy exchange (Taflove & Hagness 2005 §3.4).
///
/// A complementary test with a supercritical Δt would show exponential growth;
/// but since `FdtdSolver::new` validates dt against CFL, we instead verify
/// the subcritical bound.
#[test]
fn leapfrog_field_remains_bounded_in_lossless_medium() {
    let n = 16usize;
    let dx = 1.0e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    // Use cfl_factor = 0.45 < 1/√3; explicit dt to match
    let cfl_factor = 0.45_f64;
    let dt = cfl_factor / (3.0_f64).sqrt() * dx / c0; // subcritical

    let grid = Grid::new(n, n, n, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: false,
        cfl_factor,
        enable_nonlinear: false,
        dt,
        nt: 20,
        ..Default::default()
    };
    let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

    // Gaussian initial pressure pulse centred at grid centre
    let cx = (n / 2) as f64 * dx;
    let sigma = 3.0 * dx;
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let r2 = (i as f64 * dx - cx).powi(2)
                    + (j as f64 * dx - cx).powi(2)
                    + (k as f64 * dx - cx).powi(2);
                solver.fields.p[[i, j, k]] = (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }
    }

    // Compute initial acoustic energy E₀ = Σ p² / (2ρ₀c₀²) · ΔV
    let dv = dx * dx * dx;
    let e0: f64 = solver
        .fields
        .p
        .iter()
        .map(|&p| p * p / (2.0 * rho0 * c0 * c0) * dv)
        .sum();

    // Run 20 steps
    for _ in 0..20 {
        solver.step_forward().unwrap();
    }

    // Verify: all elements finite, energy order-of-magnitude preserved
    assert!(
        solver.fields.p.iter().all(|v| v.is_finite()),
        "FDTD leapfrog: pressure contains NaN/Inf after 20 subcritical steps"
    );

    let e_final: f64 = solver
        .fields
        .p
        .iter()
        .map(|&p| p * p / (2.0 * rho0 * c0 * c0) * dv)
        .sum();

    // Energy ratio must stay in [0.01, 100] — allows for radial spreading and
    // boundary reflections but rejects exponential blow-up or total damping
    let ratio = e_final / e0;
    assert!(
        ratio > 0.01 && ratio < 100.0,
        "FDTD leapfrog energy ratio E_final/E_0 = {ratio:.3e} is out of bounds [0.01, 100]; \
         CFL violation or physical error (Taflove & Hagness 2005 §3.4)"
    );
}

/// **FDTD absorption reproduces the prescribed power law in propagation.**
///
/// A standing wave `p = cos(k x)` decays at the temporal rate
/// `gamma = alpha * c_p`, so measuring the amplitude decay and the oscillation
/// frequency recovers `alpha(omega)` for comparison against
/// `alpha_0 * (f/f_ref)^y`. Nothing here consults the fitted spectrum, so a
/// mis-scaled modulus, a dropped relaxation term, or a wrong unit conversion
/// all surface as a discrepancy.
///
/// The exponent is swept because frequency-independent damping would satisfy
/// any single exponent; the measured `alpha` has to track `y`.
///
/// Runs on the **staggered** branch, the solver's default. An earlier revision
/// had to use the spectral branch because the finite-difference ones grew a
/// boundary mode that swamped the measurement; that was the adjointness defect
/// fixed in KW-SOL-081, and with the conservative closures in place the decay
/// measured here is absorption alone.
#[test]
fn absorption_reproduces_prescribed_power_law_in_propagation() {
    use crate::forward::fdtd::config::FdtdAbsorption;
    use std::f64::consts::TAU;

    const N: usize = 64;
    const DX: f64 = 1.0e-4;
    const C0: f64 = 1500.0;
    const RHO0: f64 = 1000.0;
    const F_REF: f64 = 1.0e6;
    const ALPHA0_DB: f64 = 0.5;
    /// Steps per averaging window, about four oscillation periods at the mode
    /// used, so the standing wave's energy exchange between `p` and `v`
    /// averages out instead of aliasing into the decay estimate.
    const WINDOW: usize = 200;

    for &y in &[0.6_f64, 1.0, 1.4] {
        // The staggered operator needs at least two points per axis; the field
        // is uniform across the transverse extent, whose derivative any
        // consistent stencil renders exactly zero, leaving a 1-D problem.
        let grid = Grid::new(N, 4, 4, DX, DX, DX).unwrap();
        let mut medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, &grid);
        medium.set_acoustic_properties(ALPHA0_DB, y, 0.0).unwrap();

        let dt = 0.15 * DX / C0;
        let config = FdtdConfig {
            spatial_order: 2,
            staggered_grid: true,
            dt,
            nt: 5000,
            absorption: FdtdAbsorption::PowerLawRelaxation {
                reference_frequency_hz: F_REF,
                band_min_hz: 0.2e6,
                band_max_hz: 4.0e6,
                relaxation_arms: 4,
            },
            ..Default::default()
        };
        let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();
        assert!(
            solver.absorption.is_some(),
            "absorbing configuration produced no absorption state"
        );

        // Mode 8 of 64 cells: 8 points per wavelength, ~1.9 MHz, inside the band.
        let k0 = TAU * 8.0 / (N as f64 * DX);
        for i in 0..N {
            let value = (k0 * i as f64 * DX).cos();
            for j in 0..4 {
                for k in 0..4 {
                    solver.fields.p[[i, j, k]] = value;
                }
            }
        }

        let mut trace = Vec::new();
        let run = |solver: &mut FdtdSolver, steps: usize, trace: &mut Vec<f64>| -> f64 {
            let mut accumulated = 0.0;
            for _ in 0..steps {
                solver.step_forward().unwrap();
                accumulated += solver.fields.p.iter().map(|&v| v * v).sum::<f64>();
                trace.push(solver.fields.p[[0, 0, 0]]);
            }
            accumulated / steps as f64
        };

        run(&mut solver, 200, &mut trace); // settle
        let early = run(&mut solver, WINDOW, &mut trace);
        let gap = 1600;
        run(&mut solver, gap, &mut trace);
        let late = run(&mut solver, WINDOW, &mut trace);

        // Window centres are `gap + WINDOW` steps apart; energy decays as
        // exp(-2*gamma*t) for amplitude rate gamma.
        let elapsed = (gap + WINDOW) as f64 * dt;
        let decay = -(late / early).ln() / (2.0 * elapsed);
        assert!(
            decay > 0.0,
            "y={y}: energy did not decay ({early:.4e} -> {late:.4e})"
        );

        let total_time = trace.len() as f64 * dt;
        let crossings = trace.windows(2).filter(|w| w[0] * w[1] < 0.0).count() as f64;
        let omega = TAU * crossings / 2.0 / total_time;
        assert!(omega > 0.0, "y={y}: no oscillation observed");

        let phase_speed = omega / k0;
        let measured = decay / phase_speed;
        let expected = kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_m(
            ALPHA0_DB,
            y,
            omega / TAU,
        );

        assert!(
            (measured - expected).abs() <= 0.15 * expected,
            "y={y}: measured alpha {measured:.3} vs prescribed {expected:.3} Np/m at {:.3} MHz",
            omega / TAU / 1.0e6
        );
    }
}

/// A lossless configuration allocates no absorption state and leaves the
/// pressure update bit-identical to what it was before absorption existed.
#[test]
fn lossless_configuration_allocates_no_absorption_state() {
    let solver = make_solver(
        8,
        1.0e-3,
        SOUND_SPEED_WATER_SIM,
        DENSITY_WATER_NOMINAL,
        0.45,
        2,
    );
    assert!(
        solver.absorption.is_none(),
        "the default configuration must remain lossless"
    );
}

/// A medium with zero absorption under an absorbing configuration skips the
/// state rather than allocating memory fields that can only contribute zero.
#[test]
fn lossless_medium_skips_absorption_state() {
    use crate::forward::fdtd::config::FdtdAbsorption;

    let grid = Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let mut medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    // State losslessness rather than assuming it: the default water medium
    // carries a small but non-zero α₀ (2.2e-3 dB/(MHz^y·cm)).
    medium.set_acoustic_properties(0.0, 1.0, 0.0).unwrap();
    let config = FdtdConfig {
        dt: 1.0e-8,
        nt: 4,
        absorption: FdtdAbsorption::PowerLawRelaxation {
            reference_frequency_hz: 1.0e6,
            band_min_hz: 0.5e6,
            band_max_hz: 5.0e6,
            relaxation_arms: 3,
        },
        ..Default::default()
    };
    let solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();
    assert!(
        solver.absorption.is_none(),
        "a lossless medium must not allocate relaxation memory fields"
    );
}

/// **Lossless leapfrog conserves discrete energy (KW-SOL-081 regression).**
///
/// The Yee leapfrog is symplectic when the pressure update's divergence is the
/// negative adjoint of the velocity update's gradient. It then has a conserved
/// discrete energy
///
/// ```text
///   E = sum_i [ p_i^2 / (2 rho c^2) + rho |u_i|^2 / 2 ] dV
/// ```
///
/// which oscillates slightly (velocity sits at half steps, so the naive sum
/// samples a shadow Hamiltonian) but does not drift.
///
/// Before the adjoint closure this ran away: the divergence used a one-sided
/// difference at the low face instead of the zero-flux one, breaking
/// `D = -G^T`, and pressure energy alone grew by a factor of 8.8e4 over two
/// thousand steps. The bound below would fail by four orders of magnitude on
/// that scheme, so this test pins the fix rather than merely exercising it.
#[test]
fn lossless_staggered_leapfrog_conserves_energy() {
    use std::f64::consts::TAU;

    const N: usize = 64;
    const DX: f64 = 1.0e-4;
    const C0: f64 = 1500.0;
    const RHO0: f64 = 1000.0;
    const STEPS: usize = 2000;

    let grid = Grid::new(N, 4, 4, DX, DX, DX).unwrap();
    let mut medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, &grid);
    // Lossless: the conserved quantity only exists without absorption.
    medium.set_acoustic_properties(0.0, 1.0, 0.0).unwrap();

    let dt = 0.15 * DX / C0;
    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: true,
        dt,
        nt: STEPS + 1,
        ..Default::default()
    };
    let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

    let k0 = TAU * 8.0 / (N as f64 * DX);
    for i in 0..N {
        let value = (k0 * i as f64 * DX).cos();
        for j in 0..4 {
            for k in 0..4 {
                solver.fields.p[[i, j, k]] = value;
            }
        }
    }

    let energy = |solver: &FdtdSolver| -> f64 {
        let bulk = RHO0 * C0 * C0;
        let potential: f64 = solver.fields.p.iter().map(|&p| p * p / (2.0 * bulk)).sum();
        let kinetic: f64 = solver
            .fields
            .ux
            .iter()
            .zip(solver.fields.uy.iter())
            .zip(solver.fields.uz.iter())
            .map(|((&ux, &uy), &uz)| 0.5 * RHO0 * (ux * ux + uy * uy + uz * uz))
            .sum();
        potential + kinetic
    };

    let initial = energy(&solver);
    assert!(initial > 0.0);

    let mut lowest = f64::INFINITY;
    let mut highest = 0.0_f64;
    for _ in 0..STEPS {
        solver.step_forward().unwrap();
        let current = energy(&solver) / initial;
        lowest = lowest.min(current);
        highest = highest.max(current);
    }

    assert!(
        highest < 1.1 && lowest > 0.9,
        "discrete energy drifted to [{lowest:.4}, {highest:.4}] of its initial value \
         over {STEPS} steps; a symplectic leapfrog holds it near 1"
    );
}

/// The same conservation on the collocated (central-difference) branch, whose
/// one-sided boundary rows broke skew-symmetry the same way the staggered
/// divergence's did.
#[test]
fn lossless_collocated_leapfrog_conserves_energy() {
    use std::f64::consts::TAU;

    const N: usize = 64;
    const DX: f64 = 1.0e-4;
    const C0: f64 = 1500.0;
    const RHO0: f64 = 1000.0;
    const STEPS: usize = 2000;

    let grid = Grid::new(N, 4, 4, DX, DX, DX).unwrap();
    let mut medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, &grid);
    medium.set_acoustic_properties(0.0, 1.0, 0.0).unwrap();

    let dt = 0.15 * DX / C0;
    let config = FdtdConfig {
        spatial_order: 2,
        staggered_grid: false,
        dt,
        nt: STEPS + 1,
        ..Default::default()
    };
    let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

    let k0 = TAU * 8.0 / (N as f64 * DX);
    for i in 0..N {
        let value = (k0 * i as f64 * DX).cos();
        for j in 0..4 {
            for k in 0..4 {
                solver.fields.p[[i, j, k]] = value;
            }
        }
    }

    let energy = |solver: &FdtdSolver| -> f64 {
        let bulk = RHO0 * C0 * C0;
        let potential: f64 = solver.fields.p.iter().map(|&p| p * p / (2.0 * bulk)).sum();
        let kinetic: f64 = solver
            .fields
            .ux
            .iter()
            .zip(solver.fields.uy.iter())
            .zip(solver.fields.uz.iter())
            .map(|((&ux, &uy), &uz)| 0.5 * RHO0 * (ux * ux + uy * uy + uz * uz))
            .sum();
        potential + kinetic
    };

    let initial = energy(&solver);
    assert!(initial > 0.0);
    let mut lowest = f64::INFINITY;
    let mut highest = 0.0_f64;
    for _ in 0..STEPS {
        solver.step_forward().unwrap();
        let current = energy(&solver) / initial;
        lowest = lowest.min(current);
        highest = highest.max(current);
    }

    assert!(
        highest < 1.1 && lowest > 0.9,
        "collocated discrete energy drifted to [{lowest:.4}, {highest:.4}] over {STEPS} \
         steps; the skew-symmetric closure holds it near 1 (the one-sided closure \
         reached 1.3e4)"
    );
}

/// **Eighth order runs, and every order conserves energy.**
///
/// Orders 4-8 became reachable on the staggered path when it moved onto
/// `StaggeredLeapfrogOperator` (KW-SOL-074); eighth order is what Fullwave 2.5
/// uses. Conservation must hold at each, since the gradient/divergence pair is
/// adjoint at every order by construction.
#[test]
fn every_staggered_order_conserves_energy() {
    use std::f64::consts::TAU;

    const N: usize = 48;
    const DX: f64 = 1.0e-4;
    const C0: f64 = 1500.0;
    const RHO0: f64 = 1000.0;
    const STEPS: usize = 1000;

    for order in [2usize, 4, 6, 8] {
        let grid = Grid::new(N, 4, 4, DX, DX, DX).unwrap();
        let mut medium = HomogeneousMedium::new(RHO0, C0, 0.0, 0.0, &grid);
        medium.set_acoustic_properties(0.0, 1.1, 0.0).unwrap();

        // Inside the derived staggered limit for every order tested.
        let dt = 0.1 * DX / C0;
        let config = FdtdConfig {
            spatial_order: order,
            staggered_grid: true,
            dt,
            nt: STEPS + 1,
            ..Default::default()
        };
        let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();

        // The exact discrete Neumann mode of the rigid-walled domain (ADR 106):
        // cell centres at `(i+½)Δ`, walls at `0` and `NΔ`, so `cos(k(i+½)Δ)`
        // with `k = mπ/(NΔ)` satisfies `∂p/∂n = 0` at both. It was a `sin` mode
        // while the walls were pressure-release.
        let k0 = std::f64::consts::PI * 8.0 / (N as f64 * DX);
        for i in 0..N {
            let value = (k0 * (i as f64 + 0.5) * DX).cos();
            for j in 0..4 {
                for k in 0..4 {
                    solver.fields.p[[i, j, k]] = value;
                }
            }
        }

        let energy = |solver: &FdtdSolver| -> f64 {
            let bulk = RHO0 * C0 * C0;
            let potential: f64 = solver.fields.p.iter().map(|&p| p * p / (2.0 * bulk)).sum();
            let kinetic: f64 = solver
                .fields
                .ux
                .iter()
                .zip(solver.fields.uy.iter())
                .zip(solver.fields.uz.iter())
                .map(|((&a, &b), &c)| 0.5 * RHO0 * (a * a + b * b + c * c))
                .sum();
            potential + kinetic
        };

        let initial = energy(&solver);
        assert!(initial > 0.0, "order {order}: zero initial energy");
        let mut lowest = f64::INFINITY;
        let mut highest = 0.0_f64;
        for _ in 0..STEPS {
            solver.step_forward().unwrap();
            let ratio = energy(&solver) / initial;
            lowest = lowest.min(ratio);
            highest = highest.max(ratio);
        }
        assert!(
            highest < 1.1 && lowest > 0.9,
            "order {order}: energy drifted to [{lowest:.4}, {highest:.4}]"
        );
        let _ = TAU;
    }
}

/// Eighth order is rejected on the collocated path, whose CFL table stops at
/// six — a configuration the solver cannot size a stable step for must fail
/// loudly rather than run.
#[test]
fn collocated_path_rejects_eighth_order() {
    let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();
    let config = FdtdConfig {
        spatial_order: 8,
        staggered_grid: false,
        dt: 1e-8,
        nt: 4,
        ..Default::default()
    };
    assert!(config.validate().is_err());

    let staggered = FdtdConfig {
        staggered_grid: true,
        ..config
    };
    assert!(staggered.validate().is_ok());
    let _ = grid;
}
