use crate::forward::fdtd::config::FdtdAbsorption;
use crate::forward::fdtd::config::FdtdConfig;
use crate::forward::fdtd::config::TemporalScheme;
use crate::forward::fdtd::solver::FdtdSolver;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::GridSource;

/// **The staggered path is second-order in time**, and this pins that.
///
/// Fullwave 2.5 runs 8th order in space and 4th in time. kwavers matched the
/// spatial order under KW-SOL-074; this records where the temporal order stands
/// and will need updating to 4 when KW-SOL-092 lands - a failure here after that
/// work is the expected signal, not a regression.
///
/// # Measuring this correctly, which took three attempts
///
/// Both of the obvious designs report the wrong number:
///
/// - Sampling after a **full period** puts the mode at an extremum, where a
///   phase error `phi ~ dt^2` enters as `cos(phi) ~ 1 - phi^2/2` and is squared.
///   That reports order 4 for this second-order scheme - the same trap as
///   KW-SOL-088.
/// - Seeding velocity with **zero** is an `O(dt)` inconsistency in the initial
///   state, because the leapfrog carries `u` at `t = -dt/2`. Mixed with the
///   above it reported order 3.
///
/// So: consistent half-step velocity, and sample at a **quarter** period, where
/// the exact pressure is identically zero and the phase error enters linearly.
#[test]
fn the_staggered_path_is_second_order_in_time() {
    let error_for = |steps: usize| temporal_error(steps, TemporalScheme::Leapfrog);
    let coarse = error_for(200);
    let mid = error_for(400);
    let fine = error_for(800);
    for (a, b, label) in [(coarse, mid, "200->400"), (mid, fine, "400->800")] {
        let order = (a / b).log2();
        assert!(
            (1.8..=2.2).contains(&order),
            "{label}: observed temporal order {order:.3}, expected 2              (errors {a:.4e} -> {b:.4e})"
        );
    }
}

/// Worst pressure error a quarter period in, for a given time scheme.
///
/// Shared so the second- and fourth-order tests measure the *same* benchmark
/// and differ only in the scheme under test. Two details are load-bearing and
/// both cost a wrong answer before they were right (KW-SOL-092):
///
/// - The sample is at a **quarter** period, where the exact pressure is
///   identically zero and a phase error enters linearly. At a full period the
///   mode sits at an extremum, which squares the phase error and reports twice
///   the true order — the KW-SOL-088 trap.
/// - Velocity is seeded at `t = -dt/2`, where the leapfrog carries it. Seeding
///   zero is an `O(dt)` inconsistency in the *initial state* that masquerades as
///   a property of the scheme.
fn temporal_error(steps: usize, scheme: TemporalScheme) -> f64 {
    const N: usize = 64;
    let dx = 1.0e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;
    let length = N as f64 * dx;
    let k = std::f64::consts::PI / length;
    let omega = c0 * k;
    let period = std::f64::consts::TAU / omega;

    let dt = period / (4 * steps) as f64;
    let grid = Grid::new(N, 1, 1, dx, dx, dx).unwrap();
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
    let config = FdtdConfig {
        spatial_order: 8,
        staggered_grid: true,
        enable_nonlinear: false,
        temporal_scheme: scheme,
        dt,
        nt: steps + 1,
        ..Default::default()
    };
    let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();
    for i in 0..N {
        let x = (i as f64 + 0.5) * dx;
        solver.fields.p[[i, 0, 0]] = (k * x).cos();
    }
    // The two schemes carry velocity at different instants: the leapfrog at
    // t = -dt/2, the symmetric composition at t = 0 (synchronized). Seeding one
    // convention into the other is an O(dt) error in the initial state and
    // reports order 1.
    let seed_time = match scheme {
        TemporalScheme::Leapfrog => -dt / 2.0,
        TemporalScheme::Yoshida4 => 0.0,
    };
    for i in 0..N {
        let x_face = (i as f64 + 1.0) * dx;
        solver.fields.ux[[i, 0, 0]] =
            (k / (rho0 * omega)) * (k * x_face).sin() * (omega * seed_time).sin();
    }
    for _ in 0..steps {
        solver.step_forward().expect("step");
    }
    (0..N).fold(0.0_f64, |worst, i| {
        worst.max(solver.fields.p[[i, 0, 0]].abs())
    })
}

/// **Yoshida composition delivers fourth order in time**, which is the half of
/// Fullwave 2.5's "8th in space, 4th in time" that kwavers was missing.
///
/// Measured on the same benchmark as the second-order test, so the two slopes
/// are directly comparable rather than each being read against its own setup.
#[test]
fn the_yoshida_composition_is_fourth_order_in_time() {
    // Step counts sit inside the stable regime. The composition's largest
    // sub-step is |w0| ~ 1.70 times dt, so it goes unstable at a step the plain
    // leapfrog tolerates - at 50 steps here the error is 6.2e-3 against 6.3e-9
    // at 100, which is the stability edge rather than a convergence rate.
    let coarse = temporal_error(100, TemporalScheme::Yoshida4);
    let mid = temporal_error(200, TemporalScheme::Yoshida4);
    let fine = temporal_error(400, TemporalScheme::Yoshida4);
    for (a, b, label) in [(coarse, mid, "100->200"), (mid, fine, "200->400")] {
        let order = (a / b).log2();
        assert!(
            (3.7..=4.3).contains(&order),
            "{label}: observed temporal order {order:.3}, expected 4              (errors {a:.4e} -> {b:.4e})"
        );
    }
}

/// The fourth-order composition still conserves energy, at every spatial order.
///
/// Yoshida composition of a symplectic method is itself symplectic, so this is
/// the property that should survive — and it is the one that would break first
/// if the sub-step weights or the half-kick structure were wrong, since a
/// non-symplectic composition leaks energy steadily rather than failing loudly.
#[test]
fn the_yoshida_composition_conserves_energy() {
    const N: usize = 32;
    const STEPS: usize = 600;
    let dx = 1.0e-3_f64;
    let c0 = SOUND_SPEED_WATER_SIM;
    let rho0 = DENSITY_WATER_NOMINAL;

    for order in [2usize, 4, 6, 8] {
        let grid = Grid::new(N, 1, 1, dx, dx, dx).unwrap();
        let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
        // Well inside the composition's reduced stability limit: its largest
        // sub-step is |w0| ~ 1.70 times dt.
        let dt = 0.1 * dx / c0;
        let config = FdtdConfig {
            spatial_order: order,
            staggered_grid: true,
            enable_nonlinear: false,
            temporal_scheme: TemporalScheme::Yoshida4,
            dt,
            nt: STEPS + 1,
            ..Default::default()
        };
        let mut solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty()).unwrap();
        for i in 0..N {
            let x = (i as f64 - N as f64 / 2.0) / 4.0;
            solver.fields.p[[i, 0, 0]] = (-x * x).exp();
        }

        let energy = |s: &FdtdSolver| -> f64 {
            (0..N)
                .map(|i| {
                    s.fields.p[[i, 0, 0]].powi(2) / (2.0 * rho0 * c0 * c0)
                        + rho0 * s.fields.ux[[i, 0, 0]].powi(2) / 2.0
                })
                .sum()
        };
        let initial = energy(&solver);
        for _ in 0..STEPS {
            solver.step_forward().expect("step");
        }
        let drift = (energy(&solver) - initial).abs() / initial;
        assert!(
            drift < 0.05,
            "order {order}: energy drifted {:.3} % under the fourth-order composition",
            100.0 * drift
        );
    }
}

/// What the composition supports, and what it still refuses.
///
/// Absorption was refused when KW-SOL-092 landed and is supported now that the
/// coefficients are derived per sub-step (KW-SOL-093); the collocated path is
/// still refused, because the composition is built on the staggered updates.
#[test]
fn fourth_order_time_accepts_absorption_and_refuses_the_collocated_path() {
    let base = FdtdConfig {
        staggered_grid: true,
        temporal_scheme: TemporalScheme::Yoshida4,
        ..Default::default()
    };
    assert!(
        base.validate().is_ok(),
        "lossless staggered must be allowed"
    );

    let absorbing = FdtdConfig {
        absorption: FdtdAbsorption::PowerLawRelaxation {
            reference_frequency_hz: 1.0e6,
            band_min_hz: 0.5e6,
            band_max_hz: 3.0e6,
            relaxation_arms: 3,
        },
        ..base.clone()
    };
    assert!(
        absorbing.validate().is_ok(),
        "absorption under composition is supported since KW-SOL-093: coefficients are          derived per sub-step rather than precomputed for one step"
    );

    let collocated = FdtdConfig {
        staggered_grid: false,
        ..base
    };
    assert!(
        collocated.validate().is_err(),
        "fourth-order time is staggered-only"
    );
}
