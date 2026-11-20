use super::make_solver;
use crate::forward::fdtd::config::FdtdConfig;
use crate::forward::fdtd::config::TemporalScheme;
use crate::forward::fdtd::solver::FdtdSolver;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::GridSource;

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
    check_absorption_power_law(TemporalScheme::Leapfrog);
}

/// **The same law, under the fourth-order composition.**
///
/// This is the acceptance for KW-SOL-093: the exponential-integrator
/// coefficients are now derived per sub-step, so each of `w1·dt`, `w0·dt`,
/// `w1·dt` decays by its own `e^{-h/tau}` instead of all three reusing the
/// full-step value. Held to the same bound as the leapfrog path, because a
/// weaker one would not demonstrate anything.
#[test]
fn absorption_reproduces_prescribed_power_law_under_composition() {
    check_absorption_power_law(TemporalScheme::Yoshida4);
}

fn check_absorption_power_law(scheme: TemporalScheme) {
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
            temporal_scheme: scheme,
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
        let absorption = solver
            .absorption
            .as_ref()
            .expect("absorbing configuration produced no absorption state");
        // The state must carry the four relaxation arms the config asked for;
        // an empty or defaulted state would satisfy a bare `is_some`.
        assert_eq!(absorption.arm_count(), 4);

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
        let expected = kwavers_medium::absorption::power_law::power_law_db_cm_to_np_m(
            ALPHA0_DB,
            y,
            omega / TAU,
        );

        assert!(
            (measured - expected).abs() <= 0.15 * expected,
            "{scheme:?}, y={y}: measured alpha {measured:.3} vs prescribed {expected:.3} Np/m              at {:.3} MHz",
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
