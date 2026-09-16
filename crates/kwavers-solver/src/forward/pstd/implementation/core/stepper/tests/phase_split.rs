//! Phase split of one split-field PSTD step, the instrument that attributes
//! `pstd_long_run`.
//!
//! Each phase runs alone in its own loop of repeats: the velocity update, the
//! density update, the pressure update, and the whole step. Each loop reports
//! its mean (total elapsed over repeats), because means add where the medians
//! of separately sampled phases do not, and its fastest repeat. A last loop
//! runs the three updates back to back, as a step does: its excess over the
//! three separate loops is the cost of alternating them, and the step minus it
//! is the rest of a step (sources, Dirichlet enforcement and sensor
//! recording). Run in release on a quiet host:
//!
//! ```text
//! cargo nextest run -p kwavers-solver --release --run-ignored only \
//!     -E 'test(pstd_step_phase_split)' --no-capture
//! ```

use crate::forward::pstd::config::{BoundaryConfig, KSpaceMethod, PSTDConfig};
use crate::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers_boundary::cpml::CPMLConfig;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::GridSource;
use leto::Array3;
use std::time::{Duration, Instant};

/// Cells per axis, the grid the FDTD split used.
const N: usize = 64;
/// Grid spacing, in metres.
const DX: f64 = 1.0e-4;
/// Repeats per phase loop.
const REPEATS: usize = 200;
/// Repeats of a phase before its loop is timed, so plans, caches and task
/// pools are warm for that phase.
const WARM_REPEATS: usize = 20;

/// Mean and fastest repeat of `phase`, in microseconds, after warming it.
fn time_phase(solver: &mut PSTDSolver, mut phase: impl FnMut(&mut PSTDSolver)) -> (f64, f64) {
    for _ in 0..WARM_REPEATS {
        phase(solver);
    }
    let mut total = Duration::ZERO;
    let mut fastest = Duration::MAX;
    for _ in 0..REPEATS {
        let start = Instant::now();
        phase(solver);
        let elapsed = start.elapsed();
        total += elapsed;
        fastest = fastest.min(elapsed);
    }
    let micros = |d: Duration| d.as_secs_f64() * 1.0e6;
    (micros(total) / REPEATS as f64, micros(fastest))
}

/// A water solver with a centred initial pressure, CPML as a long run carries.
fn probe_solver() -> PSTDSolver {
    let grid = Grid::new(N, N, N, DX, DX, DX).expect("valid grid");
    let medium = HomogeneousMedium::new(
        DENSITY_WATER_NOMINAL,
        SOUND_SPEED_WATER_SIM,
        0.0,
        0.0,
        &grid,
    );
    let mut p0 = Array3::zeros((N, N, N));
    p0[[N / 2, N / 2, N / 2]] = 1.0;
    let source = GridSource {
        p0: Some(p0),
        ..GridSource::new_empty()
    };
    let config = PSTDConfig {
        dt: 0.3 * DX / SOUND_SPEED_WATER_SIM,
        nt: 1_000,
        boundary: BoundaryConfig::CPML(CPMLConfig::with_thickness(N / 4)),
        smooth_sources: false,
        kspace_method: KSpaceMethod::StandardPSTD,
        ..PSTDConfig::default()
    };
    PSTDSolver::new(config, grid, &medium, source).expect("valid solver")
}

#[test]
#[ignore = "timing probe: run in release on a quiet host with --no-capture"]
fn pstd_step_phase_split() {
    let mut solver = probe_solver();
    let dt = solver.config.dt;

    let (step, step_fastest) = time_phase(&mut solver, |s| {
        s.step_forward().expect("step");
    });
    let (velocity, _) = time_phase(&mut solver, |s| {
        s.update_velocity(dt).expect("velocity update");
    });
    let (density, _) = time_phase(&mut solver, |s| {
        s.update_density(dt).expect("density update");
    });
    let (pressure, _) = time_phase(&mut solver, |s| {
        s.update_pressure(dt).expect("pressure update");
    });
    let (back_to_back, _) = time_phase(&mut solver, |s| {
        s.update_velocity(dt).expect("velocity update");
        s.update_density(dt).expect("density update");
        s.update_pressure(dt).expect("pressure update");
    });

    // A run that diverged would time NaN arithmetic, not the step.
    assert!(
        solver.fields.p.iter().all(|value| value.is_finite()),
        "the timed run stayed finite"
    );

    eprintln!(
        "pstd split-field: step {step:.0} us (fastest {step_fastest:.0}); \
         velocity {velocity:.0}; density {density:.0}; pressure {pressure:.0}; \
         back to back {back_to_back:.0} (interaction {:.0}); rest of step {:.0}",
        back_to_back - velocity - density - pressure,
        step - back_to_back,
    );
}
