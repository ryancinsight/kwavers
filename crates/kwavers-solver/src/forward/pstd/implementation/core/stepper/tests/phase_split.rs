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
//! recording). The two spectral phases are instead timed against their own
//! transforms, one arm per repeat in one loop, so transform and kernel time
//! separate under whatever load the host carries. Run in release:
//!
//! ```text
//! cargo nextest run -p kwavers-solver --release --run-ignored only \
//!     -E 'test(pstd_step_phase_split)' --no-capture
//! ```

use crate::forward::pstd::config::{BoundaryConfig, KSpaceMethod, PSTDConfig};
use crate::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use crate::phase_timing::PhaseTimer;
use kwavers_boundary::cpml::CPMLConfig;
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_grid::Grid;
use kwavers_math::fft::Fft3dInOutExt;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::GridSource;
use leto::Array3;

/// Cells per axis, the grid the FDTD split used.
const N: usize = 64;
/// Grid spacing, in metres.
const DX: f64 = 1.0e-4;
/// Repeats per phase loop, after warming plans, caches and task pools.
const TIMER: PhaseTimer = PhaseTimer {
    repeats: 200,
    warm: 20,
};

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

    let step = TIMER.single(&mut solver, |s| {
        s.step_forward().expect("step");
    });
    // Each spectral phase is timed against the transforms it runs, one arm per
    // repeat: the velocity update runs one forward and three inverse
    // transforms, the density update three of each. Both transform arms write
    // scratch the step rewrites every call.
    let (velocity, velocity_transforms) = TIMER.pair(
        &mut solver,
        |s| {
            s.update_velocity(dt).expect("velocity update");
        },
        |s| {
            s.fft.forward_r2c_into(&s.fields.p, &mut s.p_k);
            s.fft.inverse_c2r_into(&mut s.grad_k, &mut s.dpx);
            s.fft.inverse_c2r_into(&mut s.grad_k, &mut s.dpy);
            s.fft.inverse_c2r_into(&mut s.grad_k, &mut s.div_u);
        },
    );
    let (density, density_transforms) = TIMER.pair(
        &mut solver,
        |s| {
            s.update_density(dt).expect("density update");
        },
        |s| {
            s.fft.forward_r2c_into(&s.fields.ux, &mut s.ux_k);
            s.fft.inverse_c2r_into(&mut s.grad_k, &mut s.div_ux);
            s.fft.forward_r2c_into(&s.fields.uy, &mut s.ux_k);
            s.fft.inverse_c2r_into(&mut s.grad_k, &mut s.div_uy);
            s.fft.forward_r2c_into(&s.fields.uz, &mut s.ux_k);
            s.fft.inverse_c2r_into(&mut s.grad_k, &mut s.div_uz);
        },
    );
    let pressure = TIMER.single(&mut solver, |s| {
        s.update_pressure(dt).expect("pressure update");
    });
    let back_to_back = TIMER.single(&mut solver, |s| {
        s.update_velocity(dt).expect("velocity update");
        s.update_density(dt).expect("density update");
        s.update_pressure(dt).expect("pressure update");
    });

    let (step, step_fastest) = (step.mean, step.fastest);
    let (velocity, velocity_transforms) = (velocity.mean, velocity_transforms.mean);
    let (density, density_transforms) = (density.mean, density_transforms.mean);
    let (pressure, back_to_back) = (pressure.mean, back_to_back.mean);

    // A run that diverged would time NaN arithmetic, not the step.
    assert!(
        solver.fields.p.iter().all(|value| value.is_finite()),
        "the timed run stayed finite"
    );

    eprintln!(
        "pstd split-field: step {step:.0} us (fastest {step_fastest:.0});          velocity {velocity:.0} = transforms {velocity_transforms:.0} + kernels {:.0};          density {density:.0} = transforms {density_transforms:.0} + kernels {:.0};          pressure {pressure:.0}; back to back {back_to_back:.0} (interaction {:.0});          rest of step {:.0}",
        velocity - velocity_transforms,
        density - density_transforms,
        back_to_back - velocity - density - pressure,
        step - back_to_back,
    );
}
