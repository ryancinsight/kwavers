//! Phase split of one staggered FDTD step: the instrument behind the
//! attribution of `fdtd_step_64_cubed`.
//!
//! Each phase runs alone in its own loop of repeats: the three leapfrog
//! gradients, the velocity update that contains them, the three divergences,
//! the pressure update that contains those, and the whole step. Each loop
//! reports its mean (total elapsed over repeats), because means add where the
//! medians of separately sampled phases do not, and the fastest repeat. The
//! velocity and pressure updates minus their sweeps are the pointwise passes.
//! A last loop runs the two updates back to back, as a step does: its excess
//! over the two separate loops is the cost of alternating them, and the step
//! minus it is the finish phase (sources and sensor recording). Run in release
//! on a quiet host:
//!
//! ```text
//! cargo nextest run -p kwavers-solver --release --run-ignored only \
//!     -E 'test(fdtd_step_phase_split)' --no-capture
//! ```

use super::make_solver;
use crate::forward::fdtd::solver::FdtdSolver;
use leto_ops::Axis;
use std::time::{Duration, Instant};

/// Grid of kwavers' `fdtd_step_64_cubed` instrument, cells per axis.
const N: usize = 64;
/// That instrument's grid spacing, in metres.
const DX: f64 = 1.0e-4;
/// Its Courant factor: `make_solver` derives `dt = cfl · Δx / (√3 · c₀)`.
const CFL: f64 = 0.95;
/// Repeats per phase loop.
const REPEATS: usize = 300;
/// Repeats of a phase before its loop is timed, so plans, caches and task
/// pools are warm for that phase.
const WARM_REPEATS: usize = 20;

/// Mean and fastest repeat of `phase`, in microseconds, after warming it.
fn time_phase(solver: &mut FdtdSolver, mut phase: impl FnMut(&mut FdtdSolver)) -> (f64, f64) {
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

fn gradients(solver: &mut FdtdSolver) {
    let FdtdSolver {
        leapfrog_operator,
        fields,
        dvx_scratch,
        dvy_scratch,
        divergence_scratch,
        ..
    } = solver;
    for (axis, dst) in [
        (Axis::X, dvx_scratch),
        (Axis::Y, dvy_scratch),
        (Axis::Z, divergence_scratch),
    ] {
        leapfrog_operator
            .gradient_into(axis, fields.p.view(), &mut dst.view_mut())
            .expect("gradient");
    }
}

fn divergences(solver: &mut FdtdSolver) {
    let FdtdSolver {
        leapfrog_operator,
        fields,
        dvx_scratch,
        dvy_scratch,
        divergence_scratch,
        ..
    } = solver;
    for (axis, source, dst) in [
        (Axis::X, &fields.ux, dvx_scratch),
        (Axis::Y, &fields.uy, dvy_scratch),
        (Axis::Z, &fields.uz, divergence_scratch),
    ] {
        leapfrog_operator
            .divergence_into(axis, source.view(), &mut dst.view_mut())
            .expect("divergence");
    }
}

#[test]
#[ignore = "timing probe: run in release on a quiet host with --no-capture"]
fn fdtd_step_phase_split() {
    for spatial_order in [2, 4] {
        let mut solver = make_solver(N, DX, 1_500.0, 1_000.0, CFL, spatial_order);
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    let phase = (i as f64).mul_add(0.37, (j as f64).mul_add(0.53, k as f64 * 0.71));
                    solver.fields.p[[i, j, k]] = phase.sin();
                }
            }
        }
        let dt = solver.config.dt;

        let (step, step_fastest) = time_phase(&mut solver, |s| {
            s.step_forward().expect("step");
        });
        let (gradient_sweeps, _) = time_phase(&mut solver, gradients);
        let (velocity, _) = time_phase(&mut solver, |s| {
            s.update_velocity(dt).expect("velocity update");
        });
        let (divergence_sweeps, _) = time_phase(&mut solver, divergences);
        let (pressure, _) = time_phase(&mut solver, |s| {
            s.update_pressure(dt).expect("pressure update");
        });
        let (back_to_back, _) = time_phase(&mut solver, |s| {
            s.update_velocity(dt).expect("velocity update");
            s.update_pressure(dt).expect("pressure update");
        });

        // A run that diverged would time NaN arithmetic, not the step.
        assert!(
            solver.fields.p.iter().all(|value| value.is_finite()),
            "order {spatial_order}: the timed run stayed finite"
        );

        eprintln!(
            "order {spatial_order}: step {step:.0} us (fastest {step_fastest:.0}); \
             velocity {velocity:.0} = gradients {gradient_sweeps:.0} + pointwise {:.0}; \
             pressure {pressure:.0} = divergences {divergence_sweeps:.0} + accumulate and update {:.0}; \
             back to back {back_to_back:.0} (interaction {:.0}); finish {:.0}",
            velocity - gradient_sweeps,
            pressure - divergence_sweeps,
            back_to_back - velocity - pressure,
            step - back_to_back,
        );
    }
}
