//! Phase split of one staggered FDTD step: the instrument behind the
//! attribution of `fdtd_step_64_cubed`.
//!
//! The whole step runs in its own loop of repeats, and each update is timed
//! against the sweeps it contains — velocity against the three gradients,
//! pressure against the three divergences — one arm per repeat inside one
//! loop. Each loop reports its mean (total elapsed over repeats), because
//! means add where the medians of separately sampled phases do not, and the
//! fastest repeat. An update minus its sweeps is the pointwise pass; timing
//! the two in separate loops put that difference below zero on the PSTD probe
//! when the host load drifted between them.
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
use crate::phase_timing::PhaseTimer;
use leto_ops::Axis;

/// Grid of kwavers' `fdtd_step_64_cubed` instrument, cells per axis.
const N: usize = 64;
/// That instrument's grid spacing, in metres.
const DX: f64 = 1.0e-4;
/// Its Courant factor, which `make_solver` applies to the solver's own
/// staggered stability limit (`max_stable_dt`).
const CFL: f64 = 0.95;
/// Repeats per phase loop, after warming plans, caches and task pools.
const TIMER: PhaseTimer = PhaseTimer {
    repeats: 300,
    warm: 20,
};

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

        // The step is timed against the two updates it contains, one arm per
        // repeat: the difference is the rest of a step (sources, Dirichlet
        // enforcement and sensor recording), and timing the two in separate
        // loops made that difference negative — a step cannot cost less than
        // its own parts.
        let (step, back_to_back) = TIMER.pair(
            &mut solver,
            |s| {
                s.step_forward().expect("step");
            },
            |s| {
                s.update_velocity(dt).expect("velocity update");
                s.update_pressure(dt).expect("pressure update");
            },
        );
        // Each update is timed against the sweeps it contains, one arm per
        // repeat, so the difference between them is a reading of the pointwise
        // work rather than of whatever the host did between two loops.
        let (velocity, gradient_sweeps) = TIMER.pair(
            &mut solver,
            |s| {
                s.update_velocity(dt).expect("velocity update");
            },
            gradients,
        );
        let (pressure, divergence_sweeps) = TIMER.pair(
            &mut solver,
            |s| {
                s.update_pressure(dt).expect("pressure update");
            },
            divergences,
        );

        // A diverging run times the same arithmetic as a converging one —
        // floating point does not slow down until it reaches denormals — so
        // `is_finite` alone let this probe report a fourth-order split whose
        // field had reached 1e250. The initial condition is a unit sine, and a
        // stable run of these loops peaks near 1e5 at either order, so three
        // decades of headroom separates accumulation from divergence.
        const PEAK_BOUND: f64 = 1.0e8;
        let peak = solver
            .fields
            .p
            .iter()
            .fold(0.0_f64, |peak, value| peak.max(value.abs()));
        assert!(
            peak.is_finite() && peak < PEAK_BOUND,
            "order {spatial_order}: the timed run stayed bounded (peak {peak:e})"
        );

        // Both statistics, because they answer different questions: the means
        // add across phases and the fastest repeats compare two revisions.
        eprintln!(
            "order {spatial_order} mean: step {:.0} us; \
             velocity {:.0} = gradients {:.0} + pointwise {:.0}; \
             pressure {:.0} = divergences {:.0} + accumulate and update {:.0}; \
             back to back {:.0} (interaction {:.0}); rest of step {:.0}",
            step.mean,
            velocity.mean,
            gradient_sweeps.mean,
            velocity.mean - gradient_sweeps.mean,
            pressure.mean,
            divergence_sweeps.mean,
            pressure.mean - divergence_sweeps.mean,
            back_to_back.mean,
            back_to_back.mean - velocity.mean - pressure.mean,
            step.mean - back_to_back.mean,
        );
        eprintln!(
            "order {spatial_order} fastest: step {:.0} us; \
             velocity {:.0} = gradients {:.0} + pointwise {:.0}; \
             pressure {:.0} = divergences {:.0} + accumulate and update {:.0}",
            step.fastest,
            velocity.fastest,
            gradient_sweeps.fastest,
            velocity.fastest - gradient_sweeps.fastest,
            pressure.fastest,
            divergence_sweeps.fastest,
            pressure.fastest - divergence_sweeps.fastest,
        );
    }
}
