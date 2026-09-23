//! Phase split of one three-dimensional velocity-Verlet elastic step.
//!
//! The step is timed against its parts run back to back: two acceleration
//! evaluations, three component updates and the PML damping. The acceleration
//! is timed against the stress divergence it contains, and the component
//! updates against the damping. Each pair alternates inside one loop, so the
//! differences read the code rather than the host. Run in release:
//!
//! ```text
//! cargo nextest run -p kwavers-solver --release --run-ignored only \
//!     -E 'test(swe_step_phase_split)' --no-capture
//! ```

use super::acceleration::{SpatialStress, StressOperator};
use super::step::{kick_then_drift, update_components, KickDriftRoute};
use super::TimeIntegrator;
use crate::forward::elastic::swe::boundary::{ElasticSwePMLBoundary, SwePmlConfig};
use crate::forward::elastic::swe::scratch::ElasticStepScratch;
use crate::forward::elastic::swe::stress::{stress_kick_in_slabs, DensityScale, VelocityKick};
use crate::forward::elastic::swe::types::ElasticWaveField;
use crate::phase_timing::PhaseTimer;
use core::num::NonZeroUsize;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;

use kwavers_grid::Grid;
use leto::Array3;
use leto_ops::{Axis, FiniteDifference3D};

/// Cells per axis, the grid the FDTD and PSTD splits use.
const N: usize = 64;
/// Grid spacing, in metres.
const DX: f64 = 1.0e-3;
/// Lamé parameters of a soft solid, in pascals.
const LAMBDA: f64 = 1.0e9;
const MU: f64 = 1.0e9;
/// Squared width of the initial Gaussian pulse, in cells squared.
const PULSE_WIDTH_SQUARED: f64 = 64.0;
/// Fraction of the CFL-limited timestep.
const CFL: f64 = 0.5;
/// Repeats per phase loop, after warming caches and task pools.
const TIMER: PhaseTimer = PhaseTimer {
    repeats: 200,
    warm: 20,
};
/// The unit pulse spreads and decays into the absorbing layer; three decades
/// of headroom separates that from divergence, which times the same as a
/// valid run until denormals.
const PEAK_BOUND: f64 = 1.0e3;

struct State {
    field: ElasticWaveField,
    scratch: ElasticStepScratch,
}

#[test]
#[ignore = "timing probe: run in release on a quiet host with --no-capture"]
fn swe_step_phase_split() {
    let grid = Grid::new(N, N, N, DX, DX, DX).expect("valid grid");
    let lambda = Array3::from_elem([N; 3], LAMBDA);
    let mu = Array3::from_elem([N; 3], MU);
    let density = Array3::from_elem([N; 3], DENSITY_WATER_NOMINAL);
    let pml = ElasticSwePMLBoundary::new(&grid, SwePmlConfig::default());
    let integrator = TimeIntegrator::new(&grid, &lambda, &mu, &density, &pml);
    let dt = integrator.calculate_stable_timestep(CFL);

    let mut state = State {
        field: ElasticWaveField::new(N, N, N),
        scratch: ElasticStepScratch::new(N, N, N),
    };
    // A centred Gaussian pulse: the integrator assumes displacement that
    // vanishes towards the absorbing layer, and a field reaching the edges
    // grows secularly (`kw-swe-edge-growth`), which would time a different run.
    let centre = (N / 2) as f64;
    for i in 0..N {
        for j in 0..N {
            for k in 0..N {
                let r2 = [i, j, k]
                    .map(|n| (n as f64 - centre).powi(2))
                    .iter()
                    .sum::<f64>();
                state.field.ux[[i, j, k]] = (-r2 / PULSE_WIDTH_SQUARED).exp();
            }
        }
    }

    // Without a body force the evaluation kicks the velocities itself.
    let acceleration = |state: &mut State| {
        integrator
            .compute_acceleration::<SpatialStress>(
                &mut state.field,
                &mut state.scratch,
                None,
                0.0,
                0.5 * dt,
            )
            .expect("acceleration");
    };
    let half_velocity = |state: &mut State| {
        let State { field, scratch } = state;
        update_components::<SpatialStress>(
            &mut field.vx,
            &mut field.vy,
            &mut field.vz,
            &scratch.ax,
            &scratch.ay,
            &scratch.az,
            0.5 * dt,
        );
    };
    let displacement = |state: &mut State| {
        let ElasticWaveField {
            ux,
            uy,
            uz,
            vx,
            vy,
            vz,
            ..
        } = &mut state.field;
        update_components::<SpatialStress>(ux, uy, uz, vx, vy, vz, dt);
    };
    let damping = |state: &mut State| {
        integrator.apply_pml_damping_for::<SpatialStress>(&mut state.field, dt, &mut state.scratch);
    };

    let (step, back_to_back) = TIMER.pair(
        &mut state,
        |s| {
            integrator
                .step(&mut s.field, dt, None, &mut s.scratch)
                .expect("step");
        },
        // The same velocity-Verlet sequence `integrate` runs, so both arms
        // advance the field physically.
        |s| {
            acceleration(s);
            displacement(s);
            acceleration(s);
            damping(s);
        },
    );
    // The kick and drift as two traversals against one. Both arms advance
    // the field by the same velocity-Verlet half-step and full step, on the
    // same accelerations, so the difference is the traversal count: two
    // parallel regions writing three fields each, against one writing six.
    let (composed_kick_drift, fused_kick_drift) = TIMER.pair(
        &mut state,
        |s| {
            half_velocity(s);
            displacement(s);
        },
        |s| {
            kick_then_drift::<SpatialStress>(
                &mut s.field,
                [&s.scratch.ax, &s.scratch.ay, &s.scratch.az],
                0.5 * dt,
                dt,
                KickDriftRoute::Fused,
            );
        },
    );
    let (acceleration_total, stress) = TIMER.pair(&mut state, acceleration, |s| {
        SpatialStress::evaluate(&grid, &lambda, &mu, &s.field, &mut s.scratch);
    });
    // The evaluation against the sweeps it contains: nine derivatives of the
    // displacement components and nine of the stress components, written into
    // the same scratch fields the evaluation writes. The difference is the
    // pointwise assembly — six stress components and three divergence sums.
    let derivatives = FiniteDifference3D::central_fourth_order(grid.dx, grid.dy, grid.dz)
        .expect("a grid has positive spacing");
    let sweeps = |s: &mut State| {
        let State { field, scratch } = s;
        let sweep = |axis: Axis, from: &Array3<f64>, into: &mut Array3<f64>| {
            let mut into = into.view_mut();
            match axis {
                Axis::X => derivatives.apply_x_into(from.view(), &mut into),
                Axis::Y => derivatives.apply_y_into(from.view(), &mut into),
                Axis::Z => derivatives.apply_z_into(from.view(), &mut into),
            }
            .expect("grid-shaped fields");
        };
        for (axis, from) in [
            (Axis::X, &field.ux),
            (Axis::Y, &field.uy),
            (Axis::Z, &field.uz),
            (Axis::Y, &field.ux),
            (Axis::X, &field.uy),
            (Axis::Z, &field.ux),
            (Axis::X, &field.uz),
            (Axis::Z, &field.uy),
            (Axis::Y, &field.uz),
        ] {
            sweep(axis, from, &mut scratch.derivative);
        }
        for (axis, from) in [
            (Axis::X, &scratch.sxx),
            (Axis::Y, &scratch.sxy),
            (Axis::Z, &scratch.sxz),
            (Axis::X, &scratch.sxy),
            (Axis::Y, &scratch.syy),
            (Axis::Z, &scratch.syz),
            (Axis::X, &scratch.sxz),
            (Axis::Y, &scratch.syz),
            (Axis::Z, &scratch.szz),
        ] {
            sweep(axis, from, &mut scratch.other_derivative);
        }
    };
    let (stress_again, sweeps_only) = TIMER.pair(
        &mut state,
        |s| {
            SpatialStress::evaluate(&grid, &lambda, &mu, &s.field, &mut s.scratch);
        },
        sweeps,
    );
    // Only the two pairs above advance the field physically; the update and
    // damping arms below reapply one acceleration, so the guard reads the field
    // before them.
    let peak = [&state.field.ux, &state.field.uy, &state.field.uz]
        .into_iter()
        .flat_map(|component| component.iter())
        .fold(0.0_f64, |peak, value| peak.max(value.abs()));
    assert!(
        peak.is_finite() && peak < PEAK_BOUND,
        "the timed run stayed bounded: peak {peak}"
    );

    let (drift_total, damping_total) = TIMER.pair(&mut state, displacement, damping);

    for (label, pick) in [
        (
            "mean",
            (|p: crate::phase_timing::Phase| p.mean) as fn(_) -> f64,
        ),
        ("fastest", |p| p.fastest),
    ] {
        eprintln!(
            "swe 64 cubed {label}: step {:.0} us; back to back {:.0} = 2 x kick {:.0} \
             (stress {:.0} + assembly {:.0}) + drift {:.0} + damping {:.0}; rest of step {:.0}",
            pick(step),
            pick(back_to_back),
            pick(acceleration_total),
            pick(stress),
            pick(acceleration_total) - pick(stress),
            pick(drift_total),
            pick(damping_total),
            pick(step) - pick(back_to_back),
        );
        eprintln!(
            "swe 64 cubed {label}: kick and drift composed {:.0} us, fused {:.0}",
            pick(composed_kick_drift),
            pick(fused_kick_drift),
        );
        eprintln!(
            "swe 64 cubed {label}: stress {:.0} us = 18 sweeps {:.0} + assembly {:.0}",
            pick(stress_again),
            pick(sweeps_only),
            pick(stress_again) - pick(sweeps_only),
        );
    }
}

/// A half step, in seconds, of the order the CFL limit gives these grids;
/// the sweep times traversals, which do not depend on it.
const HALF_DT: f64 = 1.0e-7;

/// Repeats per arm of the slab sweep: 23 evaluations of each route at each
/// of eight sizes and five slab heights -- about 10 ms apiece at 128 cubed,
/// which is half the total -- keep the whole sweep near 5 s, inside the test
/// budget, while the fastest repeat still reads the same as the 200-repeat
/// probe at 96 cubed.
const SWEEP_TIMER: PhaseTimer = PhaseTimer {
    repeats: 20,
    warm: 3,
};

/// The velocity kick evaluated whole against the same evaluation in slabs of
/// x-planes through the stress window, at grid sizes either side of the
/// last-level cache. Both write every plane of every output with the same
/// arithmetic, so each pair differs only in which planes are in flight at
/// once: a slab's stresses are read back while still resident, where the
/// whole-grid form writes all of them before reading the first.
#[test]
#[ignore = "timing probe: run in release on a quiet host with --no-capture"]
fn swe_kick_slab_sweep() {
    for n in [64, 72, 76, 80, 88, 92, 96, 128] {
        let grid = Grid::new(n, n, n, DX, DX, DX).expect("valid grid");
        let lambda = Array3::from_elem([n; 3], LAMBDA);
        let mu = Array3::from_elem([n; 3], MU);
        let mut state = State {
            field: ElasticWaveField::new(n, n, n),
            scratch: ElasticStepScratch::new(n, n, n),
        };
        let centre = (n / 2) as f64;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let r2 = [i, j, k]
                        .map(|c| (c as f64 - centre).powi(2))
                        .iter()
                        .sum::<f64>();
                    state.field.ux[[i, j, k]] = (-r2 / PULSE_WIDTH_SQUARED).exp();
                }
            }
        }
        let reciprocal = DENSITY_WATER_NOMINAL.recip();
        let evaluate_in = |s: &mut State, planes: usize| {
            stress_kick_in_slabs(
                &grid,
                &lambda,
                &mu,
                &mut s.field,
                &VelocityKick {
                    scale: DensityScale::UniformReciprocal(reciprocal),
                    half_dt: HALF_DT,
                },
                &mut s.scratch,
                NonZeroUsize::new(planes).expect("a slab holds at least one plane"),
            );
        };
        for planes in [8, 12, 16, 24, 32] {
            let (whole, slabbed) = SWEEP_TIMER.pair(
                &mut state,
                |s| evaluate_in(s, n),
                |s| evaluate_in(s, planes),
            );
            eprintln!(
                "swe {n} cubed fastest: kick whole {:.0} us, in {planes}-plane slabs {:.0}",
                whole.fastest, slabbed.fastest,
            );
        }
    }
}
