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
use super::step::update_components;
use super::TimeIntegrator;
use crate::forward::elastic::swe::boundary::{ElasticSwePMLBoundary, SwePmlConfig};
use crate::forward::elastic::swe::scratch::ElasticStepScratch;
use crate::forward::elastic::swe::types::ElasticWaveField;
use crate::phase_timing::PhaseTimer;
use kwavers_core::constants::fundamental::DENSITY_WATER_NOMINAL;
use kwavers_core::traversal::{zip_mut, zip_mut_triple};
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

    let acceleration = |state: &mut State| {
        integrator
            .compute_acceleration::<SpatialStress>(&state.field, &mut state.scratch, None, 0.0)
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
    let updates = |state: &mut State| {
        half_velocity(state);
        displacement(state);
        half_velocity(state);
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
            half_velocity(s);
            displacement(s);
            acceleration(s);
            half_velocity(s);
            damping(s);
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
            let mut into = scratch.other_derivative.view_mut();
            match axis {
                Axis::X => derivatives.apply_x_into(from.view(), &mut into),
                Axis::Y => derivatives.apply_y_into(from.view(), &mut into),
                Axis::Z => derivatives.apply_z_into(from.view(), &mut into),
            }
            .expect("grid-shaped fields");
        }
    };
    // The three diagonal stresses by both routes: three sweeps into the shear
    // fields and a pointwise combination, against one fused pass that sweeps
    // the strains once and writes all three stresses.
    let composed_diagonal = |s: &mut State| {
        let State { field, scratch } = s;
        let ElasticStepScratch {
            sxx,
            syy,
            szz,
            sxy,
            sxz,
            syz,
            ..
        } = scratch;
        for (axis, (from, into)) in [Axis::X, Axis::Y, Axis::Z].into_iter().zip(
            [&field.ux, &field.uy, &field.uz]
                .into_iter()
                .zip([&mut *sxy, &mut *sxz, &mut *syz]),
        ) {
            let mut into = into.view_mut();
            match axis {
                Axis::X => derivatives.apply_x_into(from.view(), &mut into),
                Axis::Y => derivatives.apply_y_into(from.view(), &mut into),
                Axis::Z => derivatives.apply_z_into(from.view(), &mut into),
            }
            .expect("grid-shaped fields");
        }
        zip_mut_triple(
            sxx.view_mut(),
            syy.view_mut(),
            szz.view_mut(),
            (sxy.view(), sxz.view(), syz.view(), lambda.view(), mu.view()),
            |xx, yy, zz, (&exx, &eyy, &ezz, &la, &mv)| {
                let la2mu = 2.0f64.mul_add(mv, la);
                *xx = la2mu.mul_add(exx, la * (eyy + ezz));
                *yy = la2mu.mul_add(eyy, la * (exx + ezz));
                *zz = la2mu.mul_add(ezz, la * (exx + eyy));
            },
        );
    };
    let fused_diagonal = |s: &mut State| {
        let State { field, scratch } = s;
        let ElasticStepScratch { sxx, syy, szz, .. } = scratch;
        let (mut xx, mut yy, mut zz) = (sxx.view_mut(), syy.view_mut(), szz.view_mut());
        derivatives
            .map_axis_derivatives_triple(
                [
                    (Axis::X, field.ux.view()),
                    (Axis::Y, field.uy.view()),
                    (Axis::Z, field.uz.view()),
                ],
                [lambda.view(), mu.view()],
                [&mut xx, &mut yy, &mut zz],
                |[exx, eyy, ezz], [la, mv]| {
                    let la2mu = 2.0f64.mul_add(mv, la);
                    [
                        la2mu.mul_add(exx, la * (eyy + ezz)),
                        la2mu.mul_add(eyy, la * (exx + ezz)),
                        la2mu.mul_add(ezz, la * (exx + eyy)),
                    ]
                },
            )
            .expect("grid-shaped fields");
    };
    let (composed_diag, fused_diag) = TIMER.pair(&mut state, composed_diagonal, fused_diagonal);

    // The three shear stresses by both routes. Two sweeps into scratch and a
    // scaled sum, against one fused pass that reads both fields and mu once
    // per lane. Alternating inside one loop is what makes the difference
    // readable: timing the routes in separate processes measured a bimodal
    // 375/480 us split, wider than the difference itself.
    let shear_terms = [
        ((Axis::Y, 0usize), (Axis::X, 1usize)),
        ((Axis::Z, 0), (Axis::X, 2)),
        ((Axis::Z, 1), (Axis::Y, 2)),
    ];
    let composed_shears = |s: &mut State| {
        let State { field, scratch } = s;
        let components = [&field.ux, &field.uy, &field.uz];
        let ElasticStepScratch {
            sxy,
            sxz,
            syz,
            derivative,
            other_derivative,
            ..
        } = scratch;
        for (((first_axis, first), (second_axis, second)), out) in shear_terms
            .into_iter()
            .zip([&mut *sxy, &mut *sxz, &mut *syz])
        {
            let mut into = derivative.view_mut();
            match first_axis {
                Axis::X => derivatives.apply_x_into(components[first].view(), &mut into),
                Axis::Y => derivatives.apply_y_into(components[first].view(), &mut into),
                Axis::Z => derivatives.apply_z_into(components[first].view(), &mut into),
            }
            .expect("grid-shaped fields");
            let mut into = other_derivative.view_mut();
            match second_axis {
                Axis::X => derivatives.apply_x_into(components[second].view(), &mut into),
                Axis::Y => derivatives.apply_y_into(components[second].view(), &mut into),
                Axis::Z => derivatives.apply_z_into(components[second].view(), &mut into),
            }
            .expect("grid-shaped fields");
            zip_mut(
                out.view_mut(),
                (derivative.view(), other_derivative.view(), mu.view()),
                |value, (&a, &b, &scale)| *value = scale * (a + b),
            );
        }
    };
    let fused_shears = |s: &mut State| {
        let State { field, scratch } = s;
        let components = [&field.ux, &field.uy, &field.uz];
        let ElasticStepScratch { sxy, sxz, syz, .. } = scratch;
        for (((first_axis, first), (second_axis, second)), out) in shear_terms
            .into_iter()
            .zip([&mut *sxy, &mut *sxz, &mut *syz])
        {
            derivatives
                .map_axis_derivatives(
                    [
                        (first_axis, components[first].view()),
                        (second_axis, components[second].view()),
                    ],
                    [mu.view()],
                    &mut out.view_mut(),
                    |[a, b], [scale]| scale * (a + b),
                )
                .expect("grid-shaped fields");
        }
    };
    let (composed_shear, fused_shear) = TIMER.pair(&mut state, composed_shears, fused_shears);

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

    let (update_total, damping_total) = TIMER.pair(&mut state, updates, damping);

    for (label, pick) in [
        (
            "mean",
            (|p: crate::phase_timing::Phase| p.mean) as fn(_) -> f64,
        ),
        ("fastest", |p| p.fastest),
    ] {
        eprintln!(
            "swe 64 cubed {label}: step {:.0} us; back to back {:.0} = 2 x acceleration {:.0} \
             (stress {:.0} + assembly {:.0}) + updates {:.0} + damping {:.0}; rest of step {:.0}",
            pick(step),
            pick(back_to_back),
            pick(acceleration_total),
            pick(stress),
            pick(acceleration_total) - pick(stress),
            pick(update_total),
            pick(damping_total),
            pick(step) - pick(back_to_back),
        );
        eprintln!(
            "swe 64 cubed {label}: diagonal composed {:.0} us, fused {:.0}",
            pick(composed_diag),
            pick(fused_diag),
        );
        eprintln!(
            "swe 64 cubed {label}: three shears composed {:.0} us, fused {:.0}",
            pick(composed_shear),
            pick(fused_shear),
        );
        eprintln!(
            "swe 64 cubed {label}: stress {:.0} us = 18 sweeps {:.0} + assembly {:.0}",
            pick(stress_again),
            pick(sweeps_only),
            pick(stress_again) - pick(sweeps_only),
        );
    }
}
