//! Cost of the tensor DG uniform-grid projection.
//!
//! The projection evaluates a tensor-product GLL polynomial at every uniform
//! grid point. Two things changed in its cost, and this bench measures both:
//!
//! - the basis is stored once per local position rather than once per grid
//!   index per axis, so building it no longer scales with the grid extent;
//! - one per-node weight set serves every variable projected at a point, so the
//!   four-field projection no longer repeats the products the single-field
//!   projection performs.
//!
//! `fields` is the four-field entry point: one weight set, four variables.
//! `per_variable` is the same four outputs produced by four calls to the
//! single-field entry point, which is the weight-recomputation shape the shared
//! path replaces. `pressure` times the single-output path on its own.
//!
//! Extents are the solver's planned 32³ and 48³, at polynomial order 2 and 3, so
//! the node count per element varies with the extent rather than the bench.
//! Time model: six benchmarks x (1 s warm-up + 2 s measurement) with ten
//! samples, about 20 s plus the four solver constructions -- inside the 300 s
//! suite bound.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kwavers_grid::Grid;
use kwavers_solver::forward::pstd::dg::{DGConfig, DgBoundaryCondition, DGSolver};
use leto::Array3;
use std::sync::Arc;
use std::time::Duration;

/// Grid extents the solver plans. Each must divide evenly by every order's
/// node count (`polynomial_order + 1`), which is 3 and 4 here: 24 and 48 are
/// the smallest pair that does.
const EXTENTS: [usize; 2] = [24, 48];

/// Polynomial orders: two node counts per element.
const ORDERS: [usize; 2] = [2, 3];

/// Physical spacing: unit cells keep the fixture independent of the medium.
const SPACING: f64 = 1.0;

fn solver(extent: usize, polynomial_order: usize) -> DGSolver {
    let grid = Arc::new(
        Grid::new(extent, extent, extent, SPACING, SPACING, SPACING).expect("valid grid"),
    );
    let config = DGConfig {
        polynomial_order,
        sound_speed: 2.0,
        boundary_conditions: [DgBoundaryCondition::Periodic; 3],
        ..DGConfig::default()
    };
    DGSolver::new(config, grid).expect("valid DG solver")
}

/// A state whose values vary across elements, nodes and variables, so no arm
/// can be skipped as a constant field.
fn state(solver: &DGSolver) -> Array3<f64> {
    let shape = solver.acoustic_tensor_state_shape().expect("state shape");
    Array3::from_shape_fn(shape, |[elem, node, var]| {
        (0.37 * elem as f64 + 0.61 * node as f64 + 0.19 * var as f64).sin()
    })
}

fn bench_projection(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("dg_uniform_projection");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for extent in EXTENTS {
        for order in ORDERS {
            let solver = solver(extent, order);
            let state = state(&solver);
            let shape = (extent, extent, extent);
            let mut pressure = Array3::zeros(shape);
            let mut ux = Array3::zeros(shape);
            let mut uy = Array3::zeros(shape);
            let mut uz = Array3::zeros(shape);
            let label = format!("{extent}^3_p{order}");

            group.bench_with_input(BenchmarkId::new("pressure", &label), &(), |bencher, ()| {
                bencher.iter(|| {
                    solver
                        .project_acoustic_tensor_pressure_to_uniform_grid(&state, &mut pressure)
                        .expect("projection");
                });
            });

            group.bench_with_input(BenchmarkId::new("fields", &label), &(), |bencher, ()| {
                bencher.iter(|| {
                    solver
                        .project_acoustic_tensor_fields_to_uniform_grid(
                            &state, &mut pressure, &mut ux, &mut uy, &mut uz,
                        )
                        .expect("projection");
                });
            });

            let mut alone = Array3::zeros(shape);
            group.bench_with_input(
                BenchmarkId::new("per_variable", &label),
                &(),
                |bencher, ()| {
                    bencher.iter(|| {
                        for _ in 0..4 {
                            solver
                                .project_acoustic_tensor_pressure_to_uniform_grid(&state, &mut alone)
                                .expect("projection");
                        }
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_projection);
criterion_main!(benches);
