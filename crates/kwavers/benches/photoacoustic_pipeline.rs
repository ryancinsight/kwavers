//! Baseline for the photoacoustic pipeline on a 32×32×16 grid.
//!
//! Three stages are timed separately over one simulator, grid and medium:
//! the diffusion solve for the optical fluence (`compute_fluence`), the
//! generation theorem `p₀ = Γ μₐ Φ` (`compute_initial_pressure`), and the
//! 400-step wave propagation plus back-projection (`simulate`). The fluence
//! solve and the wave propagation dominate; the pressure product is a single
//! pass over the grid and shows whether anything outside those two stages
//! has become visible.
//!
//! The test `photoacoustic_validation` checks the values these stages
//! produce; elapsed time is a property of the host and the build profile, so
//! it is read from this bench and never asserted in a test.
//!
//! # Reading the numbers
//!
//! A change in the `fluence` time alone points at the diffusion solver's
//! iteration count; a change in `simulate` alone points at the FDTD or
//! reconstruction kernels.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers_grid::Grid;
use kwavers_medium::homogeneous::HomogeneousMedium;
use kwavers_simulation::modalities::photoacoustic::{
    PhotoacousticParameters, PhotoacousticSimulator,
};
use std::time::Duration;

const NX: usize = 32;
const NY: usize = 32;
const NZ: usize = 16;

fn bench_photoacoustic_pipeline(c: &mut Criterion) {
    let grid = Grid::new(NX, NY, NZ, 0.0002, 0.0002, 0.0004).expect("valid grid");
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    let mut simulator =
        PhotoacousticSimulator::new(grid, PhotoacousticParameters::default(), &medium)
            .expect("valid simulator");
    let fluence = simulator.compute_fluence().expect("fluence");
    let initial_pressure = simulator
        .compute_initial_pressure(&fluence)
        .expect("initial pressure");

    let mut group = c.benchmark_group("photoacoustic_pipeline");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5));

    group.bench_function("fluence", |b| {
        b.iter(|| black_box(&simulator).compute_fluence().expect("fluence"));
    });
    group.bench_function("initial_pressure", |b| {
        b.iter(|| {
            black_box(&simulator)
                .compute_initial_pressure(black_box(&fluence))
                .expect("initial pressure")
        });
    });
    group.bench_function("simulate", |b| {
        b.iter(|| {
            black_box(&mut simulator)
                .simulate(black_box(&initial_pressure))
                .expect("simulation")
        });
    });

    group.finish();
}

criterion_group!(benches, bench_photoacoustic_pipeline);
criterion_main!(benches);
