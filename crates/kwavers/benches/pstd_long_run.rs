//! Per-step cost of a PSTD run in its steady state.
//!
//! `fft3d_baseline` times the spectral pair a step is built on; nothing timed
//! the step itself, so the share the transforms hold against the element-wise
//! updates, the boundary and the absorption operator was inferred rather than
//! measured. A long run is that step repeated, so its wall clock is this
//! number times the step count.
//!
//! Each arm builds one solver with a Gaussian initial pressure, advances it
//! past its first steps (plan construction and thread-local scratch belong to
//! setup, not to a step), and then times `step_forward` on the same solver:
//! successive iterations are successive steps of one run, which is the
//! quantity a long run pays. The step's arithmetic does not branch on field
//! values, so the advancing field does not change the work per iteration.
//!
//! Arms:
//! - `lossless`: first-order split-field PSTD with a CPML boundary.
//! - `power_law`: the same with uniform power-law absorption, which adds the
//!   fractional-Laplacian transforms.
//! - `full_kspace`: the second-order k-space propagator.
//!
//! Extents are the solver's planned 32³ and 64³ (see `fft3d_baseline`), with a
//! CPML of a quarter of the extent per side, so the interior is half the grid.
//!
//! Time model: six benchmarks × (1 s warm-up + 3 s measurement) with ten
//! samples, about 25 s plus construction — inside the 300 s suite bound.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kwavers_boundary::cpml::CPMLConfig;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use kwavers_solver::forward::pstd::config::{BoundaryConfig, KSpaceMethod, PSTDConfig};
use kwavers_solver::forward::pstd::PSTDSolver;
use kwavers_source::GridSource;
use leto::Array3;
use std::time::Duration;

/// Steps the solver is configured for: an upper bound on what criterion can
/// run, not a target. One solver serves one benchmark for about 1 s warm-up
/// plus 3 s measurement, and a step touches at least 32^3 = 32,768 cells, so it
/// cannot take less than about 1 us; 2^32 steps would therefore take over an
/// hour against a 300 s suite bound. The solver's `nt` also sizes its
/// recording (`nt + 1` states), so `usize::MAX` -- the value this replaced --
/// was rejected at construction.
const STEP_BUDGET: usize = 1 << 32;

/// Grid extents the solver plans.
const EXTENTS: [usize; 2] = [32, 64];

/// Steps taken before timing: the first steps build plans and scratch.
const SETUP_STEPS: usize = 4;

/// Water at the simulation's nominal properties.
const DENSITY: f64 = 1_000.0;
const SOUND_SPEED: f64 = 1_500.0;
const SPACING: f64 = 1.0e-4;

/// CFL number: `dt = CFL · dx / c`, inside the PSTD stability limit.
const CFL: f64 = 0.3;

#[derive(Clone, Copy)]
enum Arm {
    Lossless,
    PowerLaw,
    FullKSpace,
}

impl Arm {
    const ALL: [Self; 3] = [Self::Lossless, Self::PowerLaw, Self::FullKSpace];

    fn name(self) -> &'static str {
        match self {
            Self::Lossless => "lossless",
            Self::PowerLaw => "power_law",
            Self::FullKSpace => "full_kspace",
        }
    }
}

/// A Gaussian pressure ball of radius `n / 8` at the grid centre.
fn initial_pressure(n: usize) -> Array3<f64> {
    let centre = (n as f64 - 1.0) / 2.0;
    let width = n as f64 / 8.0;
    let mut p0 = Array3::zeros((n, n, n));
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let r2 = [i, j, k]
                    .iter()
                    .map(|&x| (x as f64 - centre).powi(2))
                    .sum::<f64>();
                p0[[i, j, k]] = (-r2 / (width * width)).exp();
            }
        }
    }
    p0
}

fn solver(n: usize, arm: Arm) -> PSTDSolver {
    let grid = Grid::new(n, n, n, SPACING, SPACING, SPACING).expect("valid grid");
    let medium = HomogeneousMedium::new(DENSITY, SOUND_SPEED, 0.0, 0.0, &grid);
    let source = GridSource {
        p0: Some(initial_pressure(n)),
        ..GridSource::new_empty()
    };
    let (absorption_mode, kspace_method) = match arm {
        Arm::Lossless => (AbsorptionMode::Lossless, KSpaceMethod::StandardPSTD),
        Arm::PowerLaw => (
            AbsorptionMode::PowerLaw {
                alpha_coeff: Some(0.75),
                alpha_power: 1.5,
            },
            KSpaceMethod::StandardPSTD,
        ),
        Arm::FullKSpace => (AbsorptionMode::Lossless, KSpaceMethod::FullKSpace),
    };
    let config = PSTDConfig {
        dt: CFL * SPACING / SOUND_SPEED,
        nt: STEP_BUDGET,
        boundary: BoundaryConfig::CPML(CPMLConfig::with_thickness(n / 4)),
        absorption_mode,
        kspace_method,
        smooth_sources: false,
        ..Default::default()
    };
    let mut solver = PSTDSolver::new(config, grid, &medium, source).expect("valid solver");
    for _ in 0..SETUP_STEPS {
        solver.step_forward().expect("setup step");
    }
    solver
}

fn bench_pstd_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("pstd_step");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for &n in &EXTENTS {
        for arm in Arm::ALL {
            let mut solver = solver(n, arm);
            group.bench_with_input(BenchmarkId::new(arm.name(), n), &n, |b, _| {
                b.iter(|| solver.step_forward().expect("stable step"));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_pstd_step);
criterion_main!(benches);
