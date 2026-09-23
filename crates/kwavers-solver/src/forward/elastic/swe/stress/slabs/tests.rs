use super::super::super::{scratch::ElasticStepScratch, types::ElasticWaveField};
use super::super::tests::from_shape_fn_fortran;
use super::super::{stress_divergence_into, DensityScale, Lame, VelocityKick};
use super::{slab_height, stress_kick_in_slabs, window_slides, Caches};
use kwavers_grid::Grid;
use leto::Array3;

/// The velocity kick through the stress window, at every slab size from one
/// plane -- narrower than the stencil's reach, so each slab slides planes it
/// shares with the next -- to the whole grid, is the stress divergence
/// scaled by the density and added at half a step, evaluated as separate
/// passes, to the bit: under both density scales and on grids whose plane
/// count no slab size divides. The displacement it reads is left as it was.
#[test]
fn every_slab_size_kicks_the_velocities_as_the_separate_passes_do_bit_for_bit() {
    const HALF_DT: f64 = 3.7e-7;
    for (nx, ny, nz) in [(1, 4, 3), (2, 5, 4), (5, 6, 5), (11, 7, 6), (16, 6, 5)] {
        let grid = Grid::new(nx, ny, nz, 0.7e-3, 1.1e-3, 1.3e-3).expect("grid");
        let shape = (nx, ny, nz);
        let lambda =
            Array3::from_shape_fn(shape, |[i, j, k]| 2.0e6 + (i * 37 + j * 11 + k * 5) as f64);
        let mu =
            Array3::from_shape_fn(shape, |[i, j, k]| 0.8e6 + (i * 17 + j * 29 + k * 13) as f64);
        let density =
            Array3::from_shape_fn(shape, |[i, j, k]| 1000.0 + (i * 3 + j * 7 + k * 11) as f64);
        let mut field = ElasticWaveField::new(nx, ny, nz);
        field.ux = Array3::from_shape_fn(shape, |[i, j, k]| {
            ((i * 13 + j * 7 + k * 3) as f64 * 0.037).sin()
        });
        field.uy = Array3::from_shape_fn(shape, |[i, j, k]| {
            ((i * 5 + j * 19 + k * 11) as f64 * 0.041).cos()
        });
        field.uz = Array3::from_shape_fn(shape, |[i, j, k]| {
            ((i * 23 + j * 2 + k * 17) as f64 * 0.029).sin()
        });
        field.vx = Array3::from_shape_fn(shape, |[i, j, k]| {
            ((i * 3 + j * 5 + k * 7) as f64 * 0.013).cos()
        });
        field.vy = Array3::from_shape_fn(shape, |[i, j, k]| {
            ((i * 11 + j * 3 + k * 2) as f64 * 0.017).sin()
        });
        field.vz = Array3::from_shape_fn(shape, |[i, j, k]| {
            ((i * 2 + j * 13 + k * 5) as f64 * 0.019).cos()
        });

        let mut separate = ElasticStepScratch::new(nx, ny, nz);
        stress_divergence_into(&grid, &lambda, &mu, &field, &mut separate);
        let reciprocal = 997.0_f64.recip();
        for (label, scale) in [
            ("uniform", DensityScale::UniformReciprocal(reciprocal)),
            ("field", DensityScale::Field(density.view())),
        ] {
            let expected: [Array3<f64>; 3] = [
                (&field.vx, &separate.div_x),
                (&field.vy, &separate.div_y),
                (&field.vz, &separate.div_z),
            ]
            .map(|(velocity, divergence)| {
                Array3::from_shape_fn(shape, |index| {
                    let acceleration = match scale {
                        DensityScale::UniformReciprocal(r) => divergence[index] * r,
                        DensityScale::Field(rho) => divergence[index] / rho[index],
                    };
                    velocity[index] + HALF_DT * acceleration
                })
            });
            let kick = VelocityKick {
                scale,
                half_dt: HALF_DT,
            };
            for slab in 1..=nx + 1 {
                let mut kicked = field.clone();
                let mut scratch = ElasticStepScratch::new(nx, ny, nz);
                stress_kick_in_slabs(
                    &grid,
                    Lame::Field {
                        lambda: lambda.view(),
                        mu: mu.view(),
                    },
                    &mut kicked,
                    &kick,
                    &mut scratch,
                    core::num::NonZeroUsize::new(slab).expect("a slab holds a plane"),
                );
                for (component, want, got) in [
                    ("vx", &expected[0], &kicked.vx),
                    ("vy", &expected[1], &kicked.vy),
                    ("vz", &expected[2], &kicked.vz),
                    ("ux", &field.ux, &kicked.ux),
                    ("uy", &field.uy, &kicked.uy),
                    ("uz", &field.uz, &kicked.uz),
                ] {
                    for (index, (a, b)) in want.iter().zip(got.iter()).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{label} density, {nx}x{ny}x{nz}, {slab}-plane slabs: {component} \
                             at flat {index}: {a} against {b}"
                        );
                    }
                }
            }
        }
    }
}

/// The slab rule against its derivation, on the machine class it was
/// measured on: 60 MiB held by an even split across 24 workers, a 36 MiB
/// last level.
///
/// - 72 and 80 cubed with 14 live fields are 41.8 MB and 57.3 MB: past the
///   last level, inside the split, so whole-grid -- measured ahead at 72
///   and 76 cubed and level at 80.
/// - 88 cubed is 76.3 MB, past the split: slabs, measured 1.3x ahead. Its
///   plane is 867,328 bytes; the last level holds 43, 39 after the four the
///   stencil reaches, and the worker count binds: 24.
/// - 96 cubed is 1,032,192 bytes a plane: 36 fit, 32 after the reach, 24.
/// - 128 cubed is 1,835,008 bytes a plane; 20 fit, 16 after the reach, and
///   the cache binds: 16. A density field makes it 15 live fields and 15.
/// - No reported cache is whole-grid; a plane larger than the last level,
///   or one worker, is one plane; a grid of no planes is still one.
#[test]
fn the_slab_height_is_the_cache_fit_less_the_reach_capped_by_the_workers() {
    let caches = |total: usize, last_level: usize| Some(Caches { total, last_level });
    let host = || caches(60 << 20, 36 << 20);
    let height = |planes: usize, live: usize, caches: Option<Caches>, workers: usize| {
        slab_height(planes, planes * planes, live, caches, workers).get()
    };
    assert_eq!(height(72, 14, host(), 24), 72);
    assert_eq!(height(80, 14, host(), 24), 80);
    assert_eq!(height(88, 14, host(), 24), 24);
    assert_eq!(height(96, 14, host(), 24), 24);
    assert_eq!(height(128, 14, host(), 24), 16);
    assert_eq!(height(128, 15, host(), 24), 15);
    assert_eq!(height(80, 14, caches(36 << 20, 36 << 20), 24), 24);
    assert_eq!(height(128, 14, None, 24), 128);
    assert_eq!(height(128, 14, caches(1 << 20, 1 << 20), 24), 1);
    assert_eq!(height(96, 14, host(), 1), 1);
    assert_eq!(slab_height(0, 0, 14, host(), 24).get(), 1);
}

/// The window slides only through C-contiguous stress fields; one stress
/// field in any other layout sends the evaluation whole-grid.
#[test]
fn a_stress_field_out_of_c_order_stops_the_window_sliding() {
    let shape = [5, 4, 3];
    let [nx, ny, nz] = shape;
    let mut scratch = ElasticStepScratch::new(nx, ny, nz);
    assert!(window_slides(&scratch));
    scratch.sxz = from_shape_fn_fortran(shape, |_| 0.0);
    assert!(!window_slides(&scratch));
}

/// A uniform medium given as its Lamé pair kicks the velocities as the same
/// medium given as constant fields does, to the bit, for slabs narrower than
/// the stencil's reach, a middling height and the whole grid.
#[test]
fn a_uniform_lame_pair_kicks_as_its_constant_fields_do() {
    let (nx, ny, nz) = (11, 7, 6);
    let grid = Grid::new(nx, ny, nz, 0.7e-3, 1.1e-3, 1.3e-3).expect("grid");
    let (lambda_value, mu_value) = (2.25e9, 0.75e6);
    let lambda = Array3::from_elem((nx, ny, nz), lambda_value);
    let mu = Array3::from_elem((nx, ny, nz), mu_value);
    let mut field = ElasticWaveField::new(nx, ny, nz);
    field.ux = Array3::from_shape_fn((nx, ny, nz), |[i, j, k]| {
        ((i * 13 + j * 7 + k * 3) as f64 * 0.037).sin()
    });
    field.uy = Array3::from_shape_fn((nx, ny, nz), |[i, j, k]| {
        ((i * 5 + j * 19 + k * 11) as f64 * 0.041).cos()
    });
    field.uz = Array3::from_shape_fn((nx, ny, nz), |[i, j, k]| {
        ((i * 23 + j * 2 + k * 17) as f64 * 0.029).sin()
    });
    let kick = VelocityKick {
        scale: DensityScale::UniformReciprocal(997.0_f64.recip()),
        half_dt: 3.7e-7,
    };
    for slab in [1, 4, nx] {
        let slab = core::num::NonZeroUsize::new(slab).expect("a slab holds a plane");
        let kicked = |lame| {
            let mut kicked = field.clone();
            let mut scratch = ElasticStepScratch::new(nx, ny, nz);
            stress_kick_in_slabs(&grid, lame, &mut kicked, &kick, &mut scratch, slab);
            kicked
        };
        let fields = kicked(Lame::Field {
            lambda: lambda.view(),
            mu: mu.view(),
        });
        let pair = kicked(Lame::Uniform {
            lambda: lambda_value,
            mu: mu_value,
        });
        for (component, want, got) in [
            ("vx", &fields.vx, &pair.vx),
            ("vy", &fields.vy, &pair.vy),
            ("vz", &fields.vz, &pair.vz),
        ] {
            for (index, (a, b)) in want.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{slab}-plane slabs: {component} at flat {index}: {a} against {b}"
                );
            }
        }
    }
}
