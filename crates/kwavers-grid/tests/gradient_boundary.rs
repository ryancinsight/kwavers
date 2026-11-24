//! Manufactured values and an independent signed-coordinate boundary oracle.
use kwavers_grid::{
    operators::{
        curl, divergence, gradient,
        gradient_optimized::{gradient_optimized, BoundaryStrategy},
        FDCoefficients, FdAccuracyOrder, GradientCache, GradientOperatorBuilder,
    },
    Grid,
};
use leto::Array3;
use BoundaryStrategy::{Extrapolate, Mirror, Periodic, ZeroPadding};

fn extend(c: i32, n: i32, boundary: BoundaryStrategy) -> Option<usize> {
    let i = match boundary {
        ZeroPadding if !(0..n).contains(&c) => return None,
        ZeroPadding => c,
        Mirror if n == 1 => 0,
        Mirror => {
            let period = 2 * (n - 1);
            let m = c.rem_euclid(period);
            m.min(period - m)
        }
        Periodic => c.rem_euclid(n),
        Extrapolate => c.clamp(0, n - 1),
    };
    usize::try_from(i).ok()
}

#[test]
fn configured_boundaries_match_extended_samples_across_execution_modes() {
    // A strided, asymmetric field makes a shift or wrap on the wrong axis visible.
    for shape in [[1, 2, 3], [5, 7, 11]] {
        let [nx, ny, nz] = shape;
        let h = [0.25, 0.5, 2.0];
        let grid = Grid::new(nx, ny, nz, h[0], h[1], h[2]).unwrap();
        let mut source = Array3::zeros([nx, ny, 2 * nz]);
        for (i, x) in source.iter_mut().enumerate() {
            *x = f64::from(u16::try_from((i * i + 7 * i + 11) % 97).unwrap());
        }
        let view = source
            .slice(&[(0, nx, 1), (0, ny, 1), (0, 2 * nz, 2)])
            .unwrap();
        let cache = GradientCache::new(&Grid::default());
        for order in [
            FdAccuracyOrder::Second,
            FdAccuracyOrder::Fourth,
            FdAccuracyOrder::Sixth,
            FdAccuracyOrder::Eighth,
        ] {
            let weights = FDCoefficients::first_derivative::<f64>(order);
            for boundary in [ZeroPadding, Mirror, Periodic, Extrapolate] {
                for (parallel, caching, chunk) in [
                    (false, false, 1),
                    (true, true, 3),
                    (false, true, 0),
                    (true, false, usize::MAX),
                ] {
                    let op = GradientOperatorBuilder::new()
                        .with_parallel(parallel)
                        .with_caching(caching)
                        .with_chunk_size(chunk)
                        .with_boundary_strategy(boundary)
                        .build();
                    let (x, y, z) = op.compute(&view, &grid, order, Some(&cache)).unwrap();
                    for i in 0..nx {
                        for j in 0..ny {
                            for k in 0..nz {
                                let pos = [i, j, k];
                                for (axis, actual) in
                                    [x[pos], y[pos], z[pos]].into_iter().enumerate()
                                {
                                    let sample = |d| {
                                        let c = i32::try_from(pos[axis]).unwrap() + d;
                                        extend(c, i32::try_from(shape[axis]).unwrap(), boundary)
                                            .map_or(0.0, |mapped| {
                                                let mut p = pos;
                                                p[axis] = mapped;
                                                view[p]
                                            })
                                    };
                                    let expected = weights
                                        .iter()
                                        .zip(1_i32..)
                                        .map(|(w, d)| w * (sample(d) - sample(-d)))
                                        .sum::<f64>()
                                        / h[axis];
                                    assert!((actual-expected).abs()<1e-11,"{pos:?} axis {axis} {order:?} {boundary:?} {parallel} {caching}: {actual} vs {expected}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn interior_gradient_uses_the_requested_grids_spacing_when_reusing_a_cache() {
    let grid = Grid::new(9, 9, 9, 0.25, 0.5, 2.0).unwrap();
    let mut field = Array3::<f32>::zeros([9, 9, 9]);
    for i in 0_u16..9 {
        for j in 0_u16..9 {
            for k in 0_u16..9 {
                field[[i.into(), j.into(), k.into()]] = f32::from(i) + f32::from(j) + f32::from(k);
            }
        }
    }
    let cache = GradientCache::new(&Grid::new(9, 9, 9, 1.0, 1.0, 1.0).unwrap());
    let (x, y, z) =
        gradient_optimized(&field.view(), &grid, FdAccuracyOrder::Second, Some(&cache)).unwrap();
    assert_eq!([x[[4, 4, 4]], y[[4, 4, 4]], z[[4, 4, 4]]], [4.0, 2.0, 0.5]);
}

#[test]
fn narrow_axes_preserve_independent_interior_derivatives() {
    let grid = Grid::new(1, 2, 11, 1.0, 1.0, 1.0).unwrap();
    let mut field = Array3::zeros([1, 2, 11]);
    for j in 0..2 {
        for k in 0_u16..11 {
            field[[0, j, k.into()]] = f64::from(k);
        }
    }
    let view = field.view();
    let order = FdAccuracyOrder::Eighth;
    for (x, y, z) in [
        gradient(&view, &grid, order).unwrap(),
        gradient_optimized(&view, &grid, order, None).unwrap(),
    ] {
        assert!(x.iter().chain(y.iter()).all(|v| *v == 0.0));
        assert!((z[[0, 1, 5]] - 1.0).abs() < 1e-12);
        assert_eq!(z[[0, 1, 0]], 0.0);
    }
    let (x, y, z) = curl(&view, &view, &view, &grid, order).unwrap();
    assert!(x.iter().chain(y.iter()).chain(z.iter()).all(|v| *v == 0.0));
    assert!(divergence(&view, &view, &view, &grid, order)
        .unwrap()
        .iter()
        .all(|v| *v == 0.0));
}

#[test]
fn parallel_chunks_cover_partial_final_rows_in_a_large_volume() {
    let grid = Grid::new(32, 35, 33, 0.25, 0.5, 2.0).unwrap();
    let mut field = Array3::<f32>::zeros([32, 35, 33]);
    for (i, x) in field.iter_mut().enumerate() {
        *x = f32::from(u16::try_from(i % 1009).unwrap());
    }
    let builder = GradientOperatorBuilder::new()
        .with_chunk_size(13)
        .with_boundary_strategy(Periodic);
    let cache = GradientCache::new(&grid);
    let run = |parallel| {
        builder
            .clone()
            .with_parallel(parallel)
            .build()
            .compute(&field.view(), &grid, FdAccuracyOrder::Sixth, Some(&cache))
            .unwrap()
    };
    let (sx, sy, sz) = run(false);
    let (px, py, pz) = run(true);
    for (serial, parallel) in [(sx, px), (sy, py), (sz, pz)] {
        assert!(serial
            .iter()
            .zip(parallel.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    assert_eq!(cache.cache_stats(), (1, 1));
}

#[test]
fn rejects_short_or_long_outputs_and_shape_mismatch_before_writing() {
    let grid = Grid::new(2, 3, 4, 1.0, 1.0, 1.0).unwrap();
    let input = Array3::<f64>::zeros([2, 3, 4]);
    let op = GradientOperatorBuilder::new().build();
    for lens in [[23, 24, 24], [24, 25, 24], [24, 24, 23]] {
        let [mut x, mut y, mut z] = lens.map(|n| vec![91.0; n]);
        assert!(op
            .compute_into(
                &input.view(),
                &grid,
                FdAccuracyOrder::Fourth,
                None,
                [&mut x, &mut y, &mut z]
            )
            .is_err());
        assert!(x.iter().chain(&y).chain(&z).all(|v| *v == 91.0));
    }
    let other = Grid::new(3, 2, 4, 1.0, 1.0, 1.0).unwrap();
    let [mut x, mut y, mut z] = [[91.0; 24]; 3];
    assert!(op
        .compute_into(
            &input.view(),
            &other,
            FdAccuracyOrder::Fourth,
            None,
            [&mut x, &mut y, &mut z]
        )
        .is_err());
    assert_eq!([x, y, z], [[91.0; 24]; 3]);
}
