//! Independent analytical oracles for Cartesian curl and divergence.
use kwavers_grid::{
    operators::{curl, curl_into, divergence, divergence_into, FdAccuracyOrder},
    Grid,
};
use leto::Array3;
use mnemosyne::{
    counting::{measure, CountingAllocator},
    Mnemosyne,
};

#[global_allocator]
static GLOBAL: CountingAllocator<Mnemosyne> = CountingAllocator::new(Mnemosyne);

const ORDERS: [FdAccuracyOrder; 4] = [
    FdAccuracyOrder::Second,
    FdAccuracyOrder::Fourth,
    FdAccuracyOrder::Sixth,
    FdAccuracyOrder::Eighth,
];

#[test]
fn manufactured_field_matches_at_every_cell_for_dense_and_mixed_strides() {
    let shape = [11, 12, 13];
    let grid = Grid::new(11, 12, 13, 0.25, 0.5, 2.0).unwrap();
    let [mut vx, mut vy, mut vz] = std::array::from_fn(|_| Array3::<f64>::zeros(shape));
    let mut padded = Array3::from_elem([11, 12, 26], f64::NAN);
    for i in 0_u16..11 {
        for j in 0_u16..12 {
            for k in 0_u16..13 {
                let [x, y, z] = [
                    f64::from(i) * grid.dx,
                    f64::from(j) * grid.dy,
                    f64::from(k) * grid.dz,
                ];
                let p = [i.into(), j.into(), k.into()];
                vx[p] = x * x + 2.0 * y * z + 4.0 * z;
                vy[p] = 3.0 * y * y + 5.0 * z * x + 7.0 * x;
                vz[p] = 4.0 * z * z + 11.0 * x * y + 13.0 * y;
                padded[[p[0], p[1], 2 * p[2] + 1]] = vy[p];
            }
        }
    }
    let slice = padded.slice(&[(0, 11, 1), (0, 12, 1), (1, 26, 2)]).unwrap();
    for y_field in [vy.view(), slice] {
        for (r, order) in (1..=4).zip(ORDERS) {
            let (cx, cy, cz) = curl(&vx.view(), &y_field, &vz.view(), &grid, order).unwrap();
            let div = divergence(&vx.view(), &y_field, &vz.view(), &grid, order).unwrap();
            let [mut bx, mut by, mut bz, mut bd] =
                std::array::from_fn(|_| vec![f64::NAN; vx.len()]);
            curl_into(
                &vx.view(),
                &y_field,
                &vz.view(),
                &grid,
                order,
                [&mut bx, &mut by, &mut bz],
            )
            .unwrap();
            divergence_into(&vx.view(), &y_field, &vz.view(), &grid, order, &mut bd).unwrap();
            for i in 0_u16..11 {
                for j in 0_u16..12 {
                    for k in 0_u16..13 {
                        let [x, y, z] = [
                            f64::from(i) * grid.dx,
                            f64::from(j) * grid.dy,
                            f64::from(k) * grid.dz,
                        ];
                        let p: [usize; 3] = [i.into(), j.into(), k.into()];
                        let n = (p[0] * 12 + p[1]) * 13 + p[2];
                        let interior = p
                            .iter()
                            .zip(shape)
                            .all(|(&u, len)| (r..len - r).contains(&u));
                        let expected = if interior {
                            [
                                6.0 * x + 13.0,
                                4.0 - 9.0 * y,
                                3.0 * z + 7.0,
                                2.0 * x + 6.0 * y + 8.0 * z,
                            ]
                        } else {
                            [0.0; 4]
                        };
                        let allocated = [cx[p], cy[p], cz[p], div[p]];
                        let reused = [bx[n], by[n], bz[n], bd[n]];
                        assert_eq!(allocated, reused);
                        for (observed, analytic) in reused.into_iter().zip(expected) {
                            assert!(
                                (observed - analytic).abs() < 2e-12,
                                "{p:?} {order:?}: {observed} vs {analytic}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn thin_axes_clear_the_entire_reused_output() {
    for shape @ [nx, ny, nz] in [[1, 9, 10], [9, 2, 10], [9, 10, 8]] {
        let grid = Grid::new(nx, ny, nz, 0.5, 1.0, 2.0).unwrap();
        let input = Array3::<f32>::ones(shape);
        let f = input.view();
        let [mut x, mut y, mut z, mut d] = std::array::from_fn(|_| vec![7.0; input.len()]);
        curl_into(
            &f,
            &f,
            &f,
            &grid,
            FdAccuracyOrder::Eighth,
            [&mut x, &mut y, &mut z],
        )
        .unwrap();
        divergence_into(&f, &f, &f, &grid, FdAccuracyOrder::Eighth, &mut d).unwrap();
        assert!(x.iter().chain(&y).chain(&z).chain(&d).all(|v| *v == 0.0));
    }
}

#[test]
fn shape_and_output_length_errors_leave_destinations_intact() {
    let grid = Grid::new(2, 3, 4, 1.0, 1.0, 1.0).unwrap();
    let valid = Array3::<f64>::zeros([2, 3, 4]);
    let invalid = Array3::<f64>::zeros([3, 2, 4]); // same storage length
    let f = valid.view();
    let g = invalid.view();
    let order = FdAccuracyOrder::Second;
    for fields in [[&g, &f, &f], [&f, &g, &f], [&f, &f, &g]] {
        let [a, b, c] = fields;
        let [mut x, mut y, mut z, mut d] = [[91.0; 24]; 4];
        assert!(curl_into(a, b, c, &grid, order, [&mut x, &mut y, &mut z]).is_err());
        assert!(divergence_into(a, b, c, &grid, order, &mut d).is_err());
        assert_eq!([x, y, z, d], [[91.0; 24]; 4]);
    }
    for lengths in [[23, 24, 24], [24, 25, 24], [24, 24, 23]] {
        let [mut x, mut y, mut z] = lengths.map(|n| vec![91.0; n]);
        assert!(curl_into(&f, &f, &f, &grid, order, [&mut x, &mut y, &mut z]).is_err());
        assert!(x.iter().chain(&y).chain(&z).all(|v| *v == 91.0));
    }
    for n in [23, 25] {
        let mut d = vec![91.0; n];
        assert!(divergence_into(&f, &f, &f, &grid, order, &mut d).is_err());
        assert!(d.iter().all(|v| *v == 91.0));
    }
}

#[test]
fn repeated_strided_and_contiguous_calls_have_no_heap_traffic() {
    let grid = Grid::new(9, 10, 11, 1.0, 1.0, 1.0).unwrap();
    let source = Array3::<f32>::ones([9, 10, 11]);
    let wide = Array3::<f32>::ones([9, 10, 22]);
    let strided = wide.slice(&[(0, 9, 1), (0, 10, 1), (0, 22, 2)]).unwrap();
    let [mut x, mut y, mut z, mut d] = std::array::from_fn(|_| vec![f32::NAN; source.len()]);
    for field in [source.view(), strided] {
        for order in ORDERS {
            let (result, delta) = measure(|| {
                curl_into(
                    &field,
                    &field,
                    &field,
                    &grid,
                    order,
                    [&mut x, &mut y, &mut z],
                )?;
                divergence_into(&field, &field, &field, &grid, order, &mut d)
            });
            result.unwrap();
            assert_eq!((delta.allocations, delta.reallocations), (0, 0));
        }
    }
}

#[test]
fn requested_order_differentiates_its_highest_exact_polynomial() {
    // With F=(f(x),0,f(x)), div F=f'(x) and (curl F)_y=-f'(x).
    // The off-centre origin keeps the derivative nonzero. A lower-order
    // stencil fails for the fourth-, sixth- and eighth-degree fields.
    let grid = Grid::new(13, 9, 9, 0.5, 1.0, 1.0).unwrap();
    let zeros = Array3::<f64>::zeros([13, 9, 9]);
    for (degree, order) in [2, 4, 6, 8].into_iter().zip(ORDERS) {
        let mut field = Array3::zeros([13, 9, 9]);
        for i in 0_u16..13 {
            let x = (f64::from(i) - 4.0) * grid.dx;
            for j in 0..9 {
                for k in 0..9 {
                    field[[i.into(), j, k]] = x.powi(degree);
                }
            }
        }
        let f = field.view();
        let (_, cy, _) = curl(&f, &zeros.view(), &f, &grid, order).unwrap();
        let div = divergence(&f, &zeros.view(), &f, &grid, order).unwrap();
        for i in 5_u16..=8 {
            let derivative = f64::from(degree) * ((f64::from(i) - 4.0) * grid.dx).powi(degree - 1);
            let p = [i.into(), 4, 4];
            assert!((div[p] - derivative).abs() < 1e-10);
            assert!((cy[p] + derivative).abs() < 1e-10);
        }
    }
}
