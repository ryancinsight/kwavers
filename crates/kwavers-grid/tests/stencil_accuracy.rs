//! Manufactured fields for the centered Cartesian finite differences.
use kwavers_grid::{
    operators::{
        gradient,
        laplacian::{LaplacianBoundary, LaplacianConfig, LaplacianOperator},
        FdAccuracyOrder,
    },
    Grid,
};
use leto::Array3;
use std::f64::consts::PI;

const ORDERS: [FdAccuracyOrder; 4] = [
    FdAccuracyOrder::Second,
    FdAccuracyOrder::Fourth,
    FdAccuracyOrder::Sixth,
    FdAccuracyOrder::Eighth,
];

#[test]
fn gradient_differentiates_an_affine_field_in_physical_coordinates() {
    let grid = Grid::new(9, 10, 11, 0.25, 0.5, 2.0).unwrap();
    let mut field = Array3::zeros([9, 10, 11]);
    for i in 0_u16..9 {
        for j in 0_u16..10 {
            for k in 0_u16..11 {
                field[[i.into(), j.into(), k.into()]] = 2.0 * f64::from(i) * grid.dx
                    - 3.0 * f64::from(j) * grid.dy
                    + 5.0 * f64::from(k) * grid.dz;
            }
        }
    }
    for order in ORDERS {
        let (x, y, z) = gradient(&field.view(), &grid, order).unwrap();
        for (component, expected) in [(x, 2.0), (y, -3.0), (z, 5.0)] {
            assert!((component[[4, 5, 6]] - expected).abs() < 1e-12, "{order:?}");
        }
    }
}

// Eigenvalue of a centered second derivative on cos(theta * index),
// obtained by truncating the series for asin(sqrt(s))^2 at the requested order.
fn symbol(theta: f64, spacing: f64, order: FdAccuracyOrder) -> f64 {
    let s = (theta / 2.0).sin().powi(2);
    let p = match order {
        FdAccuracyOrder::Second => 1.0,
        FdAccuracyOrder::Fourth => 1.0 + s / 3.0,
        FdAccuracyOrder::Sixth => 1.0 + s / 3.0 + 8.0 * s.powi(2) / 45.0,
        FdAccuracyOrder::Eighth => 1.0 + s / 3.0 + 8.0 * s.powi(2) / 45.0 + 4.0 * s.powi(3) / 35.0,
    };
    -4.0 * s * p / spacing.powi(2)
}

#[test]
fn periodic_and_reflecting_modes_match_on_every_axis_face_edge_and_corner() {
    for dims in [[10_u16, 11, 12], [1, 2, 3], [3, 1, 2]] {
        let [nx, ny, nz] = dims.map(usize::from);
        let grid = Grid::new(nx, ny, nz, 0.25, 0.75, 1.5).unwrap();
        for boundary in [LaplacianBoundary::Periodic, LaplacianBoundary::Neumann] {
            let theta = dims.map(|n| match boundary {
                _ if n == 1 => 0.0,
                LaplacianBoundary::Periodic => 2.0 * PI / f64::from(n),
                _ => PI / f64::from(n - 1),
            });
            let mut field = Array3::zeros([nx, ny, nz]);
            for i in 0..dims[0] {
                for j in 0..dims[1] {
                    for k in 0..dims[2] {
                        field[[i.into(), j.into(), k.into()]] = (theta[0] * f64::from(i)).cos()
                            * (theta[1] * f64::from(j)).cos()
                            * (theta[2] * f64::from(k)).cos();
                    }
                }
            }
            for order in ORDERS {
                let op = LaplacianOperator::new(&grid, LaplacianConfig { order, boundary });
                let output = op.apply(field.view()).unwrap();
                let eigenvalue = symbol(theta[0], grid.dx, order)
                    + symbol(theta[1], grid.dy, order)
                    + symbol(theta[2], grid.dz, order);
                for (actual, sample) in output.iter().zip(field.iter()) {
                    assert!(
                        (actual - eigenvalue * sample).abs() < 3e-12,
                        "{dims:?} {boundary:?} {order:?} actual {actual} expected {}",
                        eigenvalue * sample
                    );
                }
            }
        }
    }
}

#[test]
fn dirichlet_overwrites_reused_strided_output_including_the_halo() {
    let grid = Grid::new(10, 10, 10, 0.5, 1.0, 2.0).unwrap();
    let mut field = Array3::zeros([10, 10, 10]);
    for k in 0_u16..10 {
        for i in 0..10 {
            for j in 0..10 {
                field[[i, j, k.into()]] = (f64::from(k) * grid.dz).powi(2);
            }
        }
    }
    for (index, order) in ORDERS.into_iter().enumerate() {
        let radius = index + 1;
        let op = LaplacianOperator::with_order(&grid, order);
        let mut storage = Array3::from_elem([10, 10, 20], 91.0);
        op.apply_mut(
            field.view(),
            storage
                .slice_mut(&[(0, 10, 1), (0, 10, 1), (0, 20, 2)])
                .unwrap(),
        )
        .unwrap();
        for i in 0..10 {
            for j in 0..10 {
                for k in 0..10 {
                    let interior = [i, j, k]
                        .into_iter()
                        .all(|v| (radius..10 - radius).contains(&v));
                    let expected = if interior { 2.0 } else { 0.0 };
                    assert!(
                        (storage[[i, j, 2 * k]] - expected).abs() < 2e-12,
                        "{order:?} at {:?}",
                        [i, j, k]
                    );
                    assert_eq!(storage[[i, j, 2 * k + 1]], 91.0);
                }
            }
        }
    }
}

#[test]
fn first_derivative_weights_reproduce_odd_polynomials_through_the_requested_order() {
    use kwavers_grid::operators::FDCoefficients;
    for (index, order) in ORDERS.into_iter().enumerate() {
        let weights = FDCoefficients::first_derivative::<f64>(order);
        assert_eq!(weights.len(), index + 1);
        // At the origin the derivative of x is one. For every higher odd
        // power the derivative is zero, and even powers cancel by symmetry.
        for degree in (1..=2 * weights.len()).step_by(2) {
            let derivative: f64 = weights
                .iter()
                .zip(1_u32..)
                .map(|(w, d)| 2.0 * w * f64::from(d).powi(i32::try_from(degree).unwrap()))
                .sum();
            let expected = if degree == 1 { 1.0 } else { 0.0 };
            assert!(
                (derivative - expected).abs() < 1e-12,
                "{order:?}, x^{degree}"
            );
        }
    }
}
