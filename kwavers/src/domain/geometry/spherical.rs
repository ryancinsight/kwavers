use ndarray::{Array1, Array2};
use std::f64::consts::PI;

use super::{GeometricDomain, GeometryDimension, PointLocation};
use crate::core::constants::numerical::{TWO_PI};

/// Circular/spherical domain
///
/// Defined as:
/// - 2D: disk of radius R centered at (x₀, y₀)
/// - 3D: ball of radius R centered at (x₀, y₀, z₀)
#[derive(Debug, Clone)]
pub struct SphericalDomain {
    pub center: Vec<f64>,
    pub radius: f64,
}

impl SphericalDomain {
    /// New 2d.
    /// # Panics
    /// - Panics if assertion fails: `Radius must be positive`.
    ///
    #[must_use]
    pub fn new_2d(x0: f64, y0: f64, radius: f64) -> Self {
        assert!(radius > 0.0, "Radius must be positive");
        Self {
            center: vec![x0, y0],
            radius,
        }
    }

    /// New 3d.
    /// # Panics
    /// - Panics if assertion fails: `Radius must be positive`.
    ///
    #[must_use]
    pub fn new_3d(x0: f64, y0: f64, z0: f64, radius: f64) -> Self {
        assert!(radius > 0.0, "Radius must be positive");
        Self {
            center: vec![x0, y0, z0],
            radius,
        }
    }

    fn distance_from_center(&self, point: &[f64]) -> f64 {
        point
            .iter()
            .zip(&self.center)
            .map(|(p, c)| (p - c).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl GeometricDomain for SphericalDomain {
    fn dimension(&self) -> GeometryDimension {
        match self.center.len() {
            2 => GeometryDimension::Two,
            3 => GeometryDimension::Three,
            _ => unreachable!("SphericalDomain only constructed with 2-3 dimensions"),
        }
    }

    fn contains(&self, point: &[f64]) -> bool {
        assert_eq!(point.len(), self.center.len(), "Point dimension mismatch");
        self.distance_from_center(point) <= self.radius
    }

    fn classify_point(&self, point: &[f64], tolerance: f64) -> PointLocation {
        let dist = self.distance_from_center(point);
        let diff = dist - self.radius;

        if diff.abs() < tolerance {
            PointLocation::Boundary
        } else if diff < 0.0 {
            PointLocation::Interior
        } else {
            PointLocation::Exterior
        }
    }

    fn bounding_box(&self) -> Vec<f64> {
        self.center
            .iter()
            .flat_map(|c| vec![c - self.radius, c + self.radius])
            .collect()
    }

    fn normal(&self, point: &[f64], tolerance: f64) -> Option<Array1<f64>> {
        if self.classify_point(point, tolerance) != PointLocation::Boundary {
            return None;
        }

        let n = self.center.len();
        let mut normal = Array1::zeros(n);

        for i in 0..n {
            normal[i] = point[i] - self.center[i];
        }

        let norm = normal.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            normal /= norm;
        }

        Some(normal)
    }

    fn measure(&self) -> f64 {
        match self.center.len() {
            2 => PI * self.radius.powi(2),
            3 => (4.0 / 3.0) * PI * self.radius.powi(3),
            _ => unreachable!("SphericalDomain only constructed with 2-3 dimensions"),
        }
    }

    fn sample_interior(&self, n_points: usize, seed: Option<u64>) -> Array2<f64> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = if let Some(s) = seed {
            ChaCha8Rng::seed_from_u64(s)
        } else {
            ChaCha8Rng::from_entropy()
        };

        let dim = self.center.len();
        let mut points = Array2::zeros((n_points, dim));

        for i in 0..n_points {
            loop {
                let mut point = vec![0.0; dim];
                let mut r_sq = 0.0;

                for coord in point.iter_mut().take(dim) {
                    let v = rng.gen_range(-self.radius..self.radius);
                    *coord = v;
                    r_sq += v * v;
                }

                if r_sq <= self.radius * self.radius {
                    for (j, coord) in point.iter().enumerate().take(dim) {
                        points[[i, j]] = self.center[j] + coord;
                    }
                    break;
                }
            }
        }

        points
    }

    fn sample_boundary(&self, n_points: usize, seed: Option<u64>) -> Array2<f64> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = if let Some(s) = seed {
            ChaCha8Rng::seed_from_u64(s)
        } else {
            ChaCha8Rng::from_entropy()
        };

        let dim = self.center.len();
        let mut points = Array2::zeros((n_points, dim));

        for i in 0..n_points {
            match dim {
                2 => {
                    let theta = rng.gen_range(0.0..(TWO_PI));
                    points[[i, 0]] = self.radius.mul_add(theta.cos(), self.center[0]);
                    points[[i, 1]] = self.radius.mul_add(theta.sin(), self.center[1]);
                }
                3 => loop {
                    let x1: f64 = rng.gen_range(-1.0..1.0);
                    let x2: f64 = rng.gen_range(-1.0..1.0);
                    let s = x1.mul_add(x1, x2 * x2);
                    if s < 1.0 {
                        let factor = 2.0 * (1.0_f64 - s).sqrt();
                        points[[i, 0]] = (self.radius * x1).mul_add(factor, self.center[0]);
                        points[[i, 1]] = (self.radius * x2).mul_add(factor, self.center[1]);
                        points[[i, 2]] =
                            self.radius.mul_add(2.0f64.mul_add(-s, 1.0), self.center[2]);
                        break;
                    }
                },
                _ => unreachable!("SphericalDomain only constructed with 2-3 dimensions"),
            }
        }

        points
    }
}
