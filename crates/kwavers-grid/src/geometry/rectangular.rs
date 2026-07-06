use leto::{Array1, Array2};

use super::{GeometricDomain, GeometryDimension, PointLocation};

/// Rectangular domain in 1D, 2D, or 3D
///
/// Defined as a Cartesian product of intervals:
/// - 1D: [x_min, x_max]
/// - 2D: [x_min, x_max] × [y_min, y_max]
/// - 3D: [x_min, x_max] × [y_min, y_max] × [z_min, z_max]
#[derive(Debug, Clone)]
pub struct RectangularDomain {
    pub min: Vec<f64>,
    pub max: Vec<f64>,
}

impl RectangularDomain {
    /// New 1d.
    /// # Panics
    /// - Panics if assertion fails: `Invalid 1D domain bounds`.
    ///
    #[must_use]
    pub fn new_1d(x_min: f64, x_max: f64) -> Self {
        assert!(x_max > x_min, "Invalid 1D domain bounds");
        Self {
            min: vec![x_min],
            max: vec![x_max],
        }
    }

    /// New 2d.
    /// # Panics
    /// - Panics if assertion fails: `Invalid 2D domain bounds`.
    ///
    #[must_use]
    pub fn new_2d(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        assert!(x_max > x_min && y_max > y_min, "Invalid 2D domain bounds");
        Self {
            min: vec![x_min, y_min],
            max: vec![x_max, y_max],
        }
    }

    /// New 3d.
    /// # Panics
    /// - Panics if assertion fails: `Invalid 3D domain bounds`.
    ///
    #[must_use]
    pub fn new_3d(x_min: f64, x_max: f64, y_min: f64, y_max: f64, z_min: f64, z_max: f64) -> Self {
        assert!(
            x_max > x_min && y_max > y_min && z_max > z_min,
            "Invalid 3D domain bounds"
        );
        Self {
            min: vec![x_min, y_min, z_min],
            max: vec![x_max, y_max, z_max],
        }
    }

    #[must_use]
    pub fn lengths(&self) -> Vec<f64> {
        self.min
            .iter()
            .zip(&self.max)
            .map(|(min, max)| max - min)
            .collect()
    }

    #[must_use]
    pub fn center(&self) -> Vec<f64> {
        self.min
            .iter()
            .zip(&self.max)
            .map(|(min, max)| 0.5 * (min + max))
            .collect()
    }
}

impl GeometricDomain for RectangularDomain {
    fn dimension(&self) -> GeometryDimension {
        match self.min.len() {
            1 => GeometryDimension::One,
            2 => GeometryDimension::Two,
            3 => GeometryDimension::Three,
            _ => unreachable!("RectangularDomain only constructed with 1-3 dimensions"),
        }
    }

    fn contains(&self, point: &[f64]) -> bool {
        assert_eq!(point.len(), self.min.len(), "Point dimension mismatch");
        point
            .iter()
            .zip(&self.min)
            .zip(&self.max)
            .all(|((p, min), max)| p >= min && p <= max)
    }

    fn classify_point(&self, point: &[f64], tolerance: f64) -> PointLocation {
        assert_eq!(point.len(), self.min.len(), "Point dimension mismatch");

        let mut on_boundary = false;
        let mut outside = false;

        for ((&p, &min), &max) in point.iter().zip(&self.min).zip(&self.max) {
            if p < min - tolerance || p > max + tolerance {
                outside = true;
                break;
            }
            if (p - min).abs() < tolerance || (p - max).abs() < tolerance {
                on_boundary = true;
            }
        }

        if outside {
            PointLocation::Exterior
        } else if on_boundary {
            PointLocation::Boundary
        } else {
            PointLocation::Interior
        }
    }

    fn bounding_box(&self) -> Vec<f64> {
        self.min
            .iter()
            .zip(&self.max)
            .flat_map(|(min, max)| vec![*min, *max])
            .collect()
    }

    fn normal(&self, point: &[f64], tolerance: f64) -> Option<Array1<f64>> {
        if self.classify_point(point, tolerance) != PointLocation::Boundary {
            return None;
        }

        let n = self.min.len();
        let mut normal = Array1::zeros([n]);

        for i in 0..n {
            if (point[i] - self.min[i]).abs() < tolerance {
                normal[i] = -1.0;
            } else if (point[i] - self.max[i]).abs() < tolerance {
                normal[i] = 1.0;
            }
        }

        let norm = normal.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            for i in 0..n {
                normal[i] /= norm;
            }
        }

        Some(normal)
    }

    fn measure(&self) -> f64 {
        self.lengths().iter().product()
    }

    fn sample_interior(&self, n_points: usize, seed: Option<u64>) -> Array2<f64> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let mut rng = if let Some(s) = seed {
            ChaCha8Rng::seed_from_u64(s)
        } else {
            ChaCha8Rng::from_entropy()
        };

        let dim = self.min.len();
        let mut points = Array2::zeros([n_points, dim]);

        for i in 0..n_points {
            for j in 0..dim {
                points[[i, j]] = rng.gen_range(self.min[j]..self.max[j]);
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

        let dim = self.min.len();
        let mut points = Array2::zeros([n_points, dim]);

        for i in 0..n_points {
            let face = rng.gen_range(0..(2 * dim));
            let axis = face / 2;
            let is_max = face % 2 == 1;

            points[[i, axis]] = if is_max {
                self.max[axis]
            } else {
                self.min[axis]
            };

            for j in 0..dim {
                if j != axis {
                    points[[i, j]] = rng.gen_range(self.min[j]..self.max[j]);
                }
            }
        }

        points
    }
}
