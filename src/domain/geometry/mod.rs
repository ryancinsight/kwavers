//! Geometric Domain Abstractions
//!
//! This module provides shared geometric primitives and domain representations
//! that are agnostic to the solver type (forward numerical solvers vs inverse
//! PINNs vs optimization methods).
//!
//! # Design Principle
//!
//! Geometry specifications define the spatial domain Ω ⊂ ℝⁿ where PDEs are solved,
//! independent of HOW they are solved. This allows:
//! - Forward solvers to use the same geometry for grid generation
//! - PINNs to use the same geometry for collocation point sampling
//! - Optimization methods to use the same geometry for parameter estimation
//!
//! # Mathematical Foundation
//!
//! For a domain Ω with boundary ∂Ω, we need:
//! - Point-in-domain test: (x₁, ..., xₙ) ∈ Ω
//! - Boundary detection: (x₁, ..., xₙ) ∈ ∂Ω
//! - Normal vector computation: n̂(x) for x ∈ ∂Ω
//! - Domain measures: volume/area |Ω|, surface area/perimeter |∂Ω|
//!
//! # Architecture
//!
//! ```text
//! Domain Layer (this module)
//!     ↓
//! Solver Layer
//!     ├─ Forward Solvers  → use for grid/mesh generation
//!     ├─ Inverse Solvers  → use for collocation sampling (PINN)
//!     └─ Optimization     → use for parameter space definition
//! ```

use ndarray::{Array1, Array2};
use std::f64::consts::PI;

/// Spatial dimension specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    One,
    Two,
    Three,
}

impl Dimension {
    /// Get dimension as integer
    pub fn as_usize(&self) -> usize {
        match self {
            Dimension::One => 1,
            Dimension::Two => 2,
            Dimension::Three => 3,
        }
    }
}

/// Boundary type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// Lower bound in x-direction
    XMin,
    /// Upper bound in x-direction
    XMax,
    /// Lower bound in y-direction
    YMin,
    /// Upper bound in y-direction
    YMax,
    /// Lower bound in z-direction
    ZMin,
    /// Upper bound in z-direction
    ZMax,
}

/// Point location relative to domain
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointLocation {
    /// Point is strictly inside domain (not on boundary)
    Interior,
    /// Point is on domain boundary
    Boundary,
    /// Point is outside domain
    Exterior,
}

/// Abstract geometric domain trait
///
/// Defines the minimal interface for a geometric domain that can be used
/// by any solver type.
pub trait GeometricDomain: Send + Sync {
    /// Get spatial dimension
    fn dimension(&self) -> Dimension;

    /// Check if point is inside domain (including boundary)
    fn contains(&self, point: &[f64]) -> bool;

    /// Classify point location (interior, boundary, or exterior)
    fn classify_point(&self, point: &[f64], tolerance: f64) -> PointLocation;

    /// Get axis-aligned bounding box [x_min, x_max, y_min, y_max, ...]
    fn bounding_box(&self) -> Vec<f64>;

    /// Compute outward unit normal vector at boundary point
    ///
    /// Returns None if point is not on boundary within tolerance
    fn normal(&self, point: &[f64], tolerance: f64) -> Option<Array1<f64>>;

    /// Compute approximate domain measure (volume/area/length)
    fn measure(&self) -> f64;

    /// Sample uniformly distributed points in domain interior
    fn sample_interior(&self, n_points: usize, seed: Option<u64>) -> Array2<f64>;

    /// Sample points on domain boundary
    fn sample_boundary(&self, n_points: usize, seed: Option<u64>) -> Array2<f64>;
}

/// Rectangular domain in 1D, 2D, or 3D
///
/// Defined as a Cartesian product of intervals:
/// - 1D: [x_min, x_max]
/// - 2D: [x_min, x_max] × [y_min, y_max]
/// - 3D: [x_min, x_max] × [y_min, y_max] × [z_min, z_max]
#[derive(Debug, Clone)]
pub struct RectangularDomain {
    /// Lower bounds [x_min, y_min, z_min]
    pub min: Vec<f64>,
    /// Upper bounds [x_max, y_max, z_max]
    pub max: Vec<f64>,
}

impl RectangularDomain {
    /// Create 1D domain [x_min, x_max]
    pub fn new_1d(x_min: f64, x_max: f64) -> Self {
        assert!(x_max > x_min, "Invalid 1D domain bounds");
        Self {
            min: vec![x_min],
            max: vec![x_max],
        }
    }

    /// Create 2D domain [x_min, x_max] × [y_min, y_max]
    pub fn new_2d(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        assert!(x_max > x_min && y_max > y_min, "Invalid 2D domain bounds");
        Self {
            min: vec![x_min, y_min],
            max: vec![x_max, y_max],
        }
    }

    /// Create 3D domain [x_min, x_max] × [y_min, y_max] × [z_min, z_max]
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

    /// Get domain lengths [Lx, Ly, Lz]
    pub fn lengths(&self) -> Vec<f64> {
        self.min
            .iter()
            .zip(&self.max)
            .map(|(min, max)| max - min)
            .collect()
    }

    /// Get domain center
    pub fn center(&self) -> Vec<f64> {
        self.min
            .iter()
            .zip(&self.max)
            .map(|(min, max)| 0.5 * (min + max))
            .collect()
    }
}

impl GeometricDomain for RectangularDomain {
    fn dimension(&self) -> Dimension {
        match self.min.len() {
            1 => Dimension::One,
            2 => Dimension::Two,
            3 => Dimension::Three,
            _ => panic!("Invalid dimension"),
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
        let mut normal = Array1::zeros(n);

        // Find which face(s) the point is on
        for i in 0..n {
            if (point[i] - self.min[i]).abs() < tolerance {
                normal[i] = -1.0; // Outward normal at x_min face
            } else if (point[i] - self.max[i]).abs() < tolerance {
                normal[i] = 1.0; // Outward normal at x_max face
            }
        }

        // Normalize (handles corner/edge cases)
        let norm = normal.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            normal /= norm;
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
        let mut points = Array2::zeros((n_points, dim));

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
        let mut points = Array2::zeros((n_points, dim));

        for i in 0..n_points {
            // Choose random face
            let face = rng.gen_range(0..(2 * dim));
            let axis = face / 2;
            let is_max = face % 2 == 1;

            // Set coordinate on chosen face
            points[[i, axis]] = if is_max {
                self.max[axis]
            } else {
                self.min[axis]
            };

            // Sample other coordinates uniformly
            for j in 0..dim {
                if j != axis {
                    points[[i, j]] = rng.gen_range(self.min[j]..self.max[j]);
                }
            }
        }

        points
    }
}

/// Circular/spherical domain
///
/// Defined as:
/// - 2D: disk of radius R centered at (x₀, y₀)
/// - 3D: ball of radius R centered at (x₀, y₀, z₀)
#[derive(Debug, Clone)]
pub struct SphericalDomain {
    /// Center coordinates
    pub center: Vec<f64>,
    /// Radius
    pub radius: f64,
}

impl SphericalDomain {
    /// Create 2D circular domain
    pub fn new_2d(x0: f64, y0: f64, radius: f64) -> Self {
        assert!(radius > 0.0, "Radius must be positive");
        Self {
            center: vec![x0, y0],
            radius,
        }
    }

    /// Create 3D spherical domain
    pub fn new_3d(x0: f64, y0: f64, z0: f64, radius: f64) -> Self {
        assert!(radius > 0.0, "Radius must be positive");
        Self {
            center: vec![x0, y0, z0],
            radius,
        }
    }

    /// Compute distance from center to point
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
    fn dimension(&self) -> Dimension {
        match self.center.len() {
            2 => Dimension::Two,
            3 => Dimension::Three,
            _ => panic!("Spherical domain only defined for 2D and 3D"),
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

        // Normalize
        let norm = normal.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            normal /= norm;
        }

        Some(normal)
    }

    fn measure(&self) -> f64 {
        match self.center.len() {
            2 => PI * self.radius.powi(2),               // Area of disk
            3 => (4.0 / 3.0) * PI * self.radius.powi(3), // Volume of ball
            _ => panic!("Invalid dimension"),
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
                // Rejection sampling
                let mut point = vec![0.0; dim];
                let mut r_sq = 0.0;

                for j in 0..dim {
                    let coord = rng.gen_range(-self.radius..self.radius);
                    point[j] = coord;
                    r_sq += coord * coord;
                }

                if r_sq <= self.radius * self.radius {
                    for j in 0..dim {
                        points[[i, j]] = self.center[j] + point[j];
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
                    // Uniform on circle
                    let theta = rng.gen_range(0.0..(2.0 * PI));
                    points[[i, 0]] = self.center[0] + self.radius * theta.cos();
                    points[[i, 1]] = self.center[1] + self.radius * theta.sin();
                }
                3 => {
                    // Uniform on sphere using Marsaglia's method
                    loop {
                        let x1 = rng.gen_range(-1.0..1.0);
                        let x2 = rng.gen_range(-1.0..1.0);
                        let s = x1 * x1 + x2 * x2;
                        if s < 1.0 {
                            let factor = 2.0 * (1.0_f64 - s).sqrt();
                            points[[i, 0]] = self.center[0] + self.radius * x1 * factor;
                            points[[i, 1]] = self.center[1] + self.radius * x2 * factor;
                            points[[i, 2]] = self.center[2] + self.radius * (1.0 - 2.0 * s);
                            break;
                        }
                    }
                }
                _ => panic!("Invalid dimension"),
            }
        }

        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangular_1d() {
        let domain = RectangularDomain::new_1d(0.0, 10.0);
        assert_eq!(domain.dimension(), Dimension::One);
        assert!(domain.contains(&[5.0]));
        assert!(!domain.contains(&[15.0]));
        assert_eq!(domain.measure(), 10.0);
    }

    #[test]
    fn test_rectangular_2d() {
        let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 2.0);
        assert_eq!(domain.dimension(), Dimension::Two);
        assert!(domain.contains(&[0.5, 1.0]));
        assert!(!domain.contains(&[1.5, 1.0]));
        assert_eq!(domain.measure(), 2.0);
    }

    #[test]
    fn test_boundary_classification() {
        let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0);
        let tol = 1e-6;

        assert_eq!(
            domain.classify_point(&[0.5, 0.5], tol),
            PointLocation::Interior
        );
        assert_eq!(
            domain.classify_point(&[0.0, 0.5], tol),
            PointLocation::Boundary
        );
        assert_eq!(
            domain.classify_point(&[1.5, 0.5], tol),
            PointLocation::Exterior
        );
    }

    #[test]
    fn test_normal_computation() {
        let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0);
        let tol = 1e-6;

        let normal = domain.normal(&[0.0, 0.5], tol).unwrap();
        assert!((normal[0] + 1.0).abs() < tol); // Outward at x_min
        assert!(normal[1].abs() < tol);

        let normal = domain.normal(&[1.0, 0.5], tol).unwrap();
        assert!((normal[0] - 1.0).abs() < tol); // Outward at x_max
    }

    #[test]
    fn test_spherical_2d() {
        let domain = SphericalDomain::new_2d(0.0, 0.0, 1.0);
        assert_eq!(domain.dimension(), Dimension::Two);
        assert!(domain.contains(&[0.5, 0.0]));
        assert!(!domain.contains(&[1.5, 0.0]));
        assert!((domain.measure() - PI).abs() < 1e-10);
    }

    #[test]
    fn test_sampling() {
        let domain = RectangularDomain::new_2d(0.0, 1.0, 0.0, 1.0);
        let interior = domain.sample_interior(100, Some(42));
        let boundary = domain.sample_boundary(100, Some(42));

        assert_eq!(interior.shape(), &[100, 2]);
        assert_eq!(boundary.shape(), &[100, 2]);

        // Check all interior points are inside
        for i in 0..interior.shape()[0] {
            let point = [interior[[i, 0]], interior[[i, 1]]];
            assert!(domain.contains(&point));
        }

        // Check all boundary points are on boundary
        let tol = 1e-10;
        for i in 0..boundary.shape()[0] {
            let point = [boundary[[i, 0]], boundary[[i, 1]]];
            assert_eq!(domain.classify_point(&point, tol), PointLocation::Boundary);
        }
    }
}
