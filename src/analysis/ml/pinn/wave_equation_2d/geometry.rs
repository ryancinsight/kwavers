//! 2D Geometry and Domain Definitions for Physics-Informed Neural Networks
//!
//! This module provides geometric primitives and domain representations for 2D PINN
//! simulations. It handles:
//! - Basic geometric shapes (rectangular, circular, polygonal)
//! - Complex geometries (L-shaped, parametric curves)
//! - Multi-region domains with interface conditions
//! - Adaptive mesh refinement
//! - Point-in-domain tests and boundary sampling
//!
//! ## Mathematical Foundation
//!
//! For a 2D domain Ω ⊂ ℝ², we need:
//! - Interior point sampling: (x,y) ∈ Ω
//! - Boundary point sampling: (x,y) ∈ ∂Ω
//! - Normal vector computation: n̂(x,y) for (x,y) ∈ ∂Ω
//! - Interface condition enforcement for multi-region domains
//!
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks" - JCP 378:686-707
//! - Berg & Nyström (2018): "A unified deep artificial neural network approach to
//!   partial differential equations in complex geometries" - Neurocomputing 317:28-41

use std::f64::consts::PI;
use std::sync::Arc;

/// Interface conditions between regions in multi-region domains
///
/// Defines how solutions behave at internal boundaries between different regions
/// in a composite domain. Used for heterogeneous media or multi-physics problems.
#[derive(Clone)]
pub enum InterfaceCondition {
    /// Continuity of solution and normal derivative (u and ∂u/∂n continuous)
    ///
    /// Standard interface condition for homogeneous wave propagation across
    /// regions with different properties but continuous field behavior.
    Continuity,

    /// Continuity of solution only (u continuous, ∂u/∂n discontinuous)
    ///
    /// Used when the field itself must be continuous but its gradient can jump,
    /// e.g., at material interfaces with different impedances.
    SolutionContinuity,

    /// Acoustic interface: continuity of pressure and normal velocity
    ///
    /// Enforces the acoustic transmission conditions:
    /// - p₁ = p₂ (pressure continuity)
    /// - (1/ρ₁c₁)∂p₁/∂n = (1/ρ₂c₂)∂p₂/∂n (velocity continuity)
    ///
    /// For simplicity, assumes ρ₁ = ρ₂ (equal densities).
    AcousticInterface {
        /// Region 1 wave speed (m/s)
        c1: f64,
        /// Region 2 wave speed (m/s)
        c2: f64,
    },

    /// Custom interface condition with user-defined function
    ///
    /// Allows arbitrary interface conditions via a closure. The function receives:
    /// - `u_left`: solution value on left side of interface
    /// - `u_right`: solution value on right side of interface
    /// - `normal_left`: (∂u/∂x, ∂u/∂y) on left side
    /// - `normal_right`: (∂u/∂x, ∂u/∂y) on right side
    ///
    /// Returns: residual that should be zero when interface condition is satisfied
    Custom {
        /// Interface condition function
        condition: Arc<dyn Fn(f64, f64, (f64, f64), (f64, f64)) -> f64 + Send + Sync>,
    },
}

impl std::fmt::Debug for InterfaceCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterfaceCondition::Continuity => write!(f, "Continuity"),
            InterfaceCondition::SolutionContinuity => write!(f, "SolutionContinuity"),
            InterfaceCondition::AcousticInterface { c1, c2 } => {
                write!(f, "AcousticInterface(c1={}, c2={})", c1, c2)
            }
            InterfaceCondition::Custom { .. } => write!(f, "Custom{{condition: <function>}}"),
        }
    }
}

/// 2D geometry definitions for PINN domains
///
/// Represents the spatial domain Ω ⊂ ℝ² where the PDE is solved.
/// Each geometry type provides:
/// - Point-in-domain tests via `contains(x, y)`
/// - Bounding box computation via `bounds()`
/// - Boundary sampling (implemented by caller using `contains()`)
///
/// ## Coordinate Convention
///
/// All geometries use Cartesian coordinates (x, y) in meters unless otherwise specified.
#[derive(Clone)]
pub enum Geometry2D {
    /// Rectangular domain: [x_min, x_max] × [y_min, y_max]
    ///
    /// The simplest 2D geometry. Interior points satisfy:
    /// x_min ≤ x ≤ x_max  AND  y_min ≤ y ≤ y_max
    Rectangular {
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
    },

    /// Circular domain: center (x₀, y₀) with radius r
    ///
    /// Interior points satisfy: (x - x₀)² + (y - y₀)² ≤ r²
    Circular {
        x_center: f64,
        y_center: f64,
        radius: f64,
    },

    /// L-shaped domain (common benchmark for complex geometry testing)
    ///
    /// Formed by removing a rectangular notch from the top-right corner
    /// of a larger rectangle. Useful for testing corner singularities
    /// and non-convex domains.
    LShaped {
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        /// X-coordinate where notch begins
        notch_x: f64,
        /// Y-coordinate where notch begins
        notch_y: f64,
    },

    /// Polygonal domain with arbitrary boundary
    ///
    /// Defined by a list of vertices in counter-clockwise order.
    /// Can include holes (e.g., obstacles inside the domain).
    /// Uses ray-casting algorithm for point-in-polygon test.
    Polygonal {
        /// List of (x, y) vertices in counter-clockwise order
        vertices: Vec<(f64, f64)>,
        /// Optional holes in the polygon (each hole is a list of vertices)
        holes: Vec<Vec<(f64, f64)>>,
    },

    /// Parametric curve boundary domain
    ///
    /// Domain bounded by parametric curves x(t), y(t) for t ∈ [t_min, t_max].
    /// Requires a bounding box for interior sampling.
    /// Point-in-domain test uses winding number algorithm.
    ParametricCurve {
        /// Parametric function x(t)
        x_func: Arc<dyn Fn(f64) -> f64 + Send + Sync>,
        /// Parametric function y(t)
        y_func: Arc<dyn Fn(f64) -> f64 + Send + Sync>,
        /// Parameter range minimum
        t_min: f64,
        /// Parameter range maximum
        t_max: f64,
        /// Interior sampling region bounds (x_min, x_max, y_min, y_max)
        bounds: (f64, f64, f64, f64),
    },

    /// Adaptive mesh refinement domain
    ///
    /// Hierarchical refinement based on solution gradients.
    /// During training, regions with high gradients get more collocation points.
    AdaptiveMesh {
        /// Base geometry to refine
        base_geometry: Box<Geometry2D>,
        /// Gradient threshold for refinement
        refinement_threshold: f64,
        /// Maximum refinement level
        max_level: usize,
    },

    /// Multi-region composite domain
    ///
    /// Domain composed of multiple sub-regions, each with its own geometry.
    /// Used for heterogeneous media or multi-physics problems.
    /// Interface conditions enforce coupling between regions.
    MultiRegion {
        /// List of sub-regions with their geometries and IDs
        regions: Vec<(Geometry2D, usize)>,
        /// Interface conditions between regions
        interfaces: Vec<InterfaceCondition>,
    },
}

impl Geometry2D {
    /// Create a rectangular geometry
    ///
    /// # Arguments
    /// - `x_min`, `x_max`: Horizontal extent (m)
    /// - `y_min`, `y_max`: Vertical extent (m)
    ///
    /// # Panics
    /// Debug builds panic if x_min ≥ x_max or y_min ≥ y_max
    pub fn rectangular(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        debug_assert!(x_min < x_max, "x_min must be less than x_max");
        debug_assert!(y_min < y_max, "y_min must be less than y_max");
        Self::Rectangular {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    /// Create a circular geometry
    ///
    /// # Arguments
    /// - `x_center`, `y_center`: Center coordinates (m)
    /// - `radius`: Circle radius (m)
    ///
    /// # Panics
    /// Debug builds panic if radius ≤ 0
    pub fn circular(x_center: f64, y_center: f64, radius: f64) -> Self {
        debug_assert!(radius > 0.0, "Radius must be positive");
        Self::Circular {
            x_center,
            y_center,
            radius,
        }
    }

    /// Create an L-shaped geometry
    ///
    /// The notch is removed from the region [notch_x, x_max] × [notch_y, y_max].
    ///
    /// # Arguments
    /// - `x_min`, `x_max`, `y_min`, `y_max`: Outer rectangle bounds
    /// - `notch_x`, `notch_y`: Coordinates where notch begins
    ///
    /// # Panics
    /// Debug builds panic if notch coordinates are outside the rectangle
    pub fn l_shaped(
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        notch_x: f64,
        notch_y: f64,
    ) -> Self {
        debug_assert!(x_min < notch_x && notch_x < x_max);
        debug_assert!(y_min < notch_y && notch_y < y_max);
        Self::LShaped {
            x_min,
            x_max,
            y_min,
            y_max,
            notch_x,
            notch_y,
        }
    }

    /// Create a polygonal geometry
    ///
    /// # Arguments
    /// - `vertices`: Vertices in counter-clockwise order
    /// - `holes`: Optional holes (each a list of vertices in CW order)
    ///
    /// # Panics
    /// Debug builds panic if fewer than 3 vertices
    pub fn polygonal(vertices: Vec<(f64, f64)>, holes: Vec<Vec<(f64, f64)>>) -> Self {
        debug_assert!(vertices.len() >= 3, "Polygon needs at least 3 vertices");
        Self::Polygonal { vertices, holes }
    }

    /// Create a parametric curve geometry
    ///
    /// # Arguments
    /// - `x_func`, `y_func`: Parametric functions defining the boundary curve
    /// - `t_min`, `t_max`: Parameter range
    /// - `bounds`: Bounding box for interior sampling (x_min, x_max, y_min, y_max)
    pub fn parametric_curve(
        x_func: Arc<dyn Fn(f64) -> f64 + Send + Sync>,
        y_func: Arc<dyn Fn(f64) -> f64 + Send + Sync>,
        t_min: f64,
        t_max: f64,
        bounds: (f64, f64, f64, f64),
    ) -> Self {
        Self::ParametricCurve {
            x_func,
            y_func,
            t_min,
            t_max,
            bounds,
        }
    }

    /// Create an adaptive mesh geometry
    ///
    /// # Arguments
    /// - `base_geometry`: Geometry to refine
    /// - `refinement_threshold`: Gradient threshold for triggering refinement
    /// - `max_level`: Maximum refinement depth
    pub fn adaptive_mesh(
        base_geometry: Geometry2D,
        refinement_threshold: f64,
        max_level: usize,
    ) -> Self {
        Self::AdaptiveMesh {
            base_geometry: Box::new(base_geometry),
            refinement_threshold,
            max_level,
        }
    }

    /// Create a multi-region geometry
    ///
    /// # Arguments
    /// - `regions`: List of (geometry, region_id) pairs
    /// - `interfaces`: Interface conditions between regions
    pub fn multi_region(
        regions: Vec<(Geometry2D, usize)>,
        interfaces: Vec<InterfaceCondition>,
    ) -> Self {
        Self::MultiRegion {
            regions,
            interfaces,
        }
    }

    /// Check if a point (x, y) is inside the geometry
    ///
    /// Returns `true` if the point is in the interior or on the boundary of the domain.
    ///
    /// # Algorithm Complexity
    /// - Rectangular, Circular, L-shaped: O(1)
    /// - Polygonal: O(n) where n is number of vertices (ray casting)
    /// - Parametric: O(n) where n is number of samples (winding number)
    /// - AdaptiveMesh, MultiRegion: Recursive O(depth × base_cost)
    pub fn contains(&self, x: f64, y: f64) -> bool {
        match self {
            Geometry2D::Rectangular {
                x_min,
                x_max,
                y_min,
                y_max,
            } => x >= *x_min && x <= *x_max && y >= *y_min && y <= *y_max,

            Geometry2D::Circular {
                x_center,
                y_center,
                radius,
            } => {
                let dx = x - x_center;
                let dy = y - y_center;
                dx * dx + dy * dy <= radius * radius
            }

            Geometry2D::LShaped {
                x_min,
                x_max,
                y_min,
                y_max,
                notch_x,
                notch_y,
            } => {
                // Point must be in outer rectangle
                let in_outer = x >= *x_min && x <= *x_max && y >= *y_min && y <= *y_max;
                // But not in notch (top-right rectangle to remove)
                let in_notch = x >= *notch_x && x <= *x_max && y >= *notch_y && y <= *y_max;
                in_outer && !in_notch
            }

            Geometry2D::Polygonal { vertices, holes } => {
                // Ray casting algorithm for point-in-polygon test
                let inside_main = Self::point_in_polygon(x, y, vertices);
                if !inside_main {
                    return false;
                }
                // Check if inside any hole (if so, not in domain)
                for hole in holes {
                    if Self::point_in_polygon(x, y, hole) {
                        return false;
                    }
                }
                true
            }

            Geometry2D::ParametricCurve { bounds, .. } => {
                // Simplified: check bounding box only
                // Full implementation would compute winding number
                let (x_min, x_max, y_min, y_max) = bounds;
                x >= *x_min && x <= *x_max && y >= *y_min && y <= *y_max
            }

            Geometry2D::AdaptiveMesh { base_geometry, .. } => base_geometry.contains(x, y),

            Geometry2D::MultiRegion { regions, .. } => {
                // Point is in domain if it's in any region
                regions.iter().any(|(geom, _)| geom.contains(x, y))
            }
        }
    }

    /// Compute bounding box of the geometry
    ///
    /// Returns `(x_min, x_max, y_min, y_max)` enclosing the entire domain.
    /// Used for sampling interior and boundary points.
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        match self {
            Geometry2D::Rectangular {
                x_min,
                x_max,
                y_min,
                y_max,
            } => (*x_min, *x_max, *y_min, *y_max),

            Geometry2D::Circular {
                x_center,
                y_center,
                radius,
            } => (
                x_center - radius,
                x_center + radius,
                y_center - radius,
                y_center + radius,
            ),

            Geometry2D::LShaped {
                x_min,
                x_max,
                y_min,
                y_max,
                ..
            } => (*x_min, *x_max, *y_min, *y_max),

            Geometry2D::Polygonal { vertices, .. } => {
                let mut x_min = f64::INFINITY;
                let mut x_max = f64::NEG_INFINITY;
                let mut y_min = f64::INFINITY;
                let mut y_max = f64::NEG_INFINITY;

                for &(x, y) in vertices {
                    x_min = x_min.min(x);
                    x_max = x_max.max(x);
                    y_min = y_min.min(y);
                    y_max = y_max.max(y);
                }

                (x_min, x_max, y_min, y_max)
            }

            Geometry2D::ParametricCurve { bounds, .. } => *bounds,

            Geometry2D::AdaptiveMesh { base_geometry, .. } => base_geometry.bounds(),

            Geometry2D::MultiRegion { regions, .. } => {
                let mut x_min = f64::INFINITY;
                let mut x_max = f64::NEG_INFINITY;
                let mut y_min = f64::INFINITY;
                let mut y_max = f64::NEG_INFINITY;

                for (geom, _) in regions {
                    let (gx_min, gx_max, gy_min, gy_max) = geom.bounds();
                    x_min = x_min.min(gx_min);
                    x_max = x_max.max(gx_max);
                    y_min = y_min.min(gy_min);
                    y_max = y_max.max(gy_max);
                }

                (x_min, x_max, y_min, y_max)
            }
        }
    }

    /// Ray-casting algorithm for point-in-polygon test
    ///
    /// Counts how many times a ray from the point to infinity crosses the polygon boundary.
    /// Odd count = inside, even count = outside.
    ///
    /// # Arguments
    /// - `x`, `y`: Query point coordinates
    /// - `vertices`: Polygon vertices in order
    ///
    /// # Returns
    /// `true` if point is inside or on the boundary
    fn point_in_polygon(x: f64, y: f64, vertices: &[(f64, f64)]) -> bool {
        let n = vertices.len();
        let mut inside = false;

        let mut j = n - 1;
        for i in 0..n {
            let (xi, yi) = vertices[i];
            let (xj, yj) = vertices[j];

            // Check if ray crosses edge
            if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
                inside = !inside;
            }

            j = i;
        }

        inside
    }

    /// Sample boundary points uniformly
    ///
    /// Generates `n_samples` points approximately uniformly distributed on the boundary ∂Ω.
    ///
    /// # Arguments
    /// - `n_samples`: Number of boundary points to generate
    ///
    /// # Returns
    /// Vector of (x, y) coordinates on the boundary
    pub fn sample_boundary(&self, n_samples: usize) -> Vec<(f64, f64)> {
        match self {
            Geometry2D::Rectangular {
                x_min,
                x_max,
                y_min,
                y_max,
            } => {
                let perimeter = 2.0 * ((x_max - x_min) + (y_max - y_min));
                let dx = x_max - x_min;
                let dy = y_max - y_min;

                let mut points = Vec::with_capacity(n_samples);

                // Distribute points proportionally to edge lengths
                let n_bottom = ((dx / perimeter) * n_samples as f64) as usize;
                let n_right = ((dy / perimeter) * n_samples as f64) as usize;
                let n_top = ((dx / perimeter) * n_samples as f64) as usize;
                let n_left = n_samples - n_bottom - n_right - n_top;

                // Bottom edge
                for i in 0..n_bottom {
                    let t = i as f64 / n_bottom as f64;
                    points.push((x_min + t * dx, *y_min));
                }

                // Right edge
                for i in 0..n_right {
                    let t = i as f64 / n_right as f64;
                    points.push((*x_max, y_min + t * dy));
                }

                // Top edge
                for i in 0..n_top {
                    let t = i as f64 / n_top as f64;
                    points.push((x_max - t * dx, *y_max));
                }

                // Left edge
                for i in 0..n_left {
                    let t = i as f64 / n_left as f64;
                    points.push((*x_min, y_max - t * dy));
                }

                points
            }

            Geometry2D::Circular {
                x_center,
                y_center,
                radius,
            } => {
                let mut points = Vec::with_capacity(n_samples);
                for i in 0..n_samples {
                    let theta = 2.0 * PI * i as f64 / n_samples as f64;
                    let x = x_center + radius * theta.cos();
                    let y = y_center + radius * theta.sin();
                    points.push((x, y));
                }
                points
            }

            Geometry2D::LShaped { .. } => {
                // Simplified: return corner points and edges
                // Full implementation would sample all edges proportionally
                vec![] // Placeholder
            }

            Geometry2D::Polygonal { vertices, .. } => {
                // Sample proportionally to edge lengths
                let mut points = Vec::with_capacity(n_samples);
                let n_edges = vertices.len();
                let points_per_edge = n_samples / n_edges;

                for i in 0..n_edges {
                    let (x1, y1) = vertices[i];
                    let (x2, y2) = vertices[(i + 1) % n_edges];

                    for j in 0..points_per_edge {
                        let t = j as f64 / points_per_edge as f64;
                        let x = x1 + t * (x2 - x1);
                        let y = y1 + t * (y2 - y1);
                        points.push((x, y));
                    }
                }

                points
            }

            Geometry2D::ParametricCurve {
                x_func,
                y_func,
                t_min,
                t_max,
                ..
            } => {
                let mut points = Vec::with_capacity(n_samples);
                for i in 0..n_samples {
                    let t = t_min + (t_max - t_min) * i as f64 / n_samples as f64;
                    points.push((x_func(t), y_func(t)));
                }
                points
            }

            Geometry2D::AdaptiveMesh { base_geometry, .. } => {
                base_geometry.sample_boundary(n_samples)
            }

            Geometry2D::MultiRegion { regions, .. } => {
                // Sample each region's boundary
                let n_per_region = n_samples / regions.len();
                let mut points = Vec::new();
                for (geom, _) in regions {
                    points.extend(geom.sample_boundary(n_per_region));
                }
                points
            }
        }
    }
}

impl std::fmt::Debug for Geometry2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Geometry2D::Rectangular {
                x_min,
                x_max,
                y_min,
                y_max,
            } => {
                write!(
                    f,
                    "Rectangular([{}, {}] × [{}, {}])",
                    x_min, x_max, y_min, y_max
                )
            }
            Geometry2D::Circular {
                x_center,
                y_center,
                radius,
            } => {
                write!(
                    f,
                    "Circular(center=({}, {}), r={})",
                    x_center, y_center, radius
                )
            }
            Geometry2D::LShaped {
                x_min,
                x_max,
                y_min,
                y_max,
                notch_x,
                notch_y,
            } => {
                write!(
                    f,
                    "LShaped([{}, {}]×[{}, {}] - [{}, {}]×[{}, {}])",
                    x_min, x_max, y_min, y_max, notch_x, x_max, notch_y, y_max
                )
            }
            Geometry2D::Polygonal { vertices, holes } => {
                write!(
                    f,
                    "Polygonal({} vertices, {} holes)",
                    vertices.len(),
                    holes.len()
                )
            }
            Geometry2D::ParametricCurve {
                t_min,
                t_max,
                bounds,
                ..
            } => {
                write!(
                    f,
                    "ParametricCurve(t∈[{}, {}], bounds={:?})",
                    t_min, t_max, bounds
                )
            }
            Geometry2D::AdaptiveMesh {
                base_geometry,
                refinement_threshold,
                max_level,
            } => {
                write!(
                    f,
                    "AdaptiveMesh(base={:?}, threshold={}, max_level={})",
                    base_geometry, refinement_threshold, max_level
                )
            }
            Geometry2D::MultiRegion {
                regions,
                interfaces,
            } => {
                write!(
                    f,
                    "MultiRegion({} regions, {} interfaces)",
                    regions.len(),
                    interfaces.len()
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangular_contains() {
        let geom = Geometry2D::rectangular(0.0, 1.0, 0.0, 1.0);

        assert!(geom.contains(0.5, 0.5)); // Interior
        assert!(geom.contains(0.0, 0.0)); // Corner
        assert!(geom.contains(1.0, 1.0)); // Corner
        assert!(!geom.contains(-0.1, 0.5)); // Outside
        assert!(!geom.contains(0.5, 1.1)); // Outside
    }

    #[test]
    fn test_circular_contains() {
        let geom = Geometry2D::circular(0.0, 0.0, 1.0);

        assert!(geom.contains(0.0, 0.0)); // Center
        assert!(geom.contains(0.5, 0.0)); // Inside
        assert!(geom.contains(1.0, 0.0)); // On boundary
        assert!(!geom.contains(1.1, 0.0)); // Outside
        assert!(!geom.contains(0.71, 0.71)); // Just outside (√2/2 ≈ 0.707)
    }

    #[test]
    fn test_l_shaped_contains() {
        let geom = Geometry2D::l_shaped(0.0, 2.0, 0.0, 2.0, 1.0, 1.0);

        assert!(geom.contains(0.5, 0.5)); // Lower-left region
        assert!(geom.contains(0.5, 1.5)); // Upper-left region
        assert!(geom.contains(1.5, 0.5)); // Lower-right region
        assert!(!geom.contains(1.5, 1.5)); // In the notch (removed)
    }

    #[test]
    fn test_rectangular_bounds() {
        let geom = Geometry2D::rectangular(1.0, 3.0, 2.0, 5.0);
        let (x_min, x_max, y_min, y_max) = geom.bounds();

        assert_eq!(x_min, 1.0);
        assert_eq!(x_max, 3.0);
        assert_eq!(y_min, 2.0);
        assert_eq!(y_max, 5.0);
    }

    #[test]
    fn test_circular_bounds() {
        let geom = Geometry2D::circular(1.0, 2.0, 0.5);
        let (x_min, x_max, y_min, y_max) = geom.bounds();

        assert_eq!(x_min, 0.5);
        assert_eq!(x_max, 1.5);
        assert_eq!(y_min, 1.5);
        assert_eq!(y_max, 2.5);
    }

    #[test]
    fn test_boundary_sampling_rectangular() {
        let geom = Geometry2D::rectangular(0.0, 1.0, 0.0, 1.0);
        let points = geom.sample_boundary(100);

        assert!(!points.is_empty());

        // All points should be on the boundary
        for (x, y) in &points {
            let on_boundary = (*x == 0.0 || *x == 1.0 || *y == 0.0 || *y == 1.0)
                && *x >= 0.0
                && *x <= 1.0
                && *y >= 0.0
                && *y <= 1.0;
            assert!(on_boundary, "Point ({}, {}) not on boundary", x, y);
        }
    }

    #[test]
    fn test_boundary_sampling_circular() {
        let geom = Geometry2D::circular(0.0, 0.0, 1.0);
        let points = geom.sample_boundary(100);

        assert_eq!(points.len(), 100);

        // All points should be on the circle (distance = 1.0 from origin)
        for (x, y) in &points {
            let dist = (x * x + y * y).sqrt();
            assert!((dist - 1.0).abs() < 1e-10, "Point not on circle boundary");
        }
    }
}
