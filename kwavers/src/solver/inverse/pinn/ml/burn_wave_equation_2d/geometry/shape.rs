//! 2D geometry definitions for PINN domains.

use super::interface::InterfaceCondition;
use ndarray::Array1;

/// 2D geometry definitions for PINN domains.
pub enum Geometry2D {
    /// Rectangular domain: [x_min, x_max] × [y_min, y_max].
    Rectangular {
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
    },
    /// Circular domain: center (x0, y0) with radius r.
    Circular {
        x_center: f64,
        y_center: f64,
        radius: f64,
    },
    /// L-shaped domain (common test case).
    LShaped {
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        notch_x: f64,
        notch_y: f64,
    },
    /// Polygonal domain with arbitrary boundary.
    Polygonal {
        /// List of (x, y) vertices in counter-clockwise order.
        vertices: Vec<(f64, f64)>,
        /// Optional holes in the polygon.
        holes: Vec<Vec<(f64, f64)>>,
    },
    /// Parametric curve boundary domain.
    ParametricCurve {
        x_func: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        y_func: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        t_min: f64,
        t_max: f64,
        /// Interior sampling region bounds: (x_min, x_max, y_min, y_max).
        bounds: (f64, f64, f64, f64),
    },
    /// Adaptive mesh refinement domain.
    AdaptiveMesh {
        base_geometry: Box<Geometry2D>,
        refinement_threshold: f64,
        max_level: usize,
    },
    /// Multi-region composite domain.
    MultiRegion {
        regions: Vec<(Geometry2D, usize)>,
        interfaces: Vec<InterfaceCondition>,
    },
}

impl Geometry2D {
    /// Create a rectangular geometry.
    pub fn rectangular(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self::Rectangular {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    /// Create a circular geometry.
    pub fn circular(x_center: f64, y_center: f64, radius: f64) -> Self {
        Self::Circular {
            x_center,
            y_center,
            radius,
        }
    }

    /// Create an L-shaped geometry.
    pub fn l_shaped(
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        notch_x: f64,
        notch_y: f64,
    ) -> Self {
        Self::LShaped {
            x_min,
            x_max,
            y_min,
            y_max,
            notch_x,
            notch_y,
        }
    }

    /// Create a polygonal geometry.
    pub fn polygonal(vertices: Vec<(f64, f64)>, holes: Vec<Vec<(f64, f64)>>) -> Self {
        Self::Polygonal { vertices, holes }
    }

    /// Create a parametric curve geometry.
    pub fn parametric_curve(
        x_func: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        y_func: Box<dyn Fn(f64) -> f64 + Send + Sync>,
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

    /// Create an adaptive mesh geometry.
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

    /// Create a multi-region geometry.
    pub fn multi_region(
        regions: Vec<(Geometry2D, usize)>,
        interfaces: Vec<InterfaceCondition>,
    ) -> Self {
        Self::MultiRegion {
            regions,
            interfaces,
        }
    }

    /// Check if a point (x, y) is inside the geometry.
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
                (dx * dx + dy * dy).sqrt() <= *radius
            }
            Geometry2D::LShaped {
                x_min,
                x_max,
                y_min,
                y_max,
                notch_x,
                notch_y,
            } => {
                let in_full_rect = x >= *x_min && x <= *x_max && y >= *y_min && y <= *y_max;
                let in_notch = x >= *notch_x && y >= *notch_y;
                in_full_rect && !in_notch
            }
            Geometry2D::Polygonal { vertices, holes } => {
                let mut inside = false;
                let n = vertices.len();
                let mut j = n - 1;
                for i in 0..n {
                    let vi = vertices[i];
                    let vj = vertices[j];
                    if ((vi.1 > y) != (vj.1 > y))
                        && (x < (vj.0 - vi.0) * (y - vi.1) / (vj.1 - vi.1) + vi.0)
                    {
                        inside = !inside;
                    }
                    j = i;
                }
                if inside {
                    for hole in holes {
                        let mut hole_inside = false;
                        let m = hole.len();
                        let mut k = m - 1;
                        for i in 0..m {
                            let vi = hole[i];
                            let vj = hole[k];
                            if ((vi.1 > y) != (vj.1 > y))
                                && (x < (vj.0 - vi.0) * (y - vi.1) / (vj.1 - vi.1) + vi.0)
                            {
                                hole_inside = !hole_inside;
                            }
                            k = i;
                        }
                        if hole_inside {
                            return false;
                        }
                    }
                }
                inside
            }
            Geometry2D::ParametricCurve {
                x_func,
                y_func,
                t_min,
                t_max,
                bounds,
            } => {
                let (x_min, x_max, y_min, y_max) = bounds;
                if !(x >= *x_min && x <= *x_max && y >= *y_min && y <= *y_max) {
                    return false;
                }
                let tolerance = 0.01;
                let n_samples = 1000;
                for i in 0..n_samples {
                    let t = t_min + (t_max - t_min) * (i as f64) / (n_samples - 1) as f64;
                    let curve_x = x_func(t);
                    let curve_y = y_func(t);
                    let dx = x - curve_x;
                    let dy = y - curve_y;
                    if (dx * dx + dy * dy).sqrt() <= tolerance {
                        return true;
                    }
                }
                false
            }
            Geometry2D::AdaptiveMesh { base_geometry, .. } => base_geometry.contains(x, y),
            Geometry2D::MultiRegion { regions, .. } => {
                regions.iter().any(|(geom, _)| geom.contains(x, y))
            }
        }
    }

    /// Get the bounding box of the geometry as (x_min, x_max, y_min, y_max).
    pub fn bounding_box(&self) -> (f64, f64, f64, f64) {
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
                for (x, y) in vertices {
                    x_min = x_min.min(*x);
                    x_max = x_max.max(*x);
                    y_min = y_min.min(*y);
                    y_max = y_max.max(*y);
                }
                (x_min, x_max, y_min, y_max)
            }
            Geometry2D::ParametricCurve { bounds, .. } => *bounds,
            Geometry2D::AdaptiveMesh { base_geometry, .. } => base_geometry.bounding_box(),
            Geometry2D::MultiRegion { regions, .. } => {
                let mut x_min = f64::INFINITY;
                let mut x_max = f64::NEG_INFINITY;
                let mut y_min = f64::INFINITY;
                let mut y_max = f64::NEG_INFINITY;
                for (geom, _) in regions {
                    let (gx_min, gx_max, gy_min, gy_max) = geom.bounding_box();
                    x_min = x_min.min(gx_min);
                    x_max = x_max.max(gx_max);
                    y_min = y_min.min(gy_min);
                    y_max = y_max.max(gy_max);
                }
                (x_min, x_max, y_min, y_max)
            }
        }
    }

    /// Generate random points inside the geometry via rejection sampling.
    pub fn sample_points(&self, n_points: usize) -> (Array1<f64>, Array1<f64>) {
        let (x_min, x_max, y_min, y_max) = self.bounding_box();
        let mut x_points = Vec::with_capacity(n_points);
        let mut y_points = Vec::with_capacity(n_points);

        while x_points.len() < n_points {
            let x = x_min + (x_max - x_min) * rand::random::<f64>();
            let y = y_min + (y_max - y_min) * rand::random::<f64>();
            if self.contains(x, y) {
                x_points.push(x);
                y_points.push(y);
            }
        }

        (Array1::from_vec(x_points), Array1::from_vec(y_points))
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
            } => write!(
                f,
                "Rectangular {{ x_min: {}, x_max: {}, y_min: {}, y_max: {} }}",
                x_min, x_max, y_min, y_max
            ),
            Geometry2D::Circular {
                x_center,
                y_center,
                radius,
            } => write!(
                f,
                "Circular {{ center: ({}, {}), radius: {} }}",
                x_center, y_center, radius
            ),
            Geometry2D::LShaped {
                x_min,
                x_max,
                y_min,
                y_max,
                notch_x,
                notch_y,
            } => write!(
                f,
                "LShaped {{ bounds: [{}, {}]×[{}, {}], notch: ({}, {}) }}",
                x_min, x_max, y_min, y_max, notch_x, notch_y
            ),
            Geometry2D::Polygonal { vertices, holes } => write!(
                f,
                "Polygonal {{ vertices: {}, holes: {} }}",
                vertices.len(),
                holes.len()
            ),
            Geometry2D::ParametricCurve {
                t_min,
                t_max,
                bounds,
                ..
            } => write!(
                f,
                "ParametricCurve {{ t: [{}, {}], bounds: {:?} }}",
                t_min, t_max, bounds
            ),
            Geometry2D::AdaptiveMesh {
                base_geometry,
                refinement_threshold,
                max_level,
            } => write!(
                f,
                "AdaptiveMesh {{ threshold: {}, max_level: {}, base: {:?} }}",
                refinement_threshold, max_level, base_geometry
            ),
            Geometry2D::MultiRegion {
                regions,
                interfaces,
            } => write!(
                f,
                "MultiRegion {{ regions: {}, interfaces: {} }}",
                regions.len(),
                interfaces.len()
            ),
        }
    }
}
