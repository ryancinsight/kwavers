//! 2D geometry definitions for PINN domains.

mod fmt;
mod query;

use super::interface::InterfaceCondition;

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
}
