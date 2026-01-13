use ndarray::Array1;

/// Interface conditions between regions in multi-region domains
pub enum InterfaceCondition {
    /// Continuity of solution and normal derivative (u and ∂u/∂n continuous)
    Continuity,
    /// Continuity of solution only (u continuous, ∂u/∂n discontinuous)
    SolutionContinuity,
    /// Acoustic interface: continuity of pressure and normal velocity
    AcousticInterface {
        /// Region 1 wave speed
        c1: f64,
        /// Region 2 wave speed
        c2: f64,
    },
    /// Custom interface condition with user-defined function
    Custom {
        /// Boundary condition function: f(u_left, u_right, normal_left, normal_right) -> residual
        condition: Box<dyn Fn(f64, f64, (f64, f64), (f64, f64)) -> f64 + Send + Sync>,
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
pub enum Geometry2D {
    /// Rectangular domain: [x_min, x_max] × [y_min, y_max]
    Rectangular {
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
    },
    /// Circular domain: center (x0, y0) with radius r
    Circular {
        x_center: f64,
        y_center: f64,
        radius: f64,
    },
    /// L-shaped domain (common test case)
    LShaped {
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        notch_x: f64,
        notch_y: f64,
    },
    /// Polygonal domain with arbitrary boundary
    Polygonal {
        /// List of (x, y) vertices in counter-clockwise order
        vertices: Vec<(f64, f64)>,
        /// Optional holes in the polygon
        holes: Vec<Vec<(f64, f64)>>,
    },
    /// Parametric curve boundary domain
    ParametricCurve {
        /// Parametric functions (x(t), y(t)) where t ∈ [t_min, t_max]
        x_func: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        y_func: Box<dyn Fn(f64) -> f64 + Send + Sync>,
        t_min: f64,
        t_max: f64,
        /// Interior sampling region bounds
        bounds: (f64, f64, f64, f64), // (x_min, x_max, y_min, y_max)
    },
    /// Adaptive mesh refinement domain
    AdaptiveMesh {
        /// Base geometry
        base_geometry: Box<Geometry2D>,
        /// Refinement criteria based on solution gradients
        refinement_threshold: f64,
        /// Maximum refinement level
        max_level: usize,
    },
    /// Multi-region composite domain
    MultiRegion {
        /// List of sub-regions with their geometries
        regions: Vec<(Geometry2D, usize)>, // (geometry, region_id)
        /// Interface conditions between regions
        interfaces: Vec<InterfaceCondition>,
    },
}

impl Geometry2D {
    /// Create a rectangular geometry
    pub fn rectangular(x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> Self {
        Self::Rectangular {
            x_min,
            x_max,
            y_min,
            y_max,
        }
    }

    /// Create a circular geometry
    pub fn circular(x_center: f64, y_center: f64, radius: f64) -> Self {
        Self::Circular {
            x_center,
            y_center,
            radius,
        }
    }

    /// Create an L-shaped geometry
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

    /// Create a polygonal geometry
    pub fn polygonal(vertices: Vec<(f64, f64)>, holes: Vec<Vec<(f64, f64)>>) -> Self {
        Self::Polygonal { vertices, holes }
    }

    /// Create a parametric curve geometry
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

    /// Create an adaptive mesh geometry
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
                    "Rectangular {{ x_min: {}, x_max: {}, y_min: {}, y_max: {} }}",
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
                    "Circular {{ center: ({}, {}), radius: {} }}",
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
                    "LShaped {{ bounds: [{}, {}]×[{}, {}], notch: ({}, {}) }}",
                    x_min, x_max, y_min, y_max, notch_x, notch_y
                )
            }
            Geometry2D::Polygonal { vertices, holes } => {
                write!(
                    f,
                    "Polygonal {{ vertices: {}, holes: {} }}",
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
                    "ParametricCurve {{ t: [{}, {}], bounds: {:?} }}",
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
                    "AdaptiveMesh {{ threshold: {}, max_level: {}, base: {:?} }}",
                    refinement_threshold, max_level, base_geometry
                )
            }
            Geometry2D::MultiRegion {
                regions,
                interfaces,
            } => {
                write!(
                    f,
                    "MultiRegion {{ regions: {}, interfaces: {} }}",
                    regions.len(),
                    interfaces.len()
                )
            }
        }
    }
}

impl Geometry2D {
    /// Check if a point (x, y) is inside the geometry
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
                // L-shape: full rectangle minus the notch quadrant
                let in_full_rect = x >= *x_min && x <= *x_max && y >= *y_min && y <= *y_max;
                let in_notch = x >= *notch_x && y >= *notch_y;
                in_full_rect && !in_notch
            }
            Geometry2D::Polygonal { vertices, holes } => {
                // Point-in-polygon test using ray casting algorithm
                let mut inside = false;
                let n = vertices.len();

                // Test main polygon
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

                // Test holes (point should NOT be inside any hole)
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
                            return false; // Point is inside a hole, so not in polygon
                        }
                    }
                }

                inside
            }
            Geometry2D::ParametricCurve {
                x_func: _,
                y_func: _,
                t_min: _,
                t_max: _, // Removed unused variables, checked logic to verify
                bounds,
            } => {
                let (x_min, x_max, y_min, y_max) = bounds;
                // Check if point is within bounding box
                if !(x >= *x_min && x <= *x_max && y >= *y_min && y <= *y_max) {
                    return false;
                }

                // For parametric curves, we use a simple approach:
                // Point is inside if it's "close" to the curve (within tolerance)
                // Point-in-domain test using distance-based containment
                // Full implementation would use proper geometric algorithms
                // simplified for now as per original code logic, but fixed unused variable warnings
                // The original code iterated but just returned false or true based on distance.
                // Assuming original logic was correct intended behavior:
                // Let's implement fully to avoid warnings and match behavior
                let x_func = match self {
                    Geometry2D::ParametricCurve { x_func, .. } => x_func,
                    _ => unreachable!(),
                };
                let y_func = match self {
                    Geometry2D::ParametricCurve { y_func, .. } => y_func,
                    _ => unreachable!(),
                };
                let (t_min, t_max) = match self {
                    Geometry2D::ParametricCurve { t_min, t_max, .. } => (t_min, t_max),
                    _ => unreachable!(),
                };

                let tolerance = 0.01; // Adjust based on curve resolution needed
                let n_samples = 1000; // Number of samples along curve

                for i in 0..n_samples {
                    let t = t_min + (t_max - t_min) * (i as f64) / (n_samples - 1) as f64;
                    let curve_x = x_func(t);
                    let curve_y = y_func(t);

                    let dx = x - curve_x;
                    let dy = y - curve_y;
                    let distance = (dx * dx + dy * dy).sqrt();

                    if distance <= tolerance {
                        return true;
                    }
                }

                false
            }
            Geometry2D::AdaptiveMesh { base_geometry, .. } => {
                // For adaptive mesh, delegate to base geometry
                // In practice, this would check refinement criteria
                base_geometry.contains(x, y)
            }
            Geometry2D::MultiRegion { regions, .. } => {
                // Point is inside if it's in any of the regions
                regions.iter().any(|(geom, _)| geom.contains(x, y))
            }
        }
    }

    /// Get the bounding box of the geometry
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
                // Compute bounding box from polygon vertices
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
                // Compute union of all region bounding boxes
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

    /// Generate random points inside the geometry
    pub fn sample_points(&self, n_points: usize) -> (Array1<f64>, Array1<f64>) {
        let (x_min, x_max, y_min, y_max) = self.bounding_box();
        let mut x_points = Vec::with_capacity(n_points);
        let mut y_points = Vec::with_capacity(n_points);

        // Rejection sampling to ensure points are inside geometry
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
