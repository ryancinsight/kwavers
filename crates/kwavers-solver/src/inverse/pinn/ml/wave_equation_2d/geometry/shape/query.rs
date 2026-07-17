//! `WaveGeometry2D` spatial queries: `contains`, `bounding_box`, `sample_points`.

use leto::Array1;

use super::WaveGeometry2D;

impl WaveGeometry2D {
    /// Check if a point (x, y) is inside the geometry.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        match self {
            WaveGeometry2D::Rectangular {
                x_min,
                x_max,
                y_min,
                y_max,
            } => x >= *x_min && x <= *x_max && y >= *y_min && y <= *y_max,
            WaveGeometry2D::Circular {
                x_center,
                y_center,
                radius,
            } => {
                let dx = x - x_center;
                let dy = y - y_center;
                (dx * dx + dy * dy).sqrt() <= *radius
            }
            WaveGeometry2D::LShaped {
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
            WaveGeometry2D::Polygonal { vertices, holes } => {
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
            WaveGeometry2D::ParametricCurve {
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
            WaveGeometry2D::AdaptiveMesh { base_geometry, .. } => base_geometry.contains(x, y),
            WaveGeometry2D::MultiRegion { regions, .. } => {
                regions.iter().any(|(geom, _)| geom.contains(x, y))
            }
        }
    }

    /// Get the bounding box of the geometry as (x_min, x_max, y_min, y_max).
    pub fn bounding_box(&self) -> (f64, f64, f64, f64) {
        match self {
            WaveGeometry2D::Rectangular {
                x_min,
                x_max,
                y_min,
                y_max,
            } => (*x_min, *x_max, *y_min, *y_max),
            WaveGeometry2D::Circular {
                x_center,
                y_center,
                radius,
            } => (
                x_center - radius,
                x_center + radius,
                y_center - radius,
                y_center + radius,
            ),
            WaveGeometry2D::LShaped {
                x_min,
                x_max,
                y_min,
                y_max,
                ..
            } => (*x_min, *x_max, *y_min, *y_max),
            WaveGeometry2D::Polygonal { vertices, .. } => {
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
            WaveGeometry2D::ParametricCurve { bounds, .. } => *bounds,
            WaveGeometry2D::AdaptiveMesh { base_geometry, .. } => base_geometry.bounding_box(),
            WaveGeometry2D::MultiRegion { regions, .. } => {
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

        while (x_points.len()) < n_points {
            let x = x_min + (x_max - x_min) * rand::random::<f64>();
            let y = y_min + (y_max - y_min) * rand::random::<f64>();
            if self.contains(x, y) {
                x_points.push(x);
                y_points.push(y);
            }
        }

        (
            Array1::from_vec([x_points.len()], x_points)
                .expect("sample_points x-shape must match x values"),
            Array1::from_vec([y_points.len()], y_points)
                .expect("sample_points y-shape must match y values"),
        )
    }
}
