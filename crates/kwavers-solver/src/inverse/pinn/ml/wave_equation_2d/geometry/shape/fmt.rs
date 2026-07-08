//! `Debug` implementation for `WaveGeometry2D`.

use super::WaveGeometry2D;

impl std::fmt::Debug for WaveGeometry2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaveGeometry2D::Rectangular {
                x_min,
                x_max,
                y_min,
                y_max,
            } => write!(
                f,
                "Rectangular {{ x_min: {}, x_max: {}, y_min: {}, y_max: {} }}",
                x_min, x_max, y_min, y_max
            ),
            WaveGeometry2D::Circular {
                x_center,
                y_center,
                radius,
            } => write!(
                f,
                "Circular {{ center: ({}, {}), radius: {} }}",
                x_center, y_center, radius
            ),
            WaveGeometry2D::LShaped {
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
            WaveGeometry2D::Polygonal { vertices, holes } => write!(
                f,
                "Polygonal {{ vertices: {}, holes: {} }}",
                vertices.len(),
                holes.len()
            ),
            WaveGeometry2D::ParametricCurve {
                t_min,
                t_max,
                bounds,
                ..
            } => write!(
                f,
                "ParametricCurve {{ t: [{}, {}], bounds: {:?} }}",
                t_min, t_max, bounds
            ),
            WaveGeometry2D::AdaptiveMesh {
                base_geometry,
                refinement_threshold,
                max_level,
            } => write!(
                f,
                "AdaptiveMesh {{ threshold: {}, max_level: {}, base: {:?} }}",
                refinement_threshold, max_level, base_geometry
            ),
            WaveGeometry2D::MultiRegion {
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
