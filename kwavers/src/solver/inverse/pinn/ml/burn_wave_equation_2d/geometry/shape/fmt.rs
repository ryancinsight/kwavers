//! `Debug` implementation for `BurnWave2dGeometry`.

use super::BurnWave2dGeometry;

impl std::fmt::Debug for BurnWave2dGeometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BurnWave2dGeometry::Rectangular {
                x_min,
                x_max,
                y_min,
                y_max,
            } => write!(
                f,
                "Rectangular {{ x_min: {}, x_max: {}, y_min: {}, y_max: {} }}",
                x_min, x_max, y_min, y_max
            ),
            BurnWave2dGeometry::Circular {
                x_center,
                y_center,
                radius,
            } => write!(
                f,
                "Circular {{ center: ({}, {}), radius: {} }}",
                x_center, y_center, radius
            ),
            BurnWave2dGeometry::LShaped {
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
            BurnWave2dGeometry::Polygonal { vertices, holes } => write!(
                f,
                "Polygonal {{ vertices: {}, holes: {} }}",
                vertices.len(),
                holes.len()
            ),
            BurnWave2dGeometry::ParametricCurve {
                t_min,
                t_max,
                bounds,
                ..
            } => write!(
                f,
                "ParametricCurve {{ t: [{}, {}], bounds: {:?} }}",
                t_min, t_max, bounds
            ),
            BurnWave2dGeometry::AdaptiveMesh {
                base_geometry,
                refinement_threshold,
                max_level,
            } => write!(
                f,
                "AdaptiveMesh {{ threshold: {}, max_level: {}, base: {:?} }}",
                refinement_threshold, max_level, base_geometry
            ),
            BurnWave2dGeometry::MultiRegion {
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
