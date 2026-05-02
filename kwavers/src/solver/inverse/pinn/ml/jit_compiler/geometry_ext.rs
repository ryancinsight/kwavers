use crate::solver::inverse::pinn::ml::Geometry2D;

impl Geometry2D {
    pub(super) fn geometry_type(&self) -> String {
        match self {
            Geometry2D::Rectangular { .. } => "rectangular".to_string(),
            Geometry2D::Circular { .. } => "circular".to_string(),
            Geometry2D::LShaped { .. } => "lshaped".to_string(),
            Geometry2D::Polygonal { .. } => "polygonal".to_string(),
            Geometry2D::ParametricCurve { .. } => "parametric".to_string(),
            Geometry2D::AdaptiveMesh { .. } => "adaptive".to_string(),
            Geometry2D::MultiRegion { .. } => "multiregion".to_string(),
        }
    }
}
