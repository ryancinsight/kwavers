use crate::inverse::pinn::ml::WaveGeometry2D;

impl WaveGeometry2D {
    pub(super) fn geometry_type(&self) -> String {
        match self {
            WaveGeometry2D::Rectangular { .. } => "rectangular".to_string(),
            WaveGeometry2D::Circular { .. } => "circular".to_string(),
            WaveGeometry2D::LShaped { .. } => "lshaped".to_string(),
            WaveGeometry2D::Polygonal { .. } => "polygonal".to_string(),
            WaveGeometry2D::ParametricCurve { .. } => "parametric".to_string(),
            WaveGeometry2D::AdaptiveMesh { .. } => "adaptive".to_string(),
            WaveGeometry2D::MultiRegion { .. } => "multiregion".to_string(),
        }
    }
}
