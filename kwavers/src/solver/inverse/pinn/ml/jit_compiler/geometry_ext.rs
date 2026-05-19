use crate::solver::inverse::pinn::ml::BurnWave2dGeometry;

impl BurnWave2dGeometry {
    pub(super) fn geometry_type(&self) -> String {
        match self {
            BurnWave2dGeometry::Rectangular { .. } => "rectangular".to_string(),
            BurnWave2dGeometry::Circular { .. } => "circular".to_string(),
            BurnWave2dGeometry::LShaped { .. } => "lshaped".to_string(),
            BurnWave2dGeometry::Polygonal { .. } => "polygonal".to_string(),
            BurnWave2dGeometry::ParametricCurve { .. } => "parametric".to_string(),
            BurnWave2dGeometry::AdaptiveMesh { .. } => "adaptive".to_string(),
            BurnWave2dGeometry::MultiRegion { .. } => "multiregion".to_string(),
        }
    }
}
