use ndarray::Array3;

/// Heterogeneous skull model with spatially varying acoustic properties.
///
/// Properties can be initialised from a binary mask, directly from CT data
/// using the legacy CTImageLoader pipeline, or from CT data via the
/// Hill-averaged BVF mixing model.
#[derive(Debug, Clone)]
pub struct HeterogeneousSkull {
    /// Sound speed distribution (m/s).
    pub sound_speed: Array3<f64>,
    /// Density distribution (kg/m³).
    pub density: Array3<f64>,
    /// Attenuation coefficient distribution (Np/m/MHz).
    pub attenuation: Array3<f64>,
}
