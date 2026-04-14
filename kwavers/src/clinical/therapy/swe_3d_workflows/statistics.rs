#[derive(Debug, Clone)]
pub struct VolumetricStatistics {
    /// Number of valid voxels in analysis
    pub valid_voxels: usize,
    /// Mean Young's modulus (Pa)
    pub mean_modulus: f64,
    /// Standard deviation of Young's modulus (Pa)
    pub std_modulus: f64,
    /// Median Young's modulus (Pa)
    pub median_modulus: f64,
    /// Minimum Young's modulus (Pa)
    pub min_modulus: f64,
    /// Maximum Young's modulus (Pa)
    pub max_modulus: f64,
    /// Mean shear wave speed (m/s)
    pub mean_speed: f64,
    /// Mean confidence score (0-1)
    pub mean_confidence: f64,
    /// Mean quality score (0-1)
    pub mean_quality: f64,
    /// Volume coverage fraction (0-1)
    pub volume_coverage: f64,
}
