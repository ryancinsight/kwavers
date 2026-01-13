use ndarray::{Array1, Array3};

/// Spectral unmixing configuration
#[derive(Debug, Clone)]
pub struct SpectralUnmixingConfig {
    /// Regularization parameter λ for Tikhonov regularization
    /// (0.0 = no regularization, >0 = increasing stability)
    pub regularization_lambda: f64,
    /// Enforce non-negative concentrations (physical constraint)
    pub non_negative: bool,
    /// Minimum acceptable condition number for extinction matrix
    pub min_condition_number: f64,
}

impl Default for SpectralUnmixingConfig {
    fn default() -> Self {
        Self {
            regularization_lambda: 1e-6, // Small regularization for stability
            non_negative: true,          // Physical constraint
            min_condition_number: 1e-10, // Warn if matrix is poorly conditioned
        }
    }
}

/// Spectral unmixing result for a single voxel
#[derive(Debug, Clone)]
pub struct UnmixingResult {
    /// Chromophore concentrations (M or arbitrary units)
    pub concentrations: Array1<f64>,
    /// Residual norm ||μ - EC||
    pub residual_norm: f64,
    /// Relative residual ||μ - EC|| / ||μ||
    pub relative_residual: f64,
}

/// Multi-voxel spectral unmixing result
#[derive(Debug, Clone)]
pub struct VolumetricUnmixingResult {
    /// Chromophore concentration maps (n_chromophores × nx × ny × nz)
    pub concentration_maps: Vec<Array3<f64>>,
    /// Residual norm map (nx × ny × nz)
    pub residual_map: Array3<f64>,
    /// Chromophore names
    pub chromophore_names: Vec<String>,
}
