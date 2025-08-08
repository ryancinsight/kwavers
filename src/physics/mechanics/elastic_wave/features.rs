/// Symmetry types for anisotropic materials
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialSymmetry {
    Isotropic,
    Cubic,
    Tetragonal,
    Hexagonal,
    Orthotropic,
    Monoclinic,
    Triclinic,
}

/// Material stiffness tensor in Voigt notation with density
#[derive(Debug, Clone)]
pub struct StiffnessTensor {
    pub c: ndarray::Array2<f64>, // 6x6 matrix
    pub density: f64,            // kg/m^3
    pub symmetry: MaterialSymmetry,
}

/// Configuration for elastic mode conversion handling
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeConversionConfig {
    pub enable_p_to_s: bool,
    pub enable_s_to_p: bool,
    pub interface_threshold: f64,
}

impl Default for ModeConversionConfig {
    fn default() -> Self {
        Self { enable_p_to_s: true, enable_s_to_p: true, interface_threshold: 0.1 }
    }
}

/// Configuration for viscoelastic damping
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ViscoelasticConfig {
    pub q_p: f64,  // P-wave quality factor
    pub q_s: f64,  // S-wave quality factor
    pub reference_frequency_hz: f64,
}

impl Default for ViscoelasticConfig {
    fn default() -> Self { Self { q_p: 100.0, q_s: 50.0, reference_frequency_hz: 1e6 } }
}