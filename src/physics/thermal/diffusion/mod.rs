//! Thermal diffusion solver parameters and physics models
//!
//! This module implements dedicated thermal diffusion physics models including:
//! - Standard heat diffusion equation
//! - Pennes bioheat equation
//! - Thermal dose calculations (CEM43)
//! - Hyperbolic heat transfer (Cattaneo-Vernotte)
//!
//! # Literature References
//!
//! 1. **Pennes, H. H. (1948)**. "Analysis of tissue and arterial blood temperatures
//!    in the resting human forearm." *Journal of Applied Physiology*, 1(2), 93-122.
//!    - Original formulation of bioheat equation
//!
//! 2. **Sapareto, S. A., & Dewey, W. C. (1984)**. "Thermal dose determination in
//!    cancer therapy." *International Journal of Radiation Oncology Biology Physics*,
//!    10(6), 787-800. DOI: 10.1016/0360-3016(84)90379-1
//!    - CEM43 thermal dose formulation
//!
//! 3. **Cattaneo, C. (1958)**. "A form of heat conduction equation which eliminates
//!    the paradox of instantaneous propagation." *Comptes Rendus*, 247, 431-433.
//!    - Hyperbolic heat transfer theory
//!
//! 4. **Liu, J., & Xu, L. X. (1999)**. "Estimation of blood perfusion using phase
//!    shift in temperature response to sinusoidal heating at the skin surface."
//!    *IEEE Transactions on Biomedical Engineering*, 46(9), 1037-1043.
//!    - Modern perfusion estimation methods

pub mod bioheat;
pub mod dose;
pub mod hyperbolic;

pub use bioheat::{BioheatParameters, PennesBioheat};
pub use dose::{thresholds, ThermalDoseCalculator};
pub use hyperbolic::{CattaneoVernotte, HyperbolicParameters};

/// Configuration for thermal diffusion solver
#[derive(Debug, Clone)]
pub struct ThermalDiffusionConfig {
    /// Enable Pennes bioheat equation terms
    pub enable_bioheat: bool,
    /// Blood perfusion rate [1/s]
    pub perfusion_rate: f64,
    /// Blood density [kg/m³]
    pub blood_density: f64,
    /// Blood specific heat [J/(kg·K)]
    pub blood_specific_heat: f64,
    /// Arterial blood temperature [K]
    pub arterial_temperature: f64,
    /// Enable hyperbolic heat transfer (Cattaneo-Vernotte)
    pub enable_hyperbolic: bool,
    /// Thermal relaxation time [s]
    pub relaxation_time: f64,
    /// Enable thermal dose tracking
    pub track_thermal_dose: bool,
    /// Reference temperature for dose calculation [°C]
    pub dose_reference_temperature: f64,
    /// Spatial discretization order (2 or 4)
    pub spatial_order: usize,
}

impl Default for ThermalDiffusionConfig {
    fn default() -> Self {
        Self {
            enable_bioheat: true,
            perfusion_rate: 0.5e-3,
            blood_density: 1050.0,
            blood_specific_heat: 3840.0,
            arterial_temperature: 310.15,
            enable_hyperbolic: false,
            relaxation_time: 20.0,
            track_thermal_dose: true,
            dose_reference_temperature: 43.0,
            spatial_order: 4,
        }
    }
}
