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

use aequitas::systems::si::quantities::{
    MassDensity, ReciprocalTime, SpecificHeatCapacity, ThermodynamicTemperature, Time,
};
use kwavers_core::constants::medical::{BLOOD_SPECIFIC_HEAT, TISSUE_PERFUSION_RATE};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_K;
use kwavers_core::constants::tissue_acoustics::DENSITY_BLOOD;

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
    /// Blood perfusion rate [1/s].
    pub perfusion_rate: ReciprocalTime<f64>,
    /// Blood density [kg/m³].
    pub blood_density: MassDensity<f64>,
    /// Blood specific heat [J/(kg·K)].
    pub blood_specific_heat: SpecificHeatCapacity<f64>,
    /// Arterial blood temperature [K].
    pub arterial_temperature: ThermodynamicTemperature<f64>,
    /// Enable hyperbolic heat transfer (Cattaneo-Vernotte)
    pub enable_hyperbolic: bool,
    /// Thermal relaxation time [s].
    pub relaxation_time: Time<f64>,
    /// Enable thermal dose tracking
    pub track_thermal_dose: bool,
    /// Spatial discretization order (2 or 4)
    pub spatial_order: usize,
}

impl Default for ThermalDiffusionConfig {
    fn default() -> Self {
        Self {
            enable_bioheat: true,
            // TISSUE_PERFUSION_RATE = 5×10⁻⁴ 1/s — generic soft tissue default
            // (Pennes 1948; Duck 1990). See `kwavers_core::constants::medical`.
            perfusion_rate: ReciprocalTime::from_base(TISSUE_PERFUSION_RATE),
            blood_density: MassDensity::from_base(DENSITY_BLOOD),
            blood_specific_heat: SpecificHeatCapacity::from_base(BLOOD_SPECIFIC_HEAT),
            arterial_temperature: ThermodynamicTemperature::from_base(BODY_TEMPERATURE_K),
            enable_hyperbolic: false,
            relaxation_time: Time::from_base(20.0),
            track_thermal_dose: true,
            spatial_order: 4,
        }
    }
}
