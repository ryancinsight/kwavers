//! Core skull attenuation model implementation

use super::types::BoneType;
use crate::core::error::{KwaversError, KwaversResult};

/// Enhanced skull attenuation calculator with frequency-dependent effects
///
/// ✅ IMPLEMENTED: Advanced frequency-dependent attenuation model
///
/// Features:
/// - Power law absorption: α_abs(f) = α₀·f^n
/// - Rayleigh/Stochastic scattering from trabecular structure
/// - Bone type differentiation (cortical vs cancellous)
/// - Temperature compensation (optional)
/// - Cumulative path length integration
///
/// Limitations (deferred to Phase 3):
/// - Kramers-Kronig dispersion (requires complex frequency analysis)
/// - Full anisotropic tensor (orientation-dependent attenuation)
/// - Nonlinear attenuation at high intensities (>1000 W/cm²)
#[derive(Debug, Clone)]
pub struct SkullAttenuation {
    /// Base attenuation coefficient (Np/m/MHz)
    pub(crate) alpha_0: f64,
    /// Power law exponent (typically 1.0-1.3)
    pub(crate) exponent: f64,
    /// Bone type for this region
    pub(crate) bone_type: BoneType,
    /// Scattering coefficient (Np/m/MHz^4 for Rayleigh)
    pub(crate) scattering_coeff: f64,
    /// Include scattering losses
    pub(crate) include_scattering: bool,
    /// Reference temperature (°C) for temperature compensation
    pub(crate) reference_temperature: f64,
}

impl Default for SkullAttenuation {
    fn default() -> Self {
        Self::cortical()
    }
}

impl SkullAttenuation {
    /// Create attenuation model for cortical bone
    ///
    /// Properties from Pinton et al. (2012):
    /// - α₀ = 60 Np/m/MHz
    /// - n = 1.0
    /// - Minimal scattering (dense structure)
    #[must_use]
    pub fn cortical() -> Self {
        Self {
            alpha_0: 60.0,
            exponent: 1.0,
            bone_type: BoneType::Cortical,
            scattering_coeff: 0.01, // Low scattering
            include_scattering: true,
            reference_temperature: 37.0, // Body temperature
        }
    }

    /// Create attenuation model for cancellous (trabecular) bone
    ///
    /// Properties from Pinton et al. (2012):
    /// - α₀ = 35 Np/m/MHz (lower absorption)
    /// - n = 1.2 (higher frequency dependence)
    /// - Higher scattering from porous microstructure
    #[must_use]
    pub fn cancellous() -> Self {
        Self {
            alpha_0: 35.0,
            exponent: 1.2,
            bone_type: BoneType::Cancellous,
            scattering_coeff: 0.1, // High scattering
            include_scattering: true,
            reference_temperature: 37.0,
        }
    }

    /// Create custom attenuation calculator
    ///
    /// # Arguments
    ///
    /// * `alpha_0` - Base attenuation (Np/m/MHz)
    /// * `exponent` - Frequency power law exponent
    /// * `bone_type` - Type of bone tissue
    pub fn new(alpha_0: f64, exponent: f64, bone_type: BoneType) -> KwaversResult<Self> {
        if alpha_0 < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Attenuation coefficient must be non-negative".to_string(),
            ));
        }

        if !(0.5..=2.0).contains(&exponent) {
            return Err(KwaversError::InvalidInput(format!(
                "Exponent {exponent} outside typical range [0.5, 2.0]"
            )));
        }

        // Set scattering based on bone type
        let (scattering_coeff, include_scattering) = match bone_type {
            BoneType::Cortical => (0.01, true),
            BoneType::Cancellous => (0.1, true),
            BoneType::Mixed { cortical_fraction } => {
                let scatter = 0.01 * cortical_fraction + 0.1 * (1.0 - cortical_fraction);
                (scatter, true)
            }
        };

        Ok(Self {
            alpha_0,
            exponent,
            bone_type,
            scattering_coeff,
            include_scattering,
            reference_temperature: 37.0,
        })
    }

    /// Get bone type
    #[must_use]
    pub fn bone_type(&self) -> BoneType {
        self.bone_type
    }

    /// Enable or disable scattering
    pub fn set_scattering(&mut self, enabled: bool) {
        self.include_scattering = enabled;
    }
}
