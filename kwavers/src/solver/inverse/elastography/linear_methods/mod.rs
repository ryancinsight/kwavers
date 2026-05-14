//! Linear Elastography Inversion Methods
//!
//! Linear inversion algorithms for reconstructing tissue elasticity from
//! shear wave propagation data. Includes time-of-flight, phase gradient,
//! and direct inversion methods.
//!
//! ## Methods Overview
//!
//! ### Time-of-Flight (TOF)
//! - Simplest method: estimates speed from wave arrival times
//! - Fast computation, suitable for real-time applications
//! - Accuracy limited by temporal sampling and noise
//!
//! ### Phase Gradient
//! - Frequency-domain method using phase information
//! - More accurate than TOF for complex geometries
//! - Requires sufficient signal bandwidth
//!
//! ### Direct Inversion
//! - Solves inverse problem directly from wave equation
//! - Most accurate but computationally expensive
//! - Requires high-quality displacement measurements
//!
//! ### Volumetric TOF
//! - 3D extension of TOF with multi-directional analysis
//! - Robust to heterogeneous tissue structures
//! - Uses multiple push locations for improved accuracy
//!
//! ### Directional Phase Gradient
//! - 3D phase gradient with directional wave analysis
//! - Accounts for anisotropic wave propagation
//! - Improved accuracy in heterogeneous 3D media
//!
//! ## Module layout
//!
//! - `time_of_flight`: scalar TOF inversion (Bercoff et al. 2004).
//! - `phase_gradient`: 1D phase-gradient inversion (McLaughlin & Renzi 2006).
//! - `direct`: Gauss-Seidel direct inversion of `∇²u + k²u = 0`.
//! - `volumetric`: 3D multi-source median TOF inversion.
//! - `directional`: 3D directional phase-gradient inversion.
//!
//! ## References
//!
//! - Bercoff, J., et al. (2004). "Supersonic shear imaging: a new technique
//!   for soft tissue elasticity mapping." *IEEE TUFFC*, 51(4), 396-409.
//! - McLaughlin, J., & Renzi, D. (2006). "Shear wave speed recovery in transient
//!   elastography and supersonic imaging using propagating fronts." *Inverse Problems*, 22(2), 681.
//! - Deffieux, T., et al. (2011). "On the effects of reflected waves in transient
//!   shear wave elastography." *IEEE TUFFC*, 58(10), 2032-2035.

mod direct;
mod directional;
mod phase_gradient;
mod time_of_flight;
mod volumetric;

#[cfg(test)]
mod tests;

use direct::direct_inversion;
use directional::directional_phase_gradient_inversion;
use phase_gradient::phase_gradient_inversion;
use time_of_flight::time_of_flight_inversion;
use volumetric::volumetric_time_of_flight_inversion;

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::imaging::ultrasound::elastography::ElasticityMap;
use crate::physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;

use super::config::ShearWaveInversionConfig;

/// Shear wave inversion processor for linear methods
#[derive(Debug)]
pub struct ShearWaveInversion {
    config: ShearWaveInversionConfig,
}

impl ShearWaveInversion {
    /// Create new shear wave inversion processor
    #[must_use]
    pub fn new(config: ShearWaveInversionConfig) -> Self {
        Self { config }
    }

    /// Get current inversion method
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn method(&self) -> crate::domain::imaging::ultrasound::elastography::InversionMethod {
        self.config.method
    }

    /// Get configuration reference
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn config(&self) -> &ShearWaveInversionConfig {
        &self.config
    }

    /// Reconstruct elasticity from displacement field
    ///
    /// # Errors
    ///
    /// Returns error if inversion fails due to insufficient data or numerical issues
    pub fn reconstruct(
        &self,
        displacement: &DisplacementField,
        grid: &Grid,
    ) -> KwaversResult<ElasticityMap> {
        use crate::domain::imaging::ultrasound::elastography::InversionMethod;

        match self.config.method {
            InversionMethod::TimeOfFlight => time_of_flight_inversion(
                displacement,
                grid,
                self.config.density,
                self.config.frequency,
            ),
            InversionMethod::PhaseGradient => phase_gradient_inversion(
                displacement,
                grid,
                self.config.density,
                self.config.frequency,
            ),
            InversionMethod::DirectInversion => direct_inversion(
                displacement,
                grid,
                self.config.density,
                self.config.frequency,
            ),
            InversionMethod::VolumetricTimeOfFlight => volumetric_time_of_flight_inversion(
                displacement,
                grid,
                self.config.density,
                self.config.frequency,
            ),
            InversionMethod::DirectionalPhaseGradient => directional_phase_gradient_inversion(
                displacement,
                grid,
                self.config.density,
                self.config.frequency,
            ),
        }
    }
}
