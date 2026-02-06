//! Source Localization Analysis Module
//!
//! Implements advanced source localization algorithms for passive acoustic mapping and
//! source detection from transducer array signals.
//!
//! ## Algorithms
//!
//! ### 1. MUSIC (Multiple Signal Classification)
//! - Super-resolution direction-of-arrival (DoA) estimation
//! - Subspace-based method
//! - Effective for multiple sources
//! - References: Schmidt (1986), Stoica & Nehorai (1989)
//!
//! ### 2. TDOA (Time-Difference-of-Arrival) Triangulation
//! - Localization from time delays between sensors
//! - Iterative Newton-Raphson refinement
//! - 2D and 3D support
//! - References: Knapp & Carter (1976)
//!
//! ### 3. Bayesian Filtering
//! - Extended Kalman Filter (EKF) for nonlinear state estimation
//! - Unscented Kalman Filter (UKF) for improved accuracy
//! - Particle filters for multi-modal distributions
//! - References: Kalman (1960), Julier & Uhlmann (1997)
//!
//! ### 4. Wavefront Analysis
//! - Spherical vs. plane wave detection
//! - Source distance estimation from wavefront curvature
//! - Plane wave detection for far-field sources
//!
//! ## Integration with Domain Layer
//!
//! All implementations satisfy the `LocalizationProcessor` trait from domain layer,
//! ensuring clean architecture and easy swapping of algorithms.

pub mod bayesian;
pub mod beamforming_search;
pub mod config;
pub mod model_order;
pub mod multilateration;
pub mod music;
pub mod tdoa;
pub mod trilateration;
pub mod wavefront;

// Re-export core types
pub use bayesian::{BayesianFilter, KalmanFilterConfig};
pub use beamforming_search::{
    BeamformSearch, BeamformingLocalizationInput, LocalizationBeamformSearchConfig,
    LocalizationBeamformingMethod, MvdrCovarianceDomain, SearchGrid,
};
pub use config::LocalizationConfig;
pub use model_order::{
    ModelOrderConfig, ModelOrderCriterion, ModelOrderEstimator, ModelOrderResult,
};
pub use multilateration::{Multilateration, MultilaterationConfig};
pub use music::{MUSICConfig, MUSICProcessor};
pub use tdoa::{TDOAConfig, TDOAProcessor};
pub use trilateration::{LocalizationResult, Trilateration};
pub use wavefront::WavefrontAnalyzer;

use crate::core::error::KwaversResult;
use crate::domain::signal_processing::localization::LocalizationProcessor;

/// Create a MUSIC-based localization processor
pub fn create_music_processor(
    config: &MUSICConfig,
) -> KwaversResult<Box<dyn LocalizationProcessor>> {
    Ok(Box::new(MUSICProcessor::new(config)?))
}

/// Create a TDOA-based localization processor
pub fn create_tdoa_processor(config: &TDOAConfig) -> KwaversResult<Box<dyn LocalizationProcessor>> {
    Ok(Box::new(TDOAProcessor::new(config)?))
}

/// Create a Bayesian filtering localization processor
pub fn create_bayesian_processor(
    config: &KalmanFilterConfig,
) -> KwaversResult<Box<dyn LocalizationProcessor>> {
    Ok(Box::new(BayesianFilter::new(config)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_music_processor_creation() {
        let config = MUSICConfig::default();
        let result = create_music_processor(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tdoa_processor_creation() {
        let config = TDOAConfig::default();
        let result = create_tdoa_processor(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bayesian_processor_creation() {
        let config = KalmanFilterConfig::default();
        let result = create_bayesian_processor(&config);
        assert!(result.is_ok());
    }
}
