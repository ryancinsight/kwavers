//! # Passive Acoustic Mapping (PAM) Module
//!
//! This module provides passive acoustic mapping algorithms for real-time monitoring
//! and localization of acoustic sources, particularly cavitation events during
//! therapeutic ultrasound procedures.
//!
//! ## Overview
//!
//! Passive Acoustic Mapping (PAM) is a technique for detecting and localizing
//! acoustic emissions from cavitation bubbles, shock waves, or other transient
//! sources. Unlike active ultrasound imaging, PAM listens passively to acoustic
//! emissions.
//!
//! ## References
//!
//! - Gyöngy & Coussios (2010). "Passive spatial mapping of inertial cavitation during HIFU exposure."
//!   *IEEE Trans. Biomed. Eng.*, 57(1), 48-56.
//! - Arvanitis et al. (2012). "Passive acoustic mapping with the angular spectrum method."
//!   *IEEE Trans. Med. Imaging*, 31(11), 2086-2091.
//! - Haworth et al. (2012). "Passive imaging with pulsed ultrasound insonations."
//!   *J. Acoust. Soc. Am.*, 132(1), 544-553.

pub mod config;
pub mod delay_and_sum;
mod eigenspace;
pub mod mapper;
pub mod processor;
#[cfg(test)]
mod tests;

pub use config::{PAMConfig, PamBeamformingConfig};
pub use delay_and_sum::{DelayAndSumConfig, DelayAndSumPAM, PamCavitationEvent, PamImagingMode};
pub use eigenspace::{eigenspace_covariance_eigenvalues, EigenspaceSpectrumError};
pub use mapper::PassiveAcousticMapper;
pub use processor::PAMProcessor;

pub use kwavers_transducer::passive_acoustic_mapping::geometry::{
    PamArrayElement, PamArrayGeometry, PamDirectivityPattern,
};

pub use kwavers_transducer::beamforming::BeamformingCoreConfig;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PamBeamformingMethod {
    DelayAndSum,
    CaponDiagonalLoading { diagonal_loading: f64 },
    Music { num_sources: usize },
    EigenspaceMinVariance { signal_subspace_dimension: usize },
    TimeExposureAcoustics,
}

pub use kwavers_math::signal::ApodizationType;
