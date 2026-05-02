#![deny(missing_docs)]
//! Beamforming-based localization (grid search) built on the shared beamforming SSOT.
//!
//! - [`LocalizationBeamformSearchConfig`]: localization-owned policy wrapper around `BeamformingCoreConfig`
//! - [`BeamformSearch`]: grid-search evaluator using shared SSOT beamforming primitives
//! - [`localize_beamforming`]: convenience entry point for time-series based localization

mod search;
#[cfg(test)]
mod tests;
mod types;

pub use search::{localize_beamforming, BeamformSearch};
pub use types::{
    BeamformingLocalizationInput, LocalizationBeamformSearchConfig, LocalizationBeamformingMethod,
    MvdrCovarianceDomain, SearchGrid,
};
