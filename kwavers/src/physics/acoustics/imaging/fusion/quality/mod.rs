// physics/acoustics/imaging/fusion/quality/mod.rs
//! Quality assessment and uncertainty quantification for multi-modal fusion.
//!
//! This module provides methods for evaluating the quality of imaging data from
//! different modalities, computing confidence maps, and quantifying uncertainty
//! in fusion results.

pub mod metrics;
pub mod modalities;
pub mod uncertainty;

#[cfg(test)]
mod tests;

pub use metrics::{calculate_image_metrics, estimate_modality_noise, ImageMetrics};
pub use modalities::{
    compute_elastography_quality, compute_optical_quality, compute_pa_quality,
};
pub use uncertainty::{
    bayesian_fusion_single_voxel, compute_confidence_map, compute_fusion_uncertainty,
};
