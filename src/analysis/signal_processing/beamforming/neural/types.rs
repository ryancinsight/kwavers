//! Type definitions for neural beamforming results and metrics.
//!
//! This module provides all result types, feedback structures, and performance
//! metrics used throughout the neural beamforming pipeline.
//!
//! ## Layer Separation
//!
//! This module now uses solver-agnostic interfaces from `pinn_interface`
//! instead of direct solver imports, maintaining clean layer separation.

use ndarray::Array3;

use crate::domain::sensor::beamforming::BeamformingConfig;

// Use solver-agnostic interface instead of direct solver imports
#[cfg(feature = "pinn")]
use super::pinn_interface::{
    PinnBeamformingConfig as InterfacePinnConfig, UncertaintyConfig as InterfaceUncertaintyConfig,
};

/// Result from hybrid neural-traditional beamforming.
#[derive(Debug)]
pub struct HybridBeamformingResult {
    /// Beamformed image
    pub image: Array3<f32>,
    /// Uncertainty map (if available)
    pub uncertainty: Option<Array3<f32>>,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Processing mode used
    pub processing_mode: String,
}

/// Feedback for beamformer adaptation.
#[derive(Debug)]
pub struct BeamformingFeedback {
    /// Performance improvement metric
    pub improvement: f64,
    /// Error gradient for learning
    pub error_gradient: f64,
    /// Signal quality assessment
    pub signal_quality: f64,
}

/// Performance metrics for hybrid beamforming.
#[derive(Debug, Default)]
pub struct HybridBeamformingMetrics {
    pub total_frames_processed: usize,
    pub average_processing_time: f64,
    pub average_confidence: f64,
    pub peak_memory_usage: usize,
}

impl HybridBeamformingMetrics {
    /// Update metrics with new frame processing results.
    pub fn update(&mut self, processing_time: f64, confidence: f64) {
        self.total_frames_processed += 1;
        self.average_processing_time = (self.average_processing_time
            * (self.total_frames_processed - 1) as f64
            + processing_time)
            / self.total_frames_processed as f64;
        self.average_confidence =
            (self.average_confidence * (self.total_frames_processed - 1) as f64 + confidence)
                / self.total_frames_processed as f64;
    }
}

/// Configuration for PINN-enhanced beamforming.
///
/// Uses solver-agnostic interface types to maintain layer separation.
#[derive(Debug, Clone)]
pub struct PINNBeamformingConfig {
    /// Base beamforming configuration
    pub base_config: BeamformingConfig,
    /// PINN training configuration (solver-agnostic)
    #[cfg(feature = "pinn")]
    pub pinn_config: InterfacePinnConfig,
    /// Uncertainty quantification settings (solver-agnostic)
    #[cfg(feature = "pinn")]
    pub uncertainty_config: InterfaceUncertaintyConfig,
    /// Learning rate for PINN optimization
    pub learning_rate: f64,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Physics constraint weight
    pub physics_weight: f64,
    /// Enable real-time adaptation
    pub adaptive_learning: bool,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Volume size (frames, channels, samples)
    pub volume_size: (usize, usize, usize),
    /// Number of RF data channels
    pub rf_data_channels: usize,
    /// Samples per channel
    pub samples_per_channel: usize,
    /// Enable PINN
    pub enable_pinn: bool,
    /// Enable Uncertainty Quantification
    pub enable_uncertainty_quantification: bool,
    /// Channel spacing (pitch) in meters
    pub channel_spacing: f64,
    /// Focal depth in meters
    pub focal_depth: f64,
}

impl Default for PINNBeamformingConfig {
    fn default() -> Self {
        Self {
            base_config: BeamformingConfig::default(),
            #[cfg(feature = "pinn")]
            pinn_config: InterfacePinnConfig::default(),
            #[cfg(feature = "pinn")]
            uncertainty_config: InterfaceUncertaintyConfig {
                bayesian_enabled: true,
                mc_samples: 10,
                confidence_level: 0.95,
            },
            learning_rate: 0.001,
            num_epochs: 1000,
            physics_weight: 1.0,
            adaptive_learning: true,
            convergence_threshold: 1e-6,
            volume_size: (1, 64, 1024),
            rf_data_channels: 64,
            samples_per_channel: 1024,
            enable_pinn: true,
            enable_uncertainty_quantification: true,
            channel_spacing: 0.0003, // 300 microns
            focal_depth: 0.05,       // 50 mm
        }
    }
}

/// Result from PINN neural beamforming processing.
///
/// Uses solver-agnostic interface types to maintain layer separation.
#[derive(Debug)]
pub struct PinnBeamformingResult {
    /// Reconstructed volume
    pub volume: Array3<f32>,
    /// Uncertainty map (variance)
    pub uncertainty: Array3<f32>,
    /// Confidence scores per voxel
    pub confidence: Array3<f32>,
    /// Processing time (ms)
    pub processing_time_ms: f64,
}

/// Alias for backward compatibility.
pub type NeuralBeamformingResult = PinnBeamformingResult;

/// Processing parameters for neural beamforming.
#[derive(Debug, Clone)]
pub struct NeuralBeamformingProcessingParams {
    pub matched_filtering: bool,
    pub dynamic_range_compression: f32,
    pub clutter_suppression: bool,
}

/// Quality metrics for neural beamforming output.
#[derive(Debug, Clone)]
pub struct NeuralBeamformingQualityMetrics {
    pub snr_db: f32,
    pub beam_width_degrees: f32,
    pub grating_lobes_suppressed: bool,
    pub side_lobe_level_db: f32,
}

/// Performance metrics for neural beamforming processor.
#[derive(Debug, Clone)]
pub struct NeuralBeamformingMetrics {
    pub total_processing_time: f64,
    pub pinn_training_time: f64,
    pub uncertainty_computation_time: f64,
    pub memory_usage_mb: f64,
    pub convergence_achieved: bool,
    pub physics_constraint_satisfaction: f64,
}

impl Default for NeuralBeamformingMetrics {
    fn default() -> Self {
        Self {
            total_processing_time: 0.0,
            pinn_training_time: 0.0,
            uncertainty_computation_time: 0.0,
            memory_usage_mb: 0.0,
            convergence_achieved: false,
            physics_constraint_satisfaction: 0.0,
        }
    }
}

/// Metrics for distributed neural beamforming across multiple GPUs.
#[derive(Debug, Default)]
pub struct DistributedNeuralBeamformingMetrics {
    pub total_processing_time: f64,
    pub communication_overhead: f64,
    pub load_imbalance_ratio: f64,
    pub memory_efficiency: f64,
    pub fault_tolerance_events: usize,
    pub active_gpus: usize,
}

/// Result from distributed neural beamforming processing.
#[derive(Debug)]
pub struct DistributedNeuralBeamformingResult {
    /// Reconstructed volume
    pub volume: Array3<f32>,
    /// Uncertainty map
    pub uncertainty: Array3<f32>,
    /// Confidence scores
    pub confidence: Array3<f32>,
    /// Total processing time (ms)
    pub processing_time_ms: f64,
    /// Number of GPUs used
    pub num_gpus_used: usize,
    /// Load balancing efficiency (0-1)
    pub load_balance_efficiency: f64,
}
