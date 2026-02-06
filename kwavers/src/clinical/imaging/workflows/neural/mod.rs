//! Neural Beamforming Clinical Decision Support
//!
//! This module provides clinical decision support functionality for neural network-enhanced
//! ultrasound beamforming, including lesion detection, tissue classification, diagnostic
//! recommendations, and clinical workflow management.
//!
//! # Module Organization
//!
//! - [`clinical`]: Clinical decision support (lesion detection, tissue classification)
//! - [`diagnosis`]: Automated diagnosis algorithm
//! - [`workflow`]: Real-time workflow manager with performance monitoring
//!
//! # Clinical Applications
//!
//! - Real-time tissue characterization
//! - Automated lesion detection with confidence scoring
//! - Clinical decision support with recommendations
//! - Diagnostic workflow orchestration
//!
//! # Clinical Safety Notice
//!
//! All neural network analysis results are for **decision support only** and require
//! clinical interpretation by qualified medical professionals. This system does not
//! replace clinical judgment and must be used in accordance with institutional protocols
//! and regulatory requirements.

pub mod ai_beamforming_processor;
pub mod clinical;
pub mod diagnosis;
pub mod feature_extraction;
pub mod workflow;

// Public re-exports for convenient access
pub use ai_beamforming_processor::{AIEnhancedBeamformingProcessor, PinnInferenceEngine};
pub use clinical::ClinicalDecisionSupport;
pub use diagnosis::DiagnosisAlgorithm;
pub use feature_extraction::FeatureExtractor;
pub use workflow::RealTimeWorkflow;

// Type re-exports
pub mod types;
pub use types::{
    AIBeamformingConfig, AIBeamformingResult, ClinicalAnalysis, ClinicalThresholds, FeatureMap,
    LesionDetection, PerformanceMetrics, TissueClassification,
};
