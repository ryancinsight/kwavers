//! AI/ML optimization algorithms for intelligent simulation control
//!
//! This module provides:
//! - Neural network-based parameter optimization
//! - Reinforcement learning for adaptive tuning
//! - Pattern recognition for cavitation and acoustic events
//! - AI-assisted convergence acceleration

pub mod acoustic_analyzer;
pub mod cavitation_detector;
pub mod convergence_predictor;
pub mod neural_network;
pub mod parameter_optimizer;
pub mod pattern_recognition;

// Re-export main types
pub use acoustic_analyzer::{AcousticEvent, AcousticEventAnalyzer, AcousticEventType};
pub use cavitation_detector::{CavitationDetector, CavitationEvent, CavitationEventType};
pub use convergence_predictor::ConvergencePredictor;
pub use neural_network::NeuralNetwork;
pub use parameter_optimizer::{OptimizationExperience, ParameterOptimizer};
pub use pattern_recognition::{PatternRecognizer, PatternSummary, SimulationPatterns};
