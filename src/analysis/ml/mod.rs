//! Machine Learning Training Infrastructure
//!
//! This module provides a comprehensive training pipeline for neural network models
//! with support for:
//! - Multiple optimization algorithms (SGD, Momentum, Adam, RMSprop)
//! - Physics-informed loss balancing
//! - Model checkpointing and serialization
//! - Training metrics and convergence monitoring
//! - Data augmentation and batch processing
//!
//! ## Architecture
//!
//! The training pipeline integrates with the Burn deep learning framework:
//! ```text
//! Domain Layer (Domain Abstractions)
//!     ↓
//! Analysis Layer (Training Pipeline) ← This module
//!     ↓
//! Solver Layer (Burn PINN Models)
//!     ↓
//! Core Layer (Types & Errors)
//! ```
//!
//! ## Key Components
//!
//! - `TrainingConfig` - Configuration management with validation
//! - `TrainingDataset` - Dataset loading and batching
//! - `TrainingMetrics` - Per-epoch monitoring
//! - `PhysicsLoss` - Physics constraint enforcement
//! - `Optimizer` - Gradient-based optimization algorithms
//! - `TrainingHistory` - Training logs and convergence analysis

pub mod beamforming_trainer;
pub mod physics_informed_loss;
pub mod training;

pub use beamforming_trainer::BeamformingTrainer;
pub use physics_informed_loss::{
    GradientMethod, LossComponents, PhysicsInformedLoss, PhysicsLossConfig, WeightSchedule,
};
pub use training::{
    Optimizer, PhysicsLoss, TrainingConfig, TrainingDataset, TrainingHistory, TrainingMetrics,
};
