//! Imaging Analysis Module
//!
//! **Architectural Note (Phase 3 Consolidation):** This module serves as a re-export hub
//! for imaging types. All type definitions are consolidated in the domain layer to ensure
//! a single source of truth.
//!
//! **Why Re-export Pattern?**
//! - Analysis layer implements post-processing algorithms, not type definitions
//! - Domain layer is the authoritative source for all imaging parameters and models
//! - Re-exporting makes types conveniently available without duplication
//! - Prevents divergence between domain types and their analysis counterparts
//!
//! **Usage Pattern:**
//! ```rust,ignore
//! // Import from analysis for convenience
//! use kwavers::analysis::imaging::{UltrasoundConfig, CEUSImagingParameters};
//!
//! // But the actual definition is in domain
//! // See: src/domain/imaging/ultrasound/ceus.rs
//! ```
//!
//! **Dependencies:**
//! - ✓ analysis::imaging depends ONLY on domain::imaging (re-export pattern)
//! - ✓ No algorithms or processing logic in analysis::imaging
//! - ✓ All physics and signal processing in appropriate layers

// Re-export all imaging types from domain as the single source of truth
pub use crate::domain::imaging::ultrasound::{
    ceus::{
        CEUSImagingParameters, Microbubble, MicrobubblePopulation, PerfusionMap,
        PerfusionStatistics, SizeDistribution,
    },
    elastography::{
        ElasticityMap, InversionMethod, NonlinearInversionMethod, NonlinearParameterMap,
    },
    hifu::{
        FeedbackChannel, HIFUTransducer, HIFUTreatmentPlan, MonitoringConfig, SafetyConstraints,
        TreatmentPhase, TreatmentProtocol, TreatmentTarget,
    },
    UltrasoundConfig, UltrasoundMode,
};

pub use crate::domain::imaging::photoacoustic::{
    InitialPressure, PhotoacousticOpticalProperties, PhotoacousticParameters, PhotoacousticResult,
};

pub use crate::domain::imaging::ceus_orchestrator::{CEUSOrchestrator, CEUSOrchestrators};
