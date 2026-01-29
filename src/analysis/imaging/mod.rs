//! Imaging Analysis Module
//!
//! This module provides imaging analysis algorithms and types for ultrasound and photoacoustic imaging.
//! It operates on data from the domain layer (sources, medium, sensors) to produce images and measurements.

pub mod ultrasound;

// Re-export main types from ultrasound
pub use ultrasound::{
    ceus::{CEUSImagingParameters, Microbubble, MicrobubblePopulation, PerfusionMap},
    elastography::{
        ElasticityMap, InversionMethod, NonlinearInversionMethod, NonlinearParameterMap,
    },
    hifu::{HIFUTransducer, HIFUTreatmentPlan, TreatmentProtocol, TreatmentTarget},
    UltrasoundConfig, UltrasoundMode,
};
