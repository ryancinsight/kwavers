//! Phase Modulation Module
//!
//! Implements various phase modulation techniques for ultrasound applications
//! including phase randomization, phase shifting, and aberration correction.
//!
//! Literature references:
//! - Tang & Clement (2010): "Standing wave suppression for transcranial ultrasound"
//! - Pernot et al. (2007): "Prediction of the skull transmission for aberration correction"
//! - Aubry et al. (2003): "Optimal focusing by spatio-temporal inverse filter"
//! - Liu et al. (2018): "Random phase modulation for reduction of peak pressures"

pub mod aberration_correction;
pub mod phase_encoding;
pub mod phase_patterns;
pub mod phase_randomization;
pub mod phase_shifting;

pub use phase_randomization::{
    PhaseDistribution, PhaseRandomizer, RandomizationScheme, SpatialRandomization,
    TimeRandomization,
};

pub use phase_shifting::{
    BeamSteering, DynamicFocusing, PhaseArray, PhaseShifter, ShiftingStrategy,
};

pub use phase_encoding::{
    BarkerCode, EncodingScheme, GolayEncoding, HadamardEncoding, PhaseEncoder, PulseCompression,
};

pub use aberration_correction::{
    AberrationCorrector, AdaptiveFocusing, CorrectionMethod, PhaseConjugation, TimeReversal,
};

pub use phase_patterns::{AiryBeam, BesselBeam, PhasePattern, SpiralPhase, VortexBeam};
