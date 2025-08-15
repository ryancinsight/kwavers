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

pub mod phase_randomization;
pub mod phase_shifting;
pub mod phase_encoding;
pub mod aberration_correction;
pub mod phase_patterns;

pub use phase_randomization::{
    PhaseRandomizer, RandomizationScheme, PhaseDistribution,
    TemporalRandomization, SpatialRandomization
};

pub use phase_shifting::{
    PhaseShifter, ShiftingStrategy, BeamSteering,
    DynamicFocusing, PhaseArray
};

pub use phase_encoding::{
    PhaseEncoder, EncodingScheme, HadamardEncoding,
    GolayEncoding, BarkerCode, PulseCompression
};

pub use aberration_correction::{
    AberrationCorrector, CorrectionMethod, TimeReversal,
    AdaptiveFocusing, PhaseConjugation
};

pub use phase_patterns::{
    PhasePattern, SpiralPhase, VortexBeam,
    BesselBeam, AiryBeam
};