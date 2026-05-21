//! CT-derived transcranial ultrasound tomography with a focused-bowl array.
//!
//! The module implements a deterministic finite-frequency encoded
//! reconstruction used by the Python book chapter.  RITK owns image I/O in the
//! Python wrapper; this module owns the acoustic model, masks, transducer
//! geometry, synthetic encoded data, and inversion.

mod born;
mod conditioning;
mod config;
mod linear_algebra;
mod medium;
mod sensitivity;
mod transducer;
mod volume;
mod volume_born;
mod volume_operator;
mod volume_regularization;

pub use born::{
    reconstruct_brain_slice, TranscranialUstBornInversionMetrics,
    TranscranialUstBornInversionResult,
};
pub use config::{
    TranscranialUstBornInversionConfig, C_BONE_M_S, C_BRAIN_REF_M_S, C_WATER_M_S,
    TRANSCRANIAL_FOCUSED_BOWL_ELEMENT_COUNT,
};
pub use medium::{resample_head_slice, select_head_slice, AcousticSlice, CtResampledSlice};
pub use transducer::{ElementPosition, TranscranialBowlGeometry};
pub use volume::{resample_head_volume, AcousticVolume, CtResampledVolume};
pub use volume_born::{
    reconstruct_brain_volume, TranscranialUstBornInversionVolumeMetrics,
    TranscranialUstBornInversionVolumeResult,
};

#[cfg(test)]
mod tests;
