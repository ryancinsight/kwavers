//! Detection modules for recording

pub mod cavitation;
pub mod sonoluminescence;

pub use cavitation::{CavitationDetector, CavitationRegion, CavitationStatistics};
pub use sonoluminescence::{SonoluminescenceDetector, SonoluminescenceEvent};