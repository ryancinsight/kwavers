pub mod amplitude;
pub mod windowing;

pub use crate::signal::FrequencyFilter;
pub use amplitude::AmplitudeCorrector;
pub use windowing::{apply_spatial_window, tukey_window};
