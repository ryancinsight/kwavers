pub mod amplitude;
pub mod windowing;

pub use amplitude::AmplitudeCorrector;
pub use kwavers_signal::FrequencyFilter;
pub use windowing::{apply_spatial_window, tukey_window};
