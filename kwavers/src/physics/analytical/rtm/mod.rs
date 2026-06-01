//! Reverse-time migration (RTM) and adaptive beamforming physics for
//! book chapter ch25.
//!
//! Covers: focused Gaussian beam 2-D field, Green's function backpropagation,
//! RTM imaging condition, multi-frequency fusion, temporal modulation
//! frequency schedule, and standing-wave suppression gain.

mod backprop;
mod beam;
mod condition;
mod temporal;
#[cfg(test)]
mod tests;

pub use backprop::{backprop_green_function_2d, backprop_green_function_3d};
pub use beam::focused_gaussian_beam_2d;
pub use condition::{
    rtm_aperture_weighted_fusion, rtm_imaging_condition, rtm_multi_frequency_fusion,
    rtm_source_normalized_condition,
};
pub use temporal::{
    standing_wave_field_1d, standing_wave_intensity_statistics, standing_wave_modulation_period_hz,
    standing_wave_spatial_frequency_cycles_m, standing_wave_suppression_gain,
    temporal_modulation_frequencies,
};
