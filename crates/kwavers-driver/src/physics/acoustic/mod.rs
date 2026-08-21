//! Phased-array acoustics.
//!
//! This module supplies the in-crate analytical fallback for the driver acoustic seam. Under
//! the `kwavers` Cargo feature the experiment layer delegates focused propagation to
//! `kwavers_transducer::propagate_focused_linear_array` instead; this is the surface that
//! stands in when the feature is off:
//!
//! * [`wavelength`] — [`wavelength::wavelength_m`] + [`wavelength::bvd_series_resonance_hz`] +
//!   [`wavelength::bvd_anti_resonance_hz`] (the textbook BVD anti-resonance per Kino
//!   *Acoustic Waves* §3.4 / IEEE Std 176; couples the motional branch with the static
//!   dielectric `C_0`; couples with the series-branch fn for matching-network design).
//! * [`grating`] — [`grating::max_grating_free_steer_deg`] + [`grating::grating_lobe_angle_deg`] +
//!   [`grating::array_factor`] (element-pitch steering bounds + ULA beam pattern).
//! * [`focus`] — [`focus::focused_delay_profile_s`] + [`focus::quantize_delays_s`] +
//!   [`focus::max_delay_quantization_error_s`] (relative transmit delays + nearest-step
//!   quantisation + worst-case quantisation error).
//! * [`element`] — [`element::near_field_distance_m`] + [`element::element_factor`] +
//!   [`element::f_number`] + [`element::pitch_from_aperture_m`] + [`element::focal_pressure_gain`]
//!   (per-element Fresnel-range, directivity, f-number, span→pitch, and coherent focal gain).
//! * [`safety`] — [`safety::mechanical_index`] + [`safety::tissue_attenuation_db`] +
//!   [`safety::pressure_derating`] + [`safety::acoustic_intensity_w_per_m2`] +
//!   [`safety::isppa_w_per_m2`] (FDA Track-3 spatial-peak pulse-average intensity,
//!   distinct from the continuous-RMS [`safety::acoustic_intensity_w_per_m2`]) +
//!   [`safety::round_trip_attenuation_db`] (`2 · α · f · z` — pulse-echo two-way loss,
//!   the complement to the one-way [`safety::tissue_attenuation_db`] for TGC budgets).
//! * [`nonlinear`] — [`nonlinear::nonlinear_shock_parameter`] (Earnshaw normalised shock
//!   parameter, harmonic-distortion regime indicator).
//!
//! Every function here is pure math — `f64` in, `f64` out, no state and no cross-slice
//! dependency — and is re-exported at the crate root by
//! `src/lib.rs::pub use physics::acoustic::{…}`.
//!
//! # Units
//!
//! Signatures pass plain `f64` in their documented units (metres, MHz, henries, farads,
//! dB/cm/MHz). Typing them with the [`crate::units`] newtypes — which needs a `Rayl` for
//! acoustic impedance — is tracked in `docs/MIGRATION.md`.
//!
//! [`crate::units`]: crate::units

pub mod element;
pub mod focus;
pub mod grating;
pub mod nonlinear;
pub mod safety;
pub mod wavelength;

pub use element::{
    element_factor, f_number, focal_pressure_gain, near_field_distance_m, pitch_from_aperture_m,
};
pub use focus::{focused_delay_profile_s, max_delay_quantization_error_s, quantize_delays_s};
pub use grating::{array_factor, grating_lobe_angle_deg, max_grating_free_steer_deg};
pub use nonlinear::nonlinear_shock_parameter;
pub use safety::{
    acoustic_intensity_w_per_m2, isppa_w_per_m2, mechanical_index, pressure_derating,
    round_trip_attenuation_db, tissue_attenuation_db,
};
pub use wavelength::{bvd_anti_resonance_hz, bvd_series_resonance_hz, wavelength_m};

#[cfg(test)]
mod tests;
