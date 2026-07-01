//! Phased-array acoustics vertical slice — Phase 3g cut-over from flat `src/acoustic.rs`.
//!
//! This module supplies the in-crate analytical fallback for the driver acoustic seam. Under the
//! `kwavers` Cargo feature, the experiment layer delegates focused propagation to
//! `kwavers_transducer::propagate_focused_linear_array`. At Phase 3g this module carries the full
//! 21-fn fallback surface:
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
//!   [`safety::isppa_w_per_m2`] (FDA Track-3 spatial-peak pulse-average intensity —
//!   **NEW** — distinct from continuous RMS intensity) +
//!   [`safety::round_trip_attenuation_db`] (`2 · α · f · z` — pulse-echo two-way loss —
//!   **NEW** — complement to the one-way form for TGC/pulse-echo budgets).
//! * [`nonlinear`] — [`nonlinear::nonlinear_shock_parameter`] (Earnshaw normalised shock
//!   parameter, harmonic-distortion regime indicator).
//!
//! All 21 free fns (18 prior + 3 NEW) are pure-math (`f64`-in/`f64`-out, no state, no
//! cross-slice dep) and feed the slice facade's named `pub use` re-export chain.
//!
//! # Phase 1a migration roadmap
//!
//! The fn signatures today are `f64`-passing. Phase 2 will replace `w, h, z, L, C, depth_cm,
//! alpha_db, freq_mhz, …` with the typed [`crate::units`] newtypes as the slice sits alongside
//! `physics::thermal`, `physics::emi`, `physics::pdn`, and `physics::si` in the vertical-slice
//! units migration: `Meter` for `w`, `MHz` for `freq_mhz`, `Henry` for `L`, `Farad` for `C`,
//! `Z₀` as `Rayl` (new newtype at Phase 2), etc. **No signature change at Phase 3g** — keeping
//! the API as `f64` preserves every existing call-site and test fixture until the vertical-slice
//! units land.
//!
//! [`crate::units`]: crate::units
//!
//! # SSOT for the slice
//!
//! * `pub mod wavelength` — fundamental numbers `λ = c/f` and the BVD equivalent-circuit
//!   resonances (both branches).
//! * `pub mod grating` — spatial-sampling half of phased-array acoustics (steering limits +
//!   beam pattern).
//! * `pub mod focus` — timing-half of focus synthesis (delays + quantisation + error).
//! * `pub mod element` — per-element Fresnel range + directivity + f-number + focal gain.
//! * `pub mod safety` — FDA/regulatory safety + intensity + tissue budgets (MI + I +
//!   I_sppa + one-way + round-trip attenuation).
//! * `pub mod nonlinear` — propagation nonlinearity indicator.
//!
//! # Phase 3g cut-over status
//!
//! The flat `src/acoustic.rs` (Phase 0 surface, 437 LOC + 13 inline tests + 18 `pub fn`s) has
//! been retired: all 18 prior fns have migrated into the per-concern submodules above, plus 3
//! NEW APIs added to fill out the FDA safety/regulatory surface —
//! `bvd_anti_resonance_hz` (the textbook BVD anti-resonance per Kino *Acoustic Waves* §3.4 /
//! IEEE Std 176; couples the motional series branch `L_s·C_s` with the static dielectric
//! `C_0`; completes the matching-network design picture),
//! `isppa_w_per_m2` (FDA Track-3 spatial-peak pulse-average intensity for pulsed operation,
//! SSOT-distinct from the continuous-RMS `acoustic_intensity_w_per_m2`), and
//! `round_trip_attenuation_db` (pulse-echo two-way loss for TGC curves and time-gain
//! compensation, SSOT-distinct from the one-way `tissue_attenuation_db`). The crate-root
//! re-export at `src/lib.rs::pub use physics::acoustic::{...}` covers the 21-fn surface for
//! downstream callers; `crate::acoustic::*` is now retired.

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
