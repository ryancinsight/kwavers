//! PyO3 bindings for `kwavers_physics::analytical::transducer`.

mod aperture;
mod basic;
mod beam;
mod interpolation;
mod lens;
mod multi_focus;
mod optoacoustic;
mod steering;

pub use aperture::{
    delay_law_focus_3d, focused_bowl_element_positions_3d, focused_bowl_steered_pressure_profile,
    linear_array_positions, steered_aperture_pressure_3d,
};
pub use basic::{
    apodization_weights, apodization_window_response, circular_piston_directivity,
    circular_piston_onaxis, focused_bowl_onaxis, grating_lobe_angles, linear_array_factor,
};
pub use beam::{
    beam_pattern_2d, beam_pattern_2d_magnitude, beam_pattern_magnitude, delay_law_focus_2d,
};
pub use interpolation::{bli_interpolation_error_curves, bli_stencil_weights};
pub use lens::{
    acoustic_lens_delay_profile, corrective_lens_thickness, fresnel_zone_radii,
    isoplanatic_steering_curve,
};
pub use multi_focus::{multi_focus_delay_laws_2d, multi_focus_field_magnitude_2d};
pub use optoacoustic::{
    acoustic_resolution_lateral, f_number_from_na, numerical_aperture_from_geometry,
    soap_focal_gain,
};
pub use steering::{
    delay_law_steer_2d, electronic_steering_efficiency, linear_array_aperiodic_positions,
    near_field_distance, safe_steering_halfangle, steered_beam_pattern_1d, steering_focus_point,
    steering_grating_lobe_ratio_1d,
};
