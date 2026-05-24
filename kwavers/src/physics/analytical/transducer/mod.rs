//! Transducer array and beamforming physics for book chapters ch04, ch11.
//!
//! Covers: circular piston directivity, linear array factor, grating lobes,
//! apodization windows, delay laws, 2-D beam patterns, on-axis pressure
//! profiles, and bandlimited interpolation stencils.

pub mod array_factor;
pub mod beam;
pub mod interpolation;

pub use array_factor::{
    apodization_weights, circular_piston_directivity, grating_lobe_angles, linear_array_factor,
};
pub use beam::{
    beam_pattern_2d, circular_piston_onaxis, delay_law_focus_2d, focused_bowl_onaxis,
};
pub use interpolation::bli_stencil_weights;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use crate::core::constants::numerical::MHZ_TO_HZ;
    use std::f64::consts::PI;

    #[test]
    fn piston_directivity_on_axis() {
        let d = circular_piston_directivity(&[0.0], 5.0);
        assert!((d[0] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn array_factor_at_steering_angle_is_one() {
        let steer = 0.1_f64;
        let k = 2.0 * PI * 2.0 * MHZ_TO_HZ / SOUND_SPEED_WATER_SIM;
        let af = linear_array_factor(&[steer], k, 0.3e-3, 64, steer);
        assert!((af[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apodization_uniform_sum() {
        let w = apodization_weights(64, "uniform");
        let s: f64 = w.iter().sum();
        assert!((s - 64.0).abs() < 1e-10);
    }

    #[test]
    fn bli_stencil_dc_preservation() {
        let ws = bli_stencil_weights(&[0.0, 0.25, 0.5, 0.75], 8);
        for w in &ws {
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "sum={}", s);
        }
    }

    #[test]
    fn delay_law_max_is_zero() {
        // The element closest to the focus should have delay approaching 0
        let ex = vec![0.0];
        let ez = vec![0.0];
        let d = delay_law_focus_2d(&ex, &ez, 0.0, 0.0, SOUND_SPEED_WATER_SIM);
        assert!((d[0]).abs() < 1e-15);
    }
}
