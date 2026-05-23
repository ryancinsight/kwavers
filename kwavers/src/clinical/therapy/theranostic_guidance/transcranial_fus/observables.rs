use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};
use ndarray::Array3;

/// Compute acoustic observables from a pressure field.
///
/// # Returns
/// `(intensity_w_m2, mechanical_index, cavitation_probability)`
pub fn acoustic_fus_observables(
    pressure_pa: &Array3<f32>,
    frequency_hz: f64,
    density_kg_m3: f64,
    sound_speed_m_s: f64,
    inertial_mi_threshold: f64,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let two_rho_c = 2.0 * density_kg_m3 * sound_speed_m_s;
    let f_mhz = frequency_hz / MHZ_TO_HZ;
    let sqrt_f = f_mhz.sqrt();

    let intensity = pressure_pa.mapv(|p| {
        let p64 = p as f64;
        (p64 * p64 / two_rho_c) as f32
    });
    let mi = pressure_pa.mapv(|p| {
        let p64 = p as f64;
        (p64 / MPA_TO_PA / sqrt_f) as f32
    });
    let cavitation = mi.mapv(|m| {
        let m64 = m as f64;
        let exponent = -(m64 - inertial_mi_threshold) / 0.10;
        (1.0 / (1.0 + exponent.exp())) as f32
    });
    (intensity, mi, cavitation)
}
