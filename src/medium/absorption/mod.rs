// medium/absorption/mod.rs

use log::{debug, trace};

pub mod power_law_absorption;

pub fn absorption_coefficient(frequency: f64, temperature: f64, bubble_radius: Option<f64>) -> f64 {
    debug!(
        "Computing basic absorption: freq = {:.2e} Hz, temp = {:.2} K",
        frequency, temperature
    );
    assert!(
        frequency > 0.0 && temperature > 0.0,
        "Frequency and temperature must be positive"
    );

    let alpha_base = 0.022 / (100.0 * 8.686); // Base absorption in water (Np/m/MHz²)
    let f_mhz = frequency / 1e6;
    let alpha_f = alpha_base * f_mhz * f_mhz;

    let t_c = temperature - 273.15;
    let t_adjust = (1.0 - 0.0005 * (t_c - 20.0)).max(0.1);

    let alpha_bubble = bubble_radius.map_or(0.0, |r| {
        let r_clamped = r.max(1e-10);
        0.01 * f_mhz * r_clamped * 1e6 / 1500.0
    });

    let total_alpha = (alpha_f * t_adjust + alpha_bubble).max(0.0);
    trace!("Absorption: alpha = {:.6e} Np/m", total_alpha);
    total_alpha
}
