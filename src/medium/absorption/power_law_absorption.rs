// medium/absorption/power_law_absorption.rs

use log::debug;

/// Calculates absorption coefficient using a power law model: α = α0 * f^δ.
/// - frequency: Acoustic frequency (Hz)
/// - alpha0: Absorption coefficient at 1 MHz (Np/m/MHz^δ)
/// - delta: Power law exponent (typically 1 to 2 for tissues)
pub fn power_law_absorption_coefficient(frequency: f64, alpha0: f64, delta: f64) -> f64 {
    assert!(
        frequency > 0.0 && alpha0 >= 0.0 && delta >= 0.0,
        "Invalid parameters for power law absorption"
    );
    debug!(
        "Computing power law absorption: freq = {:.2e} Hz, alpha0 = {:.2e}, delta = {:.2}",
        frequency, alpha0, delta
    );
    if delta == 0.0 {
        // Frequency-independent absorption
        alpha0
    } else {
        alpha0 * (frequency / 1e6).powf(delta) // Convert frequency to MHz for consistency
    }
}
