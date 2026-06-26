//! PDN power-plane cavity-mode kernel.
//!
//! Single free function [`plane_resonance_hz`] that computes the `(m, n)` mode frequency of an
//! `a × b` power-plane pair in dielectric `er`. The decoupling must stay effective at least up
//! to the lowest mode — for a 100 × 80 mm FR4 pair the first mode is ~700 MHz, far above any
//! realistic HV-pulser drive band, so the cavity is effectively transparent to the design.

/// Power-plane cavity resonance (Hz) for mode `(m, n)` of an `a × b` plane pair in dielectric
/// `er`: `f = c/(2√er)·√((m/a)² + (n/b)²)`. The decoupling must stay effective below the lowest
/// mode; for a 100 × 80 mm FR4 pair the first mode is ~700 MHz, far above the 2 MHz drive.
#[must_use]
pub fn plane_resonance_hz(a_m: f64, b_m: f64, er: f64, m: u32, n: u32) -> f64 {
    if a_m <= 0.0 || b_m <= 0.0 || er <= 0.0 {
        return f64::INFINITY;
    }
    let c = 2.998e8;
    let term = (m as f64 / a_m).powi(2) + (n as f64 / b_m).powi(2);
    c / (2.0 * er.sqrt()) * term.sqrt()
}
