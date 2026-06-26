//! First-order radiated-EMI estimate (CISPR-22 small-loop antenna).

/// First-order radiated EMI estimate (dBµV/m) at test distance `r_m` from a rectangular
/// current loop radiating at frequency `f_hz` with peak current `i_pk_a`.
///
/// Based on the small-loop antenna formula (CISPR 22 / FCC Part 15 regulatory context):
///
/// `E = (1.316e-14 · f² · A · I) / r`  (V/m),   then → dBµV/m
///
/// where:
/// - `f_hz` — fundamental switching frequency (Hz); HV7355 at 2 MHz
/// - `loop_area_mm2` — commutation-loop area (mm²) — from [`super::scene::CommutationLoop::area_mm2`]
/// - `i_pk_a` — peak current through the loop (A); for the HV driver: `C·dV/dt ≈ 1.5 A`
/// - `r_m` — measurement distance (m); CISPR 22 class B uses 3 m or 10 m
///
/// The formula gives E in V/m at the fundamental; harmonic content adds ~20 dB/decade for a
/// trapezoidal waveform. **This is a first-order estimate** for design margin checks — actual
/// radiated emission testing on a CISPR-22 antenna is the compliance oracle.
///
/// Returns the field level in dBµV/m (`20·log₁₀(E_µV/m)`).
#[must_use]
pub fn radiated_emi_dbuv_m(f_hz: f64, loop_area_mm2: f64, i_pk_a: f64, r_m: f64) -> f64 {
    if f_hz <= 0.0 || loop_area_mm2 <= 0.0 || i_pk_a <= 0.0 || r_m <= 0.0 {
        return f64::NEG_INFINITY;
    }
    // Convert loop area mm² → m²
    let area_m2 = loop_area_mm2 * 1.0e-6;
    // Small-loop antenna field (V/m): E = µ₀·π·f²·A·I / (c·r) = 1.316e-14·f²·A·I/r
    // derivation: E ≈ (µ₀ · 2πf)² · A · I / (4π · c · r) = µ₀·π·f²·A·I/(c·r)
    // µ₀ = 1.2566e-6, c = 2.998e8 → µ₀·π/c ≈ 1.316e-14
    let e_v_per_m = 1.316e-14 * f_hz * f_hz * area_m2 * i_pk_a / r_m;
    // Convert V/m → µV/m → dBµV/m
    let e_uv_per_m = e_v_per_m * 1.0e6;
    if e_uv_per_m <= 0.0 {
        return f64::NEG_INFINITY;
    }
    20.0 * e_uv_per_m.log10()
}
