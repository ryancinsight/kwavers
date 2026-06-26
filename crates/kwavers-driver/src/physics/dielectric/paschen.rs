//! Paschen air-breakdown kinetics — [`paschen_breakdown_v`], [`paschen_min_air`],
//! [`air_breakdown_possible`].
//!
//! The breakdown voltage of a gap in air is a function of `pd` (pressure × distance), with a
//! minimum near 327 V. The key engineering consequence for a 150 V design: 150 V is **below
//! the Paschen minimum**, so air will not break down across *any* gap — the HV spacing is
//! governed entirely by surface creepage, which the router already enforces.
//!
//! Air constants (intentionally `const` private to this sub-slice — NOT `pub const` — because
//! they are physics-internal and the [`super`] facade re-exports only the 3 fns above, so the
//! API surface stays clean):
//! * `A = 15 (cm·Torr)⁻¹` — saturation ionisation rate
//! * `B = 365 V/(cm·Torr)` — electron-mobility constant
//! * `γ ≈ 0.01` — secondary-electron emission coefficient

// Slice-private air constants (NOT `pub const`). Physics-internal — only consumed by the
// 3 `paschen_*` fns in this file. Future contributors wanting a pinning test on these
// values would need to add `pub(crate) use paschen::{A_AIR,B_AIR,GAMMA};` in the parent
// `mod.rs` (don't change visibility here without also updating the slice API).
const A_AIR: f64 = 15.0;
const B_AIR: f64 = 365.0;
const GAMMA: f64 = 0.01;

/// The `ln(ln(1 + 1/γ))` denominator term in Paschen's law. Computed at runtime so the value
/// tracks any future edit to `GAMMA`.
fn ln_term() -> f64 {
    (1.0 + 1.0 / GAMMA).ln().ln()
}

/// Paschen breakdown voltage (V) of an air gap at `pd` in Torr·cm. Returns `+∞` left of the
/// discharge-sustaining limit (no breakdown possible).
#[must_use]
pub fn paschen_breakdown_v(pd_torr_cm: f64) -> f64 {
    if pd_torr_cm <= 0.0 {
        return f64::INFINITY;
    }
    let denom = (A_AIR * pd_torr_cm).ln() - ln_term();
    if denom <= 0.0 {
        return f64::INFINITY; // below the sustaining limit — cannot break down
    }
    B_AIR * pd_torr_cm / denom
}

/// The Paschen minimum breakdown voltage for air (V) and the `pd` (Torr·cm) at which it occurs.
#[must_use]
pub fn paschen_min_air() -> (f64, f64) {
    // Minimum at pd* = e · ln(1+1/γ) / A.
    let pd_min = std::f64::consts::E * (1.0 + 1.0 / GAMMA).ln() / A_AIR;
    (paschen_breakdown_v(pd_min), pd_min)
}

/// Whether `voltage_v` can ever break down an air gap (i.e. exceeds the Paschen minimum). If
/// false, air breakdown is impossible at any gap and spacing is creepage-limited.
#[must_use]
pub fn air_breakdown_possible(voltage_v: f64) -> bool {
    voltage_v > paschen_min_air().0
}
