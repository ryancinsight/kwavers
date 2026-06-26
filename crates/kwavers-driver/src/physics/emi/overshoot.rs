//! Capacitive-drive current + inductive voltage overshoot — the two transient peaks on a
//! class-D switching node.

/// Current (A) to charge a capacitive load `c_f` through `dv` volts in `risetime_s`: `I = C·dV/dt`.
/// For the BVD transducer load (~50 pF) at 150 V in ~5 ns this is the HV7355's ~1.5 A peak rating —
/// the layout's output path must carry it.
#[must_use]
pub fn capacitive_drive_current_a(c_f: f64, dv: f64, risetime_s: f64) -> f64 {
    if risetime_s <= 0.0 {
        return f64::INFINITY;
    }
    c_f * dv / risetime_s
}

/// Inductive voltage overshoot (V) on a switching node: `V = L·dI/dt`.
#[must_use]
pub fn inductive_overshoot_v(inductance_nh: f64, current_a: f64, risetime_s: f64) -> f64 {
    if risetime_s <= 0.0 {
        return f64::INFINITY;
    }
    inductance_nh * 1.0e-9 * current_a / risetime_s
}
