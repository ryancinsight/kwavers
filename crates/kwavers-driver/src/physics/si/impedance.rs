//! Controlled-impedance characteristic values for microstrip / stripline / differential
//! signal-integrity design.
//!
//! All four characteristic-impedance fns accept `f64` lengths / impedances — the docstrings
//! say "any consistent length unit". Phase 2 will replace the `w, h, er, t, b` parameters with
//! `Meter, Meter, f64, Meter, Meter` for the dimensioned quantities and return types as the
//! typed [`Ohm`] wrapper (not yet wrapped at Phase 1a; carried in plain backticks to keep
//! informational references rustdoc-warning-clean). **No signature change at Phase 3f** —
//! keeping the API as `f64` preserves every existing call-site and test fixture until the
//! vertical-slice units land.
//!
//! [`Ohm`]: crate::units::Ohm

/// Effective relative permittivity of a microstrip of width `w` over height `h` in a dielectric
/// `er` (Hammerstad).
#[must_use]
pub fn microstrip_eeff(w: f64, h: f64, er: f64) -> f64 {
    let wh = (w / h).max(1.0e-6);
    (er + 1.0) / 2.0 + (er - 1.0) / 2.0 * (1.0 + 12.0 / wh).powf(-0.5)
}

/// Characteristic impedance (Ω) of a microstrip (Hammerstad). `w`, `h` any consistent unit.
#[must_use]
pub fn microstrip_impedance(w: f64, h: f64, er: f64) -> f64 {
    let wh = (w / h).max(1.0e-6);
    let eeff = microstrip_eeff(w, h, er).sqrt();
    if wh <= 1.0 {
        60.0 / eeff * (8.0 / wh + wh / 4.0).ln()
    } else {
        120.0 * std::f64::consts::PI / (eeff * (wh + 1.393 + 0.667 * (wh + 1.444).ln()))
    }
}

/// Characteristic impedance (Ω) of a centered stripline (Wadell approximation).
///
/// A stripline trace of width `w` and thickness `t` is embedded symmetrically between two ground
/// planes separated by `b` (centre-to-centre), in a dielectric of relative permittivity `er`.
/// The effective width includes a fringe correction. Wadell formula:
///
/// `Z = (60 / √er) · ln(4b / (0.67π · (0.8w + t)))`, valid for `w/b < 0.85`.
///
/// For inner-layer signal routing where both adjacent planes are ground (the preferred arrangement
/// for controlled impedance and EMI), this is the correct impedance model — [`microstrip_impedance`]
/// applies only to outer traces above a single reference.
///
/// All dimensions in any consistent length unit.
#[must_use]
pub fn stripline_impedance(w: f64, t: f64, b: f64, er: f64) -> f64 {
    if b <= 0.0 || er <= 0.0 {
        return 0.0;
    }
    let w_eff = 0.8 * w + t;
    let arg = 4.0 * b / (0.67 * std::f64::consts::PI * w_eff.max(1.0e-12));
    if arg <= 1.0 {
        return 0.0; // trace wider than the plane gap — not a valid stripline geometry
    }
    60.0 / er.sqrt() * arg.ln()
}

/// Differential (odd-mode) impedance (Ω) of a closely-coupled microstrip pair.
///
/// The odd-mode impedance `Z_odd = Z₀ · (1 − k)` where `k` is the backward-crosstalk
/// coupling coefficient ([`crate::physics::si::crosstalk::crosstalk_coupling`]) and `Z₀`
/// is the single-ended microstrip impedance. The differential impedance is
/// `Z_diff = 2 · Z_odd` (each driver sees Z_odd, the receiver senses 2× for differential
/// signalling).
///
/// Valid for loosely coupled pairs (`k ≪ 1`); for tight coupling (s < h) the formula
/// overpredicts coupling and the differential impedance may need a full 2D field solver.
#[must_use]
pub fn differential_microstrip_impedance(w: f64, h: f64, s: f64, er: f64) -> f64 {
    let z0 = microstrip_impedance(w, h, er);
    let k = crate::physics::si::crosstalk::crosstalk_coupling(s, h);
    2.0 * z0 * (1.0 - k)
}

/// Impedance target (Ω) for a controlled-impedance signal line, given the driver's source
/// impedance and a tolerated reflection coefficient.
///
/// The single-line **branching-matching** problem: a driver with `z_driver` feeds a line
/// whose impedance must be chosen so the worst-case reflection coefficient stays below
/// `max_reflection_coef`. Solving the mismatch identity
/// `Γ = (z_line − z_driver) / (z_line + z_driver)` for `z_line` yields
///
/// `z_target = z_driver · (1 + Γ_max) / (1 − Γ_max)`
///
/// which is the larger of the two mirror-symmetric solutions (the `+Γ_max` branch driving
/// `z_line > z_driver`; the `−Γ_max` branch would yield `z_line < z_driver` if a downward
/// regulator is preferred, but the upward branch is the conventional digital-impedance
/// convention).
///
/// **SSOT distinction**: this is the **signal-line** target. For the **PDN power-rail**
/// target impedance (`V_tolerance / I_step`), see
/// [`crate::physics::pdn::target_impedance_ohm`]. The two functions solve different physical
/// problems; importing both at the same call site is normal.
///
/// Boundary behaviour:
/// * `max_reflection_coef ∈ (0, 1)` — normal range; returns a positive multiplier of
///   `z_driver`.
/// * `max_reflection_coef == 0` — perfect match; returns `z_driver` exactly.
/// * `max_reflection_coef ≥ 1` — any-mismatch tolerance; the only feasible line Z is
///   arbitrarily large (`∞`); degenerate to `z_driver` for a finite answer.
/// * `z_driver <= 0` — degenerate driver; returns `0.0`.
#[must_use]
pub fn impedance_target(z_driver_ohm: f64, max_reflection_coef: f64) -> f64 {
    if z_driver_ohm <= 0.0 {
        return 0.0;
    }
    if !(0.0..1.0).contains(&max_reflection_coef) {
        return z_driver_ohm;
    }
    z_driver_ohm * (1.0 + max_reflection_coef) / (1.0 - max_reflection_coef)
}

/// Return loss (dB) at a controlled-impedance boundary, given driver and line impedances.
///
/// `Γ = (z_line − z_driver) / (z_line + z_driver)`; `RL = −20·log10(|Γ|)`. Two key boundary
/// cases:
///
/// * `z_line == z_driver` ⇒ `Γ = 0` ⇒ `RL = +∞` (perfect match, no reflection). Returned
///   as [`f64::INFINITY`] so the caller can compare against a budget threshold.
/// * `z_line` open (≡ `∞`) or short (≡ `0`) ⇒ `|Γ| = 1` ⇒ `RL = 0 dB` (full reflection).
///
/// Frequency is **not** threaded in: the function assumes resistive scalar Z, which holds
/// when skin-effect-induced series resistance is rolled into the line impedance at each
/// frequency the caller evaluates. To compute RL **per frequency band**, call this in a loop
/// with the per-band Z(f) values — that keeps the function zero-cost (no `Vec` heap allocation,
/// no caller-side `Vec<f64>` for the band range).
///
/// Boundary behaviour for negative or non-finite inputs: returns [`f64::NEG_INFINITY`] (the
/// caller has fed nonsense; an arbitrary finite result is preferred over a `NaN` chain).
#[must_use]
pub fn return_loss_db(z_driver: f64, z_line: f64) -> f64 {
    if z_driver < 0.0 || z_line < 0.0 {
        return f64::NEG_INFINITY;
    }
    let denom = z_line + z_driver;
    if denom <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let gamma_mag = ((z_line - z_driver) / denom).abs();
    if gamma_mag == 0.0 {
        return f64::INFINITY;
    }
    -20.0 * gamma_mag.log10()
}
