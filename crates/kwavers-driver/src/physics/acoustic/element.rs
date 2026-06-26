//! Element-level geometry + directivity + the focal-pressure coherent-gain estimate.
//!
//! This submodule carries the per-element half of phased-array acoustics: where the
//! near-field / far-field transition lies (`N = d²/4λ`), how a single element's
//! directivity rolls off off-axis (the sinc that multiplies the array factor), what
//! `f/#` a focused aperture of given focal-depth + aperture realizes, how to derive
//! the centre-to-centre pitch for an `n`-element span, and the coherent linear
//! pressure gain at the geometric focus.
//!
//! Each kernel is pure-math (`f64`-in/`f64`-out, no state, no cross-slice dep) so it
//! feeds straight into the slice facade's named `pub use` re-export chain.

/// Near-field (Fraunhofer) distance (m) of an aperture `d`: `N = d²/(4λ)`. Beyond `N` the beam
/// diverges and focusing/steering behaves as far-field; the paper focuses at 10 mm.
#[must_use]
pub fn near_field_distance_m(aperture_m: f64, lambda_m: f64) -> f64 {
    if lambda_m <= 0.0 {
        return f64::INFINITY;
    }
    aperture_m * aperture_m / (4.0 * lambda_m)
}

/// Element directivity (normalised pressure) of a single element of width `w` at off-axis angle
/// `θ`: the far-field `sinc` pattern `sin(x)/x`, `x = π·w·sinθ/λ`. A wider element rolls off faster
/// with steering — the element factor that multiplies the array factor.
#[must_use]
pub fn element_factor(width_m: f64, lambda_m: f64, angle_deg: f64) -> f64 {
    if lambda_m <= 0.0 {
        return 1.0;
    }
    let x = std::f64::consts::PI * width_m / lambda_m * angle_deg.to_radians().sin();
    if x.abs() < 1.0e-9 {
        1.0
    } else {
        x.sin() / x
    }
}

/// f-number of a focused aperture: `focal_depth / aperture` (lower = tighter focus).
#[must_use]
pub fn f_number(focal_m: f64, aperture_m: f64) -> f64 {
    if aperture_m <= 0.0 {
        return f64::INFINITY;
    }
    focal_m / aperture_m
}

/// Centre-to-centre pitch for `n` elements spanning an aperture from first to last element centre.
#[must_use]
pub fn pitch_from_aperture_m(aperture_m: f64, n: usize) -> f64 {
    if n < 2 {
        0.0
    } else {
        aperture_m / (n as f64 - 1.0)
    }
}

/// Coherent pressure gain at the geometric focus of an N-element focused array (linear, ≥ 1).
///
/// In the far-field of each element the coherent summation gives gain = N (pressure doubles per
/// 6 dB of element count). In the near-field the focus concentrates energy: the geometric gain
/// factor is `N · (λ / (4 · F_number · d))^0.5` — i.e. proportional to array size and inversely
/// to f-number. For far-field targets (depth > N·d²/4λ) the gain saturates near N.
///
/// This uses the simpler aperture-gain estimate `G ≈ N` (coherent sum normalised to N elements),
/// which is an analytically exact upper bound — the geometric near-field factor is ≤ 1.
/// Returns the linear pressure amplitude ratio relative to a single element.
#[must_use]
pub fn focal_pressure_gain(n: usize) -> f64 {
    n as f64
}
