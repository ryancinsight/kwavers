//! Shared aperture constraints for abdominal same-device arrays.

use super::config::TheranosticFwiConfig;

pub(crate) const ABDOMINAL_SKIN_CLEARANCE_M: f64 = 0.003;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AbdominalArcSpec {
    pub radius_m: f64,
    pub half_angle_rad: f64,
    pub cutout_angle_rad: f64,
}

/// Compute a skin-exterior abdominal arc for a circular focused array.
///
/// # Theorem
///
/// Let the focus be at signed depth `d = x_f - x_s >= 0` from the external
/// skin point, and let the array arc be
/// `x(theta) = x_f - r cos(theta)`. If `|theta| <= acos((d + c) / r)`,
/// every array element satisfies `x(theta) <= x_s - c`.
///
/// # Proof sketch
///
/// For `|theta| <= alpha = acos((d + c) / r)`, monotonicity of cosine on
/// `[0, pi]` gives `cos(theta) >= cos(alpha) = (d + c) / r`. Therefore
/// `x(theta) = x_f - r cos(theta) <= x_f - d - c = x_s - c`.
///
/// The central cutout requires a minimum angle
/// `asin((w / 2) / r)`. The radius is bounded below by
/// `sqrt((d + c)^2 + (w / 2)^2)`, which makes the skin-limited angle at least
/// the cutout angle, so the aperture remains realizable without placing
/// elements inside the patient support.
#[must_use]
pub(crate) fn abdominal_arc_spec(
    config: &TheranosticFwiConfig,
    target_depth_m: f64,
) -> AbdominalArcSpec {
    let depth_m = target_depth_m.max(0.0);
    let clearance_depth_m = depth_m + ABDOMINAL_SKIN_CLEARANCE_M;
    let cutout_half_width_m = (0.5 * config.central_cutout_m).max(0.0);
    let contact_safe_radius_m = clearance_depth_m.hypot(cutout_half_width_m);
    let radius_m = config
        .focal_radius_m
        .max(depth_m + 0.02)
        .max(contact_safe_radius_m);
    let cutout_angle_rad = asin_unit(cutout_half_width_m / radius_m);
    let requested_half_angle_rad = asin_unit((0.5 * config.lateral_extent_m.max(0.0)) / radius_m);
    let skin_limited_half_angle_rad = acos_unit(clearance_depth_m / radius_m);
    let half_angle_rad = requested_half_angle_rad
        .max(cutout_angle_rad)
        .min(skin_limited_half_angle_rad);
    AbdominalArcSpec {
        radius_m,
        half_angle_rad,
        cutout_angle_rad,
    }
}

fn asin_unit(value: f64) -> f64 {
    value.clamp(0.0, 1.0).asin()
}

fn acos_unit(value: f64) -> f64 {
    value.clamp(-1.0, 1.0).acos()
}
