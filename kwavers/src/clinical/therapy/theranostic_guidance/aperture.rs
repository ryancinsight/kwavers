//! Shared aperture constraints for abdominal same-device arrays.

use super::config::TheranosticFwiConfig;

pub(crate) const ABDOMINAL_SKIN_CLEARANCE_M: f64 = 0.003;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AbdominalArcSpec {
    pub radius_m: f64,
    pub half_angle_rad: f64,
    pub cutout_angle_rad: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct AbdominalApertureFrame {
    pub normal_x: f64,
    pub normal_y: f64,
    pub tangent_x: f64,
    pub tangent_y: f64,
    pub depth_m: f64,
}

/// Compute a skin-exterior abdominal arc for a circular focused array.
///
/// # Theorem
///
/// Let `n` be the unit vector from the external skin point to the focus, let
/// `d = dot(focus - skin, n) >= 0`, and let the array arc be
/// `p(theta) = focus - r cos(theta) n + r sin(theta) t`, where `t` is the
/// local tangent. If `|theta| <= acos((d + c) / r)`, every element lies outside
/// the patient half-space by `dot(p(theta) - skin, n) <= -c`.
///
/// # Proof sketch
///
/// For `|theta| <= alpha = acos((d + c) / r)`, monotonicity of cosine on
/// `[0, pi]` gives `cos(theta) >= cos(alpha) = (d + c) / r`. Therefore
/// `dot(p(theta) - skin, n) = d - r cos(theta) <= -c`.
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

#[must_use]
pub(crate) fn abdominal_aperture_frame(
    focus_x_m: f64,
    focus_y_m: f64,
    skin_x_m: f64,
    skin_y_m: f64,
) -> AbdominalApertureFrame {
    let dx = focus_x_m - skin_x_m;
    let dy = focus_y_m - skin_y_m;
    let depth_m = dx.hypot(dy);
    if depth_m <= f64::EPSILON || !depth_m.is_finite() {
        return AbdominalApertureFrame {
            normal_x: 1.0,
            normal_y: 0.0,
            tangent_x: 0.0,
            tangent_y: 1.0,
            depth_m: 0.0,
        };
    }
    let normal_x = dx / depth_m;
    let normal_y = dy / depth_m;
    AbdominalApertureFrame {
        normal_x,
        normal_y,
        tangent_x: -normal_y,
        tangent_y: normal_x,
        depth_m,
    }
}

#[must_use]
pub(crate) fn abdominal_arc_point_2d(
    frame: AbdominalApertureFrame,
    focus_x_m: f64,
    focus_y_m: f64,
    radius_m: f64,
    theta_rad: f64,
) -> (f64, f64) {
    let normal_offset_m = radius_m * theta_rad.cos();
    let tangent_offset_m = radius_m * theta_rad.sin();
    (
        focus_x_m - frame.normal_x * normal_offset_m + frame.tangent_x * tangent_offset_m,
        focus_y_m - frame.normal_y * normal_offset_m + frame.tangent_y * tangent_offset_m,
    )
}

#[must_use]
pub(crate) fn abdominal_imaging_point_2d(
    frame: AbdominalApertureFrame,
    skin_x_m: f64,
    skin_y_m: f64,
    aperture_m: f64,
    t: f64,
) -> (f64, f64) {
    let lateral_m = (t - 0.5) * aperture_m;
    (
        skin_x_m - frame.normal_x * ABDOMINAL_SKIN_CLEARANCE_M + frame.tangent_x * lateral_m,
        skin_y_m - frame.normal_y * ABDOMINAL_SKIN_CLEARANCE_M + frame.tangent_y * lateral_m,
    )
}

fn asin_unit(value: f64) -> f64 {
    value.clamp(0.0, 1.0).asin()
}

fn acos_unit(value: f64) -> f64 {
    value.clamp(-1.0, 1.0).acos()
}
