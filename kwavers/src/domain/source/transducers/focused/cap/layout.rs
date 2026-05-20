//! Equal-area spherical-cap sampling for focused bowl transducers.
//!
//! # Theorem
//!
//! For a spherical cap of radius `R` centered at focus `F`, with axis `d` from
//! cap vertex toward focus, the element position
//! `P = F - R (cos(theta)d + sin(theta)(cos(phi)e1 + sin(phi)e2))` lies on the
//! sphere `||P - F|| = R`. Equal-area sampling over
//! `theta in [theta_min, theta_max]` is obtained by sampling `cos(theta)`
//! uniformly because the cap area element is `R^2 sin(theta) dtheta dphi`.

use crate::core::error::{KwaversError, KwaversResult};
use std::f64::consts::{FRAC_PI_2, PI};

/// Configuration for equal-area focused spherical-cap element placement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalCapConfig {
    /// Number of elements on the cap.
    pub element_count: usize,
    /// Radius from the acoustic focus to the cap surface [m].
    pub radius_m: f64,
    /// Acoustic focus [m].
    pub focus_m: [f64; 3],
    /// Unit-agnostic axis from the cap vertex toward the focus.
    pub axis_vertex_to_focus: [f64; 3],
    /// Minimum polar angle from the cap axis [rad].
    pub theta_min_rad: f64,
    /// Maximum polar angle from the cap axis [rad].
    pub theta_max_rad: f64,
}

impl SphericalCapConfig {
    /// Full hemispherical focused cap.
    #[must_use]
    pub fn hemisphere(
        element_count: usize,
        radius_m: f64,
        focus_m: [f64; 3],
        axis_vertex_to_focus: [f64; 3],
    ) -> Self {
        Self {
            element_count,
            radius_m,
            focus_m,
            axis_vertex_to_focus,
            theta_min_rad: 0.0,
            theta_max_rad: FRAC_PI_2,
        }
    }
}

/// One focused spherical-cap element.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphericalCapElement {
    /// Element center position [m].
    pub position_m: [f64; 3],
    /// Unit normal pointing from element toward the acoustic focus.
    pub normal_to_focus: [f64; 3],
    /// Equal surface-area weight represented by this element [m^2].
    pub area_weight_m2: f64,
}

/// Equal-area focused spherical-cap element layout.
#[derive(Clone, Debug, PartialEq)]
pub struct SphericalCapLayout {
    elements: Vec<SphericalCapElement>,
}

impl SphericalCapLayout {
    /// Generate an equal-area focused spherical-cap layout.
    ///
    /// # Errors
    ///
    /// Returns an error when the count, radius, focus, axis, or angular span is
    /// outside the physical spherical-cap domain.
    pub fn new(config: SphericalCapConfig) -> KwaversResult<Self> {
        validate_config(config)?;

        let axis = normalize(config.axis_vertex_to_focus).expect("validated nonzero axis");
        let (e1, e2) = perpendicular_frame(axis);
        let cos_min = config.theta_min_rad.cos();
        let cos_max = config.theta_max_rad.cos();
        let cap_area = 2.0 * PI * config.radius_m * config.radius_m * (cos_min - cos_max).abs();
        let area_weight = cap_area / config.element_count as f64;
        let golden_angle = PI * (3.0 - 5.0_f64.sqrt());

        let elements = (0..config.element_count)
            .map(|idx| {
                let t = (idx as f64 + 0.5) / config.element_count as f64;
                let cos_theta = cos_min - t * (cos_min - cos_max);
                let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
                let phi = idx as f64 * golden_angle;
                let radial = add3(scale3(e1, phi.cos()), scale3(e2, phi.sin()));
                let normal = add3(scale3(axis, cos_theta), scale3(radial, sin_theta));
                let position = sub3(config.focus_m, scale3(normal, config.radius_m));
                SphericalCapElement {
                    position_m: position,
                    normal_to_focus: normal,
                    area_weight_m2: area_weight,
                }
            })
            .collect();

        Ok(Self { elements })
    }

    /// Borrow generated elements.
    #[must_use]
    pub fn elements(&self) -> &[SphericalCapElement] {
        &self.elements
    }

    /// Borrow generated element positions.
    #[must_use]
    pub fn positions(&self) -> impl ExactSizeIterator<Item = [f64; 3]> + '_ {
        self.elements.iter().map(|element| element.position_m)
    }
}

fn validate_config(config: SphericalCapConfig) -> KwaversResult<()> {
    if config.element_count == 0 {
        return Err(KwaversError::InvalidInput(
            "spherical-cap layout requires at least one element".to_owned(),
        ));
    }
    if !positive_finite(config.radius_m) {
        return Err(KwaversError::InvalidInput(
            "spherical-cap radius must be positive and finite".to_owned(),
        ));
    }
    if !config.focus_m.iter().all(|value| value.is_finite()) {
        return Err(KwaversError::InvalidInput(
            "spherical-cap focus must be finite".to_owned(),
        ));
    }
    if normalize(config.axis_vertex_to_focus).is_none() {
        return Err(KwaversError::InvalidInput(
            "spherical-cap axis must be finite and nonzero".to_owned(),
        ));
    }
    if !config.theta_min_rad.is_finite()
        || !config.theta_max_rad.is_finite()
        || config.theta_min_rad < 0.0
        || config.theta_min_rad >= config.theta_max_rad
        || config.theta_max_rad > PI
    {
        return Err(KwaversError::InvalidInput(
            "spherical-cap angles must satisfy 0 <= theta_min < theta_max <= pi".to_owned(),
        ));
    }
    Ok(())
}

fn perpendicular_frame(axis: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let reference = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let e1 = normalize(cross3(axis, reference)).expect("reference is not parallel to axis");
    let e2 = cross3(axis, e1);
    (e1, e2)
}

fn normalize(vector: [f64; 3]) -> Option<[f64; 3]> {
    if !vector.iter().all(|value| value.is_finite()) {
        return None;
    }
    let norm = dot3(vector, vector).sqrt();
    positive_finite(norm).then(|| scale3(vector, norm.recip()))
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale3(vector: [f64; 3], scale: f64) -> [f64; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1].mul_add(b[2], -a[2] * b[1]),
        a[2].mul_add(b[0], -a[0] * b[2]),
        a[0].mul_add(b[1], -a[1] * b[0]),
    ]
}
