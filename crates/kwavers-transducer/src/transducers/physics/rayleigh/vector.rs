//! Three-vector arithmetic and plane frames.

use kwavers_core::error::KwaversResult;
use kwavers_core::validation::{validate_finite_point, validate_positive_finite};

pub(super) fn plane_basis(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let seed = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let tangent = unit(subtract(seed, scale(normal, dot(seed, normal))));
    (tangent, cross(normal, tangent))
}

pub(super) fn normalized_normal(normal: [f64; 3]) -> KwaversResult<[f64; 3]> {
    validate_finite_point("normal", normal, "all coordinates finite")?;
    let normal_norm = norm(normal);
    validate_positive_finite("normal_norm", normal_norm, "must be finite and positive")?;
    Ok(scale(normal, 1.0 / normal_norm))
}

pub(super) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub(super) fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub(super) fn scale(vector: [f64; 3], factor: f64) -> [f64; 3] {
    vector.map(|value| value * factor)
}
pub(super) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].mul_add(b[0], a[1].mul_add(b[1], a[2] * b[2]))
}
pub(super) fn norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}
pub(super) fn unit(vector: [f64; 3]) -> [f64; 3] {
    scale(vector, 1.0 / norm(vector))
}
pub(super) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1].mul_add(b[2], -a[2] * b[1]),
        a[2].mul_add(b[0], -a[0] * b[2]),
        a[0].mul_add(b[1], -a[1] * b[0]),
    ]
}
