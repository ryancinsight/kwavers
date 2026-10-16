//! Gauss-Legendre rules on the unit interval and the surface rules built from them.

use super::aperture::PlanarApertureShape;
use super::input::invalid;
use eunomia::FloatElement;
use kwavers_core::error::KwaversResult;
use std::f64::consts::PI;

const LEGENDRE_ROOT_STEPS: usize = 64;
const LEGENDRE_ROOT_TOLERANCE: f64 = 8.0 * f64::EPSILON;

pub(super) fn gauss_legendre_unit(order: usize) -> KwaversResult<Vec<(f64, f64)>> {
    let mut rule = vec![(0.0, 0.0); order];
    let paired_roots = order.div_ceil(2);
    for root_index in 0..paired_roots {
        let mut root =
            (PI * (f64::from_count(root_index) + 0.75) / (f64::from_count(order) + 0.5)).cos();
        let mut converged = false;
        for _ in 0..LEGENDRE_ROOT_STEPS {
            let (polynomial, previous) = legendre_pair(order, root);
            let derivative =
                f64::from_count(order) * (root * polynomial - previous) / (root * root - 1.0);
            let next = root - polynomial / derivative;
            if (next - root).abs() <= LEGENDRE_ROOT_TOLERANCE * next.abs().max(1.0) {
                root = next;
                converged = true;
                break;
            }
            root = next;
        }
        if !converged {
            return Err(invalid(
                "quadrature_order",
                order.to_string(),
                "Gauss-Legendre roots converge",
            ));
        }
        let (polynomial, previous) = legendre_pair(order, root);
        let derivative =
            f64::from_count(order) * (root * polynomial - previous) / (root * root - 1.0);
        let weight = 1.0 / ((1.0 - root * root) * derivative * derivative);
        let lower = root_index;
        let upper = order - 1 - root_index;
        rule[lower] = (0.5 * (1.0 - root), weight);
        rule[upper] = (0.5 * (1.0 + root), weight);
    }
    Ok(rule)
}

pub(super) fn legendre_pair(order: usize, x: f64) -> (f64, f64) {
    let mut previous = 1.0;
    if order == 0 {
        return (previous, 0.0);
    }
    let mut current = x;
    for degree in 2..=order {
        let next = (f64::from_count(2 * degree - 1) * x * current
            - f64::from_count(degree - 1) * previous)
            / f64::from_count(degree);
        previous = current;
        current = next;
    }
    (current, previous)
}

/// Local surface nodes `([u, v], weight)` of a planar shape.
///
/// `u` runs along the first axis and `v` along `normal × first_axis`; the
/// weights sum to the shape's area. A polar shape takes `primary_order`
/// Gauss-Legendre nodes in normalized squared radius, which integrate the
/// area element exactly, and `secondary_order` periodic midpoint nodes in
/// angle. A rectangle takes a tensor-product Gauss-Legendre rule:
/// `primary_order` nodes across the width and `secondary_order` across the
/// height, each mapped from the unit interval by `x = (t − ½)·L` with
/// Jacobian `L`.
pub(super) fn surface_rule(
    shape: PlanarApertureShape,
    primary_order: usize,
    secondary_order: usize,
) -> KwaversResult<Vec<([f64; 2], f64)>> {
    let primary = gauss_legendre_unit(primary_order)?;
    let mut nodes = Vec::with_capacity(primary_order * secondary_order);
    if let PlanarApertureShape::Rectangle { width_m, height_m } = shape {
        let secondary = gauss_legendre_unit(secondary_order)?;
        for &(width_fraction, width_weight) in &primary {
            for &(height_fraction, height_weight) in &secondary {
                nodes.push((
                    [
                        (width_fraction - 0.5) * width_m,
                        (height_fraction - 0.5) * height_m,
                    ],
                    width_m * width_weight * height_m * height_weight,
                ));
            }
        }
        return Ok(nodes);
    }
    // Fewer than three midpoint nodes cannot resolve the first angular
    // harmonic of a ring's phase; a rectangle has no angular rule.
    if secondary_order < 3 {
        return Err(invalid(
            "secondary_order",
            secondary_order.to_string(),
            ">= 3 for a polar aperture",
        ));
    }
    let (inner, outer, start, span) = shape
        .polar_bounds()
        .expect("invariant: every non-rectangular shape is polar");
    let squared_span = outer * outer - inner * inner;
    let angular_weight = 0.5 * squared_span * span / f64::from_count(secondary_order);
    for &(radial_fraction, radial_weight) in &primary {
        let radius = (inner * inner + radial_fraction * squared_span).sqrt();
        for angular_index in 0..secondary_order {
            let angle = start
                + span * (f64::from_count(angular_index) + 0.5) / f64::from_count(secondary_order);
            nodes.push((
                [radius * angle.cos(), radius * angle.sin()],
                angular_weight * radial_weight,
            ));
        }
    }
    Ok(nodes)
}
