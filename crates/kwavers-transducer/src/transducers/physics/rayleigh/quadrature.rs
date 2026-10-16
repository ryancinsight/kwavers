//! Gauss-Legendre rules on the unit interval.

use super::input::invalid;
use kwavers_core::error::KwaversResult;
use std::f64::consts::PI;

const LEGENDRE_ROOT_STEPS: usize = 64;
const LEGENDRE_ROOT_TOLERANCE: f64 = 8.0 * f64::EPSILON;

pub(super) fn gauss_legendre_unit(order: usize) -> KwaversResult<Vec<(f64, f64)>> {
    let mut rule = vec![(0.0, 0.0); order];
    let paired_roots = order.div_ceil(2);
    for root_index in 0..paired_roots {
        let mut root = (PI * (root_index as f64 + 0.75) / (order as f64 + 0.5)).cos();
        let mut converged = false;
        for _ in 0..LEGENDRE_ROOT_STEPS {
            let (polynomial, previous) = legendre_pair(order, root);
            let derivative = order as f64 * (root * polynomial - previous) / (root * root - 1.0);
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
                "radial_order",
                order.to_string(),
                "Gauss-Legendre roots converge",
            ));
        }
        let (polynomial, previous) = legendre_pair(order, root);
        let derivative = order as f64 * (root * polynomial - previous) / (root * root - 1.0);
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
        let next = ((2 * degree - 1) as f64 * x * current - (degree - 1) as f64 * previous)
            / degree as f64;
        previous = current;
        current = next;
    }
    (current, previous)
}
