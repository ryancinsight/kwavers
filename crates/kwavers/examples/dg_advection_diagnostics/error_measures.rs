//! Quadrature-weighted error, phase, amplitude, and energy measures for the DG diagnostics.

use super::reference_solutions::physical_coordinate;
use super::{DENSITY, ELEMENTS, SOUND_SPEED};
use leto::{Array1, Array3};
use std::f64::consts::PI;

pub(crate) fn weighted_mass(coeffs: &Array3<f64>, weights: &Array1<f64>) -> f64 {
    let mut mass = 0.0;
    for elem in 0..ELEMENTS {
        for node in 0..weights.len() {
            mass += weights[node] * coeffs[[elem, node, 0]];
        }
    }
    mass
}

pub(crate) fn relative_l2(
    actual: &Array3<f64>,
    expected: &Array3<f64>,
    weights: &Array1<f64>,
) -> f64 {
    let mut diff_sq = 0.0;
    let mut expected_sq = 0.0;
    for elem in 0..ELEMENTS {
        for node in 0..weights.len() {
            let diff = actual[[elem, node, 0]] - expected[[elem, node, 0]];
            diff_sq += weights[node] * diff * diff;
            expected_sq += weights[node] * expected[[elem, node, 0]] * expected[[elem, node, 0]];
        }
    }
    diff_sq.sqrt() / expected_sq.sqrt().max(f64::EPSILON)
}

pub(crate) fn left_going_invariant_error(pressure: &Array3<f64>, velocity: &Array3<f64>) -> f64 {
    pressure
        .iter()
        .zip(velocity.iter())
        .map(|(&p, &u)| (p - DENSITY * SOUND_SPEED * u).abs())
        .fold(0.0, f64::max)
}

pub(crate) fn acoustic_energy(
    pressure: &Array3<f64>,
    velocity: &Array3<f64>,
    weights: &Array1<f64>,
) -> f64 {
    let mut energy = 0.0;
    for elem in 0..ELEMENTS {
        for node in 0..weights.len() {
            let p = pressure[[elem, node, 0]];
            let u = velocity[[elem, node, 0]];
            energy += weights[node]
                * (p * p / (2.0 * DENSITY * SOUND_SPEED * SOUND_SPEED) + 0.5 * DENSITY * u * u);
        }
    }
    energy
}

pub(crate) fn phase_error(
    coeffs: &Array3<f64>,
    weights: &Array1<f64>,
    xi_nodes: &Array1<f64>,
    k: f64,
    time: f64,
) -> f64 {
    let expected_phase = wrap_angle(k * SOUND_SPEED * time);
    let measured_phase = measured_phase(coeffs, weights, xi_nodes, k);
    wrap_angle(measured_phase - expected_phase).abs()
}

fn measured_phase(
    coeffs: &Array3<f64>,
    weights: &Array1<f64>,
    xi_nodes: &Array1<f64>,
    k: f64,
) -> f64 {
    let mut sin_coeff = 0.0;
    let mut cos_coeff = 0.0;
    for elem in 0..ELEMENTS {
        for node in 0..weights.len() {
            let x = physical_coordinate(elem, xi_nodes[node]);
            let value = coeffs[[elem, node, 0]];
            sin_coeff += weights[node] * value * (k * x).sin();
            cos_coeff += weights[node] * value * (k * x).cos();
        }
    }
    wrap_angle((-cos_coeff).atan2(sin_coeff))
}

pub(crate) fn amplitude(
    coeffs: &Array3<f64>,
    weights: &Array1<f64>,
    xi_nodes: &Array1<f64>,
    k: f64,
) -> f64 {
    let mut sin_coeff = 0.0;
    let mut cos_coeff = 0.0;
    for elem in 0..ELEMENTS {
        for node in 0..weights.len() {
            let x = physical_coordinate(elem, xi_nodes[node]);
            let value = coeffs[[elem, node, 0]];
            sin_coeff += weights[node] * value * (k * x).sin();
            cos_coeff += weights[node] * value * (k * x).cos();
        }
    }
    sin_coeff.hypot(cos_coeff)
}

fn wrap_angle(angle: f64) -> f64 {
    (angle + PI).rem_euclid(2.0 * PI) - PI
}
