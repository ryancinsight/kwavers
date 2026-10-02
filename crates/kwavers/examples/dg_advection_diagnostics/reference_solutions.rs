//! Exact periodic solutions and initial conditions for the DG diagnostics.

use super::{DENSITY, ELEMENTS, SOUND_SPEED};
use leto::{Array1, Array3};

pub(crate) fn initialize_sine_coefficients(
    coeffs: &mut Array3<f64>,
    xi_nodes: &Array1<f64>,
    k: f64,
) {
    for elem in 0..ELEMENTS {
        for node in 0..xi_nodes.len() {
            let x = physical_coordinate(elem, xi_nodes[node]);
            coeffs[[elem, node, 0]] = (k * x).sin();
        }
    }
}

pub(crate) fn initialize_right_going_characteristic(
    coeffs: &mut Array3<f64>,
    xi_nodes: &Array1<f64>,
    k: f64,
) {
    for elem in 0..ELEMENTS {
        for node in 0..xi_nodes.len() {
            let x = physical_coordinate(elem, xi_nodes[node]);
            coeffs[[elem, node, 0]] = 2.0 * (k * x).sin();
        }
    }
}

pub(crate) fn exact_shifted_coefficients(
    xi_nodes: &Array1<f64>,
    k: f64,
    displacement: f64,
) -> Array3<f64> {
    let mut exact = Array3::zeros((ELEMENTS, xi_nodes.len(), 1));
    for elem in 0..ELEMENTS {
        for node in 0..xi_nodes.len() {
            let x = physical_coordinate(elem, xi_nodes[node]);
            exact[[elem, node, 0]] = (k * (x - displacement)).sin();
        }
    }
    exact
}

pub(crate) fn exact_shifted_characteristic(
    xi_nodes: &Array1<f64>,
    k: f64,
    displacement: f64,
) -> Array3<f64> {
    &exact_shifted_coefficients(xi_nodes, k, displacement) * 2.0
}

pub(crate) fn reflect_coefficients(coeffs: &Array3<f64>) -> Array3<f64> {
    let mut reflected = Array3::zeros(coeffs.shape());
    let n_nodes = coeffs.shape()[1];
    for elem in 0..ELEMENTS {
        for node in 0..n_nodes {
            reflected[[elem, node, 0]] = coeffs[[ELEMENTS - 1 - elem, n_nodes - 1 - node, 0]];
        }
    }
    reflected
}

pub(crate) fn pressure_velocity_from_characteristics(
    w_plus: &Array3<f64>,
    w_minus: &Array3<f64>,
) -> (Array3<f64>, Array3<f64>) {
    let pressure = &(w_plus + w_minus) * 0.5;
    let velocity = &(w_plus - w_minus) / (2.0 * DENSITY * SOUND_SPEED);
    (pressure, velocity)
}

pub(crate) fn exact_bidirectional_acoustic(
    xi_nodes: &Array1<f64>,
    k: f64,
    displacement: f64,
) -> (Array3<f64>, Array3<f64>) {
    let mut w_plus = Array3::zeros((ELEMENTS, xi_nodes.len(), 1));
    let mut w_minus = Array3::zeros((ELEMENTS, xi_nodes.len(), 1));
    for elem in 0..ELEMENTS {
        for node in 0..xi_nodes.len() {
            let x = physical_coordinate(elem, xi_nodes[node]);
            w_plus[[elem, node, 0]] = (k * (x - displacement)).sin();
            w_minus[[elem, node, 0]] = (k * (x + displacement)).sin();
        }
    }
    pressure_velocity_from_characteristics(&w_plus, &w_minus)
}

pub(crate) fn physical_coordinate(elem: usize, xi: f64) -> f64 {
    2.0 * elem as f64 + xi + 1.0
}

pub(crate) fn pressure_from_characteristic(characteristic: &Array3<f64>) -> Array3<f64> {
    characteristic * 0.5
}

pub(crate) fn velocity_from_characteristic(characteristic: &Array3<f64>) -> Array3<f64> {
    characteristic / (2.0 * DENSITY * SOUND_SPEED)
}
