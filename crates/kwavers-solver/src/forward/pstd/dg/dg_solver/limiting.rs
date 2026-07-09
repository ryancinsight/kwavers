//! Troubled-cell limiting for DG time stepping.

use super::super::config::DGConfig;
use super::super::flux::apply_limiter;
use super::topology::DgTopology;
use kwavers_core::constants::numerical::EPSILON;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use leto::{
    Array1,
    Array3,
};

pub(super) fn apply_shock_capture_to_coeffs(
    config: DGConfig,
    xi_nodes: &Array1<f64>,
    weights: &Array1<f64>,
    coeffs: &mut Array3<f64>,
    scratch: &mut Array3<f64>,
) -> KwaversResult<()> {
    if !shock_capture_enabled(config) {
        return Ok(());
    }
    if coeffs.shape() != scratch.shape() {
        return Err(KwaversError::Validation(
            ValidationError::DimensionMismatch {
                expected: format!("{:?}", coeffs.shape()),
                actual: format!("{:?}", scratch.shape()),
            },
        ));
    }

    scratch.assign(coeffs);
    let [n_elements, n_nodes, n_vars] = coeffs.shape();
    if n_elements == 0 || n_nodes == 0 || n_vars == 0 {
        return Ok(());
    }
    if (xi_nodes.shape()[0] * xi_nodes.shape()[1] * xi_nodes.shape()[2]) != n_nodes || (weights.shape()[0] * weights.shape()[1] * weights.shape()[2]) != n_nodes {
        return Err(KwaversError::Validation(
            ValidationError::DimensionMismatch {
                expected: format!("{} DG nodes", n_nodes),
                actual: format!("xi={}, weights={}", (xi_nodes.shape()[0] * xi_nodes.shape()[1] * xi_nodes.shape()[2]), (weights.shape()[0] * weights.shape()[1] * weights.shape()[2])),
            },
        ));
    }

    let threshold = shock_threshold(config);
    let xi_mean = weighted_node_mean(xi_nodes, weights, n_nodes);
    for var in 0..n_vars {
        for elem in 0..n_elements {
            let mean = element_mean(coeffs, weights, elem, var, n_nodes);
            let left_elem = if elem == 0 { n_elements - 1 } else { elem - 1 };
            let right_elem = (elem + 1) % n_elements;
            let left_mean = element_mean(coeffs, weights, left_elem, var, n_nodes);
            let right_mean = element_mean(coeffs, weights, right_elem, var, n_nodes);
            let left_jump = mean - left_mean;
            let right_jump = right_mean - mean;
            let internal_variation = max_internal_variation(coeffs, elem, var, n_nodes, mean);
            let scale = mean
                .abs()
                .max(left_mean.abs())
                .max(right_mean.abs())
                .max(internal_variation)
                .max(EPSILON);
            let indicator = left_jump
                .abs()
                .max(right_jump.abs())
                .max(internal_variation)
                / scale;

            if indicator > threshold {
                let limited_slope = apply_limiter(config.limiter_type, left_jump, right_jump);
                for node in 0..n_nodes {
                    scratch[(elem, node, var)] =
                        mean + 0.5 * (xi_nodes[node] - xi_mean) * limited_slope;
                }
            }
        }
    }
    coeffs.assign(scratch);
    Ok(())
}

pub(super) fn apply_shock_capture_to_tensor_coeffs(
    config: DGConfig,
    topology: DgTopology,
    weights: &Array1<f64>,
    coeffs: &mut Array3<f64>,
    scratch: &mut Array3<f64>,
) -> KwaversResult<()> {
    if !shock_capture_enabled(config) {
        return Ok(());
    }
    if coeffs.shape() != scratch.shape() {
        return Err(KwaversError::Validation(
            ValidationError::DimensionMismatch {
                expected: format!("{:?}", coeffs.shape()),
                actual: format!("{:?}", scratch.shape()),
            },
        ));
    }
    if coeffs.shape()[0] != topology.n_elements || coeffs.shape()[1] != topology.nodes_per_element {
        return Err(KwaversError::Validation(
            ValidationError::DimensionMismatch {
                expected: format!(
                    "({}, {}, n_vars)",
                    topology.n_elements, topology.nodes_per_element
                ),
                actual: format!("{:?}", coeffs.shape()),
            },
        ));
    }
    if (weights.shape()[0] * weights.shape()[1] * weights.shape()[2]) != topology.n_nodes {
        return Err(KwaversError::Validation(
            ValidationError::DimensionMismatch {
                expected: format!("{} one-dimensional DG weights", topology.n_nodes),
                actual: format!("{}", (weights.shape()[0] * weights.shape()[1] * weights.shape()[2])),
            },
        ));
    }

    scratch.assign(coeffs);
    let [_, _, n_vars] = coeffs.shape();
    let threshold = shock_threshold(config);
    for var in 0..n_vars {
        for elem in 0..topology.n_elements {
            let mean = tensor_element_mean(coeffs, weights, topology, elem, var);
            let neighbor_jump = max_neighbor_mean_jump(coeffs, weights, topology, elem, var, mean);
            let internal_variation = tensor_internal_variation(coeffs, topology, elem, var, mean);
            let scale = mean
                .abs()
                .max(neighbor_jump)
                .max(internal_variation)
                .max(EPSILON);
            let indicator = neighbor_jump.max(internal_variation) / scale;

            if indicator > threshold {
                for node in 0..topology.nodes_per_element {
                    scratch[(elem, node, var)] = mean;
                }
            }
        }
    }
    coeffs.assign(scratch);
    Ok(())
}

fn shock_capture_enabled(config: DGConfig) -> bool {
    config.shock_capture.enabled || config.use_limiter
}

fn shock_threshold(config: DGConfig) -> f64 {
    if config.shock_capture.enabled {
        config.shock_capture.threshold
    } else {
        config.shock_threshold
    }
    .max(EPSILON)
}

fn element_mean(
    coeffs: &Array3<f64>,
    weights: &Array1<f64>,
    elem: usize,
    var: usize,
    n_nodes: usize,
) -> f64 {
    let weighted_sum: f64 = (0..n_nodes)
        .map(|node| weights[node] * coeffs[(elem, node, var)])
        .sum();
    weighted_sum / weight_sum(weights, n_nodes)
}

fn weighted_node_mean(xi_nodes: &Array1<f64>, weights: &Array1<f64>, n_nodes: usize) -> f64 {
    let weighted_sum: f64 = (0..n_nodes)
        .map(|node| weights[node] * xi_nodes[node])
        .sum();
    weighted_sum / weight_sum(weights, n_nodes)
}

fn weight_sum(weights: &Array1<f64>, n_nodes: usize) -> f64 {
    (0..n_nodes).map(|node| weights[node]).sum::<f64>()
}

fn max_internal_variation(
    coeffs: &Array3<f64>,
    elem: usize,
    var: usize,
    n_nodes: usize,
    mean: f64,
) -> f64 {
    (0..n_nodes)
        .map(|node| (coeffs[(elem, node, var)] - mean).abs())
        .fold(0.0, f64::max)
}

fn tensor_element_mean(
    coeffs: &Array3<f64>,
    weights: &Array1<f64>,
    topology: DgTopology,
    elem: usize,
    var: usize,
) -> f64 {
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;
    for node in 0..topology.nodes_per_element {
        let weight = topology.node_weight(node, weights);
        weighted_sum += weight * coeffs[(elem, node, var)];
        total_weight += weight;
    }
    weighted_sum / total_weight
}

fn max_neighbor_mean_jump(
    coeffs: &Array3<f64>,
    weights: &Array1<f64>,
    topology: DgTopology,
    elem: usize,
    var: usize,
    mean: f64,
) -> f64 {
    let mut max_jump = 0.0;
    for axis in 0..3 {
        if !topology.active_axes[axis] {
            continue;
        }
        let left = topology.neighbor(elem, axis, false);
        let right = topology.neighbor(elem, axis, true);
        let left_mean = tensor_element_mean(coeffs, weights, topology, left, var);
        let right_mean = tensor_element_mean(coeffs, weights, topology, right, var);
        let jump = (mean - left_mean).abs().max((right_mean - mean).abs());
        if jump > max_jump {
            max_jump = jump;
        }
    }
    max_jump
}

fn tensor_internal_variation(
    coeffs: &Array3<f64>,
    topology: DgTopology,
    elem: usize,
    var: usize,
    mean: f64,
) -> f64 {
    (0..topology.nodes_per_element)
        .map(|node| (coeffs[(elem, node, var)] - mean).abs())
        .fold(0.0, f64::max)
}
