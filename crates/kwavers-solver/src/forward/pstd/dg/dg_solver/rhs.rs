//! DG semi-discrete RHS assembly for line and tensor-product layouts.
//!
//! ## Conservation Contract
//!
//! For periodic scalar advection, the semi-discrete DG operator must satisfy
//! `sum_e sum_i w_i L(u)_{e,i} = 0` for the one-dimensional line layout and
//! `sum_e sum_n W_n L(u)_{e,n} = 0` for tensor-product layouts, where `W_n` is
//! the product quadrature weight for the active coordinate axes. The volume
//! derivative contributes only boundary traces under the SBP identity for GLL
//! nodes, and the left/right Lax-Friedrichs residuals telescope pairwise across
//! periodic element interfaces.
//!
//! The left face residual uses `f* - c u_left_interior`, not its negation. This
//! sign is required for the weighted global mass derivative to vanish.
//!
//! References: Hesthaven & Warburton (2008) §3, §6.3; Cockburn & Shu (2001)
//! §4; Kopriva (2009) §3.4.

use super::topology::{CoefficientLayout, DgTopology};
use leto::{Array2, Array3};

pub(super) struct RhsOperator<'a> {
    pub(super) n_nodes: usize,
    pub(super) d_matrix: &'a Array2<f64>,
    pub(super) lift: &'a Array2<f64>,
    pub(super) layout: CoefficientLayout,
    pub(super) wave_speed: f64,
    pub(super) axis_scales: [f64; 3],
}

pub(super) fn compute_rhs_from_coeffs_into(
    operator: RhsOperator<'_>,
    coeffs: &Array3<f64>,
    rhs: &mut Array3<f64>,
) {
    match operator.layout {
        CoefficientLayout::TensorProduct(topology)
            if coeffs.shape()[1] == topology.nodes_per_element =>
        {
            compute_tensor_rhs(&operator, topology, coeffs, rhs);
        }
        _ => compute_line_rhs(&operator, coeffs, rhs),
    }
}

fn compute_line_rhs(operator: &RhsOperator<'_>, coeffs: &Array3<f64>, rhs: &mut Array3<f64>) {
    rhs.fill(0.0);
    let n_elements = coeffs.shape()[0];
    let n_vars = coeffs.shape()[2];
    let n_face = operator.lift.shape()[1];

    for var in 0..n_vars {
        for elem in 0..n_elements {
            for i in 0..operator.n_nodes {
                let mut du = 0.0;
                for j in 0..operator.n_nodes {
                    du += operator.d_matrix[[i, j]] * coeffs[[elem, j, var]];
                }
                rhs[[elem, i, var]] -= operator.wave_speed * du;
            }
        }
    }

    for var in 0..n_vars {
        for elem in 0..n_elements {
            let left_elem = if elem == 0 { n_elements - 1 } else { elem - 1 };
            let right_elem = (elem + 1) % n_elements;

            let u_minus_left = coeffs[[left_elem, operator.n_nodes - 1, var]];
            let u_plus_left = coeffs[[elem, 0, var]];
            let u_minus_right = coeffs[[elem, operator.n_nodes - 1, var]];
            let u_plus_right = coeffs[[right_elem, 0, var]];

            let flux_left = lax_friedrichs_flux(operator.wave_speed, u_minus_left, u_plus_left);
            let flux_right = lax_friedrichs_flux(operator.wave_speed, u_minus_right, u_plus_right);
            let face_res_left = flux_left - operator.wave_speed * u_plus_left;
            let face_res_right = flux_right - operator.wave_speed * u_minus_right;

            for i in 0..operator.n_nodes {
                if n_face > 0 {
                    rhs[[elem, i, var]] += operator.lift[[i, 0]] * face_res_left;
                }
                if n_face > 1 {
                    rhs[[elem, i, var]] += operator.lift[[i, 1]] * face_res_right;
                }
            }
        }
    }
}

fn compute_tensor_rhs(
    operator: &RhsOperator<'_>,
    topology: DgTopology,
    coeffs: &Array3<f64>,
    rhs: &mut Array3<f64>,
) {
    rhs.fill(0.0);
    let n_vars = coeffs.shape()[2];

    for var in 0..n_vars {
        for elem in 0..topology.n_elements {
            for node in 0..topology.nodes_per_element {
                let node_coords = topology.node_coords(node);
                for (axis, axis_scale) in operator.axis_scales.iter().copied().enumerate() {
                    if !topology.active_axes[axis] {
                        continue;
                    }
                    let mut du = 0.0;
                    for j in 0..operator.n_nodes {
                        let source_node = topology.node_with_axis(node, axis, j);
                        du += operator.d_matrix[[node_coords[axis], j]]
                            * coeffs[[elem, source_node, var]];
                    }
                    rhs[[elem, node, var]] -= operator.wave_speed * axis_scale * du;
                }
            }
        }
    }

    for var in 0..n_vars {
        for elem in 0..topology.n_elements {
            for axis in 0..3 {
                if !topology.active_axes[axis] {
                    continue;
                }
                add_axis_surface_flux(operator, topology, coeffs, elem, axis, var, rhs);
            }
        }
    }
}

fn add_axis_surface_flux(
    operator: &RhsOperator<'_>,
    topology: DgTopology,
    coeffs: &Array3<f64>,
    elem: usize,
    axis: usize,
    var: usize,
    rhs: &mut Array3<f64>,
) {
    let left_elem = topology.neighbor(elem, axis, false);
    let right_elem = topology.neighbor(elem, axis, true);

    for transverse in 0..topology.nodes_per_element {
        let transverse_coords = topology.node_coords(transverse);
        if transverse_coords[axis] != 0 {
            continue;
        }

        let left_node = topology.node_with_axis(transverse, axis, 0);
        let right_node = topology.node_with_axis(transverse, axis, operator.n_nodes - 1);

        let u_minus_left = coeffs[[left_elem, right_node, var]];
        let u_plus_left = coeffs[[elem, left_node, var]];
        let u_minus_right = coeffs[[elem, right_node, var]];
        let u_plus_right = coeffs[[right_elem, left_node, var]];

        let flux_left = lax_friedrichs_flux(operator.wave_speed, u_minus_left, u_plus_left);
        let flux_right = lax_friedrichs_flux(operator.wave_speed, u_minus_right, u_plus_right);
        let face_res_left = flux_left - operator.wave_speed * u_plus_left;
        let face_res_right = flux_right - operator.wave_speed * u_minus_right;

        for i in 0..operator.n_nodes {
            let lifted_node = topology.node_with_axis(transverse, axis, i);
            rhs[[elem, lifted_node, var]] += operator.axis_scales[axis]
                * (operator.lift[[i, 0]] * face_res_left + operator.lift[[i, 1]] * face_res_right);
        }
    }
}

#[inline]
fn lax_friedrichs_flux(wave_speed: f64, left: f64, right: f64) -> f64 {
    (0.5 * wave_speed).mul_add(left + right, -(0.5 * wave_speed * (right - left)))
}
