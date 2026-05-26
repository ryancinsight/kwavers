//! Acoustic tensor DG face-state policies.
//!
//! ## Characteristic absorbing boundary theorem
//!
//! For the one-dimensional acoustic subsystem along an active axis,
//! `p_t + K u_x = 0`, `u_t + rho^{-1} p_x = 0`, with impedance
//! `Z = rho c`, the characteristic variables are
//! `w_plus = p + Z u` and `w_minus = p - Z u`. `w_plus` propagates in the
//! positive coordinate direction and `w_minus` propagates in the negative
//! coordinate direction. A non-reflecting exterior state at the negative
//! boundary sets incoming `w_plus = 0` and preserves outgoing `w_minus`; at the
//! positive boundary it sets incoming `w_minus = 0` and preserves outgoing
//! `w_plus`. This is exact for normally incident linear acoustic plane waves.

use super::{velocity_var, ACOUSTIC_PRESSURE_VAR};
use crate::solver::forward::pstd::dg::config::DgBoundaryCondition;
use crate::solver::forward::pstd::dg::dg_solver::core::DGSolver;
use crate::solver::forward::pstd::dg::dg_solver::topology::DgTopology;
use ndarray::Array3;

#[derive(Debug, Clone, Copy)]
pub(super) struct NormalState {
    pub(super) pressure: f64,
    pub(super) velocity_normal: f64,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn add_axis_surface_flux(
    solver: &DGSolver,
    topology: DgTopology,
    state: &Array3<f64>,
    elem: usize,
    axis: usize,
    bulk: f64,
    inv_density: f64,
    rhs: &mut Array3<f64>,
) {
    let velocity_var = velocity_var(axis);
    let axis_scale = solver.acoustic_axis_scales()[axis];
    let wave_speed = solver.config().sound_speed;

    for transverse in 0..topology.nodes_per_element {
        let transverse_coords = topology.node_coords(transverse);
        if transverse_coords[axis] != 0 {
            continue;
        }

        let left_node = topology.node_with_axis(transverse, axis, 0);
        let right_node = topology.node_with_axis(transverse, axis, solver.n_nodes - 1);
        let left_interior = normal_state(state, elem, left_node, velocity_var);
        let right_interior = normal_state(state, elem, right_node, velocity_var);
        let left_exterior = exterior_normal_state(
            solver,
            topology,
            state,
            elem,
            axis,
            false,
            left_node,
            right_node,
            velocity_var,
            inv_density,
            left_interior,
        );
        let right_exterior = exterior_normal_state(
            solver,
            topology,
            state,
            elem,
            axis,
            true,
            left_node,
            right_node,
            velocity_var,
            inv_density,
            right_interior,
        );

        let flux_left =
            lax_friedrichs_flux(wave_speed, bulk, inv_density, left_exterior, left_interior);
        let flux_right = lax_friedrichs_flux(
            wave_speed,
            bulk,
            inv_density,
            right_interior,
            right_exterior,
        );
        let interior_flux_left = acoustic_flux(bulk, inv_density, left_interior);
        let interior_flux_right = acoustic_flux(bulk, inv_density, right_interior);
        // Strong-form GLL DG uses opposite residual orientation on the two
        // faces so periodic interface fluxes telescope in the weighted sum.
        let face_res_left = NormalState {
            pressure: flux_left.pressure - interior_flux_left.pressure,
            velocity_normal: flux_left.velocity_normal - interior_flux_left.velocity_normal,
        };
        let face_res_right = NormalState {
            pressure: interior_flux_right.pressure - flux_right.pressure,
            velocity_normal: interior_flux_right.velocity_normal - flux_right.velocity_normal,
        };

        for i in 0..solver.n_nodes {
            let lifted_node = topology.node_with_axis(transverse, axis, i);
            rhs[(elem, lifted_node, ACOUSTIC_PRESSURE_VAR)] += axis_scale
                * (solver.lift_matrix[[i, 0]] * face_res_left.pressure
                    + solver.lift_matrix[[i, 1]] * face_res_right.pressure);
            rhs[(elem, lifted_node, velocity_var)] += axis_scale
                * (solver.lift_matrix[[i, 0]] * face_res_left.velocity_normal
                    + solver.lift_matrix[[i, 1]] * face_res_right.velocity_normal);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn exterior_normal_state(
    solver: &DGSolver,
    topology: DgTopology,
    state: &Array3<f64>,
    elem: usize,
    axis: usize,
    positive_face: bool,
    left_node: usize,
    right_node: usize,
    velocity_var: usize,
    inv_density: f64,
    interior: NormalState,
) -> NormalState {
    match solver.config().boundary_conditions[axis] {
        DgBoundaryCondition::Periodic => periodic_exterior_state(
            topology,
            state,
            elem,
            axis,
            positive_face,
            left_node,
            right_node,
            velocity_var,
        ),
        DgBoundaryCondition::AbsorbingCharacteristic => {
            if is_physical_boundary(topology, elem, axis, positive_face) {
                let impedance = solver.config().sound_speed / inv_density;
                absorbing_characteristic_external_state(interior, impedance, positive_face)
            } else {
                periodic_exterior_state(
                    topology,
                    state,
                    elem,
                    axis,
                    positive_face,
                    left_node,
                    right_node,
                    velocity_var,
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn periodic_exterior_state(
    topology: DgTopology,
    state: &Array3<f64>,
    elem: usize,
    axis: usize,
    positive_face: bool,
    left_node: usize,
    right_node: usize,
    velocity_var: usize,
) -> NormalState {
    let neighbor = topology.neighbor(elem, axis, positive_face);
    let node = if positive_face { left_node } else { right_node };
    normal_state(state, neighbor, node, velocity_var)
}

#[inline]
fn is_physical_boundary(
    topology: DgTopology,
    elem: usize,
    axis: usize,
    positive_face: bool,
) -> bool {
    let coords = topology.element_coords(elem);
    if positive_face {
        coords[axis] + 1 == topology.element_counts[axis]
    } else {
        coords[axis] == 0
    }
}

#[inline]
pub(super) fn absorbing_characteristic_external_state(
    interior: NormalState,
    impedance: f64,
    positive_face: bool,
) -> NormalState {
    if positive_face {
        let outgoing = interior.pressure + impedance * interior.velocity_normal;
        NormalState {
            pressure: 0.5 * outgoing,
            velocity_normal: 0.5 * outgoing / impedance,
        }
    } else {
        let outgoing = interior.pressure - impedance * interior.velocity_normal;
        NormalState {
            pressure: 0.5 * outgoing,
            velocity_normal: -0.5 * outgoing / impedance,
        }
    }
}

#[inline]
fn normal_state(state: &Array3<f64>, elem: usize, node: usize, velocity_var: usize) -> NormalState {
    NormalState {
        pressure: state[(elem, node, ACOUSTIC_PRESSURE_VAR)],
        velocity_normal: state[(elem, node, velocity_var)],
    }
}

#[inline]
fn acoustic_flux(bulk: f64, inv_density: f64, state: NormalState) -> NormalState {
    NormalState {
        pressure: bulk * state.velocity_normal,
        velocity_normal: inv_density * state.pressure,
    }
}

#[inline]
fn lax_friedrichs_flux(
    wave_speed: f64,
    bulk: f64,
    inv_density: f64,
    left: NormalState,
    right: NormalState,
) -> NormalState {
    let left_flux = acoustic_flux(bulk, inv_density, left);
    let right_flux = acoustic_flux(bulk, inv_density, right);
    NormalState {
        pressure: 0.5 * (left_flux.pressure + right_flux.pressure)
            - 0.5 * wave_speed * (right.pressure - left.pressure),
        velocity_normal: 0.5 * (left_flux.velocity_normal + right_flux.velocity_normal)
            - 0.5 * wave_speed * (right.velocity_normal - left.velocity_normal),
    }
}
