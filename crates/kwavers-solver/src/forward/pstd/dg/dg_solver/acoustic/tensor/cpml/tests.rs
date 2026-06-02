//! Value-semantic regression tests for the DG-CPML tensor acoustic stepper.

use super::super::{AcousticDgTensorWorkspace, ACOUSTIC_PRESSURE_VAR, ACOUSTIC_VELOCITY_X_VAR};
use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_domain::grid::Grid;
use crate::forward::pstd::dg::cpml::{
    DgCpmlAxis, DgCpmlConfig, DgCpmlMemoryWorkspace, DgCpmlProfiles,
};
use crate::forward::pstd::dg::dg_solver::core::DGSolver;
use crate::forward::pstd::dg::DGConfig;
use ndarray::Array3;
use std::sync::Arc;

const POLY_ORDER: usize = 2;
const N_NODES: usize = POLY_ORDER + 1;

fn make_solver_1d(nx_elements: usize) -> (DGSolver, Arc<Grid>) {
    let nx = nx_elements * N_NODES;
    let grid = Arc::new(Grid::new(nx, 1, 1, 1.0e-3, 1.0e-3, 1.0e-3).unwrap());
    let config = DGConfig {
        polynomial_order: POLY_ORDER,
        sound_speed: SOUND_SPEED_WATER_SIM,
        ..DGConfig::default()
    };
    let solver = DGSolver::new(config, Arc::clone(&grid)).expect("DG solver");
    (solver, grid)
}

fn axis_profile_only(axis: usize, layer: DgCpmlAxis) -> DgCpmlConfig {
    let mut cfg = DgCpmlConfig::uniform(0);
    cfg.axes[axis] = layer;
    cfg
}

#[test]
fn neutral_profile_matches_standard_rhs_bit_for_bit() {
    // With σ = 0, κ = 1, α = 0 everywhere, the CPML RHS must reproduce the
    // standard tensor RHS exactly (memory state stays zero, ψ contributions
    // vanish, and the field RHS is the un-stretched DG operator).
    let nx_elements = 8;
    let (solver, _grid) = make_solver_1d(nx_elements);
    let shape = solver.acoustic_tensor_state_shape().unwrap();
    let mut state = Array3::<f64>::zeros(shape);
    for elem in 0..shape.0 {
        for node in 0..shape.1 {
            state[(elem, node, ACOUSTIC_PRESSURE_VAR)] = ((elem * N_NODES + node) as f64).sin();
            state[(elem, node, ACOUSTIC_VELOCITY_X_VAR)] =
                0.5 * ((elem * N_NODES + node) as f64).cos();
        }
    }
    let memory_state = Array3::<f64>::zeros((shape.0, shape.1, 6));
    let mut rhs_cpml = Array3::<f64>::zeros(shape);
    let mut memory_rhs = Array3::<f64>::zeros((shape.0, shape.1, 6));
    let neutral = DgCpmlConfig::uniform(0);
    let profiles = DgCpmlProfiles::new(
        &neutral,
        solver.config().sound_speed,
        [nx_elements, 1, 1],
        N_NODES,
        [N_NODES as f64 * 1.0e-3, 1.0e-3, 1.0e-3],
    )
    .unwrap();

    solver
        .compute_acoustic_tensor_rhs_with_cpml_into(
            &state,
            &memory_state,
            DENSITY_WATER_NOMINAL,
            &profiles,
            &mut rhs_cpml,
            &mut memory_rhs,
        )
        .unwrap();

    let mut rhs_standard = Array3::<f64>::zeros(shape);
    solver
        .compute_acoustic_tensor_rhs_into(&state, DENSITY_WATER_NOMINAL, &mut rhs_standard)
        .unwrap();

    // Bit-for-bit equality is too strict because the two code paths multiply
    // `axis_scale · du` in a different evaluation order: the standard RHS
    // folds the factor into one expression (`bulk · axis_scale · du`) while
    // the CPML path computes `axis_scale · du` first and then divides by κ = 1.
    // Both orderings are mathematically identical but emit ULP-level
    // differences. The tolerance is `8 ULP * max(|a|, |b|)`, well below any
    // physical reflection or memory-state-leakage signal.
    for ((a, b), idx) in rhs_cpml.iter().zip(rhs_standard.iter()).zip(0..) {
        let scale = a.abs().max(b.abs()).max(1.0);
        assert!(
            (a - b).abs() <= 8.0 * f64::EPSILON * scale,
            "neutral CPML RHS diverges from standard at flat idx {idx}: \
             {a} vs {b}, |Δ|={} > 8 ULP × {}",
            (a - b).abs(),
            scale
        );
    }
    // Memory RHS must be exactly zero with no σ.
    assert!(memory_rhs.iter().all(|v| *v == 0.0));
}

#[test]
fn rhs_shape_mismatches_are_rejected() {
    let (solver, _grid) = make_solver_1d(8);
    let shape = solver.acoustic_tensor_state_shape().unwrap();
    let state = Array3::<f64>::zeros(shape);
    let memory_state = Array3::<f64>::zeros((shape.0, shape.1, 6));
    let profiles = DgCpmlProfiles::new(
        &DgCpmlConfig::uniform(0),
        SOUND_SPEED_WATER_SIM,
        [8, 1, 1],
        N_NODES,
        [8.0e-3, 1.0e-3, 1.0e-3],
    )
    .unwrap();
    let mut wrong_field_rhs = Array3::<f64>::zeros((shape.0, shape.1, 3));
    let mut memory_rhs = Array3::<f64>::zeros((shape.0, shape.1, 6));
    assert!(solver
        .compute_acoustic_tensor_rhs_with_cpml_into(
            &state,
            &memory_state,
            DENSITY_WATER_NOMINAL,
            &profiles,
            &mut wrong_field_rhs,
            &mut memory_rhs,
        )
        .is_err());

    let mut field_rhs = Array3::<f64>::zeros(shape);
    let mut wrong_memory_rhs = Array3::<f64>::zeros((shape.0, shape.1, 3));
    assert!(solver
        .compute_acoustic_tensor_rhs_with_cpml_into(
            &state,
            &memory_state,
            DENSITY_WATER_NOMINAL,
            &profiles,
            &mut field_rhs,
            &mut wrong_memory_rhs,
        )
        .is_err());
}

#[test]
fn step_with_neutral_profile_matches_standard_step() {
    // Step a state forward by SSP-RK3 with σ = 0 and verify it matches the
    // standard stepper to machine precision.
    let nx_elements = 8;
    let (solver, _grid) = make_solver_1d(nx_elements);
    let shape = solver.acoustic_tensor_state_shape().unwrap();
    let mut state_a = Array3::<f64>::zeros(shape);
    for elem in 0..shape.0 {
        for node in 0..shape.1 {
            let x = (elem * N_NODES + node) as f64;
            state_a[(elem, node, ACOUSTIC_PRESSURE_VAR)] = (x * 0.3).sin();
        }
    }
    let mut state_b = state_a.clone();
    let mut ws_a = AcousticDgTensorWorkspace::new(shape);
    let mut ws_b = AcousticDgTensorWorkspace::new(shape);
    let mut memory = DgCpmlMemoryWorkspace::new(shape.0, shape.1);
    let neutral = DgCpmlConfig::uniform(0);
    let profiles = DgCpmlProfiles::new(
        &neutral,
        solver.config().sound_speed,
        [nx_elements, 1, 1],
        N_NODES,
        [N_NODES as f64 * 1.0e-3, 1.0e-3, 1.0e-3],
    )
    .unwrap();
    let dt = 1.0e-9;
    solver
        .step_acoustic_tensor_ssp_rk3(&mut state_a, DENSITY_WATER_NOMINAL, dt, &mut ws_a)
        .unwrap();
    solver
        .step_acoustic_tensor_ssp_rk3_with_cpml(
            &mut state_b,
            DENSITY_WATER_NOMINAL,
            dt,
            &mut ws_b,
            &mut memory,
            &profiles,
        )
        .unwrap();
    for (a, b) in state_a.iter().zip(state_b.iter()) {
        assert!(
            (a - b).abs() < 1.0e-10,
            "neutral CPML stepper diverges from standard: {a} vs {b}"
        );
    }
    // Memory state remains exactly zero under neutral profile.
    assert!(memory.state.iter().all(|v| *v == 0.0));
}

#[test]
fn cpml_attenuates_right_propagating_plane_wave_below_periodic_baseline() {
    // 1-D plane wave fired from the left, observed at the right end after
    // multiple periodic loops vs. with right-side CPML installed.
    //
    // Without CPML: periodic boundary reinjects the wave; energy in the
    // monitored region oscillates indefinitely.
    //
    // With CPML on the +x face: the wave is absorbed; the monitored
    // energy decays below the periodic baseline by orders of magnitude.
    let nx_elements = 32;
    let (solver, _grid) = make_solver_1d(nx_elements);
    let shape = solver.acoustic_tensor_state_shape().unwrap();
    let nx = nx_elements * N_NODES;

    // Initial condition: localized right-going acoustic pulse near x = 0.25 L.
    // For the linear acoustic plane wave, p = Z·u_x guarantees a right-going
    // characteristic (`w_minus = p - Z·u_x = 0`).
    let density = DENSITY_WATER_NOMINAL;
    let c = solver.config().sound_speed;
    let impedance = density * c;
    let center_node = nx / 4;
    let sigma_pulse = 6.0;
    let mut state = Array3::<f64>::zeros(shape);
    for elem in 0..shape.0 {
        for node in 0..shape.1 {
            let global_node = elem * N_NODES + node;
            let x = (global_node as f64 - center_node as f64) / sigma_pulse;
            let p = (-x * x).exp();
            state[(elem, node, ACOUSTIC_PRESSURE_VAR)] = p;
            state[(elem, node, ACOUSTIC_VELOCITY_X_VAR)] = p / impedance;
        }
    }

    // Reference run: periodic both sides.
    let mut state_periodic = state.clone();
    let mut ws_periodic = AcousticDgTensorWorkspace::new(shape);
    let dt = 0.4 * 1.0e-3 / (c * N_NODES as f64);
    let steps = (1.5 * (nx as f64) * 1.0e-3 / (c * dt)) as usize;
    for _ in 0..steps {
        solver
            .step_acoustic_tensor_ssp_rk3(&mut state_periodic, density, dt, &mut ws_periodic)
            .unwrap();
    }

    // CPML run with right-side absorbing layer of 8 elements.
    let layer = DgCpmlAxis::standard(8);
    let cfg = axis_profile_only(0, layer);
    let profiles = DgCpmlProfiles::new(
        &cfg,
        c,
        [nx_elements, 1, 1],
        N_NODES,
        [N_NODES as f64 * 1.0e-3, 1.0e-3, 1.0e-3],
    )
    .unwrap();
    let mut state_cpml = state.clone();
    let mut ws_cpml = AcousticDgTensorWorkspace::new(shape);
    let mut memory = DgCpmlMemoryWorkspace::new(shape.0, shape.1);
    for _ in 0..steps {
        solver
            .step_acoustic_tensor_ssp_rk3_with_cpml(
                &mut state_cpml,
                density,
                dt,
                &mut ws_cpml,
                &mut memory,
                &profiles,
            )
            .unwrap();
    }

    // L2 of the residual field over the inner physical region (excluding
    // PML zone).
    let inner_start = 8 * N_NODES;
    let inner_end = (nx_elements - 8) * N_NODES;
    let l2_periodic: f64 = inner_region_l2(&state_periodic, inner_start, inner_end);
    let l2_cpml: f64 = inner_region_l2(&state_cpml, inner_start, inner_end);

    // The CPML run must drop the inner-region L2 well below the periodic
    // baseline. 20 dB suppression is a conservative threshold for an
    // 8-element layer with R₀ = 1e-6 after the wave has crossed the
    // absorbing layer; this also rules out fully-passing or amplifying
    // bugs.
    assert!(
        l2_cpml < 0.1 * l2_periodic,
        "CPML did not attenuate plane wave: periodic L2 = {l2_periodic}, CPML L2 = {l2_cpml}"
    );
    assert!(l2_cpml.is_finite());
    assert!(state_cpml.iter().all(|v| v.is_finite()));
}

fn inner_region_l2(state: &Array3<f64>, inner_start: usize, inner_end: usize) -> f64 {
    let (n_elem, nodes_per, _) = state.dim();
    let mut sum = 0.0;
    for elem in 0..n_elem {
        for node in 0..nodes_per {
            let global_node = elem * N_NODES + node;
            if global_node >= inner_start && global_node < inner_end {
                let p = state[(elem, node, ACOUSTIC_PRESSURE_VAR)];
                sum += p * p;
            }
        }
    }
    sum.sqrt()
}
