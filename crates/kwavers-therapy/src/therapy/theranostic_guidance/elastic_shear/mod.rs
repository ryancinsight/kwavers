//! Iterative nonlinear elastic FWI reconstruction for Chapter 29.
//!
//! The reconstruction uses the existing ElasticPSTD propagator instead of the
//! reduced acoustic same-aperture row operator. The observed data are generated
//! by a lesion-perturbed shear medium, and each FWI iteration resimulates the
//! current shear map, computes receiver residuals, migrates the residual with
//! `t = (min_s |s - x| + |x - x_r|) / c_s`, and accepts a bounded nonlinear
//! shear-speed update only when the receiver-trace objective decreases.

mod geometry;
mod inversion;
mod medium;
mod propagation;
mod sampling;
mod signal;
mod types;

pub use types::{ElasticShearReconstructionResult, THERANOSTIC_ELASTIC_SHEAR_MODEL};

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_solver::inverse::same_aperture::C_REF_M_S;
use leto::Array2;

use super::config::TheranosticInverseConfig;
use super::exposure::normalize_positive;
use super::geometry::DeviceLayout;
use super::medium::PreparedTheranosticSlice;
use geometry::{mask_indices, mask_points_m, source_mask};
use inversion::run_iterative_elastic_fwi;
use medium::{elastic_medium, receiver_mask};
use propagation::propagate_traces;
use sampling::{center_frequency, stable_time_step, time_steps, trace_energy};
use signal::velocity_source;

pub fn reconstruct_elastic_shear(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    config: &TheranosticInverseConfig,
    lesion_target: &Array2<f64>,
) -> KwaversResult<ElasticShearReconstructionResult> {
    let source_mask_2d = source_mask(&prepared.target_mask, prepared.spacing_m, layout.focus_m)?;
    let source_points = mask_points_m(&source_mask_2d, prepared.spacing_m)?;
    let recv_mask = receiver_mask(prepared, layout)?;
    let receiver_indices = mask_indices(&recv_mask);
    if receiver_indices.is_empty() {
        return Err(KwaversError::InvalidInput(
            "elastic shear reconstruction requires at least one receiver".to_owned(),
        ));
    }
    let receiver_points = receiver_indices
        .iter()
        .map(|(ix, iy, _)| {
            geometry::index_point_m(*ix, *iy, prepared.ct_hu.dim(), prepared.spacing_m)
        })
        .collect::<Vec<_>>();

    let lesion_fraction = config.lesion_delta_c_m_s / C_REF_M_S;
    let zero_model = Array2::<f64>::zeros(prepared.ct_hu.dim());
    let baseline = elastic_medium(prepared, &zero_model, config.elastic_shear_speed_m_s, 0.0);
    let observed_medium = elastic_medium(
        prepared,
        lesion_target,
        config.elastic_shear_speed_m_s,
        lesion_fraction,
    );
    let dt_s = stable_time_step(prepared.spacing_m, config.elastic_shear_speed_m_s);
    let center_frequency_hz = center_frequency(&config.elastic_frequencies_hz)?;
    let n_time = time_steps(
        &source_points,
        &receiver_points,
        config.elastic_shear_speed_m_s,
        center_frequency_hz,
        dt_s,
    );
    let source = velocity_source(
        &source_mask_2d,
        prepared.ct_hu.dim(),
        n_time,
        dt_s,
        center_frequency_hz,
        config.elastic_shear_speed_m_s,
    );
    let grid = Grid::new(
        prepared.ct_hu.dim().0,
        prepared.ct_hu.dim().1,
        1,
        prepared.spacing_m,
        prepared.spacing_m,
        prepared.spacing_m,
    )?;

    let baseline_traces = propagate_traces(&grid, baseline, dt_s, n_time, &source, &recv_mask)?;
    let lesion_traces =
        propagate_traces(&grid, observed_medium, dt_s, n_time, &source, &recv_mask)?;
    let inversion = run_iterative_elastic_fwi(
        prepared,
        config,
        &grid,
        &source,
        &recv_mask,
        &lesion_traces,
        &baseline_traces,
        dt_s,
        center_frequency_hz,
        &source_points,
        &receiver_points,
        lesion_fraction,
    )?;

    Ok(ElasticShearReconstructionResult {
        reconstruction: normalize_positive(&inversion.model, &prepared.body_mask),
        model_name: THERANOSTIC_ELASTIC_SHEAR_MODEL.to_owned(),
        receiver_count: receiver_points.len(),
        time_steps: n_time,
        dt_s,
        iteration_count: inversion.iteration_count,
        accepted_step_count: inversion.accepted_step_count,
        objective_history: inversion.objective_history,
        baseline_trace_energy: trace_energy(&baseline_traces),
        lesion_trace_energy: trace_energy(&lesion_traces),
        residual_trace_energy: inversion.final_residual_energy,
    })
}
