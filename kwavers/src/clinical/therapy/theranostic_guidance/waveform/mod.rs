//! Source-encoded time-domain receive simulation for same-device guidance.
//!
//! The solver in this module is deliberately separate from the reduced
//! finite-frequency normal-equation inverse. It integrates the two-dimensional
//! scalar acoustic wave equation on the CT-derived slice, records receiver
//! traces on the treatment/imaging aperture, and backpropagates the trace
//! residual to produce an adjoint RTM image. It is not nonlinear wave
//! propagation and it is not an iterative full-waveform inversion.
//!
//! # Numerical methods
//!
//! - **4th-order spatial FD stencil** (Fornberg 1988, Table 2, row k=2)
//! - **CPML absorbing boundaries** (Komatitsch & Martin 2007, Geophysics 72:SM155)
//! - **Rayon** inner-loop parallelism over the ix (row) dimension
//! - **Power-law attenuation** (Treeby & Cox 2010, per-cell amplitude decay)
//! - **Checkpointed adjoint** (Griewank 1992) — O(√T · N) memory vs O(T · N)

mod adjoint;
mod forward;
mod grid;
mod medium;
mod types;
mod utils;

pub use types::WaveformSimulationResult;

use super::config::TheranosticInverseConfig;
use super::exposure::normalize_positive;
use super::geometry::DeviceLayout;
use super::medium::PreparedTheranosticSlice;
use super::misfit::evaluate_trace_residual;
use adjoint::adjoint_image;
use forward::propagate;
use grid::acoustic_grid;
use medium::lesion_speed;
use utils::energy;

pub const THERANOSTIC_WAVEFORM_MODEL: &str = "source_encoded_time_domain_acoustic_adjoint_rtm";

/// Run source-encoded forward modeling and adjoint residual backpropagation.
///
/// # Mathematical contract
///
/// Forward: `p_tt = c(x)² Δ₄p + s(x,t)` — 4th-order FD Laplacian
/// (Fornberg 1988), CPML absorbing boundary conditions
/// (Komatitsch & Martin 2007), and per-cell amplitude attenuation
/// (Treeby & Cox 2010).
///
/// Adjoint imaging condition: `I(x) = Σ_t p_fwd(x,t) · p_adj(x,t)`.
///
/// Memory: checkpointing (Griewank 1992) reduces forward storage from
/// O(T·N²) to O(√T·N²) at the cost of ≈1.5 forward passes.
#[must_use]
pub fn simulate_waveform_adjoint_rtm(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    config: &TheranosticInverseConfig,
    lesion: &ndarray::Array2<f64>,
) -> WaveformSimulationResult {
    let true_speed = lesion_speed(prepared, config, lesion);
    let grid = acoustic_grid(
        prepared,
        layout,
        config,
        &prepared.sound_speed_m_s,
        &true_speed,
    );
    let observed = propagate(&grid, &true_speed, config, false);
    let predicted = propagate(&grid, &prepared.sound_speed_m_s, config, true);
    let residual = evaluate_trace_residual(
        &observed.traces,
        &predicted.traces,
        config.waveform_misfit,
        config.waveform_misfit_scale_fraction,
    );
    let residual_energy = energy(&residual.adjoint_source);
    let observed_energy = energy(&observed.traces);
    let checkpoints = predicted
        .checkpoints
        .as_ref()
        .expect("forward checkpoints required for adjoint imaging");
    let image = adjoint_image(
        &grid,
        &prepared.sound_speed_m_s,
        &residual.adjoint_source,
        checkpoints,
        predicted.checkpoint_interval,
    );
    let reconstruction = normalize_positive(&image, &prepared.body_mask);
    WaveformSimulationResult {
        reconstruction,
        residual_energy,
        observed_energy,
        receiver_count: grid.receiver_cells.len(),
        time_steps: grid.time_steps,
        dt_s: grid.dt_s,
        model_name: THERANOSTIC_WAVEFORM_MODEL,
        misfit_name: residual.misfit.label(),
        misfit_scale: residual.scale,
        objective_value: residual.objective_value,
    }
}
