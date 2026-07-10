//! Source-encoded time-domain receive simulation for same-device guidance.
//!
//! The solver in this module is deliberately separate from the reduced
//! finite-frequency normal-equation inverse. It integrates the two-dimensional
//! scalar acoustic wave equation on the CT-derived slice, records receiver
//! traces on the treatment/imaging aperture, and backpropagates the trace
//! residual to produce an adjoint RTM image. The same forward solver also
//! computes the planned exposure as a heterogeneous peak-pressure field. It is
//! not nonlinear wave propagation and it is not an iterative acoustic
//! full-waveform inversion.
//!
//! # Numerical methods
//!
//! - **4th-order spatial FD stencil** (Fornberg 1988, Table 2, row k=2)
//! - **CPML absorbing boundaries** (Komatitsch & Martin 2007, Geophysics 72:SM155)
//! - **Rayon** inner-loop parallelism over the ix (row) dimension
//! - **Power-law attenuation** (Treeby & Cox 2010, per-cell amplitude decay)
//! - **Checkpointed adjoint** (Griewank 1992) — O(√T · N) memory vs O(T · N)

mod adjoint;
mod backend;
mod cavitation;
mod eikonal;
mod emission;
mod forward;
mod grid;
mod medium;
mod types;
mod utils;

pub use emission::passive_acoustic_maps;
pub use types::{PeakPressureExposureResult, WaveformSimulationResult};

use super::config::TheranosticInverseConfig;
use super::exposure::normalize_positive;
use super::geometry::DeviceLayout;
use super::medium::PreparedTheranosticSlice;
use super::misfit::evaluate_trace_residual;
use adjoint::adjoint_image;
use backend::{downsample_max, simulate_peak_pressure_with_backend, ReferenceFdtdCpmlBackend};
use forward::propagate;
use grid::acoustic_grid;
use medium::lesion_speed;
use utils::energy;

pub const THERANOSTIC_WAVEFORM_MODEL: &str = "source_encoded_time_domain_acoustic_adjoint_rtm";
pub const THERANOSTIC_WAVE_EXPOSURE_MODEL: &str =
    "source_encoded_time_domain_acoustic_peak_pressure_exposure";
pub const THERANOSTIC_WAVE_EXPOSURE_BACKEND: &str = "reference_fdtd_cpml_2d";
pub const THERANOSTIC_HYBRID_PSTD_FDTD_EXPOSURE_READY: bool = false;

/// Run heterogeneous forward propagation and return the body-normalized peak
/// pressure exposure plus the raw peak-pressure field.
///
/// # Mathematical contract
///
/// Forward: `p_tt = c(x)^2 Δ_4 p + s(x,t)` with the same 4th-order stencil,
/// CPML boundary condition, attenuation map, source encoding, and electronic
/// steering delays used by the RTM acquisition. The reported exposure is the
/// body-masked normalized raw peak pressure scaled to the configured source
/// pressure so downstream figures retain their pressure-display contract.
///
/// Memory: the peak solve stores three rolling pressure fields, two CPML memory
/// fields, and one peak accumulator: `6 · nx · ny` scalar values, with no
/// receiver traces and no forward checkpoints.
#[must_use]
pub fn simulate_peak_pressure_exposure(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    config: &TheranosticInverseConfig,
) -> PeakPressureExposureResult {
    simulate_peak_pressure_with_backend::<ReferenceFdtdCpmlBackend>(prepared, layout, config)
}

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
    lesion: &leto::Array2<f64>,
) -> WaveformSimulationResult {
    let true_speed = lesion_speed(prepared, config, lesion);
    let sim = acoustic_grid(
        prepared,
        layout,
        config,
        &prepared.sound_speed_m_s,
        &true_speed,
    );
    let observed = propagate(&sim.grid, &sim.speed_true, false);
    let predicted = propagate(&sim.grid, &sim.speed_baseline, true);
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
        &sim.grid,
        &sim.speed_baseline,
        &residual.adjoint_source,
        checkpoints,
        predicted.checkpoint_interval,
    );
    // Crop the padded refined adjoint image to the body sub-region, then
    // max-pool down to the caller-visible coarse body grid.  Matches the
    // exposure backend's downsampling so the reconstruction lives in the
    // same coarse coordinate system as `prepared.body_mask` and the rest
    // of the inverse pipeline.
    let (nx_b, ny_b) = sim.body_dims;
    let (ox, oy) = sim.body_offset;
    let image_cropped_refined =
        leto::Array2::from_shape_fn((nx_b, ny_b), |[ix, iy]| image[[ix + ox, iy + oy]]);
    let image_cropped =
        downsample_max(&image_cropped_refined, sim.body_dims_coarse, sim.refinement);
    let reconstruction = normalize_positive(&image_cropped, &prepared.body_mask);
    WaveformSimulationResult {
        reconstruction,
        residual_energy,
        observed_energy,
        receiver_count: sim.grid.receiver_cells.len(),
        time_steps: sim.grid.time_steps,
        dt_s: sim.grid.dt_s,
        model_name: THERANOSTIC_WAVEFORM_MODEL,
        misfit_name: residual.misfit.label(),
        misfit_scale: residual.scale,
        objective_value: residual.objective_value,
    }
}

#[cfg(test)]
mod tests;
