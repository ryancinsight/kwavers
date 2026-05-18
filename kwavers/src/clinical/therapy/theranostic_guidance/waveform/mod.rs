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
mod forward;
mod grid;
mod medium;
mod types;
mod utils;

pub use types::{PeakPressureExposureResult, WaveformSimulationResult};

use super::config::TheranosticInverseConfig;
use super::exposure::normalize_positive;
use super::geometry::DeviceLayout;
use super::medium::PreparedTheranosticSlice;
use super::misfit::evaluate_trace_residual;
use adjoint::adjoint_image;
use backend::{simulate_peak_pressure_with_backend, ReferenceFdtdCpmlBackend};
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

#[cfg(test)]
mod tests {
    use ndarray::Array2;

    use super::*;
    use crate::clinical::therapy::theranostic_guidance::config::AnatomyKind;
    use crate::clinical::therapy::theranostic_guidance::geometry::Point2;

    #[test]
    fn peak_pressure_exposure_records_bounded_workspace() {
        let prepared = prepared_fixture(Array2::from_elem((28, 28), 1540.0));
        let config = exposure_config();
        let layout = exposure_layout();

        let result = simulate_peak_pressure_exposure(&prepared, &layout, &config);
        let exposure_peak = result.exposure.iter().copied().fold(0.0, f64::max);
        let raw_peak = result.raw_peak_pressure.iter().copied().fold(0.0, f64::max);

        assert_eq!(result.model_name, THERANOSTIC_WAVE_EXPOSURE_MODEL);
        assert_eq!(result.backend_name, THERANOSTIC_WAVE_EXPOSURE_BACKEND);
        assert!(!result.uses_hybrid_pstd_fdtd);
        assert_eq!(result.source_count, layout.therapy_elements.len());
        assert_eq!(result.workspace_values, 6 * 28 * 28);
        assert!(result.time_steps >= 96);
        assert!(result.dt_s > 0.0);
        assert!(raw_peak > 0.0);
        assert!(
            (exposure_peak - config.source_pressure_pa).abs() <= config.source_pressure_pa * 1.0e-6,
            "exposure_peak={exposure_peak}, expected={}",
            config.source_pressure_pa
        );
    }

    #[test]
    fn peak_pressure_exposure_responds_to_internal_gas_scattering() {
        let config = exposure_config();
        let layout = exposure_layout();
        let homogeneous = simulate_peak_pressure_exposure(
            &prepared_fixture(Array2::from_elem((28, 28), 1540.0)),
            &layout,
            &config,
        );

        let mut speed = Array2::from_elem((28, 28), 1540.0);
        for ix in 12..16 {
            for iy in 8..20 {
                speed[[ix, iy]] = 343.0;
            }
        }
        let scattered = simulate_peak_pressure_exposure(&prepared_fixture(speed), &layout, &config);

        let downstream_difference =
            mean_abs_difference(&homogeneous.exposure, &scattered.exposure, 18..24, 8..20);
        assert!(
            downstream_difference > 1.0e-3,
            "heterogeneous wave solve must alter downstream peak pressure; diff={downstream_difference}"
        );
    }

    fn prepared_fixture(speed: Array2<f64>) -> PreparedTheranosticSlice {
        let (nx, ny) = speed.dim();
        let mut target_mask = Array2::from_elem((nx, ny), false);
        for ix in 13..15 {
            for iy in 13..15 {
                target_mask[[ix, iy]] = true;
            }
        }
        PreparedTheranosticSlice {
            anatomy: AnatomyKind::Liver,
            ct_hu: Array2::from_elem((nx, ny), 40.0),
            label: Array2::zeros((nx, ny)),
            sound_speed_m_s: speed,
            attenuation_np_per_m_mhz: Array2::from_elem((nx, ny), 0.1),
            body_mask: Array2::from_elem((nx, ny), true),
            organ_mask: Array2::from_elem((nx, ny), true),
            target_mask,
            spacing_m: 0.002,
            source_slice_index: 0,
            source_dimensions: [nx, ny],
            source_spacing_m: [0.002, 0.002],
            crop_bounds_index: [0, nx - 1, 0, ny - 1],
        }
    }

    fn exposure_config() -> TheranosticInverseConfig {
        let mut config = TheranosticInverseConfig::new(AnatomyKind::Liver);
        config.frequencies_hz = vec![800_000.0];
        config.source_pressure_pa = 10_000.0;
        config
    }

    fn exposure_layout() -> DeviceLayout {
        let therapy_elements = (0..8)
            .map(|idx| Point2 {
                x_m: -0.022,
                y_m: -0.014 + idx as f64 * 0.004,
            })
            .collect::<Vec<_>>();
        DeviceLayout {
            therapy_elements,
            imaging_receivers: Vec::new(),
            focus_m: Point2 { x_m: 0.0, y_m: 0.0 },
            skin_contact_m: Point2 {
                x_m: -0.026,
                y_m: 0.0,
            },
            model_name: "test_linear_array".to_owned(),
        }
    }

    fn mean_abs_difference(
        lhs: &Array2<f64>,
        rhs: &Array2<f64>,
        x_range: std::ops::Range<usize>,
        y_range: std::ops::Range<usize>,
    ) -> f64 {
        let mut sum = 0.0;
        let mut count = 0usize;
        for ix in x_range {
            for iy in y_range.clone() {
                sum += (lhs[[ix, iy]] - rhs[[ix, iy]]).abs();
                count += 1;
            }
        }
        sum / count.max(1) as f64
    }
}
