//! Backend contract for theranostic acoustic exposure propagation.

use super::forward::{peak_pressure_workspace_values, propagate_peak_pressure};
use super::grid::acoustic_grid;
use super::types::PeakPressureExposureResult;
use super::utils::linear;
use super::{THERANOSTIC_WAVE_EXPOSURE_BACKEND, THERANOSTIC_WAVE_EXPOSURE_MODEL};
use crate::clinical::therapy::theranostic_guidance::config::TheranosticInverseConfig;
use crate::clinical::therapy::theranostic_guidance::exposure::normalize_positive;
use crate::clinical::therapy::theranostic_guidance::geometry::DeviceLayout;
use crate::clinical::therapy::theranostic_guidance::medium::PreparedTheranosticSlice;

/// Zero-cost backend contract for peak-pressure exposure solves.
///
/// # Contract
///
/// Implementors must solve the same scalar acoustic initial-boundary-value
/// problem and must expose retained workspace. Backend selection is static:
/// the generic type parameter monomorphizes the call site, so no dynamic
/// dispatch or heap-erased solver object sits in the propagation hot path.
pub(super) trait PeakPressureBackend {
    const BACKEND_NAME: &'static str;
    const USES_HYBRID_PSTD_FDTD: bool;

    fn simulate(
        prepared: &PreparedTheranosticSlice,
        layout: &DeviceLayout,
        config: &TheranosticInverseConfig,
    ) -> PeakPressureExposureResult;
}

/// Reference 4th-order finite-difference CPML backend.
#[derive(Clone, Copy, Debug)]
pub(super) struct ReferenceFdtdCpmlBackend;

impl PeakPressureBackend for ReferenceFdtdCpmlBackend {
    const BACKEND_NAME: &'static str = THERANOSTIC_WAVE_EXPOSURE_BACKEND;
    const USES_HYBRID_PSTD_FDTD: bool = false;

    fn simulate(
        prepared: &PreparedTheranosticSlice,
        layout: &DeviceLayout,
        config: &TheranosticInverseConfig,
    ) -> PeakPressureExposureResult {
        let sim = acoustic_grid(
            prepared,
            layout,
            config,
            &prepared.sound_speed_m_s,
            &prepared.sound_speed_m_s,
        );
        let peak = propagate_peak_pressure(&sim.grid, &sim.speed_true);
        let (nx_b, ny_b) = sim.body_dims;
        let (ox, oy) = sim.body_offset;
        let padded_ny = sim.grid.ny;
        // Crop the padded refined peak field to the body sub-region, then
        // max-pool down to the caller-visible coarse body grid.  Max-pool
        // preserves focal-spot localization across the refinement; mean-pool
        // would smear the peak across `R²` cells and reduce the contrast.
        let raw_peak_refined = ndarray::Array2::from_shape_fn((nx_b, ny_b), |(ix, iy)| {
            peak[linear(ix + ox, iy + oy, padded_ny)] as f64
        });
        let raw_peak_pressure =
            downsample_max(&raw_peak_refined, sim.body_dims_coarse, sim.refinement);
        let exposure = normalize_positive(&raw_peak_pressure, &prepared.body_mask)
            .mapv(|value| value * config.source_pressure_pa);

        PeakPressureExposureResult {
            exposure,
            raw_peak_pressure,
            source_count: sim.grid.source_cells.len(),
            time_steps: sim.grid.time_steps,
            dt_s: sim.grid.dt_s,
            workspace_values: peak_pressure_workspace_values(sim.grid.nx, sim.grid.ny),
            model_name: THERANOSTIC_WAVE_EXPOSURE_MODEL,
            backend_name: Self::BACKEND_NAME,
            uses_hybrid_pstd_fdtd: Self::USES_HYBRID_PSTD_FDTD,
        }
    }
}

#[inline]
pub(super) fn simulate_peak_pressure_with_backend<B: PeakPressureBackend>(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    config: &TheranosticInverseConfig,
) -> PeakPressureExposureResult {
    B::simulate(prepared, layout, config)
}

/// Max-pool an `(nx_r, ny_r) = (nx_c · R, ny_c · R)` refined field into the
/// `(nx_c, ny_c)` coarse body grid.  Used to map the FDTD peak-pressure
/// field (and the RTM image) back to the caller-visible body resolution
/// after the internally refined propagation.  Max-pool is the correct
/// reduction for peak-pressure visualisation (it preserves the focal-spot
/// maximum across the `R²` refined cells per coarse cell).
pub(super) fn downsample_max(
    refined: &ndarray::Array2<f64>,
    coarse_dims: (usize, usize),
    refinement: usize,
) -> ndarray::Array2<f64> {
    let (nx_c, ny_c) = coarse_dims;
    if refinement <= 1 {
        return refined.clone();
    }
    ndarray::Array2::from_shape_fn((nx_c, ny_c), |(ix, iy)| {
        let x0 = ix * refinement;
        let y0 = iy * refinement;
        let mut best = f64::NEG_INFINITY;
        for di in 0..refinement {
            for dj in 0..refinement {
                let v = refined[[x0 + di, y0 + dj]];
                if v > best {
                    best = v;
                }
            }
        }
        if best.is_finite() {
            best
        } else {
            0.0
        }
    })
}
