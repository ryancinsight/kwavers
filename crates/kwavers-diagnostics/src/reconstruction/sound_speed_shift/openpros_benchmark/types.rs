//! OpenPros benchmark data contracts.

use leto::Array2;

use super::super::{SoundSpeedShiftConfig, SoundSpeedShiftImage, SoundSpeedShiftSample};
use kwavers_core::constants::numerical::MHZ_TO_HZ;

/// Hugging Face and arXiv identifier for the OpenPros structural reference.
pub const OPENPROS_PAPER_ID: &str = "2505.12261";

const REFERENCE_SPACING_M: f64 = 0.000_375;
const REFERENCE_AXIAL_POINTS: usize = 401;
const REFERENCE_LATERAL_POINTS: usize = 161;
const REFERENCE_PEAK_FREQUENCY_HZ: f64 = MHZ_TO_HZ;
const REFERENCE_TIME_STEPS: usize = 1_000;
const REFERENCE_ABSORBING_BOUNDARY_POINTS: usize = 120;

/// Configuration for the reduced OpenPros-style benchmark fixture.
///
/// `spatial_decimation = 10` maps the paper's 401 x 161 SOS lattice and
/// 0.375 mm spacing to a 41 x 17 benchmark lattice with 3.75 mm spacing. This
/// preserves the limited-view geometry while keeping unit tests bounded.
#[derive(Clone, Debug)]
pub struct OpenProsShiftBenchmarkConfig {
    pub spatial_decimation: usize,
    pub source_count_per_probe: usize,
    pub receiver_count_per_probe: usize,
    pub dense_iterations: usize,
    pub sparse_iterations: usize,
    pub sparse_stride: usize,
    pub tikhonov_weight: f64,
    pub smoothness_weight: f64,
    pub sparsity_weight: f64,
}

impl Default for OpenProsShiftBenchmarkConfig {
    fn default() -> Self {
        Self {
            spatial_decimation: 10,
            source_count_per_probe: 10,
            receiver_count_per_probe: 17,
            dense_iterations: 48,
            sparse_iterations: 96,
            sparse_stride: 4,
            tikhonov_weight: 1.0e-8,
            smoothness_weight: 1.0e-8,
            sparsity_weight: 1.0e-5,
        }
    }
}

impl OpenProsShiftBenchmarkConfig {
    #[must_use]
    pub fn spacing_m(&self) -> f64 {
        REFERENCE_SPACING_M * self.spatial_decimation as f64
    }

    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        (
            decimated_points(REFERENCE_AXIAL_POINTS, self.spatial_decimation),
            decimated_points(REFERENCE_LATERAL_POINTS, self.spatial_decimation),
        )
    }

    #[must_use]
    pub fn waveform_expectation(&self) -> OpenProsWaveformExpectation {
        let shape = self.shape();
        let spacing_m = self.spacing_m();
        OpenProsWaveformExpectation {
            paper_id: OPENPROS_PAPER_ID,
            source_channels: 4 * self.source_count_per_probe,
            receivers_per_channel: self.receiver_count_per_probe,
            time_steps: REFERENCE_TIME_STEPS,
            peak_frequency_hz: REFERENCE_PEAK_FREQUENCY_HZ,
            absorbing_boundary_points: REFERENCE_ABSORBING_BOUNDARY_POINTS,
            grid_spacing_m: spacing_m,
            sos_shape: shape,
            field_of_view_m: (shape.0 as f64 * spacing_m, shape.1 as f64 * spacing_m),
        }
    }
}

/// Waveform and phantom dimensions mirrored by the benchmark fixture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpenProsWaveformExpectation {
    pub paper_id: &'static str,
    pub source_channels: usize,
    pub receivers_per_channel: usize,
    pub time_steps: usize,
    pub peak_frequency_hz: f64,
    pub absorbing_boundary_points: usize,
    pub grid_spacing_m: f64,
    pub sos_shape: (usize, usize),
    pub field_of_view_m: (f64, f64),
}

/// Deterministic OpenPros-style benchmark case.
#[derive(Clone, Debug)]
pub struct OpenProsShiftBenchmarkCase {
    pub active_mask: Array2<bool>,
    pub truth_shift_m_s: Array2<f64>,
    pub samples: Vec<SoundSpeedShiftSample>,
    pub frame_time_shifts_s: Vec<f64>,
    pub dense_config: SoundSpeedShiftConfig,
    pub sparse_config: SoundSpeedShiftConfig,
    pub waveform: OpenProsWaveformExpectation,
}

/// Value metrics for one reconstruction branch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OpenProsShiftReconstructionMetrics {
    pub rows_available: usize,
    pub rows_used: usize,
    pub active_voxels: usize,
    pub stored_weight_count: usize,
    pub mean_absolute_error_m_s: f64,
    pub root_mean_square_error_m_s: f64,
    pub normalized_root_mean_square_error: f64,
    pub pearson_correlation: f64,
    pub objective_initial: f64,
    pub objective_final: f64,
    pub objective_reduction_fraction: f64,
}

/// Dense-versus-sparse benchmark output.
#[derive(Clone, Debug)]
pub struct OpenProsShiftBenchmarkResult {
    pub dense_reconstruction: SoundSpeedShiftImage,
    pub sparse_reconstruction: SoundSpeedShiftImage,
    pub dense_metrics: OpenProsShiftReconstructionMetrics,
    pub sparse_metrics: OpenProsShiftReconstructionMetrics,
    pub waveform: OpenProsWaveformExpectation,
}

const fn decimated_points(reference_points: usize, decimation: usize) -> usize {
    let checked_decimation = if decimation == 0 { 1 } else { decimation };
    ((reference_points - 1) / checked_decimation) + 1
}
