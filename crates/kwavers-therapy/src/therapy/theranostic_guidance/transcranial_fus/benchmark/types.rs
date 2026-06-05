use ndarray::{Array1, Array2, Array3};

use super::super::types::TranscranialFusPlanConfig;

#[derive(Clone, Debug)]
pub struct SkullAdaptiveBenchmarkConfig {
    pub fus: TranscranialFusPlanConfig,
    pub aperture_diameter_m: f64,
    pub minimum_active_elements: usize,
}

impl Default for SkullAdaptiveBenchmarkConfig {
    fn default() -> Self {
        let fus = TranscranialFusPlanConfig {
            element_count: 1024,
            chunk_size: 512,
            ..TranscranialFusPlanConfig::default()
        };
        Self {
            fus,
            aperture_diameter_m: 0.120,
            minimum_active_elements: 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SkullAwareTransducerPlacement {
    pub element_positions_m: Array2<f64>,
    pub active_elements: Array1<bool>,
    pub aperture_anchor_index: usize,
    pub active_element_count: usize,
    pub aperture_diameter_m: f64,
    pub radius_of_curvature_m: f64,
    pub focal_length_m: f64,
    pub mean_skull_length_m: f64,
    pub mean_amplitude_weight: f64,
    pub min_amplitude_weight: f64,
    pub max_amplitude_weight: f64,
}

#[derive(Clone, Debug)]
pub struct PressureFieldMetrics {
    pub relative_l2: f64,
    pub focal_position_error_m: f64,
    pub max_pressure_error_percent: f64,
    pub reference_peak_pa: f64,
    pub candidate_peak_pa: f64,
    pub reference_focus_index: [usize; 3],
    pub candidate_focus_index: [usize; 3],
}

#[derive(Debug)]
pub struct SkullAdaptiveBenchmarkResult {
    pub reference_pressure_pa: Array3<f32>,
    pub baseline_pressure_pa: Array3<f32>,
    pub metrics: PressureFieldMetrics,
    pub placement: SkullAwareTransducerPlacement,
    pub phases_rad: Array1<f64>,
    pub delays_s: Array1<f64>,
    pub skull_lengths_m: Array1<f64>,
    pub amplitude_weights: Array1<f64>,
    pub focus_index: [usize; 3],
    pub spacing_m: [f64; 3],
    pub frequency_hz: f64,
    pub target_peak_pa: f64,
}
