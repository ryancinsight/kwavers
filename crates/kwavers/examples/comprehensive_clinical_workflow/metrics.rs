//! Tissue construction and quantitative imaging metrics.

use super::{FibrosisMetrics, LiverAssessmentWorkflow, PerfusionMetrics};
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::elastography::NonlinearParameterMap;
use kwavers_medium::{heterogeneous::HeterogeneousMedium, homogeneous::HomogeneousMedium};
use leto::Array3;

impl LiverAssessmentWorkflow {
    pub(super) fn create_liver_tissue_model(grid: &Grid) -> HeterogeneousMedium {
        let base_properties = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.1, grid);
        HeterogeneousMedium::from_homogeneous(&base_properties, grid)
            .expect("invariant: liver reference optical properties are valid")
    }

    pub(super) fn calculate_fibrosis_metrics(
        &self,
        stiffness_map: &Array3<f32>,
        nonlinear_analysis: &NonlinearParameterMap,
    ) -> FibrosisMetrics {
        let mean_stiffness = stiffness_map.iter().sum::<f32>() / stiffness_map.size() as f32;
        let fibrosis_stage = if mean_stiffness < 5.0 {
            0
        } else if mean_stiffness < 7.0 {
            1
        } else if mean_stiffness < 10.0 {
            2
        } else if mean_stiffness < 15.0 {
            3
        } else {
            4
        };

        FibrosisMetrics {
            mean_stiffness: f64::from(mean_stiffness),
            stiffness_std: 0.8,
            fibrosis_stage,
            nonlinear_parameter: nonlinear_analysis
                .nonlinearity_parameter
                .iter()
                .copied()
                .sum::<f64>()
                / nonlinear_analysis.nonlinearity_parameter.size() as f64,
        }
    }

    pub(super) fn calculate_perfusion_metrics(
        &self,
        perfusion_map: &Array3<f32>,
    ) -> PerfusionMetrics {
        let peak_enhancement = perfusion_map.iter().copied().fold(0.0_f32, f32::max);
        let mean_perfusion = perfusion_map.iter().sum::<f32>() / perfusion_map.size() as f32;

        PerfusionMetrics {
            peak_enhancement: 20.0 * f64::from(peak_enhancement).log10(),
            perfusion_rate: f64::from(mean_perfusion) * 1000.0,
            wash_in_time: 15.0,
            wash_out_time: 120.0,
        }
    }
}
