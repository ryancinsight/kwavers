use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::source::{GridSource, Source as KwaversSource};
use kwavers::solver::forward::pstd::config::CompatibilityMode;

use crate::medium_py::MediumInner;
use crate::sensor_py::Sensor;
use crate::simulation_result_py::SimulationRunResult;
use crate::transducer_array_py::TransducerArray2D;

use super::super::Simulation;

impl Simulation {
    /// Dispatch GPU-resident PSTD if the `gpu` feature is enabled, otherwise
    /// fall back to CPU PSTD.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_gpu_pstd_or_cpu_fallback(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        compatibility_mode: CompatibilityMode,
        alpha_coeff_db: f64,
        alpha_power: f64,
        grid_source: GridSource,
        sources: Vec<Box<dyn KwaversSource>>,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        pml_size: Option<usize>,
        pml_size_xyz: Option<(usize, usize, usize)>,
        pml_inside: bool,
        pml_alpha_xyz: Option<(f64, f64, f64)>,
        enable_nonlinear: bool,
        axisymmetric: bool,
        record_modes: &[String],
        record_start_index: usize,
    ) -> KwaversResult<SimulationRunResult> {
        #[cfg(feature = "gpu")]
        {
            match super::super::gpu::run_gpu_pstd_impl(
                grid,
                medium,
                time_steps,
                dt,
                alpha_coeff_db,
                alpha_power,
                &grid_source,
                sensor,
                transducer_sensor,
                pml_size,
                pml_size_xyz,
                pml_inside,
                pml_alpha_xyz,
            ) {
                Ok((sensor_data, stats)) => {
                    let sensor_data = Self::trim_initial_recorder_sample(
                        sensor_data,
                        time_steps,
                        record_start_index,
                    );
                    return Ok(SimulationRunResult {
                        sensor_data,
                        stats,
                        ux_data: None,
                        uy_data: None,
                        uz_data: None,
                        ix_data: None,
                        iy_data: None,
                        iz_data: None,
                        i_avg_x: None,
                        i_avg_y: None,
                        i_avg_z: None,
                        velocity_stats: None,
                        full_grid_stats: None,
                        thermal_temperature: None,
                        thermal_dose: None,
                    });
                }
                Err(e) => {
                    eprintln!("[PstdGpu] GPU path failed ({e}), falling back to CPU PSTD");
                }
            }
        }
        Self::run_pstd_impl(
            grid,
            medium,
            time_steps,
            dt,
            compatibility_mode,
            enable_nonlinear,
            alpha_coeff_db,
            alpha_power,
            grid_source,
            sources,
            sensor,
            transducer_sensor,
            pml_size,
            pml_size_xyz,
            pml_inside,
            pml_alpha_xyz,
            axisymmetric,
            record_modes,
            record_start_index,
        )
    }
}
