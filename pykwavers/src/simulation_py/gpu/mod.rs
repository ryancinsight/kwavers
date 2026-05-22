mod session;

pub use session::GpuPstdSession;

#[cfg(feature = "gpu")]
use kwavers::core::error::KwaversResult;
#[cfg(feature = "gpu")]
use kwavers::domain::grid::Grid as KwaversGrid;
#[cfg(feature = "gpu")]
use kwavers::domain::sensor::recorder::SampledStatistics;
#[cfg(feature = "gpu")]
use kwavers::domain::source::GridSource;

#[cfg(feature = "gpu")]
use crate::medium_py::MediumInner;
#[cfg(feature = "gpu")]
use crate::sensor_py::Sensor;
#[cfg(feature = "gpu")]
use crate::simulation_py::Simulation;
#[cfg(feature = "gpu")]
use crate::transducer_array_py::TransducerArray2D;

/// GPU-resident PSTD implementation (requires `gpu` feature).
///
/// This is a thin Python-binding adapter: it builds the sensor mask from
/// the pykwavers `Sensor` / `TransducerArray2D` wrappers and the GPU PSTD
/// run config from the per-call knobs, then delegates to the canonical
/// kwavers-side entry [`kwavers::solver::forward::pstd::gpu_pstd::run_gpu_pstd`].
/// All buffer preparation, CPML profile evaluation, source / sensor indexing,
/// and `GpuPstdSolver` dispatch live in the kwavers crate so non-Python
/// callers (clinical adapters, examples, benches) can drive the GPU path
/// without going through PyO3.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_gpu_pstd_impl(
    grid: &KwaversGrid,
    medium: &MediumInner,
    time_steps: usize,
    dt: f64,
    alpha_coeff_db: f64,
    alpha_power: f64,
    grid_source: &GridSource,
    sensor: Option<&Sensor>,
    transducer_sensor: Option<&TransducerArray2D>,
    pml_size: Option<usize>,
    pml_size_xyz: Option<(usize, usize, usize)>,
    pml_inside: bool,
    pml_alpha_xyz: Option<(f64, f64, f64)>,
) -> KwaversResult<(ndarray::Array2<f64>, Option<SampledStatistics>)> {
    use kwavers::solver::forward::pstd::gpu_pstd::{run_gpu_pstd, GpuPstdRunConfig};

    let sensor_mask = Simulation::create_sensor_mask(grid, sensor, transducer_sensor);
    let config = GpuPstdRunConfig {
        time_steps,
        dt,
        alpha_coeff_db,
        alpha_power,
        pml_size,
        pml_size_xyz,
        pml_inside,
        pml_alpha_xyz,
    };
    let sensor_data = run_gpu_pstd(grid, medium.as_medium(), grid_source, &sensor_mask, config)?;
    Ok((sensor_data, None))
}
