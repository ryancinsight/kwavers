//! Simulation run request/response types.
//!
//! These types are shared between the runner dispatch and per-solver modules.

use ndarray::Array3 as NdArray3;

use crate::configs::{
    HelmholtzConfig, NonlinearConfig, PmlConfig, PoroelasticConfig, ThermalConfig,
};
use kwavers_grid::Grid;
use kwavers_medium::traits::Medium as MediumTrait;
use kwavers_receiver::recorder::pressure_statistics::SampledStatistics;
use kwavers_receiver::recorder::simple::SensorRecorder;
use kwavers_solver::config::SolverType;
use kwavers_solver::forward::fdtd::config::KSpaceCorrectionMode;
use kwavers_solver::forward::pstd::config::CompatibilityMode;
use kwavers_solver::forward::pstd::extensions::ElasticPstdVelocitySource;
use kwavers_source::GridSource;
use leto::Array3 as LetoArray3;

// ============================================================================
// Full-grid statistics
// ============================================================================

/// Full-grid pressure statistics bundle: (p_max, p_min, p_rms, p_final).
pub type FullGridStats = Option<(LetoArray3<f64>, LetoArray3<f64>, LetoArray3<f64>, LetoArray3<f64>)>;

/// Extract full-grid `(p_max, p_min, p_rms, p_final)` from a recorder.
pub fn extract_full_grid_stats(recorder: &SensorRecorder) -> FullGridStats {
    let stats = recorder.full_pressure_statistics()?;
    Some((
        stats.get_p_max().clone(),
        stats.get_p_min().clone(),
        stats.p_rms(),
        stats.get_p_final().clone(),
    ))
}

// ============================================================================
// Simulation run result
// ============================================================================

/// Bundle returned by every solver run function.
#[derive(Debug)]
pub struct SimulationRunResult {
    /// Pressure time series at sensor positions: `(n_sensors, time_steps)`.
    pub sensor_data: ndarray::Array2<f64>,
    /// Pressure spatial statistics (p_max/min/rms/final sampled at sensors).
    pub stats: Option<SampledStatistics>,
    /// Staggered ux time series at sensor positions: `(n_sensors, time_steps)`.
    pub ux_data: Option<ndarray::Array2<f64>>,
    /// Staggered uy time series at sensor positions.
    pub uy_data: Option<ndarray::Array2<f64>>,
    /// Staggered uz time series at sensor positions.
    pub uz_data: Option<ndarray::Array2<f64>>,
    /// Acoustic x-intensity time series at sensor positions.
    pub ix_data: Option<ndarray::Array2<f64>>,
    /// Acoustic y-intensity time series at sensor positions.
    pub iy_data: Option<ndarray::Array2<f64>>,
    /// Acoustic z-intensity time series at sensor positions.
    pub iz_data: Option<ndarray::Array2<f64>>,
    /// Time-averaged x-intensity at sensor positions.
    pub i_avg_x: Option<ndarray::Array1<f64>>,
    /// Time-averaged y-intensity at sensor positions.
    pub i_avg_y: Option<ndarray::Array1<f64>>,
    /// Time-averaged z-intensity at sensor positions.
    pub i_avg_z: Option<ndarray::Array1<f64>>,
    /// Per-component velocity statistics sampled at sensor positions.
    pub velocity_stats:
        Option<kwavers_receiver::recorder::velocity_statistics::SampledVelocityStats>,
    /// Full-grid pressure-statistics field.
    pub full_grid_stats: FullGridStats,
    /// Final temperature field (nx, ny, nz) [K]. `None` for acoustic-only runs.
    pub thermal_temperature: Option<NdArray3<f64>>,
    /// CEM43 thermal dose field (nx, ny, nz) [min].
    pub thermal_dose: Option<NdArray3<f64>>,
}

// ============================================================================
// Simulation run request
// ============================================================================

/// All parameters needed to dispatch and run a simulation.
#[allow(clippy::too_many_arguments)]
#[derive(Debug)]
pub struct SimulationRunRequest<'a> {
    pub grid: &'a Grid,
    pub medium: &'a dyn MediumTrait,
    pub time_steps: usize,
    pub dt: f64,
    pub solver_type: SolverType,
    pub pml: Option<&'a PmlConfig>,
    pub helmholtz: Option<&'a HelmholtzConfig>,
    pub nonlinear: Option<&'a NonlinearConfig>,
    pub thermal: Option<&'a ThermalConfig>,
    pub poroelastic: Option<&'a PoroelasticConfig>,
    pub compatibility_mode: CompatibilityMode,
    pub kspace_correction: KSpaceCorrectionMode,
    pub axisymmetric: bool,

    pub grid_source: GridSource,
    pub sensor_mask: Option<NdArray3<bool>>,
    pub transducer_ordered_indices: Option<Vec<(usize, usize, usize)>>,
    pub record_modes: Vec<String>,
    pub record_start_index: usize,
    pub transducers_for_rs: &'a [kwavers_transducer::array_2d::TransducerArray2D],
    /// ElasticPSTD velocity source (mask + per-axis signals + injection mode).
    pub elastic_velocity_source: Option<ElasticPstdVelocitySource>,
    /// ElasticPSTD initial-displacement IVP axis (0=x, 1=y, 2=z). When set, the
    /// `grid_source.p0` field is interpreted as the initial displacement along
    /// this axis and seeds the initial stress (σ = λ(∇·u)I + μ(∇u + ∇uᵀ)).
    pub elastic_ivp_axis: Option<usize>,
}
