use ndarray::{Array1, Array3};

pub use crate::physics::acoustics::mechanics::elastic_wave::ElasticBodyForceConfig;

/// Velocity-source injection mode for [`ElasticVelocitySource`].
///
/// Matches k-Wave's MATLAB `source.u_mode` semantics for `pstdElastic*`:
///
/// - **Dirichlet**: at each step, the integrator's post-step velocity
///   field is **assigned** at masked grid points with the signal sample
///   (`v[idx] = signal[step]`). Effectively replaces the integrator's
///   contribution at those points.
/// - **Additive**: the signal sample is **added** to the integrator's
///   post-step velocity field (`v[idx] += signal[step]`). The integrator
///   contribution at masked points is preserved and the signal acts as
///   a forcing term. This is the default in MATLAB k-Wave's elastic
///   solvers and KWave.jl's `pstd_elastic_2d`.
///
/// For a tone-burst drive on a dense plane mask, the two modes produce
/// different amplitude scaling (Additive accumulates, Dirichlet replaces),
/// so cross-engine parity requires matching the mode used by the
/// reference. The k-wave-julia parity harness in
/// `external/elastic_julia_parity/` uses this enum to drive both runs
/// in the same mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ElasticVelocitySourceMode {
    /// Replace velocity at masked points with the signal sample.
    Dirichlet,
    /// Add the signal sample to the integrator's velocity update.
    /// Default — matches MATLAB k-Wave / KWave.jl elastic solvers.
    #[default]
    Additive,
}

/// Particle-velocity source configuration for the elastic wave solver.
///
/// Implements k-Wave's `source.u_mask` / `source.ux` / `source.uy` /
/// `source.uz` semantics for `pstdElastic2D` / `pstdElastic3D`. At each
/// time step, if the corresponding component signal is supplied, the
/// integrator's post-step velocity field is updated at every grid point
/// inside `mask` per the configured [`ElasticVelocitySourceMode`]
/// (broadcast 1-D signal across all mask points; phase A.3 does not yet
/// support per-point signal matrices).
///
/// Phase A.3 of ADR 007. Phase A.3.1 added `mode` selection. Stress-
/// tensor sources (`source.s_mask`, `source.sxx`, `source.syy`, etc.)
/// ship in Phase A.3.5.
#[derive(Debug, Clone)]
pub struct ElasticVelocitySource {
    /// Boolean grid mask marking source-active points.
    pub mask: Array3<bool>,
    /// Time series for ux at each step (length must equal `Nt`). `None`
    /// disables vx injection.
    pub ux_signal: Option<Array1<f64>>,
    /// Time series for uy at each step.
    pub uy_signal: Option<Array1<f64>>,
    /// Time series for uz at each step.
    pub uz_signal: Option<Array1<f64>>,
    /// Injection mode (Dirichlet vs Additive). Default: Additive,
    /// matching MATLAB k-Wave and KWave.jl elastic solvers.
    pub mode: ElasticVelocitySourceMode,
}

impl ElasticVelocitySource {
    /// Returns `true` if at least one component signal is present.
    #[must_use]
    pub fn has_any_component(&self) -> bool {
        self.ux_signal.is_some() || self.uy_signal.is_some() || self.uz_signal.is_some()
    }
}

/// Configuration for elastic wave simulation
#[derive(Debug, Clone)]
pub struct ElasticWaveConfig {
    /// Time step in seconds (0.0 for auto-CFL)
    pub time_step: f64,
    /// CFL number for auto-time step (default 0.5)
    pub cfl_number: f64,
    pub cfl_factor: f64,
    pub simulation_time: f64,
    pub save_every: usize,
    pub pml_thickness: usize,
    /// Sensor mask for recording time series data at specific locations
    pub sensor_mask: Option<Array3<bool>>,
    /// Optional particle-velocity source. When present, at each step
    /// `n`, the integrator's post-step velocity field is assigned at
    /// every masked grid point with `signal[n]` for any provided
    /// component. Phase A.3 of ADR 007.
    pub velocity_source: Option<ElasticVelocitySource>,
}

impl Default for ElasticWaveConfig {
    fn default() -> Self {
        Self {
            time_step: 0.0,
            cfl_number: 0.5,
            cfl_factor: 0.5,
            simulation_time: 10e-3,
            save_every: 10,
            pml_thickness: 10,
            sensor_mask: None,
            velocity_source: None,
        }
    }
}

/// State of the elastic wave field (displacement and velocity)
#[derive(Debug, Clone)]
pub struct ElasticWaveField {
    /// Displacement X component (m)
    pub ux: Array3<f64>,
    /// Displacement Y component (m)
    pub uy: Array3<f64>,
    /// Displacement Z component (m)
    pub uz: Array3<f64>,

    /// Velocity X component (m/s)
    pub vx: Array3<f64>,
    /// Velocity Y component (m/s)
    pub vy: Array3<f64>,
    /// Velocity Z component (m/s)
    pub vz: Array3<f64>,

    /// Simulation time (s)
    pub time: f64,
}

impl ElasticWaveField {
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            ux: Array3::zeros((nx, ny, nz)),
            uy: Array3::zeros((nx, ny, nz)),
            uz: Array3::zeros((nx, ny, nz)),
            vx: Array3::zeros((nx, ny, nz)),
            vy: Array3::zeros((nx, ny, nz)),
            vz: Array3::zeros((nx, ny, nz)),
            time: 0.0,
        }
    }

    #[must_use]
    pub fn displacement_magnitude(&self) -> Array3<f64> {
        let mut out = self.ux.clone();
        out.zip_mut_with(&self.uy, |o, &y| {
            *o = (*o).mul_add(*o, y * y);
        });
        out.zip_mut_with(&self.uz, |o, &z| {
            *o += z * z;
        });
        out.par_mapv_inplace(f64::sqrt);
        out
    }
}

#[derive(Debug, Clone)]
pub enum ArrivalDetection {
    EnergyThreshold { threshold: f64 },
    MatchedFilter { template: Vec<f64>, min_corr: f64 },
}

impl Default for ArrivalDetection {
    fn default() -> Self {
        Self::EnergyThreshold { threshold: 0.0 }
    }
}

#[derive(Debug, Clone)]
pub struct VolumetricSource {
    pub location_m: [f64; 3],
    pub time_offset_s: f64,
}

#[derive(Debug, Clone)]
pub struct VolumetricWaveConfig {
    pub volumetric_boundaries: bool,
    pub interference_tracking: bool,
    pub volumetric_attenuation: bool,
    pub dispersion_correction: bool,
    pub arrival_detection: ArrivalDetection,
    pub tracking_decimation: [usize; 3],
    pub duration_s: f64,
    pub max_snapshots: usize,
}

impl Default for VolumetricWaveConfig {
    fn default() -> Self {
        Self {
            volumetric_boundaries: false,
            interference_tracking: false,
            volumetric_attenuation: false,
            dispersion_correction: false,
            arrival_detection: ArrivalDetection::default(),
            tracking_decimation: [1, 1, 1],
            duration_s: 10e-3,
            max_snapshots: 256,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WaveFrontTracker {
    pub arrival_times: Array3<f64>,
    pub amplitudes: Array3<f64>,
    pub tracking_quality: Array3<f64>,
}

#[derive(Debug, Clone)]
pub struct VolumetricQualityMetrics {
    pub coverage: f64,
    pub average_quality: f64,
    pub valid_tracking_points: usize,
}
