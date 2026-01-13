use ndarray::Array3;

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
            *o = *o * *o + y * y;
        });
        out.zip_mut_with(&self.uz, |o, &z| {
            *o += z * z;
        });
        out.mapv_inplace(f64::sqrt);
        out
    }
}

/// Configuration for body force excitation
#[derive(Debug, Clone)]
pub enum ElasticBodyForceConfig {
    GaussianImpulse {
        /// Center position [x, y, z] (m)
        center_m: [f64; 3],
        /// Spatial width [sigma_x, sigma_y, sigma_z] (m)
        sigma_m: [f64; 3],
        /// Force direction vector [x, y, z] (normalized)
        direction: [f64; 3],
        /// Center time (s)
        t0_s: f64,
        /// Temporal width (s)
        sigma_t_s: f64,
        /// Impulse magnitude (N·s/m³)
        impulse_n_per_m3_s: f64,
    },
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
