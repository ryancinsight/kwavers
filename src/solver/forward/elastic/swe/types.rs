//! Type definitions for elastic wave solver
//!
//! This module contains configuration types, enumerations, and data structures
//! used throughout the elastic wave solver system.

use ndarray::Array3;

/// The definition of "arrival time" used for volumetric SWE wave-front tracking.
///
/// This is a *semantic choice* that must be explicit to avoid conflating unrelated quantities
/// (e.g., amplitude thresholds vs spatial resolution).
///
/// All variants operate on the displacement magnitude:
/// `|u(x,t)| = sqrt(ux^2 + uy^2 + uz^2)`
/// where `u` is in meters.
///
/// The chosen strategy determines how `WaveFrontTracker.arrival_times` is filled.
///
/// ## Invariants
/// - Arrival times are expressed in seconds.
/// - Undetected voxels keep `arrival_times = +∞`.
/// - For multi-source configurations, arrival and quality evaluation are defined w.r.t. the
///   earliest predicted source arrival (min over sources), unless otherwise stated.
#[derive(Debug, Clone)]
pub enum ArrivalDetection {
    /// Arrival time is the first time `|u(x,t)|` exceeds `threshold_m`.
    ///
    /// Mathematical definition:
    /// `t_arrival(x) = inf { t : |u(x,t)| >= threshold_m }`.
    ///
    /// Units:
    /// - `threshold_m`: meters (displacement magnitude).
    AmplitudeThresholdCrossing { threshold_m: f64 },

    /// Arrival time is the time at which `|u(x,t)|` reaches its maximum over the simulated window.
    ///
    /// Mathematical definition:
    /// `t_arrival(x) = argmax_t |u(x,t)|`.
    ///
    /// This is the most robust detection strategy if the entire shear-wave pulse is simulated.
    MaximumAmplitude,

    /// Arrival time is when the spatial gradient of `|u(x,t)|` exceeds a threshold.
    ///
    /// This can be used to detect the wave-front edge more sharply than amplitude-based methods.
    ///
    /// Mathematical definition (approximate):
    /// `t_arrival(x) = inf { t : ||∇|u(x,t)|| >= threshold_per_m }`.
    ///
    /// Units:
    /// - `threshold_per_m`: 1/meter (inverse length).
    GradientThreshold { threshold_per_m: f64 },
}

/// Volumetric source configuration for multi-push SWE sequences.
///
/// Each source has:
/// - A spatial center `(x, y, z)` in meters.
/// - An application time offset `t_s` in seconds (relative to simulation start).
///
/// This information is used for:
/// 1. Wave-front tracking quality evaluation (expected arrival time model).
/// 2. Multi-source interference analysis.
#[derive(Debug, Clone)]
pub struct VolumetricSource {
    pub center_m: [f64; 3],
    pub application_time_s: f64,
}

/// Configuration for elastic wave solver
///
/// ## Physical Parameters
/// - `simulation_time`: Total simulation duration (seconds)
/// - `dt`: Time step size (seconds, optional - auto-computed if None)
/// - `save_every`: Save field state every N time steps
///
/// ## Boundary Conditions
/// - `pml_size`: Size of perfectly matched layer (grid points)
/// - `pml_alpha`: PML absorption coefficient
///
/// ## Source Configuration
/// - `body_force`: Optional body force configuration for ARFI
#[derive(Debug, Clone)]
pub struct ElasticWaveConfig {
    pub simulation_time: f64,
    pub dt: Option<f64>,
    pub save_every: usize,
    pub pml_size: usize,
    pub pml_alpha: f64,
    pub body_force: Option<ElasticBodyForceConfig>,
}

impl Default for ElasticWaveConfig {
    /// Create default elastic wave configuration
    ///
    /// Default values:
    /// - simulation_time: 0.01 s (10 ms)
    /// - dt: None (auto-computed from CFL condition)
    /// - save_every: 10 time steps
    /// - pml_size: 10 grid points
    /// - pml_alpha: 2.0
    /// - body_force: None
    fn default() -> Self {
        Self {
            simulation_time: 0.01,
            dt: None,
            save_every: 10,
            pml_size: 10,
            pml_alpha: 2.0,
            body_force: None,
        }
    }
}

/// Body force configuration for ARFI (Acoustic Radiation Force Impulse) imaging
///
/// Represents a spatially localized force applied during a time window.
///
/// ## Mathematical Model
/// ```text
/// f(x,t) = amplitude * spatial_profile(x) * temporal_profile(t)
/// ```
///
/// Where:
/// - `spatial_profile(x)`: Gaussian or rectangular spatial distribution
/// - `temporal_profile(t)`: Pulse envelope (typically Gaussian or rectangular)
#[derive(Debug, Clone)]
pub enum ElasticBodyForceConfig {
    /// Gaussian spatial distribution with rectangular temporal pulse
    ///
    /// Force profile:
    /// ```text
    /// f(x,t) = amplitude * exp(-|x-center|²/(2*sigma²)) * rect(t, t_start, t_end)
    /// ```
    GaussianPulse {
        center_m: [f64; 3],
        sigma_m: f64,
        amplitude_n_per_m3: f64,
        start_time_s: f64,
        end_time_s: f64,
        direction: [f64; 3],
    },

    /// Rectangular (uniform) spatial distribution
    ///
    /// Force profile:
    /// ```text
    /// f(x,t) = amplitude * rect(x, bbox) * rect(t, t_start, t_end)
    /// ```
    RectangularPulse {
        bbox_min_m: [f64; 3],
        bbox_max_m: [f64; 3],
        amplitude_n_per_m3: f64,
        start_time_s: f64,
        end_time_s: f64,
        direction: [f64; 3],
    },
}

/// Elastic wave field state at a single time instant
///
/// Contains displacement and velocity components in all three spatial dimensions.
///
/// ## Units
/// - Displacement: meters (m)
/// - Velocity: meters per second (m/s)
/// - Time: seconds (s)
#[derive(Debug, Clone)]
pub struct ElasticWaveField {
    pub ux: Array3<f64>,
    pub uy: Array3<f64>,
    pub uz: Array3<f64>,
    pub vx: Array3<f64>,
    pub vy: Array3<f64>,
    pub vz: Array3<f64>,
    pub time: f64,
}

impl ElasticWaveField {
    /// Create new elastic wave field with zeros
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            ux: Array3::<f64>::zeros((nx, ny, nz)),
            uy: Array3::<f64>::zeros((nx, ny, nz)),
            uz: Array3::<f64>::zeros((nx, ny, nz)),
            vx: Array3::<f64>::zeros((nx, ny, nz)),
            vy: Array3::<f64>::zeros((nx, ny, nz)),
            vz: Array3::<f64>::zeros((nx, ny, nz)),
            time: 0.0,
        }
    }

    /// Initialize displacement field from provided data
    pub fn initialize_displacement(&mut self, displacement: &Array3<f64>) {
        self.ux.assign(displacement);
        self.uy.assign(displacement);
        self.uz.assign(displacement);
    }

    /// Calculate displacement magnitude at a point
    ///
    /// Returns: `sqrt(ux² + uy² + uz²)` in meters
    #[must_use]
    pub fn displacement_magnitude(&self, i: usize, j: usize, k: usize) -> f64 {
        let ux = self.ux[[i, j, k]];
        let uy = self.uy[[i, j, k]];
        let uz = self.uz[[i, j, k]];
        (ux * ux + uy * uy + uz * uz).sqrt()
    }

    /// Calculate total kinetic energy
    ///
    /// Returns: `½ ∫ ρ v² dV` (approximated by grid sum)
    #[must_use]
    pub fn kinetic_energy(&self, density: &Array3<f64>) -> f64 {
        let mut energy = 0.0;
        let (nx, ny, nz) = self.vx.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let rho = density[[i, j, k]];
                    let vx = self.vx[[i, j, k]];
                    let vy = self.vy[[i, j, k]];
                    let vz = self.vz[[i, j, k]];
                    energy += 0.5 * rho * (vx * vx + vy * vy + vz * vz);
                }
            }
        }
        energy
    }
}

/// Configuration for volumetric wave propagation with attenuation and dispersion
///
/// Extends basic elastic wave propagation with:
/// - Frequency-dependent attenuation
/// - Material dispersion
/// - Multi-layer tissue models
#[derive(Debug, Clone)]
pub struct VolumetricWaveConfig {
    pub attenuation_db_per_cm_per_mhz: f64,
    pub dispersion_power_law_exponent: f64,
    pub enable_nonlinear_effects: bool,
}

impl Default for VolumetricWaveConfig {
    /// Default volumetric wave configuration
    ///
    /// Values typical for soft tissue:
    /// - Attenuation: 0.5 dB/(cm·MHz)
    /// - Dispersion exponent: 1.0 (linear frequency dependence)
    /// - Nonlinear effects: disabled (linear regime)
    fn default() -> Self {
        Self {
            attenuation_db_per_cm_per_mhz: 0.5,
            dispersion_power_law_exponent: 1.0,
            enable_nonlinear_effects: false,
        }
    }
}

/// Wave-front tracker for volumetric SWE
///
/// Tracks arrival times, amplitudes, and propagation directions
/// for shear wave elastography applications.
#[derive(Debug, Clone)]
pub struct WaveFrontTracker {
    pub arrival_times: Array3<f64>,
    pub wave_amplitudes: Array3<f64>,
    pub wave_directions: Array3<[f64; 3]>,
    pub detection_strategy: ArrivalDetection,
}

impl WaveFrontTracker {
    /// Create new wave-front tracker
    #[must_use]
    pub fn new(nx: usize, ny: usize, nz: usize, strategy: ArrivalDetection) -> Self {
        Self {
            arrival_times: Array3::<f64>::from_elem((nx, ny, nz), f64::INFINITY),
            wave_amplitudes: Array3::<f64>::zeros((nx, ny, nz)),
            wave_directions: Array3::<[f64; 3]>::from_elem((nx, ny, nz), [0.0, 0.0, 0.0]),
            detection_strategy: strategy,
        }
    }

    /// Check if arrival detected at a voxel
    #[must_use]
    pub fn is_detected(&self, i: usize, j: usize, k: usize) -> bool {
        self.arrival_times[[i, j, k]].is_finite()
    }

    /// Get detection fraction (percentage of voxels with detected arrivals)
    #[must_use]
    pub fn detection_fraction(&self) -> f64 {
        let total = self.arrival_times.len();
        let detected = self
            .arrival_times
            .iter()
            .filter(|&&t| t.is_finite())
            .count();
        detected as f64 / total as f64
    }
}

/// Quality metrics for volumetric SWE tracking
///
/// Provides quantitative measures of wave-front tracking quality
/// for clinical validation and algorithm tuning.
#[derive(Debug, Clone)]
pub struct VolumetricQualityMetrics {
    /// Fraction of voxels with detected arrivals [0, 1]
    pub detection_coverage: f64,

    /// Mean absolute timing error vs expected arrival (seconds)
    pub mean_timing_error_s: f64,

    /// Standard deviation of timing error (seconds)
    pub std_timing_error_s: f64,

    /// Maximum timing error (seconds)
    pub max_timing_error_s: f64,

    /// Mean signal-to-noise ratio (dB)
    pub mean_snr_db: f64,
}

impl VolumetricQualityMetrics {
    /// Create quality metrics with default (invalid) values
    #[must_use]
    pub fn new() -> Self {
        Self {
            detection_coverage: 0.0,
            mean_timing_error_s: f64::NAN,
            std_timing_error_s: f64::NAN,
            max_timing_error_s: f64::NAN,
            mean_snr_db: f64::NAN,
        }
    }
}

impl Default for VolumetricQualityMetrics {
    fn default() -> Self {
        Self::new()
    }
}
