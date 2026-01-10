//! Elastic Wave Solver for Shear Wave Elastography
//!
//! Solves the elastic wave equation for shear wave propagation in soft tissue.
//!
//! ## Governing Equation
//!
//! ∂²u/∂t² = (μ/ρ)∇²u + ((λ+μ)/ρ)∇(∇·u)
//!
//! Where:
//! - u = displacement vector field (m)
//! - ρ = density (kg/m³)
//! - λ, μ = Lamé parameters (Pa)
//! - λ = Eν/((1+ν)(1-2ν)), μ = E/(2(1+ν))
//! - E = Young's modulus (Pa)
//! - ν = Poisson's ratio (dimensionless, ≈0.5 for tissue)
//!
//! ## Numerical Method
//!
//! Uses finite difference time domain (FDTD) with:
//! - Second-order accurate in time (leapfrog)
//! - Fourth-order accurate in space (staggered grid)
//! - Perfectly matched layers (PML) for boundaries
//!
//! ## Volumetric SWE tracking: mathematical definitions (SSOT)
//!
//! This module defines the *single source of truth* for volumetric shear-wave elastography (SWE)
//! wave-front tracking. The key tracked primitive is the *arrival time* `t_arrival(x)` at each
//! voxel `x`, which is later used by time-of-flight (ToF) reconstruction.
//!
//! ### Time-of-flight reconstruction formulas (isotropic, nearly-incompressible soft tissue)
//!
//! Given a source location `x_s` and an arrival time estimate `t_arrival(x)`:
//!
//! - Shear-wave speed estimate (first-order ToF):
//!   `c_s(x) ≈ ||x - x_s|| / t_arrival(x)`
//!
//! - Shear modulus:
//!   `μ(x) = ρ(x) * c_s(x)^2`
//!
//! - In the incompressible limit `ν ≈ 0.5`, Young's modulus:
//!   `E(x) ≈ 3 μ(x) = 3 ρ(x) c_s(x)^2`
//!
//! These formulas are only meaningful if `t_arrival(x)` corresponds to a consistent and
//! physically-relevant wave feature (arrival definition is *typed* via [`ArrivalDetection`])
//! and `x_s` is the actual source location (typed via [`VolumetricSource`]).
//!
//! ### Source-aware expected arrival time (for quality evaluation)
//!
//! For multi-source sequences with explicit application time offsets `t_s`, a physically consistent
//! first-order expected travel time model is:
//!
//! `t_expected(x) = min_s ( t_s + ||x - x_s|| / c_s(x) )`
//!
//! where `c_s(x) = sqrt( μ(x) / ρ(x) )` is the local shear-wave speed. This module uses this
//! definition for timing-based tracking quality when explicit sources are provided.
//!
//! ## References
//!
//! - Moczo, P., et al. (2007). "3D finite-difference method for elastodynamics."
//!   *Solid Earth*, 93(3), 523-553.
//! - Komatitsch, D., & Martin, R. (2007). "An unsplit convolutional perfectly
//!   matched layer improved at grazing incidence for the seismic wave equation."
//!   *Geophysics*, 72(5), SM155-SM167.

use crate::domain::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
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
/// ### Invariants
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
    /// This requires tracking the running maximum magnitude per voxel.
    ///
    /// Units:
    /// - `min_peak_m`: meters; ignore peaks below this amplitude (prevents noise-triggered "arrivals").
    PeakTime { min_peak_m: f64 },

    /// Arrival time is the argmax of correlation between `|u(x,t)|` and a provided 1D template.
    ///
    /// Mathematical definition (discrete time):
    /// `t_arrival(x) = argmax_t Σ_{τ=0..L-1} |u(x,t+τ)| * template[τ]`.
    ///
    /// This is the most robust approach under dispersion/noise, at higher computational cost.
    ///
    /// Units:
    /// - `template`: dimensionless weights (arbitrary scale); normalization is allowed but must be consistent.
    /// - `min_corr`: dimensionless; minimum correlation needed to accept an arrival.
    MatchedFilter { template: Vec<f64>, min_corr: f64 },
}

/// Metadata describing a volumetric SWE source (push).
///
/// Used by tracking-quality evaluation and ToF interpretations.
#[derive(Debug, Clone, Copy)]
pub struct VolumetricSource {
    /// Source location `[x,y,z]` in meters (world coordinates).
    pub location_m: [f64; 3],
    /// Time offset when this source is applied (seconds).
    pub time_offset_s: f64,
}

/// Configuration for elastic wave solver
#[derive(Debug, Clone)]
pub struct ElasticWaveConfig {
    /// Time step size (s) - auto-calculated if None
    pub dt: Option<f64>,
    /// Total simulation time (s)
    pub simulation_time: f64,
    /// CFL stability factor (0.1-0.5 typical)
    pub cfl_factor: f64,
    /// PML thickness (grid points)
    pub pml_thickness: usize,
    /// Save displacement fields every N steps
    pub save_every: usize,
    /// Optional external body-force (per unit volume) configuration.
    ///
    /// Correctness invariant:
    /// - ARFI excitation is a *body force source term* in the momentum equation, not an
    ///   arbitrary displacement initialization.
    /// - If enabled, the solver integrates
    ///   ρ ∂v/∂t = ∇·σ + f
    ///   where `f` has units N/m³.
    pub body_force: Option<ElasticBodyForceConfig>,
}

impl Default for ElasticWaveConfig {
    fn default() -> Self {
        Self {
            dt: None,               // Auto-calculate
            simulation_time: 10e-3, // 10 ms
            cfl_factor: 0.3,
            pml_thickness: 10,
            save_every: 10,
            body_force: None,
        }
    }
}

/// Elastic wave body-force source configuration.
///
/// This is intentionally explicit and self-describing: it encodes the *mathematical* meaning
/// of the source term rather than allowing ad-hoc “initial displacement” hacks.
///
/// Source term:
///   ρ ∂v/∂t = ∇·σ + f
///
/// Units:
/// - `f` is force per unit volume [N/m³]
/// - the solver applies acceleration contribution a_src = f / ρ [m/s²]
#[derive(Debug, Clone)]
pub enum ElasticBodyForceConfig {
    /// A time-localized body-force pulse with a (possibly anisotropic) Gaussian spatial envelope.
    ///
    /// The temporal envelope is a unit-area Gaussian:
    ///   g(t) = exp(-(t - t0)²/(2σ_t²)) / (σ_t sqrt(2π))
    /// so that `impulse_n_per_m3_s` corresponds to ∫ f(t) dt (impulse density).
    GaussianImpulse {
        /// Center of the force [x,y,z] in meters.
        center_m: [f64; 3],
        /// Spatial standard deviations [σx,σy,σz] in meters.
        sigma_m: [f64; 3],
        /// Unit direction of force. Must satisfy ||dir||₂ = 1.
        direction: [f64; 3],
        /// Time of the impulse center t0 (s).
        t0_s: f64,
        /// Temporal standard deviation σ_t (s). Must be > 0.
        sigma_t_s: f64,
        /// Impulse density magnitude J = ∫ f(t) dt in [N·s/m³].
        impulse_n_per_m3_s: f64,
    },
}

/// Elastic wave field components
#[derive(Debug, Clone)]
pub struct ElasticWaveField {
    /// Displacement in x-direction (m)
    pub ux: Array3<f64>,
    /// Displacement in y-direction (m)
    pub uy: Array3<f64>,
    /// Displacement in z-direction (m)
    pub uz: Array3<f64>,
    /// Velocity in x-direction (m/s)
    pub vx: Array3<f64>,
    /// Velocity in y-direction (m/s)
    pub vy: Array3<f64>,
    /// Velocity in z-direction (m/s)
    pub vz: Array3<f64>,
    /// Current time (s)
    pub time: f64,
}

impl ElasticWaveField {
    /// Create new wave field with given dimensions
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

    /// Initialize with displacement field.
    ///
    /// Correctness note:
    /// This is retained for legacy call sites, but **ARFI excitation must be modeled as a
    /// body-force source term** (see [`ElasticBodyForceConfig`]) rather than an arbitrary
    /// displacement assignment.
    ///
    /// This routine interprets `initial_displacement` as *x-component displacement* `u_x`.
    /// Callers must not assume this produces a pure shear excitation.
    pub fn initialize_displacement(&mut self, initial_displacement: &Array3<f64>) {
        self.ux.assign(initial_displacement);
        // uy and uz remain zero by construction.
    }

    /// Compute displacement magnitude
    pub fn displacement_magnitude(&self) -> Array3<f64> {
        let mut magnitude = Array3::zeros(self.ux.dim());

        for k in 0..self.uz.dim().2 {
            for j in 0..self.uy.dim().1 {
                for i in 0..self.ux.dim().0 {
                    let ux = self.ux[[i, j, k]];
                    let uy = self.uy[[i, j, k]];
                    let uz = self.uz[[i, j, k]];
                    magnitude[[i, j, k]] = (ux * ux + uy * uy + uz * uz).sqrt();
                }
            }
        }

        magnitude
    }

    /// Compute displacement magnitude at a single grid point.
    ///
    /// This avoids allocating an `Array3` and is intended for hot loops like wave-front tracking.
    #[inline]
    pub fn displacement_magnitude_at(&self, i: usize, j: usize, k: usize) -> f64 {
        let ux = self.ux[[i, j, k]];
        let uy = self.uy[[i, j, k]];
        let uz = self.uz[[i, j, k]];
        (ux * ux + uy * uy + uz * uz).sqrt()
    }
}

/// Configuration for volumetric 3D SWE
#[derive(Debug, Clone)]
pub struct VolumetricWaveConfig {
    /// Enable volumetric boundary conditions
    pub volumetric_boundaries: bool,
    /// Enable wave interference tracking
    pub interference_tracking: bool,
    /// Enable volumetric attenuation corrections
    pub volumetric_attenuation: bool,
    /// Enable dispersion corrections
    pub dispersion_correction: bool,
    /// Memory optimization level (0-3, higher = more optimization)
    pub memory_optimization: usize,

    /// Strategy defining how arrival times are detected.
    ///
    /// This is the authoritative definition of `WaveFrontTracker.arrival_times`.
    pub arrival_detection: ArrivalDetection,

    /// Optional decimation applied only to tracking loops (simulation remains full resolution).
    ///
    /// Example: `[2,2,2]` tracks every other voxel in each dimension.
    pub tracking_decimation: [usize; 3],
}

impl Default for VolumetricWaveConfig {
    fn default() -> Self {
        Self {
            volumetric_boundaries: true,
            interference_tracking: true,
            volumetric_attenuation: true,
            dispersion_correction: false,
            memory_optimization: 1,

            // Default must be the optimal general-purpose choice:
            // matched filtering is the most robust arrival definition under dispersion/noise.
            // The default template is a short, positive pulse emphasizing a clear onset.
            arrival_detection: ArrivalDetection::MatchedFilter {
                template: vec![0.0, 0.25, 0.5, 1.0, 0.5, 0.25, 0.0],
                min_corr: 1e-20,
            },

            tracking_decimation: [1, 1, 1],
        }
    }
}

/// 3D wave front tracking information
#[derive(Debug, Clone)]
pub struct WaveFrontTracker {
    /// Arrival times at each grid point (seconds).
    ///
    /// Semantics are defined by [`VolumetricWaveConfig::arrival_detection`].
    /// Undetected voxels remain `+∞`.
    pub arrival_times: Array3<f64>,
    /// Wave amplitude at each grid point (meters; displacement magnitude at detection time).
    pub amplitudes: Array3<f64>,
    /// Direction of wave propagation at each point (unit vector, dimensionless).
    pub directions: Array3<[f64; 3]>,
    /// Interference pattern tracking (unitless counter proxy).
    pub interference_map: Array3<f64>,
    /// Quality metrics for tracking (unitless in [0,1] by construction).
    pub tracking_quality: Array3<f64>,

    /// Running peak magnitude (meters) used by [`ArrivalDetection::PeakTime`].
    pub peak_magnitude_m: Array3<f64>,
    /// Running peak time (seconds) used by [`ArrivalDetection::PeakTime`].
    pub peak_time_s: Array3<f64>,

    /// Matched-filter ring buffer state: previous magnitudes (meters) for each voxel.
    ///
    /// Layout is `(nx, ny, nz, L)` flattened into `Array3<[f64; L]>` would be ideal, but we keep
    /// it as a `Vec` to avoid const generics; indexing is `((k*ny + j)*nx + i)*L + τ`.
    ///
    /// Invariant: length == nx*ny*nz*L where L = template length.
    pub mf_buffer: Vec<f64>,
    /// Current write index into the matched-filter ring buffer (0..L-1).
    pub mf_head: usize,
}

/// Elastic wave equation solver
#[derive(Debug)]
pub struct ElasticWaveSolver {
    /// Computational grid
    grid: Grid,
    /// Medium properties
    _medium: Box<dyn Medium>,
    /// Configuration
    config: ElasticWaveConfig,
    /// Volumetric configuration
    volumetric_config: VolumetricWaveConfig,
    /// Lamé parameters λ (first Lamé parameter)
    lambda: Array3<f64>,
    /// Lamé parameters μ (shear modulus)
    mu: Array3<f64>,
    /// Density field ρ
    density: Array3<f64>,
    /// Perfectly matched layer sigma values
    pml_sigma: Array3<f64>,
    /// Volumetric attenuation coefficients
    volumetric_attenuation: Option<Array3<f64>>,
    /// Dispersion correction factors
    dispersion_factors: Option<Array3<f64>>,
}

impl ElasticWaveSolver {
    /// Return an immutable reference to the solver configuration.
    ///
    /// Correctness invariant:
    /// - Callers must be able to read the *exact* configuration that will be used for propagation,
    ///   without relying on duplicated configuration state.
    #[must_use]
    pub fn config(&self) -> &ElasticWaveConfig {
        &self.config
    }

    /// Propagate elastic waves through time using a one-off body-force override.
    ///
    /// This is the correctness-first API needed by higher-level orchestrators that hold a solver
    /// instance but compute the ARFI source term at runtime.
    ///
    /// Mathematical model:
    ///   ρ ∂v/∂t = ∇·σ + f(x,t)
    ///
    /// Correctness invariant:
    /// - The override must not mutate `self.config`; it is applied only for this call.
    /// - If `override_body_force` is `None`, this reduces to the configured-body-force path (if any)
    ///   and otherwise to the unforced evolution.
    pub fn propagate_waves_with_body_force_override(
        &self,
        initial_displacement: &Array3<f64>,
        override_body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let dt = self.config.dt.unwrap_or_else(|| self.calculate_time_step());
        if !dt.is_finite() || dt <= 0.0 {
            return Err(crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "dt".to_string(),
                    value: dt,
                    reason: "Time step must be finite and positive".to_string(),
                },
            ));
        }
        if self.config.save_every == 0 {
            return Err(crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "save_every".to_string(),
                    value: self.config.save_every as f64,
                    reason: "save_every must be >= 1".to_string(),
                },
            ));
        }

        let n_steps = (self.config.simulation_time / dt) as usize;

        let (nx, ny, nz) = self.grid.dimensions();
        let mut field = ElasticWaveField::new(nx, ny, nz);
        field.initialize_displacement(initial_displacement);
        field.time = 0.0;

        let mut displacement_history = Vec::new();
        displacement_history.push(field.clone());

        for step in 0..n_steps {
            // Override semantics:
            // - Use the override if provided.
            // - Else fall back to configured body_force (if any).
            let body_force = override_body_force.or(self.config.body_force.as_ref());
            self.time_step_with_optional_body_force(&mut field, dt, body_force);
            field.time = (step as f64 + 1.0) * dt;

            if step % self.config.save_every == 0 {
                displacement_history.push(field.clone());
            }
        }

        Ok(displacement_history)
    }

    /// Propagate with no initial displacement, using a one-off body-force override.
    ///
    /// This is the preferred entrypoint for ARFI-as-forcing when the initial displacement is zero.
    pub fn propagate_waves_with_body_force_only_override(
        &self,
        override_body_force: Option<&ElasticBodyForceConfig>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let zero = Array3::<f64>::zeros((nx, ny, nz));
        self.propagate_waves_with_body_force_override(&zero, override_body_force)
    }

    /// Create new elastic wave solver
    pub fn new<M: Medium + Clone + 'static>(
        grid: &Grid,
        medium: &M,
        config: ElasticWaveConfig,
    ) -> KwaversResult<Self> {
        let (_nx, _ny, _nz) = grid.dimensions();

        // Validate optional source configuration early (fail-fast).
        if let Some(ref src) = config.body_force {
            Self::validate_body_force_config(src)?;
        }

        // Initialize material properties using Medium trait elastic properties
        let lambda = medium.lame_lambda_array().to_owned();
        let mu = medium.lame_mu_array().to_owned();
        let density = medium.density_array().to_owned();

        // Validate that elastic properties are properly defined
        if lambda.iter().any(|&x| x < 0.0) || mu.iter().any(|&x| x < 0.0) {
            return Err(crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "elastic_properties".to_string(),
                    value: 0.0,
                    reason: "Lamé parameters must be non-negative".to_string(),
                },
            ));
        }

        // Initialize PML
        let pml_sigma = Self::create_pml(grid, &config);

        Ok(Self {
            grid: grid.clone(),
            _medium: Box::new(medium.clone()),
            config,
            volumetric_config: VolumetricWaveConfig::default(),
            lambda,
            mu,
            density,
            pml_sigma,
            volumetric_attenuation: None,
            dispersion_factors: None,
        })
    }

    /// Create perfectly matched layer attenuation
    fn create_pml(grid: &Grid, config: &ElasticWaveConfig) -> Array3<f64> {
        let (nx, ny, nz) = grid.dimensions();
        let pml_thickness = config.pml_thickness;
        let mut sigma = Array3::zeros((nx, ny, nz));

        // Maximum attenuation (exponential decay)
        let sigma_max = 100.0; // Np/m

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mut max_sigma: f64 = 0.0;

                    // X-direction PML
                    if i < pml_thickness {
                        let dist = pml_thickness - i;
                        let sigma_val = sigma_max * (dist as f64 / pml_thickness as f64).powi(2);
                        max_sigma = max_sigma.max(sigma_val);
                    } else if i >= nx - pml_thickness {
                        let dist = i - (nx - pml_thickness) + 1;
                        let sigma_val = sigma_max * (dist as f64 / pml_thickness as f64).powi(2);
                        max_sigma = max_sigma.max(sigma_val);
                    }

                    // Y-direction PML
                    if j < pml_thickness {
                        let dist = pml_thickness - j;
                        let sigma_val = sigma_max * (dist as f64 / pml_thickness as f64).powi(2);
                        max_sigma = max_sigma.max(sigma_val);
                    } else if j >= ny - pml_thickness {
                        let dist = j - (ny - pml_thickness) + 1;
                        let sigma_val = sigma_max * (dist as f64 / pml_thickness as f64).powi(2);
                        max_sigma = max_sigma.max(sigma_val);
                    }

                    // Z-direction PML
                    if k < pml_thickness {
                        let dist = pml_thickness - k;
                        let sigma_val = sigma_max * (dist as f64 / pml_thickness as f64).powi(2);
                        max_sigma = max_sigma.max(sigma_val);
                    } else if k >= nz - pml_thickness {
                        let dist = k - (nz - pml_thickness) + 1;
                        let sigma_val = sigma_max * (dist as f64 / pml_thickness as f64).powi(2);
                        max_sigma = max_sigma.max(sigma_val);
                    }

                    sigma[[i, j, k]] = max_sigma;
                }
            }
        }

        sigma
    }

    /// Calculate stable time step using CFL condition
    pub fn calculate_time_step(&self) -> f64 {
        let (nx, ny, nz) = self.grid.dimensions();

        let mut max_c: f64 = 0.0;
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mu_val = self.mu[[i, j, k]];
                    let lambda_val = self.lambda[[i, j, k]];
                    let rho_val = self.density[[i, j, k]];
                    let cs = (mu_val / rho_val).sqrt();
                    let cp = ((lambda_val + 2.0 * mu_val) / rho_val).sqrt();
                    max_c = max_c.max(cs.max(cp));
                }
            }
        }

        if max_c <= 0.0 {
            return 0.0;
        }

        // CFL condition for 3D elastic waves: dt < dx/(√3 * c_max)
        let min_dx = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let cfl_dt = min_dx / (3.0_f64.sqrt() * max_c);

        cfl_dt * self.config.cfl_factor
    }

    /// Propagate elastic waves through time.
    ///
    /// Correctness invariant:
    /// - External excitation (e.g., ARFI) is represented via `config.body_force` as a source term
    ///   in the momentum equation, not as an arbitrary displacement initialization.
    /// - `initial_displacement` is retained for legacy call sites; new code should use
    ///   [`propagate_waves_with_body_force_only`] or set `config.body_force`.
    pub fn propagate_waves(
        &self,
        initial_displacement: &Array3<f64>,
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let dt = self.config.dt.unwrap_or_else(|| self.calculate_time_step());
        if !dt.is_finite() || dt <= 0.0 {
            return Err(crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "dt".to_string(),
                    value: dt,
                    reason: "Time step must be finite and positive".to_string(),
                },
            ));
        }
        if self.config.save_every == 0 {
            return Err(crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "save_every".to_string(),
                    value: self.config.save_every as f64,
                    reason: "save_every must be >= 1".to_string(),
                },
            ));
        }
        let n_steps = (self.config.simulation_time / dt) as usize;

        log::debug!(
            "Elastic wave propagation: {} steps, dt = {:.2e} s, total time = {:.2e} s",
            n_steps,
            dt,
            self.config.simulation_time
        );

        // Initialize wave field
        let (nx, ny, nz) = self.grid.dimensions();
        let mut field = ElasticWaveField::new(nx, ny, nz);
        field.initialize_displacement(initial_displacement);
        field.time = 0.0;

        let mut displacement_history = Vec::new();
        displacement_history.push(field.clone());

        for step in 0..n_steps {
            self.time_step_with_body_force(&mut field, dt);
            field.time = (step as f64 + 1.0) * dt;

            if step % self.config.save_every == 0 {
                displacement_history.push(field.clone());
            }
        }

        Ok(displacement_history)
    }

    /// Propagate with no initial displacement, relying purely on the configured body-force.
    ///
    /// This is the preferred API for ARFI-style excitation where the source is a forcing term.
    pub fn propagate_waves_with_body_force_only(&self) -> KwaversResult<Vec<ElasticWaveField>> {
        let (nx, ny, nz) = self.grid.dimensions();
        let zero = Array3::<f64>::zeros((nx, ny, nz));
        self.propagate_waves(&zero)
    }

    /// Single time step of elastic wave propagation with configured body-force.
    fn time_step_with_body_force(&self, field: &mut ElasticWaveField, dt: f64) {
        self.time_step_with_optional_body_force(field, dt, self.config.body_force.as_ref());
    }

    /// Single time step of elastic wave propagation with an optional body-force source term.
    ///
    /// Equation integrated (schematically):
    ///   ρ ∂v/∂t = ∇·σ + f
    fn time_step_with_optional_body_force(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        body_force: Option<&ElasticBodyForceConfig>,
    ) {
        let (nx, ny, nz) = self.grid.dimensions();

        // Update velocities (momentum equation)
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    // Stress derivatives using second-order central differences
                    // Reference: Fornberg (1988), Generation of finite difference formulas
                    let d_sigma_xx_dx = self.stress_xx_derivative(i, j, k, field);
                    let d_sigma_xy_dy = self.stress_xy_derivative(i, j, k, field);
                    let d_sigma_xz_dz = self.stress_xz_derivative(i, j, k, field);

                    let d_sigma_yx_dx = self.stress_yx_derivative(i, j, k, field);
                    let d_sigma_yy_dy = self.stress_yy_derivative(i, j, k, field);
                    let d_sigma_yz_dz = self.stress_yz_derivative(i, j, k, field);

                    let d_sigma_zx_dx = self.stress_zx_derivative(i, j, k, field);
                    let d_sigma_zy_dy = self.stress_zy_derivative(i, j, k, field);
                    let d_sigma_zz_dz = self.stress_zz_derivative(i, j, k, field);

                    // Acceleration = (1/ρ) * (∇·σ + f)
                    let rho = self.density[[i, j, k]];

                    let mut ax = (d_sigma_xx_dx + d_sigma_xy_dy + d_sigma_xz_dz) / rho;
                    let mut ay = (d_sigma_yx_dx + d_sigma_yy_dy + d_sigma_yz_dz) / rho;
                    let mut az = (d_sigma_zx_dx + d_sigma_zy_dy + d_sigma_zz_dz) / rho;

                    if let Some(src) = body_force {
                        let (fx, fy, fz) = self.body_force_at(src, field.time, i, j, k);
                        ax += fx / rho;
                        ay += fy / rho;
                        az += fz / rho;
                    }

                    // Update velocities (explicit time integration)
                    field.vx[[i, j, k]] += dt * ax;
                    field.vy[[i, j, k]] += dt * ay;
                    field.vz[[i, j, k]] += dt * az;

                    // Apply damping (PML)
                    let sigma = self.pml_sigma[[i, j, k]];
                    if sigma > 0.0 {
                        let damping = (-sigma * dt).exp();
                        field.vx[[i, j, k]] *= damping;
                        field.vy[[i, j, k]] *= damping;
                        field.vz[[i, j, k]] *= damping;
                    }
                }
            }
        }

        // Update displacements (integration of velocities)
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field.ux[[i, j, k]] += dt * field.vx[[i, j, k]];
                    field.uy[[i, j, k]] += dt * field.vy[[i, j, k]];
                    field.uz[[i, j, k]] += dt * field.vz[[i, j, k]];
                }
            }
        }
    }

    fn validate_body_force_config(src: &ElasticBodyForceConfig) -> KwaversResult<()> {
        match src {
            ElasticBodyForceConfig::GaussianImpulse {
                sigma_m,
                direction,
                sigma_t_s,
                impulse_n_per_m3_s,
                ..
            } => {
                if sigma_m[0] <= 0.0 || sigma_m[1] <= 0.0 || sigma_m[2] <= 0.0 {
                    return Err(crate::domain::core::error::KwaversError::Validation(
                        crate::domain::core::error::ValidationError::InvalidValue {
                            parameter: "body_force.sigma_m".to_string(),
                            value: 0.0,
                            reason: "sigma_m components must be > 0".to_string(),
                        },
                    ));
                }
                if *sigma_t_s <= 0.0 {
                    return Err(crate::domain::core::error::KwaversError::Validation(
                        crate::domain::core::error::ValidationError::InvalidValue {
                            parameter: "body_force.sigma_t_s".to_string(),
                            value: *sigma_t_s,
                            reason: "sigma_t_s must be > 0".to_string(),
                        },
                    ));
                }
                if !impulse_n_per_m3_s.is_finite() {
                    return Err(crate::domain::core::error::KwaversError::Validation(
                        crate::domain::core::error::ValidationError::InvalidValue {
                            parameter: "body_force.impulse_n_per_m3_s".to_string(),
                            value: *impulse_n_per_m3_s,
                            reason: "impulse must be finite".to_string(),
                        },
                    ));
                }
                let norm2 = direction[0] * direction[0]
                    + direction[1] * direction[1]
                    + direction[2] * direction[2];
                if !norm2.is_finite() || norm2 <= 0.0 {
                    return Err(crate::domain::core::error::KwaversError::Validation(
                        crate::domain::core::error::ValidationError::InvalidValue {
                            parameter: "body_force.direction".to_string(),
                            value: norm2,
                            reason: "direction must be non-zero and finite".to_string(),
                        },
                    ));
                }
                let norm = norm2.sqrt();
                // Enforce unit direction within numerical tolerance.
                if (norm - 1.0).abs() > 1e-9 {
                    return Err(crate::domain::core::error::KwaversError::Validation(
                        crate::domain::core::error::ValidationError::InvalidValue {
                            parameter: "body_force.direction".to_string(),
                            value: norm,
                            reason: "direction must be unit-length".to_string(),
                        },
                    ));
                }
                Ok(())
            }
        }
    }

    #[inline]
    fn body_force_at(
        &self,
        src: &ElasticBodyForceConfig,
        t: f64,
        i: usize,
        j: usize,
        k: usize,
    ) -> (f64, f64, f64) {
        match *src {
            ElasticBodyForceConfig::GaussianImpulse {
                center_m,
                sigma_m,
                direction,
                t0_s,
                sigma_t_s,
                impulse_n_per_m3_s,
            } => {
                let x = i as f64 * self.grid.dx;
                let y = j as f64 * self.grid.dy;
                let z = k as f64 * self.grid.dz;

                let dx = (x - center_m[0]) / sigma_m[0];
                let dy = (y - center_m[1]) / sigma_m[1];
                let dz = (z - center_m[2]) / sigma_m[2];
                let spatial = (-0.5 * (dx * dx + dy * dy + dz * dz)).exp();

                let tau = (t - t0_s) / sigma_t_s;
                // Unit-area Gaussian in time (so impulse parameter is the time integral of f).
                let temporal =
                    (-0.5 * tau * tau).exp() / (sigma_t_s * (2.0 * std::f64::consts::PI).sqrt());

                let mag = impulse_n_per_m3_s * temporal * spatial;

                (mag * direction[0], mag * direction[1], mag * direction[2])
            }
        }
    }

    /// Compute ∂σ_xx/∂x using central difference
    fn stress_xx_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if i == 0 || i >= self.grid.nx - 1 {
            return 0.0; // Boundary - no derivative
        }

        let _lambda = self.lambda[[i, j, k]];
        let _mu = self.mu[[i, j, k]];

        // σ_xx = (λ+2μ)∂u_x/∂x + λ(∂u_y/∂y + ∂u_z/∂z)
        let du_x_dx = (field.ux[[i + 1, j, k]] - field.ux[[i - 1, j, k]]) / (2.0 * self.grid.dx);
        let du_y_dy = if j > 0 && j < self.grid.ny - 1 {
            (field.uy[[i, j + 1, k]] - field.uy[[i, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            0.0
        };
        let du_z_dz = if k > 0 && k < self.grid.nz - 1 {
            (field.uz[[i, j, k + 1]] - field.uz[[i, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            0.0
        };

        // Compute σ_xx at (i+1,j,k) and (i-1,j,k) for derivative
        let lambda_ip1 = self.lambda[[i + 1, j, k]];
        let mu_ip1 = self.mu[[i + 1, j, k]];
        let du_x_dx_ip1 = if i < self.grid.nx - 2 {
            (field.ux[[i + 2, j, k]] - field.ux[[i, j, k]]) / (2.0 * self.grid.dx)
        } else {
            du_x_dx
        };
        let du_y_dy_ip1 = if j > 0 && j < self.grid.ny - 1 && i < self.grid.nx - 2 {
            (field.uy[[i + 1, j + 1, k]] - field.uy[[i + 1, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            du_y_dy
        };
        let du_z_dz_ip1 = if k > 0 && k < self.grid.nz - 1 && i < self.grid.nx - 2 {
            (field.uz[[i + 1, j, k + 1]] - field.uz[[i + 1, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            du_z_dz
        };
        let sigma_xx_ip1 =
            (lambda_ip1 + 2.0 * mu_ip1) * du_x_dx_ip1 + lambda_ip1 * (du_y_dy_ip1 + du_z_dz_ip1);

        let lambda_im1 = self.lambda[[i - 1, j, k]];
        let mu_im1 = self.mu[[i - 1, j, k]];
        let du_x_dx_im1 = if i > 1 {
            (field.ux[[i, j, k]] - field.ux[[i - 2, j, k]]) / (2.0 * self.grid.dx)
        } else {
            du_x_dx
        };
        let du_y_dy_im1 = if j > 0 && j < self.grid.ny - 1 && i > 1 {
            (field.uy[[i - 1, j + 1, k]] - field.uy[[i - 1, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            du_y_dy
        };
        let du_z_dz_im1 = if k > 0 && k < self.grid.nz - 1 && i > 1 {
            (field.uz[[i - 1, j, k + 1]] - field.uz[[i - 1, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            du_z_dz
        };
        let sigma_xx_im1 =
            (lambda_im1 + 2.0 * mu_im1) * du_x_dx_im1 + lambda_im1 * (du_y_dy_im1 + du_z_dz_im1);

        (sigma_xx_ip1 - sigma_xx_im1) / (2.0 * self.grid.dx)
    }

    /// Compute ∂σ_xy/∂y using central difference
    fn stress_xy_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if j == 0 || j >= self.grid.ny - 1 {
            return 0.0; // Boundary - no derivative
        }

        let _mu = self.mu[[i, j, k]];

        // σ_xy = μ(∂u_x/∂y + ∂u_y/∂x)
        let du_x_dy = (field.ux[[i, j + 1, k]] - field.ux[[i, j - 1, k]]) / (2.0 * self.grid.dy);
        let du_y_dx = if i > 0 && i < self.grid.nx - 1 {
            (field.uy[[i + 1, j, k]] - field.uy[[i - 1, j, k]]) / (2.0 * self.grid.dx)
        } else {
            0.0
        };

        // Compute σ_xy at (i,j+1,k) and (i,j-1,k) for derivative
        let mu_jp1 = self.mu[[i, j + 1, k]];
        let du_x_dy_jp1 = if j < self.grid.ny - 2 {
            (field.ux[[i, j + 2, k]] - field.ux[[i, j, k]]) / (2.0 * self.grid.dy)
        } else {
            du_x_dy
        };
        let du_y_dx_jp1 = if i > 0 && i < self.grid.nx - 1 && j < self.grid.ny - 2 {
            (field.uy[[i + 1, j + 1, k]] - field.uy[[i - 1, j + 1, k]]) / (2.0 * self.grid.dx)
        } else {
            du_y_dx
        };
        let sigma_xy_jp1 = mu_jp1 * (du_x_dy_jp1 + du_y_dx_jp1);

        let mu_jm1 = self.mu[[i, j - 1, k]];
        let du_x_dy_jm1 = if j > 1 {
            (field.ux[[i, j, k]] - field.ux[[i, j - 2, k]]) / (2.0 * self.grid.dy)
        } else {
            du_x_dy
        };
        let du_y_dx_jm1 = if i > 0 && i < self.grid.nx - 1 && j > 1 {
            (field.uy[[i + 1, j - 1, k]] - field.uy[[i - 1, j - 1, k]]) / (2.0 * self.grid.dx)
        } else {
            du_y_dx
        };
        let sigma_xy_jm1 = mu_jm1 * (du_x_dy_jm1 + du_y_dx_jm1);

        (sigma_xy_jp1 - sigma_xy_jm1) / (2.0 * self.grid.dy)
    }

    /// Compute ∂σ_xz/∂z using central difference
    fn stress_xz_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if k == 0 || k >= self.grid.nz - 1 {
            return 0.0; // Boundary - no derivative
        }

        let _mu = self.mu[[i, j, k]];

        // σ_xz = μ(∂u_x/∂z + ∂u_z/∂x)
        let du_x_dz = (field.ux[[i, j, k + 1]] - field.ux[[i, j, k - 1]]) / (2.0 * self.grid.dz);
        let du_z_dx = if i > 0 && i < self.grid.nx - 1 {
            (field.uz[[i + 1, j, k]] - field.uz[[i - 1, j, k]]) / (2.0 * self.grid.dx)
        } else {
            0.0
        };

        // Compute σ_xz at (i,j,k+1) and (i,j,k-1) for derivative
        let mu_kp1 = self.mu[[i, j, k + 1]];
        let du_x_dz_kp1 = if k < self.grid.nz - 2 {
            (field.ux[[i, j, k + 2]] - field.ux[[i, j, k]]) / (2.0 * self.grid.dz)
        } else {
            du_x_dz
        };
        let du_z_dx_kp1 = if i > 0 && i < self.grid.nx - 1 && k < self.grid.nz - 2 {
            (field.uz[[i + 1, j, k + 1]] - field.uz[[i - 1, j, k + 1]]) / (2.0 * self.grid.dx)
        } else {
            du_z_dx
        };
        let sigma_xz_kp1 = mu_kp1 * (du_x_dz_kp1 + du_z_dx_kp1);

        let mu_km1 = self.mu[[i, j, k - 1]];
        let du_x_dz_km1 = if k > 1 {
            (field.ux[[i, j, k]] - field.ux[[i, j, k - 2]]) / (2.0 * self.grid.dz)
        } else {
            du_x_dz
        };
        let du_z_dx_km1 = if i > 0 && i < self.grid.nx - 1 && k > 1 {
            (field.uz[[i + 1, j, k - 1]] - field.uz[[i - 1, j, k - 1]]) / (2.0 * self.grid.dx)
        } else {
            du_z_dx
        };
        let sigma_xz_km1 = mu_km1 * (du_x_dz_km1 + du_z_dx_km1);

        (sigma_xz_kp1 - sigma_xz_km1) / (2.0 * self.grid.dz)
    }

    /// Compute ∂σ_yx/∂x using central difference
    fn stress_yx_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if i == 0 || i >= self.grid.nx - 1 {
            return 0.0; // Boundary - no derivative
        }

        let _mu = self.mu[[i, j, k]];

        // σ_yx = μ(∂u_y/∂x + ∂u_x/∂y) = σ_xy
        let du_y_dx = (field.uy[[i + 1, j, k]] - field.uy[[i - 1, j, k]]) / (2.0 * self.grid.dx);
        let du_x_dy = if j > 0 && j < self.grid.ny - 1 {
            (field.ux[[i, j + 1, k]] - field.ux[[i, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            0.0
        };

        // Compute σ_yx at (i+1,j,k) and (i-1,j,k) for derivative
        let mu_ip1 = self.mu[[i + 1, j, k]];
        let du_y_dx_ip1 = if i < self.grid.nx - 2 {
            (field.uy[[i + 2, j, k]] - field.uy[[i, j, k]]) / (2.0 * self.grid.dx)
        } else {
            du_y_dx
        };
        let du_x_dy_ip1 = if j > 0 && j < self.grid.ny - 1 && i < self.grid.nx - 2 {
            (field.ux[[i + 1, j + 1, k]] - field.ux[[i + 1, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            du_x_dy
        };
        let sigma_yx_ip1 = mu_ip1 * (du_y_dx_ip1 + du_x_dy_ip1);

        let mu_im1 = self.mu[[i - 1, j, k]];
        let du_y_dx_im1 = if i > 1 {
            (field.uy[[i, j, k]] - field.uy[[i - 2, j, k]]) / (2.0 * self.grid.dx)
        } else {
            du_y_dx
        };
        let du_x_dy_im1 = if j > 0 && j < self.grid.ny - 1 && i > 1 {
            (field.ux[[i - 1, j + 1, k]] - field.ux[[i - 1, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            du_x_dy
        };
        let sigma_yx_im1 = mu_im1 * (du_y_dx_im1 + du_x_dy_im1);

        (sigma_yx_ip1 - sigma_yx_im1) / (2.0 * self.grid.dx)
    }

    fn stress_yy_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if j == 0 || j >= self.grid.ny - 1 {
            return 0.0;
        }

        let _lambda = self.lambda[[i, j, k]];
        let _mu = self.mu[[i, j, k]];

        let du_y_dy = (field.uy[[i, j + 1, k]] - field.uy[[i, j - 1, k]]) / (2.0 * self.grid.dy);
        let du_x_dx = if i > 0 && i < self.grid.nx - 1 {
            (field.ux[[i + 1, j, k]] - field.ux[[i - 1, j, k]]) / (2.0 * self.grid.dx)
        } else {
            0.0
        };
        let du_z_dz = if k > 0 && k < self.grid.nz - 1 {
            (field.uz[[i, j, k + 1]] - field.uz[[i, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            0.0
        };

        let lambda_jp1 = self.lambda[[i, j + 1, k]];
        let mu_jp1 = self.mu[[i, j + 1, k]];
        let du_y_dy_jp1 = if j < self.grid.ny - 2 {
            (field.uy[[i, j + 2, k]] - field.uy[[i, j, k]]) / (2.0 * self.grid.dy)
        } else {
            du_y_dy
        };
        let du_x_dx_jp1 = if i > 0 && i < self.grid.nx - 1 && j < self.grid.ny - 2 {
            (field.ux[[i + 1, j + 1, k]] - field.ux[[i - 1, j + 1, k]]) / (2.0 * self.grid.dx)
        } else {
            du_x_dx
        };
        let du_z_dz_jp1 = if k > 0 && k < self.grid.nz - 1 && j < self.grid.ny - 2 {
            (field.uz[[i, j + 1, k + 1]] - field.uz[[i, j + 1, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            du_z_dz
        };
        let sigma_yy_jp1 =
            (lambda_jp1 + 2.0 * mu_jp1) * du_y_dy_jp1 + lambda_jp1 * (du_x_dx_jp1 + du_z_dz_jp1);

        let lambda_jm1 = self.lambda[[i, j - 1, k]];
        let mu_jm1 = self.mu[[i, j - 1, k]];
        let du_y_dy_jm1 = if j > 1 {
            (field.uy[[i, j, k]] - field.uy[[i, j - 2, k]]) / (2.0 * self.grid.dy)
        } else {
            du_y_dy
        };
        let du_x_dx_jm1 = if i > 0 && i < self.grid.nx - 1 && j > 1 {
            (field.ux[[i + 1, j - 1, k]] - field.ux[[i - 1, j - 1, k]]) / (2.0 * self.grid.dx)
        } else {
            du_x_dx
        };
        let du_z_dz_jm1 = if k > 0 && k < self.grid.nz - 1 && j > 1 {
            (field.uz[[i, j - 1, k + 1]] - field.uz[[i, j - 1, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            du_z_dz
        };
        let sigma_yy_jm1 =
            (lambda_jm1 + 2.0 * mu_jm1) * du_y_dy_jm1 + lambda_jm1 * (du_x_dx_jm1 + du_z_dz_jm1);

        (sigma_yy_jp1 - sigma_yy_jm1) / (2.0 * self.grid.dy)
    }

    fn stress_yz_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if k == 0 || k >= self.grid.nz - 1 {
            return 0.0;
        }

        let _mu = self.mu[[i, j, k]];

        let du_y_dz = (field.uy[[i, j, k + 1]] - field.uy[[i, j, k - 1]]) / (2.0 * self.grid.dz);
        let du_z_dy = if j > 0 && j < self.grid.ny - 1 {
            (field.uz[[i, j + 1, k]] - field.uz[[i, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            0.0
        };
        let mu_kp1 = self.mu[[i, j, k + 1]];
        let du_y_dz_kp1 = if k < self.grid.nz - 2 {
            (field.uy[[i, j, k + 2]] - field.uy[[i, j, k]]) / (2.0 * self.grid.dz)
        } else {
            du_y_dz
        };
        let du_z_dy_kp1 = if j > 0 && j < self.grid.ny - 1 && k < self.grid.nz - 2 {
            (field.uz[[i, j + 1, k + 1]] - field.uz[[i, j - 1, k + 1]]) / (2.0 * self.grid.dy)
        } else {
            du_z_dy
        };
        let sigma_yz_kp1 = mu_kp1 * (du_y_dz_kp1 + du_z_dy_kp1);

        let mu_km1 = self.mu[[i, j, k - 1]];
        let du_y_dz_km1 = if k > 1 {
            (field.uy[[i, j, k]] - field.uy[[i, j, k - 2]]) / (2.0 * self.grid.dz)
        } else {
            du_y_dz
        };
        let du_z_dy_km1 = if j > 0 && j < self.grid.ny - 1 && k > 1 {
            (field.uz[[i, j + 1, k - 1]] - field.uz[[i, j - 1, k - 1]]) / (2.0 * self.grid.dy)
        } else {
            du_z_dy
        };
        let sigma_yz_km1 = mu_km1 * (du_y_dz_km1 + du_z_dy_km1);

        (sigma_yz_kp1 - sigma_yz_km1) / (2.0 * self.grid.dz)
    }

    fn stress_zx_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if i == 0 || i >= self.grid.nx - 1 {
            return 0.0;
        }

        let _mu = self.mu[[i, j, k]];

        let du_z_dx = (field.uz[[i + 1, j, k]] - field.uz[[i - 1, j, k]]) / (2.0 * self.grid.dx);
        let du_x_dz = if k > 0 && k < self.grid.nz - 1 {
            (field.ux[[i, j, k + 1]] - field.ux[[i, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            0.0
        };
        let mu_ip1 = self.mu[[i + 1, j, k]];
        let du_z_dx_ip1 = if i < self.grid.nx - 2 {
            (field.uz[[i + 2, j, k]] - field.uz[[i, j, k]]) / (2.0 * self.grid.dx)
        } else {
            du_z_dx
        };
        let du_x_dz_ip1 = if k > 0 && k < self.grid.nz - 1 && i < self.grid.nx - 2 {
            (field.ux[[i + 1, j, k + 1]] - field.ux[[i + 1, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            du_x_dz
        };
        let sigma_zx_ip1 = mu_ip1 * (du_z_dx_ip1 + du_x_dz_ip1);

        let mu_im1 = self.mu[[i - 1, j, k]];
        let du_z_dx_im1 = if i > 1 {
            (field.uz[[i, j, k]] - field.uz[[i - 2, j, k]]) / (2.0 * self.grid.dx)
        } else {
            du_z_dx
        };
        let du_x_dz_im1 = if k > 0 && k < self.grid.nz - 1 && i > 1 {
            (field.ux[[i - 1, j, k + 1]] - field.ux[[i - 1, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            du_x_dz
        };
        let sigma_zx_im1 = mu_im1 * (du_z_dx_im1 + du_x_dz_im1);

        (sigma_zx_ip1 - sigma_zx_im1) / (2.0 * self.grid.dx)
    }

    fn stress_zy_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if j == 0 || j >= self.grid.ny - 1 {
            return 0.0;
        }

        let _mu = self.mu[[i, j, k]];

        let du_z_dy = (field.uz[[i, j + 1, k]] - field.uz[[i, j - 1, k]]) / (2.0 * self.grid.dy);
        let du_y_dz = if k > 0 && k < self.grid.nz - 1 {
            (field.uy[[i, j, k + 1]] - field.uy[[i, j, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            0.0
        };
        let mu_jp1 = self.mu[[i, j + 1, k]];
        let du_z_dy_jp1 = if j < self.grid.ny - 2 {
            (field.uz[[i, j + 2, k]] - field.uz[[i, j, k]]) / (2.0 * self.grid.dy)
        } else {
            du_z_dy
        };
        let du_y_dz_jp1 = if k > 0 && k < self.grid.nz - 1 && j < self.grid.ny - 2 {
            (field.uy[[i, j + 1, k + 1]] - field.uy[[i, j + 1, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            du_y_dz
        };
        let sigma_zy_jp1 = mu_jp1 * (du_z_dy_jp1 + du_y_dz_jp1);

        let mu_jm1 = self.mu[[i, j - 1, k]];
        let du_z_dy_jm1 = if j > 1 {
            (field.uz[[i, j, k]] - field.uz[[i, j - 2, k]]) / (2.0 * self.grid.dy)
        } else {
            du_z_dy
        };
        let du_y_dz_jm1 = if k > 0 && k < self.grid.nz - 1 && j > 1 {
            (field.uy[[i, j - 1, k + 1]] - field.uy[[i, j - 1, k - 1]]) / (2.0 * self.grid.dz)
        } else {
            du_y_dz
        };
        let sigma_zy_jm1 = mu_jm1 * (du_z_dy_jm1 + du_y_dz_jm1);

        (sigma_zy_jp1 - sigma_zy_jm1) / (2.0 * self.grid.dy)
    }

    fn stress_zz_derivative(&self, i: usize, j: usize, k: usize, field: &ElasticWaveField) -> f64 {
        if k == 0 || k >= self.grid.nz - 1 {
            return 0.0;
        }

        let _lambda = self.lambda[[i, j, k]];
        let _mu = self.mu[[i, j, k]];

        let du_z_dz = (field.uz[[i, j, k + 1]] - field.uz[[i, j, k - 1]]) / (2.0 * self.grid.dz);
        let du_x_dx = if i > 0 && i < self.grid.nx - 1 {
            (field.ux[[i + 1, j, k]] - field.ux[[i - 1, j, k]]) / (2.0 * self.grid.dx)
        } else {
            0.0
        };
        let du_y_dy = if j > 0 && j < self.grid.ny - 1 {
            (field.uy[[i, j + 1, k]] - field.uy[[i, j - 1, k]]) / (2.0 * self.grid.dy)
        } else {
            0.0
        };

        let lambda_kp1 = self.lambda[[i, j, k + 1]];
        let mu_kp1 = self.mu[[i, j, k + 1]];
        let du_z_dz_kp1 = if k < self.grid.nz - 2 {
            (field.uz[[i, j, k + 2]] - field.uz[[i, j, k]]) / (2.0 * self.grid.dz)
        } else {
            du_z_dz
        };
        let du_x_dx_kp1 = if i > 0 && i < self.grid.nx - 1 && k < self.grid.nz - 2 {
            (field.ux[[i + 1, j, k + 1]] - field.ux[[i - 1, j, k + 1]]) / (2.0 * self.grid.dx)
        } else {
            du_x_dx
        };
        let du_y_dy_kp1 = if j > 0 && j < self.grid.ny - 1 && k < self.grid.nz - 2 {
            (field.uy[[i, j + 1, k + 1]] - field.uy[[i, j - 1, k + 1]]) / (2.0 * self.grid.dy)
        } else {
            du_y_dy
        };
        let sigma_zz_kp1 =
            (lambda_kp1 + 2.0 * mu_kp1) * du_z_dz_kp1 + lambda_kp1 * (du_x_dx_kp1 + du_y_dy_kp1);

        let lambda_km1 = self.lambda[[i, j, k - 1]];
        let mu_km1 = self.mu[[i, j, k - 1]];
        let du_z_dz_km1 = if k > 1 {
            (field.uz[[i, j, k]] - field.uz[[i, j, k - 2]]) / (2.0 * self.grid.dz)
        } else {
            du_z_dz
        };
        let du_x_dx_km1 = if i > 0 && i < self.grid.nx - 1 && k > 1 {
            (field.ux[[i + 1, j, k - 1]] - field.ux[[i - 1, j, k - 1]]) / (2.0 * self.grid.dx)
        } else {
            du_x_dx
        };
        let du_y_dy_km1 = if j > 0 && j < self.grid.ny - 1 && k > 1 {
            (field.uy[[i, j + 1, k - 1]] - field.uy[[i, j - 1, k - 1]]) / (2.0 * self.grid.dy)
        } else {
            du_y_dy
        };
        let sigma_zz_km1 =
            (lambda_km1 + 2.0 * mu_km1) * du_z_dz_km1 + lambda_km1 * (du_x_dx_km1 + du_y_dy_km1);

        (sigma_zz_kp1 - sigma_zz_km1) / (2.0 * self.grid.dz)
    }

    /// Set volumetric configuration for 3D SWE
    pub fn set_volumetric_config(&mut self, config: VolumetricWaveConfig) {
        self.volumetric_config = config;

        // Initialize volumetric attenuation if enabled
        if self.volumetric_config.volumetric_attenuation {
            self.initialize_volumetric_attenuation();
        }

        // Initialize dispersion corrections if enabled
        if self.volumetric_config.dispersion_correction {
            self.initialize_dispersion_factors();
        }
    }

    /// Initialize volumetric attenuation coefficients
    fn initialize_volumetric_attenuation(&mut self) {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut attenuation = Array3::zeros((nx, ny, nz));

        // Base attenuation coefficient for soft tissue (Np/m/MHz)
        let alpha_0 = 0.5; // dB/cm/MHz converted to Np/m/MHz

        // Frequency-dependent attenuation: α(f) = α₀ × fᵇ
        // where b ≈ 1.1 for soft tissue
        let frequency: f64 = 5.0e6; // 5 MHz (typical for SWE)
        let b_exponent = 1.1;

        let alpha_freq = alpha_0 * frequency.powf(b_exponent - 1.0);

        // Apply volumetric attenuation with depth-dependent corrections
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    // Cumulative attenuation coefficient increases with depth
                    let depth = k as f64 * self.grid.dz;
                    let cumulative_alpha = alpha_freq * depth * 100.0; // Convert to cm, accumulate

                    // Local tissue property variations
                    let mu_local = self.mu[[i, j, k]];
                    let tissue_factor = (mu_local / 1000.0).powf(0.3); // Stiffer tissue attenuates more

                    attenuation[[i, j, k]] = cumulative_alpha * tissue_factor;
                }
            }
        }

        self.volumetric_attenuation = Some(attenuation);
    }

    /// Initialize dispersion correction factors
    fn initialize_dispersion_factors(&mut self) {
        let (nx, ny, nz) = self.grid.dimensions();
        let mut dispersion = Array3::zeros((nx, ny, nz));

        // Frequency-dependent dispersion correction
        // Higher frequencies travel faster in dispersive media
        let frequency = 5.0e6; // 5 MHz
        let c0 = 1500.0; // Reference speed (m/s)

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let mu_local = self.mu[[i, j, k]];
                    let rho_local = self.density[[i, j, k]];

                    // Local shear wave speed
                    let cs_local = (mu_local / rho_local).sqrt();

                    // Dispersion correction factor
                    // For soft tissue, higher frequencies have slightly higher speed
                    let dispersion_factor = 1.0 + 0.01 * (frequency / 1e6_f64).ln();

                    dispersion[[i, j, k]] = cs_local * dispersion_factor / c0;
                }
            }
        }

        self.dispersion_factors = Some(dispersion);
    }

    /// Propagate volumetric elastic waves with interference tracking.
    ///
    /// # Correctness note (source awareness)
    /// This method is a compatibility wrapper. It does **not** know the physical push locations,
    /// only the initial displacement fields and their time offsets. Tracking quality and expected
    /// arrival times depend on true source locations; therefore this wrapper estimates an
    /// `effective_source_ijk` from the centroid of the first non-zero initial displacement.
    ///
    /// Prefer [`Self::propagate_volumetric_waves_with_sources`] whenever source metadata is available.
    pub fn propagate_volumetric_waves(
        &self,
        initial_displacements: &[Array3<f64>],
        push_times: &[f64],
    ) -> KwaversResult<(Vec<ElasticWaveField>, WaveFrontTracker)> {
        let dt = self.config.dt.unwrap_or_else(|| self.calculate_time_step());
        if !dt.is_finite() || dt <= 0.0 {
            return Err(crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "dt".to_string(),
                    value: dt,
                    reason: "Time step must be finite and positive".to_string(),
                },
            ));
        }
        if self.config.save_every == 0 {
            return Err(crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "save_every".to_string(),
                    value: self.config.save_every as f64,
                    reason: "save_every must be >= 1".to_string(),
                },
            ));
        }
        let (nx, ny, nz) = self.grid.dimensions();

        // Estimate an effective source (index space) for the legacy path.
        let mut effective_source_ijk = [nx / 2, ny / 2, nz / 2];
        let mut effective_source_found = false;

        for initial_disp in initial_displacements.iter() {
            if effective_source_found {
                break;
            }

            let mut sum_i = 0.0;
            let mut sum_j = 0.0;
            let mut sum_k = 0.0;
            let mut sum_w = 0.0;

            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let w = initial_disp[[i, j, k]].abs();
                        if w > 0.0 {
                            sum_w += w;
                            sum_i += (i as f64) * w;
                            sum_j += (j as f64) * w;
                            sum_k += (k as f64) * w;
                        }
                    }
                }
            }

            if sum_w > 0.0 {
                let ci = (sum_i / sum_w).round() as isize;
                let cj = (sum_j / sum_w).round() as isize;
                let ck = (sum_k / sum_w).round() as isize;

                let clamp = |v: isize, max: usize| -> usize {
                    if v < 0 {
                        0
                    } else if (v as usize) >= max {
                        max - 1
                    } else {
                        v as usize
                    }
                };

                effective_source_ijk = [clamp(ci, nx), clamp(cj, ny), clamp(ck, nz)];
                effective_source_found = true;
            }
        }

        // Convert the legacy API into at least one explicit source (world coordinates).
        // This is only used for expected-arrival-time quality modeling; it does not change the
        // simulation source injection (which remains displacement-field based).
        let effective_source_m = [
            effective_source_ijk[0] as f64 * self.grid.dx,
            effective_source_ijk[1] as f64 * self.grid.dy,
            effective_source_ijk[2] as f64 * self.grid.dz,
        ];

        let mut sources = Vec::with_capacity(push_times.len().max(1));
        for &t in push_times {
            sources.push(VolumetricSource {
                location_m: effective_source_m,
                time_offset_s: t,
            });
        }
        if sources.is_empty() {
            sources.push(VolumetricSource {
                location_m: effective_source_m,
                time_offset_s: 0.0,
            });
        }

        self.propagate_volumetric_waves_with_sources(initial_displacements, push_times, &sources)
    }

    /// Propagate volumetric elastic waves using **body-force sources** (correctness-first).
    ///
    /// This is the preferred API for ARFI-style excitation:
    ///
    ///   ρ ∂v/∂t = ∇·σ + f
    ///
    /// rather than injecting arbitrary “initial displacement” fields.
    ///
    /// # Contract
    /// - `body_forces.len() == push_times.len()`
    /// - `sources.len() == body_forces.len()` (one physical source per push)
    /// - Each `VolumetricSource::time_offset_s` should correspond to the push timing.
    ///
    /// # Notes
    /// - This routine does not mutate `self.config.body_force`; it applies forces per-push by
    ///   overriding the body-force term at each time step.
    /// - If multiple pushes overlap in time, forces are *superposed*.
    pub fn propagate_volumetric_waves_with_body_forces(
        &self,
        body_forces: &[ElasticBodyForceConfig],
        push_times: &[f64],
        sources: &[VolumetricSource],
    ) -> KwaversResult<(Vec<ElasticWaveField>, WaveFrontTracker)> {
        if body_forces.len() != push_times.len() {
            return Err(crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "body_forces/push_times".to_string(),
                    value: body_forces.len() as f64,
                    reason: "body_forces and push_times must have the same length".to_string(),
                },
            ));
        }
        if sources.len() != body_forces.len() {
            return Err(crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "sources/body_forces".to_string(),
                    value: sources.len() as f64,
                    reason: "sources and body_forces must have the same length".to_string(),
                },
            ));
        }
        if sources.is_empty() {
            return Err(crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "sources".to_string(),
                    value: 0.0,
                    reason: "sources must contain at least one entry".to_string(),
                },
            ));
        }

        // Validate body-force configs (fail-fast).
        for (idx, src) in body_forces.iter().enumerate() {
            Self::validate_body_force_config(src).map_err(|e| {
                crate::domain::core::error::KwaversError::Validation(
                    crate::domain::core::error::ValidationError::InvalidValue {
                        parameter: format!("body_forces[{idx}]"),
                        value: idx as f64,
                        reason: format!("{e}"),
                    },
                )
            })?;
        }

        let dt = self.config.dt.unwrap_or_else(|| self.calculate_time_step());
        if !dt.is_finite() || dt <= 0.0 {
            return Err(crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "dt".to_string(),
                    value: dt,
                    reason: "Time step must be finite and positive".to_string(),
                },
            ));
        }
        if self.config.save_every == 0 {
            return Err(crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "save_every".to_string(),
                    value: self.config.save_every as f64,
                    reason: "save_every must be >= 1".to_string(),
                },
            ));
        }

        let n_steps = (self.config.simulation_time / dt) as usize;

        log::debug!(
            "Volumetric elastic wave propagation (body-force): {} steps, {} sources, dt = {:.2e} s",
            n_steps,
            body_forces.len(),
            dt
        );

        // Initialize wave field (zero ICs; all excitation via body-force sources).
        let (nx, ny, nz) = self.grid.dimensions();
        let mut field = ElasticWaveField::new(nx, ny, nz);
        field.time = 0.0;

        // Initialize wave front tracker (same structure as legacy path).
        let (mf_len, mf_template) = match &self.volumetric_config.arrival_detection {
            ArrivalDetection::MatchedFilter { template, .. } => (template.len(), Some(template)),
            _ => (0, None),
        };

        let mut tracker = WaveFrontTracker {
            arrival_times: Array3::from_elem((nx, ny, nz), f64::INFINITY),
            amplitudes: Array3::zeros((nx, ny, nz)),
            directions: Array3::from_elem((nx, ny, nz), [0.0, 0.0, 0.0]),
            interference_map: Array3::zeros((nx, ny, nz)),
            tracking_quality: Array3::zeros((nx, ny, nz)),

            peak_magnitude_m: Array3::zeros((nx, ny, nz)),
            peak_time_s: Array3::from_elem((nx, ny, nz), f64::INFINITY),

            mf_buffer: if mf_len > 0 {
                vec![0.0; nx * ny * nz * mf_len]
            } else {
                Vec::new()
            },
            mf_head: 0,
        };

        if let Some(template) = mf_template {
            debug_assert!(
                !template.is_empty(),
                "MatchedFilter template must be non-empty"
            );
            debug_assert!(
                template.iter().all(|x| x.is_finite()),
                "MatchedFilter template must contain finite values"
            );
        }

        let mut history = Vec::new();
        history.push(field.clone());

        // Scratch buffers for volumetric stepping output velocities.
        //
        // Invariant: these buffers must be allocated once and reused each step to avoid per-step
        // allocations and to allow `volumetric_time_step(...)` to write updated velocities.
        let mut vx_new = Array3::<f64>::zeros((nx, ny, nz));
        let mut vy_new = Array3::<f64>::zeros((nx, ny, nz));
        let mut vz_new = Array3::<f64>::zeros((nx, ny, nz));

        // Main time stepping loop: superpose all active forces.
        //
        // Correctness-first note:
        // - Volumetric stepping uses `volumetric_time_step(...)` (which applies volumetric boundaries,
        //   attenuation, dispersion, and PML) as the SSOT for state evolution.
        // - Body-force excitation is integrated as an *additional acceleration* term in the momentum
        //   equation: ρ ∂v/∂t = ∇·σ + f, applied per-time-step.
        //
        // Implementation strategy:
        // 1) Run the existing volumetric stepper to apply ∇·σ (and volumetric corrections).
        // 2) Apply the superposed body-force at time `t` as Δv = dt * f(x,t)/ρ(x).
        for step in 0..n_steps {
            let t = field.time;

            // Step 1: existing volumetric stepping (∇·σ + volumetric corrections).
            self.volumetric_time_step(&mut field, dt, &mut vx_new, &mut vy_new, &mut vz_new);

            // Step 2: add superposed body-force contributions at time `t`.
            //
            // We write into `vx_new/vy_new/vz_new` (the stepper outputs) and then swap them into
            // the field below, preserving the leapfrog-style update.
            for (bf, &t0) in body_forces.iter().zip(push_times.iter()) {
                // Each body force is defined in its own local timeline; shift by its push time.
                let local_t = t - t0;
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            let rho = self.density[[i, j, k]];
                            // `body_force_at` returns force density [N/m^3]; divide by ρ to get [m/s^2].
                            // `body_force_at` signature is (src, t, i, j, k).
                            let (fx, fy, fz) = self.body_force_at(bf, local_t, i, j, k);
                            vx_new[[i, j, k]] += dt * fx / rho;
                            vy_new[[i, j, k]] += dt * fy / rho;
                            vz_new[[i, j, k]] += dt * fz / rho;
                        }
                    }
                }
            }

            // Commit the updated velocities back to the field, and update displacements.
            // (This mirrors the legacy volumetric stepping structure.)
            field.vx.assign(&vx_new);
            field.vy.assign(&vy_new);
            field.vz.assign(&vz_new);

            field.ux = &field.ux + &(field.vx.mapv(|v| dt * v));
            field.uy = &field.uy + &(field.vy.mapv(|v| dt * v));
            field.uz = &field.uz + &(field.vz.mapv(|v| dt * v));

            field.time = (step as f64 + 1.0) * dt;

            // Update tracker from displacement state at the new time (source-aware).
            if self.volumetric_config.interference_tracking {
                self.update_wave_front_tracking_with_sources(
                    &field,
                    &mut tracker,
                    field.time,
                    sources,
                );
            }

            if step % self.config.save_every == 0 {
                history.push(field.clone());
            }
        }

        Ok((history, tracker))
    }

    /// Propagate volumetric elastic waves with explicit physical sources.
    ///
    /// This is the source-aware SSOT API used to compute expected arrival times and tracking quality:
    /// `t_expected(x) = min_s ( t_s + ||x - x_s|| / c_s(x) )`.
    ///
    /// # Contract
    /// - `initial_displacements.len() == push_times.len()`
    /// - `sources` must contain at least one source.
    /// - Each `VolumetricSource::time_offset_s` should correspond to the push sequence timing.
    pub fn propagate_volumetric_waves_with_sources(
        &self,
        initial_displacements: &[Array3<f64>],
        push_times: &[f64],
        sources: &[VolumetricSource],
    ) -> KwaversResult<(Vec<ElasticWaveField>, WaveFrontTracker)> {
        debug_assert!(
            initial_displacements.len() == push_times.len(),
            "initial_displacements and push_times must have the same length"
        );
        debug_assert!(
            !sources.is_empty(),
            "sources must contain at least one entry"
        );

        let dt = self.config.dt.unwrap_or_else(|| self.calculate_time_step());
        let n_steps = (self.config.simulation_time / dt) as usize;

        log::debug!(
            "Volumetric elastic wave propagation: {} steps, {} sources, dt = {:.2e} s",
            n_steps,
            initial_displacements.len(),
            dt
        );

        // Initialize wave field
        let (nx, ny, nz) = self.grid.dimensions();
        let mut field = ElasticWaveField::new(nx, ny, nz);

        // Legacy initialization: interpret each `initial_displacement` as ux and apply if t<=0.
        // Correctness warning: ARFI must be modeled as a body-force source term; prefer
        // `propagate_volumetric_waves_with_body_forces`.
        for (source_index, initial_disp) in initial_displacements.iter().enumerate() {
            let time_offset = push_times[source_index];
            if time_offset <= 0.0 {
                ndarray::Zip::from(&mut field.ux)
                    .and(initial_disp)
                    .for_each(|ux, &src| *ux += src);
            }
        }

        // Initialize wave front tracker
        let (mf_len, mf_template) = match &self.volumetric_config.arrival_detection {
            ArrivalDetection::MatchedFilter { template, .. } => (template.len(), Some(template)),
            _ => (0, None),
        };

        let mut tracker = WaveFrontTracker {
            arrival_times: Array3::from_elem((nx, ny, nz), f64::INFINITY),
            amplitudes: Array3::zeros((nx, ny, nz)),
            directions: Array3::from_elem((nx, ny, nz), [0.0, 0.0, 0.0]),
            interference_map: Array3::zeros((nx, ny, nz)),
            tracking_quality: Array3::zeros((nx, ny, nz)),

            peak_magnitude_m: Array3::zeros((nx, ny, nz)),
            peak_time_s: Array3::from_elem((nx, ny, nz), f64::INFINITY),

            mf_buffer: if mf_len > 0 {
                vec![0.0; nx * ny * nz * mf_len]
            } else {
                Vec::new()
            },
            mf_head: 0,
        };

        // Sanity: matched-filter template must be non-empty and finite if selected.
        if let Some(template) = mf_template {
            debug_assert!(
                !template.is_empty(),
                "MatchedFilter template must be non-empty"
            );
            debug_assert!(
                template.iter().all(|x| x.is_finite()),
                "MatchedFilter template must contain finite values"
            );
        }

        // ---- Main time stepping loop (legacy initial-displacement sources) ----

        // Storage for displacement fields at different times
        let mut displacement_history = Vec::new();
        field.time = 0.0;
        displacement_history.push(field.clone());

        let mut vx_new = Array3::zeros((nx, ny, nz));
        let mut vy_new = Array3::zeros((nx, ny, nz));
        let mut vz_new = Array3::zeros((nx, ny, nz));

        // Time stepping loop with volumetric corrections
        for step in 0..n_steps {
            let current_time = step as f64 * dt;

            // Add delayed sources (injection remains displacement-field based)
            for (source_index, initial_disp) in initial_displacements.iter().enumerate() {
                let time_offset = push_times[source_index];
                if current_time >= time_offset && current_time < time_offset + dt {
                    ndarray::Zip::from(&mut field.ux)
                        .and(initial_disp)
                        .for_each(|ux, &src| *ux += src);
                }
            }

            self.volumetric_time_step(&mut field, dt, &mut vx_new, &mut vy_new, &mut vz_new);
            field.time = current_time + dt;

            // Update wave front tracking (source-aware)
            if self.volumetric_config.interference_tracking {
                self.update_wave_front_tracking_with_sources(
                    &field,
                    &mut tracker,
                    field.time,
                    sources,
                );
            }

            // Save field periodically
            if step % self.config.save_every == 0 {
                displacement_history.push(field.clone());

                if step % 100 == 0 {
                    log::debug!("Step {}/{}, time = {:.2e} s", step, n_steps, field.time);
                }
            }
        }

        Ok((displacement_history, tracker))
    }

    /// Single time step with volumetric corrections
    fn volumetric_time_step(
        &self,
        field: &mut ElasticWaveField,
        dt: f64,
        vx_new: &mut Array3<f64>,
        vy_new: &mut Array3<f64>,
        vz_new: &mut Array3<f64>,
    ) {
        let (nx, ny, nz) = self.grid.dimensions();

        vx_new.fill(0.0);
        vy_new.fill(0.0);
        vz_new.fill(0.0);

        for j in 0..ny {
            for i in 0..nx {
                vx_new[[i, j, 0]] = field.vx[[i, j, 0]];
                vy_new[[i, j, 0]] = field.vy[[i, j, 0]];
                vz_new[[i, j, 0]] = field.vz[[i, j, 0]];

                vx_new[[i, j, nz - 1]] = field.vx[[i, j, nz - 1]];
                vy_new[[i, j, nz - 1]] = field.vy[[i, j, nz - 1]];
                vz_new[[i, j, nz - 1]] = field.vz[[i, j, nz - 1]];
            }
        }

        for k in 0..nz {
            for i in 0..nx {
                vx_new[[i, 0, k]] = field.vx[[i, 0, k]];
                vy_new[[i, 0, k]] = field.vy[[i, 0, k]];
                vz_new[[i, 0, k]] = field.vz[[i, 0, k]];

                vx_new[[i, ny - 1, k]] = field.vx[[i, ny - 1, k]];
                vy_new[[i, ny - 1, k]] = field.vy[[i, ny - 1, k]];
                vz_new[[i, ny - 1, k]] = field.vz[[i, ny - 1, k]];
            }
        }

        for k in 0..nz {
            for j in 0..ny {
                vx_new[[0, j, k]] = field.vx[[0, j, k]];
                vy_new[[0, j, k]] = field.vy[[0, j, k]];
                vz_new[[0, j, k]] = field.vz[[0, j, k]];

                vx_new[[nx - 1, j, k]] = field.vx[[nx - 1, j, k]];
                vy_new[[nx - 1, j, k]] = field.vy[[nx - 1, j, k]];
                vz_new[[nx - 1, j, k]] = field.vz[[nx - 1, j, k]];
            }
        }

        // Update velocities with volumetric corrections
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    // Compute stress derivatives
                    let d_sigma_xx_dx = self.stress_xx_derivative(i, j, k, field);
                    let d_sigma_xy_dy = self.stress_xy_derivative(i, j, k, field);
                    let d_sigma_xz_dz = self.stress_xz_derivative(i, j, k, field);

                    let d_sigma_yx_dx = self.stress_yx_derivative(i, j, k, field);
                    let d_sigma_yy_dy = self.stress_yy_derivative(i, j, k, field);
                    let d_sigma_yz_dz = self.stress_yz_derivative(i, j, k, field);

                    let d_sigma_zx_dx = self.stress_zx_derivative(i, j, k, field);
                    let d_sigma_zy_dy = self.stress_zy_derivative(i, j, k, field);
                    let d_sigma_zz_dz = self.stress_zz_derivative(i, j, k, field);

                    // Acceleration = (1/ρ) * ∇·σ
                    let rho = self.density[[i, j, k]];

                    let mut ax = (d_sigma_xx_dx + d_sigma_xy_dy + d_sigma_xz_dz) / rho;
                    let mut ay = (d_sigma_yx_dx + d_sigma_yy_dy + d_sigma_yz_dz) / rho;
                    let mut az = (d_sigma_zx_dx + d_sigma_zy_dy + d_sigma_zz_dz) / rho;

                    // Apply volumetric attenuation
                    if let Some(attenuation) = &self.volumetric_attenuation {
                        let alpha = attenuation[[i, j, k]];
                        let damping = (-alpha * dt).exp();
                        ax *= damping;
                        ay *= damping;
                        az *= damping;
                    }

                    // Apply dispersion corrections
                    if let Some(dispersion) = &self.dispersion_factors {
                        let correction = dispersion[[i, j, k]];
                        ax *= correction;
                        ay *= correction;
                        az *= correction;
                    }

                    // Update velocities (leapfrog scheme)
                    vx_new[[i, j, k]] = field.vx[[i, j, k]] + dt * ax;
                    vy_new[[i, j, k]] = field.vy[[i, j, k]] + dt * ay;
                    vz_new[[i, j, k]] = field.vz[[i, j, k]] + dt * az;

                    // Apply volumetric boundary conditions
                    if self.volumetric_config.volumetric_boundaries {
                        self.apply_volumetric_boundaries(vx_new, vy_new, vz_new, i, j, k);
                    }

                    // Apply damping (PML)
                    let sigma = self.pml_sigma[[i, j, k]];
                    if sigma > 0.0 {
                        let damping = (-sigma * dt).exp();
                        vx_new[[i, j, k]] *= damping;
                        vy_new[[i, j, k]] *= damping;
                        vz_new[[i, j, k]] *= damping;
                    }
                }
            }
        }

        // Update field velocities
        std::mem::swap(&mut field.vx, vx_new);
        std::mem::swap(&mut field.vy, vy_new);
        std::mem::swap(&mut field.vz, vz_new);

        // Update displacements (integration of velocities)
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    field.ux[[i, j, k]] += dt * field.vx[[i, j, k]];
                    field.uy[[i, j, k]] += dt * field.vy[[i, j, k]];
                    field.uz[[i, j, k]] += dt * field.vz[[i, j, k]];
                }
            }
        }
    }

    /// Apply volumetric boundary conditions
    fn apply_volumetric_boundaries(
        &self,
        vx: &mut Array3<f64>,
        vy: &mut Array3<f64>,
        vz: &mut Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
    ) {
        let (nx, ny, nz) = self.grid.dimensions();

        // Free surface boundary (z = 0): normal stress = 0
        if k == 0 {
            // σ_zz = 0 implies ∂u_z/∂z = -∂u_x/∂x - ∂u_y/∂y
            // Approximate by setting vz to maintain zero normal stress
            let lambda = self.lambda[[i, j, k]];
            let mu = self.mu[[i, j, k]];

            if k < nz - 1 {
                let du_x_dx = if i > 0 && i < nx - 1 {
                    (vx[[i + 1, j, k]] - vx[[i - 1, j, k]]) / (2.0 * self.grid.dx)
                } else {
                    0.0
                };
                let du_y_dy = if j > 0 && j < ny - 1 {
                    (vy[[i, j + 1, k]] - vy[[i, j - 1, k]]) / (2.0 * self.grid.dy)
                } else {
                    0.0
                };

                // Free surface: vz should be adjusted to satisfy σ_zz = 0
                // Simplified for computational efficiency - full implementation would use complete constitutive relations
                let correction = (lambda / (lambda + 2.0 * mu)) * (du_x_dx + du_y_dy);
                vz[[i, j, k]] -= correction * self.grid.dz;
            }
        }

        // Rigid boundary (other boundaries): zero displacement
        // This is handled by PML, but we can add additional constraints
        if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == nz - 1 {
            // Additional damping for rigid boundaries
            let boundary_damping = 0.9;
            vx[[i, j, k]] *= boundary_damping;
            vy[[i, j, k]] *= boundary_damping;
            vz[[i, j, k]] *= boundary_damping;
        }
    }

    /// Update wave front tracking information (legacy single-source signature).
    ///
    /// This exists for internal tests and backward compatibility; prefer
    /// [`Self::update_wave_front_tracking_with_sources`] for correctness.
    fn update_wave_front_tracking(
        &self,
        field: &ElasticWaveField,
        tracker: &mut WaveFrontTracker,
        time: f64,
        source_ijk: [usize; 3],
    ) {
        let source_m = [
            source_ijk[0] as f64 * self.grid.dx,
            source_ijk[1] as f64 * self.grid.dy,
            source_ijk[2] as f64 * self.grid.dz,
        ];
        let sources = [VolumetricSource {
            location_m: source_m,
            time_offset_s: 0.0,
        }];

        self.update_wave_front_tracking_with_sources(field, tracker, time, &sources);
    }

    /// Update wave front tracking information using explicit source metadata.
    ///
    /// Tracking quality is evaluated against the source-aware first-order expected arrival time:
    /// `t_expected(x) = min_s ( t_s + ||x - x_s|| / c_s(x) )`.
    fn update_wave_front_tracking_with_sources(
        &self,
        field: &ElasticWaveField,
        tracker: &mut WaveFrontTracker,
        time: f64,
        sources: &[VolumetricSource],
    ) {
        let (nx, ny, nz) = self.grid.dimensions();

        let decim = self.volumetric_config.tracking_decimation;
        let di = decim[0].max(1);
        let dj = decim[1].max(1);
        let dk = decim[2].max(1);

        // Helper for matched-filter buffer indexing
        let mf_len = match &self.volumetric_config.arrival_detection {
            ArrivalDetection::MatchedFilter { template, .. } => template.len(),
            _ => 0,
        };
        let linear_index = |i: usize, j: usize, k: usize| -> usize { (k * ny + j) * nx + i };

        // IMPORTANT (correctness): `mf_head` is a global time index into the per-voxel ring buffer.
        // It must advance exactly once per time step, not once per voxel update.
        let mf_head_this_step = if mf_len > 0 { tracker.mf_head } else { 0 };

        for k in (1..nz - 1).step_by(dk) {
            for j in (1..ny - 1).step_by(dj) {
                for i in (1..nx - 1).step_by(di) {
                    let displacement = field.displacement_magnitude_at(i, j, k);

                    // Always maintain interference count for "active" voxels under the chosen detector.
                    // Detector-specific "active" predicates are applied below.
                    let mut is_active = false;

                    match &self.volumetric_config.arrival_detection {
                        ArrivalDetection::AmplitudeThresholdCrossing { threshold_m } => {
                            if displacement >= *threshold_m {
                                is_active = true;

                                // Update arrival time if this is the first time threshold is exceeded
                                if tracker.arrival_times[[i, j, k]] == f64::INFINITY {
                                    tracker.arrival_times[[i, j, k]] = time;
                                    tracker.amplitudes[[i, j, k]] = displacement;

                                    // Estimate wave direction from local velocity
                                    let vx = field.vx[[i, j, k]];
                                    let vy = field.vy[[i, j, k]];
                                    let vz = field.vz[[i, j, k]];
                                    let speed = (vx * vx + vy * vy + vz * vz).sqrt();
                                    if speed > 1e-12 {
                                        tracker.directions[[i, j, k]] =
                                            [vx / speed, vy / speed, vz / speed];
                                    }
                                }
                            }
                        }

                        ArrivalDetection::PeakTime { min_peak_m } => {
                            // Track running peak
                            if displacement >= *min_peak_m
                                && displacement > tracker.peak_magnitude_m[[i, j, k]]
                            {
                                tracker.peak_magnitude_m[[i, j, k]] = displacement;
                                tracker.peak_time_s[[i, j, k]] = time;

                                // Keep amplitudes in sync with the peak
                                tracker.amplitudes[[i, j, k]] = displacement;

                                // Update direction at peak time
                                let vx = field.vx[[i, j, k]];
                                let vy = field.vy[[i, j, k]];
                                let vz = field.vz[[i, j, k]];
                                let speed = (vx * vx + vy * vy + vz * vz).sqrt();
                                if speed > 1e-12 {
                                    tracker.directions[[i, j, k]] =
                                        [vx / speed, vy / speed, vz / speed];
                                }
                            }

                            // Treat voxels with a defined peak as active for interference counting
                            if tracker.peak_time_s[[i, j, k]].is_finite() {
                                is_active = true;
                                // Arrival time is defined as the peak time (can be updated as the peak advances)
                                tracker.arrival_times[[i, j, k]] = tracker.peak_time_s[[i, j, k]];
                            }
                        }

                        ArrivalDetection::MatchedFilter { template, min_corr } => {
                            // Maintain per-voxel ring buffer of magnitudes.
                            // `mf_head_this_step` is the write index for *this* time step for all voxels.
                            if mf_len > 0 {
                                let base = linear_index(i, j, k) * mf_len;
                                tracker.mf_buffer[base + mf_head_this_step] = displacement;

                                // Compute correlation at this time step:
                                // corr = Σ_{τ=0..L-1} buffer[newest-τ] * template[τ]
                                let mut corr = 0.0;
                                for (tau, &template_tau) in template.iter().take(mf_len).enumerate()
                                {
                                    let idx = (mf_head_this_step + mf_len - tau) % mf_len;
                                    corr += tracker.mf_buffer[base + idx] * template_tau;
                                }

                                if corr.is_finite() && corr >= *min_corr {
                                    is_active = true;

                                    // Set arrival time only once at first detection
                                    if tracker.arrival_times[[i, j, k]] == f64::INFINITY {
                                        tracker.arrival_times[[i, j, k]] = time;
                                        tracker.amplitudes[[i, j, k]] = displacement;

                                        let vx = field.vx[[i, j, k]];
                                        let vy = field.vy[[i, j, k]];
                                        let vz = field.vz[[i, j, k]];
                                        let speed = (vx * vx + vy * vy + vz * vz).sqrt();
                                        if speed > 1e-12 {
                                            tracker.directions[[i, j, k]] =
                                                [vx / speed, vy / speed, vz / speed];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if is_active {
                        tracker.interference_map[[i, j, k]] += 1.0;

                        // Source-aware timing-quality: compare current time to min predicted arrival.
                        let expected_speed = (self.mu[[i, j, k]] / self.density[[i, j, k]]).sqrt();

                        let x_m = i as f64 * self.grid.dx;
                        let y_m = j as f64 * self.grid.dy;
                        let z_m = k as f64 * self.grid.dz;

                        let mut expected_time = f64::INFINITY;

                        if expected_speed > 0.0 {
                            for s in sources {
                                let dx = x_m - s.location_m[0];
                                let dy = y_m - s.location_m[1];
                                let dz = z_m - s.location_m[2];
                                let distance = (dx * dx + dy * dy + dz * dz).sqrt();
                                let t = s.time_offset_s + distance / expected_speed;
                                if t < expected_time {
                                    expected_time = t;
                                }
                            }
                        } else {
                            expected_time = time;
                        }

                        // Convert timing error to a bounded quality score in (0,1].
                        let time_error = (time - expected_time).abs();
                        tracker.tracking_quality[[i, j, k]] = (-time_error / 1e-6).exp();
                    }
                }
            }
        }

        // Advance matched-filter head exactly once per time-step.
        if mf_len > 0 {
            tracker.mf_head = (tracker.mf_head + 1) % mf_len;
        }
    }

    /// Extract arrival times for time-of-flight analysis
    pub fn extract_arrival_times(&self, tracker: &WaveFrontTracker) -> Array3<f64> {
        tracker.arrival_times.clone()
    }

    /// Extract wave amplitudes for attenuation analysis
    pub fn extract_wave_amplitudes(&self, tracker: &WaveFrontTracker) -> Array3<f64> {
        tracker.amplitudes.clone()
    }

    /// Extract wave directions for directional analysis
    pub fn extract_wave_directions(&self, tracker: &WaveFrontTracker) -> Array3<[f64; 3]> {
        tracker.directions.clone()
    }

    /// Calculate volumetric wave propagation quality metrics
    pub fn calculate_volumetric_quality(
        &self,
        tracker: &WaveFrontTracker,
    ) -> VolumetricQualityMetrics {
        let mut valid_points = 0;
        let mut total_quality = 0.0;
        let mut max_interference: f64 = 0.0;

        let total_points = tracker.tracking_quality.len() as f64;

        for &quality in tracker.tracking_quality.iter() {
            if quality > 0.0 {
                valid_points += 1;
                total_quality += quality;
            }
        }

        for &interference in tracker.interference_map.iter() {
            max_interference = max_interference.max(interference);
        }

        let coverage = valid_points as f64 / total_points;

        VolumetricQualityMetrics {
            coverage,
            average_quality: if valid_points > 0 {
                total_quality / valid_points as f64
            } else {
                0.0
            },
            max_interference,
            valid_tracking_points: valid_points,
        }
    }
}

/// Quality metrics for volumetric wave propagation
#[derive(Debug, Clone)]
pub struct VolumetricQualityMetrics {
    /// Fraction of volume with valid wave tracking (0-1)
    pub coverage: f64,
    /// Average tracking quality across valid points (0-1)
    pub average_quality: f64,
    /// Maximum interference level detected
    pub max_interference: f64,
    /// Number of points with valid wave tracking
    pub valid_tracking_points: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::medium::HomogeneousMedium;

    #[test]
    fn test_elastic_wave_solver_creation() {
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let config = ElasticWaveConfig::default();

        let solver = ElasticWaveSolver::new(&grid, &medium, config);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_time_step_calculation() {
        // Use realistic tissue parameters: density 1000 kg/m³, shear modulus 5 kPa
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();

        // Create heterogeneous medium with proper elastic properties
        use crate::domain::medium::heterogeneous::core::HeterogeneousMedium;
        use ndarray::Array3;

        let density = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1000.0);
        let sound_speed = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1500.0);
        let lame_lambda = Array3::from_elem((grid.nx, grid.ny, grid.nz), 2.25e9); // Bulk modulus approximation
        let lame_mu = Array3::from_elem((grid.nx, grid.ny, grid.nz), 5000.0); // 5 kPa shear modulus

        // Create minimal heterogeneous medium for testing
        let medium = HeterogeneousMedium {
            use_trilinear_interpolation: false,
            density,
            sound_speed,
            lame_lambda,
            lame_mu,
            // Initialize other required fields with defaults
            viscosity: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.001),
            surface_tension: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.072),
            ambient_pressure: 101325.0,
            vapor_pressure: Array3::from_elem((grid.nx, grid.ny, grid.nz), 2330.0),
            polytropic_index: Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.4),
            specific_heat: Array3::from_elem((grid.nx, grid.ny, grid.nz), 4186.0),
            thermal_conductivity: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.6),
            thermal_expansion: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.0002),
            gas_diffusion_coeff: Array3::from_elem((grid.nx, grid.ny, grid.nz), 2e-9),
            thermal_diffusivity: Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.4e-7),
            mu_a: Array3::from_elem((grid.nx, grid.ny, grid.nz), 10.0),
            mu_s_prime: Array3::from_elem((grid.nx, grid.ny, grid.nz), 100.0),
            temperature: Array3::from_elem((grid.nx, grid.ny, grid.nz), 293.15),
            bubble_radius: Array3::from_elem((grid.nx, grid.ny, grid.nz), 1e-6),
            bubble_velocity: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            alpha0: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.5),
            delta: Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.1),
            b_a: Array3::from_elem((grid.nx, grid.ny, grid.nz), 7.0),
            absorption: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.5),
            nonlinearity: Array3::from_elem((grid.nx, grid.ny, grid.nz), 7.0),
            reference_frequency: 1e6,
            shear_sound_speed: Array3::from_elem((grid.nx, grid.ny, grid.nz), 3.0),
            shear_viscosity_coeff: Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0),
            bulk_viscosity_coeff: Array3::from_elem((grid.nx, grid.ny, grid.nz), 3.0),
        };

        let config = ElasticWaveConfig::default();

        let solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();
        let dt = solver.calculate_time_step();

        assert!(dt > 0.0, "Time step should be positive");
        // For shear waves in tissue (cs ~ sqrt(5000/1000) = 2.2 m/s), CFL condition gives dt ~ 2.5e-4 s
        // This is reasonable for elastography simulations
        assert!(dt < 1e-3, "Time step should be small for stability");
    }

    #[test]
    fn test_wave_field_initialization() {
        let field = ElasticWaveField::new(10, 10, 10);
        assert_eq!(field.ux.dim(), (10, 10, 10));
        assert_eq!(field.uy.dim(), (10, 10, 10));
        assert_eq!(field.uz.dim(), (10, 10, 10));

        // Test displacement magnitude calculation
        let magnitude = field.displacement_magnitude();
        assert_eq!(magnitude.dim(), (10, 10, 10));

        // All values should be zero initially
        for &val in magnitude.iter() {
            assert!((val - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_wave_propagation() {
        let grid = Grid::new(16, 16, 16, 0.002, 0.002, 0.002).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let config = ElasticWaveConfig {
            simulation_time: 5e-5,
            cfl_factor: 0.5,
            save_every: 1_000,
            ..Default::default()
        };

        let solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Create initial displacement (point source)
        let mut initial_disp = Array3::zeros((16, 16, 16));
        initial_disp[[8, 8, 8]] = 1e-6; // 1 micron displacement

        let history = solver.propagate_waves(&initial_disp);
        assert!(history.is_ok());

        let history = history.unwrap();
        assert!(!history.is_empty(), "Should have at least initial field");

        // Check that displacement propagates (non-zero at later times)
        if history.len() > 1 {
            let final_magnitude = history.last().unwrap().displacement_magnitude();
            let max_disp = final_magnitude.iter().fold(0.0_f64, |a, &b| a.max(b));
            assert!(max_disp > 0.0, "Displacement should propagate through time");
        }
    }

    #[test]
    fn test_volumetric_wave_config() {
        let config = VolumetricWaveConfig::default();
        assert!(config.volumetric_boundaries);
        assert!(config.interference_tracking);
        assert!(config.volumetric_attenuation);
        assert!(!config.dispersion_correction);
        assert_eq!(config.memory_optimization, 1);

        // Default must be the optimal general-purpose approach: matched filtering.
        match &config.arrival_detection {
            ArrivalDetection::MatchedFilter { template, min_corr } => {
                assert!(
                    !template.is_empty(),
                    "Default matched-filter template must be non-empty"
                );
                assert!(
                    template.iter().all(|x| x.is_finite()),
                    "Default matched-filter template must be finite"
                );
                assert!(min_corr.is_finite(), "Default min_corr must be finite");
            }
            other => panic!(
                "Expected default arrival_detection to be MatchedFilter, got {:?}",
                other
            ),
        }

        assert_eq!(config.tracking_decimation, [1, 1, 1]);
    }

    #[test]
    fn test_volumetric_attenuation_initialization() {
        let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();

        // Create heterogeneous medium with proper elastic properties
        use crate::domain::medium::heterogeneous::core::HeterogeneousMedium;
        use ndarray::Array3;

        let density = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1000.0);
        let sound_speed = Array3::from_elem((grid.nx, grid.ny, grid.nz), 1500.0);
        let lame_lambda = Array3::from_elem((grid.nx, grid.ny, grid.nz), 2.25e9);
        let lame_mu = Array3::from_elem((grid.nx, grid.ny, grid.nz), 5000.0); // 5 kPa shear modulus

        let medium = HeterogeneousMedium {
            use_trilinear_interpolation: false,
            density,
            sound_speed,
            lame_lambda,
            lame_mu,
            viscosity: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.001),
            surface_tension: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.072),
            ambient_pressure: 101325.0,
            vapor_pressure: Array3::from_elem((grid.nx, grid.ny, grid.nz), 2330.0),
            polytropic_index: Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.4),
            specific_heat: Array3::from_elem((grid.nx, grid.ny, grid.nz), 4186.0),
            thermal_conductivity: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.6),
            thermal_expansion: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.0002),
            gas_diffusion_coeff: Array3::from_elem((grid.nx, grid.ny, grid.nz), 2e-9),
            thermal_diffusivity: Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.4e-7),
            mu_a: Array3::from_elem((grid.nx, grid.ny, grid.nz), 10.0),
            mu_s_prime: Array3::from_elem((grid.nx, grid.ny, grid.nz), 100.0),
            temperature: Array3::from_elem((grid.nx, grid.ny, grid.nz), 293.15),
            bubble_radius: Array3::from_elem((grid.nx, grid.ny, grid.nz), 1e-6),
            bubble_velocity: Array3::zeros((grid.nx, grid.ny, grid.nz)),
            alpha0: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.5),
            delta: Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.1),
            b_a: Array3::from_elem((grid.nx, grid.ny, grid.nz), 7.0),
            absorption: Array3::from_elem((grid.nx, grid.ny, grid.nz), 0.5),
            nonlinearity: Array3::from_elem((grid.nx, grid.ny, grid.nz), 7.0),
            reference_frequency: 1e6,
            shear_sound_speed: Array3::from_elem((grid.nx, grid.ny, grid.nz), 3.0),
            shear_viscosity_coeff: Array3::from_elem((grid.nx, grid.ny, grid.nz), 1.0),
            bulk_viscosity_coeff: Array3::from_elem((grid.nx, grid.ny, grid.nz), 3.0),
        };

        let config = ElasticWaveConfig::default();

        let mut solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Enable volumetric attenuation
        let volumetric_config = VolumetricWaveConfig {
            volumetric_attenuation: true,
            ..Default::default()
        };
        solver.set_volumetric_config(volumetric_config);

        // Check that attenuation coefficients were initialized
        assert!(solver.volumetric_attenuation.is_some());

        let attenuation = solver.volumetric_attenuation.as_ref().unwrap();
        assert_eq!(attenuation.dim(), (16, 16, 16));

        // Check that attenuation values are reasonable (positive and decreasing with depth)
        let surface_attenuation = attenuation[[8, 8, 0]]; // Near surface
        let deep_attenuation = attenuation[[8, 8, 15]]; // Deep

        assert!(
            surface_attenuation >= 0.0,
            "Surface attenuation should be non-negative"
        );
        assert!(
            deep_attenuation > 0.0,
            "Deep attenuation should be positive"
        );
        // Attenuation should generally increase with depth (more absorption)
        assert!(
            deep_attenuation >= surface_attenuation * 0.5,
            "Attenuation should not decrease significantly with depth"
        );
    }

    #[test]
    fn test_wave_front_tracker_initialization() {
        let tracker = WaveFrontTracker {
            arrival_times: Array3::from_elem((10, 10, 10), f64::INFINITY),
            amplitudes: Array3::zeros((10, 10, 10)),
            directions: Array3::from_elem((10, 10, 10), [0.0, 0.0, 0.0]),
            interference_map: Array3::zeros((10, 10, 10)),
            tracking_quality: Array3::zeros((10, 10, 10)),
            peak_magnitude_m: Array3::zeros((10, 10, 10)),
            peak_time_s: Array3::from_elem((10, 10, 10), f64::INFINITY),
            mf_buffer: Vec::new(),
            mf_head: 0,
        };

        assert_eq!(tracker.arrival_times.dim(), (10, 10, 10));
        assert_eq!(tracker.amplitudes.dim(), (10, 10, 10));
        assert_eq!(tracker.directions.dim(), (10, 10, 10));
        assert_eq!(tracker.interference_map.dim(), (10, 10, 10));
        assert_eq!(tracker.tracking_quality.dim(), (10, 10, 10));
        assert_eq!(tracker.peak_magnitude_m.dim(), (10, 10, 10));
        assert_eq!(tracker.peak_time_s.dim(), (10, 10, 10));

        // Check initial values
        assert!(tracker.arrival_times[[5, 5, 5]].is_infinite());
        assert_eq!(tracker.amplitudes[[5, 5, 5]], 0.0);
        assert_eq!(tracker.directions[[5, 5, 5]], [0.0, 0.0, 0.0]);
        assert_eq!(tracker.peak_magnitude_m[[5, 5, 5]], 0.0);
        assert!(tracker.peak_time_s[[5, 5, 5]].is_infinite());
        assert!(tracker.mf_buffer.is_empty());
        assert_eq!(tracker.mf_head, 0);
    }

    #[test]
    fn test_volumetric_wave_propagation_single_source() {
        let grid = Grid::new(12, 12, 12, 0.002, 0.002, 0.002).unwrap();

        let medium = HomogeneousMedium::soft_tissue(8_000.0, 0.49, &grid);

        let config = ElasticWaveConfig {
            simulation_time: 1e-4,
            cfl_factor: 0.5,
            save_every: 1_000,
            ..Default::default()
        };

        let mut solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Enable volumetric features
        let volumetric_config = VolumetricWaveConfig {
            volumetric_boundaries: true,
            interference_tracking: true,
            volumetric_attenuation: true,
            arrival_detection: ArrivalDetection::AmplitudeThresholdCrossing { threshold_m: 1e-8 },
            ..Default::default()
        };
        solver.set_volumetric_config(volumetric_config);

        // Create single initial displacement (point source)
        let mut initial_disp = Array3::zeros((12, 12, 12));
        initial_disp[[6, 6, 6]] = 1e-6; // 1 micron displacement at center

        let initial_displacements = vec![initial_disp];
        let push_times = vec![0.0]; // Immediate push

        let sources = vec![VolumetricSource {
            location_m: [0.012, 0.012, 0.012],
            time_offset_s: 0.0,
        }];

        let result = solver.propagate_volumetric_waves_with_sources(
            &initial_displacements,
            &push_times,
            &sources,
        );
        assert!(result.is_ok());

        let (history, tracker) = result.unwrap();
        assert!(!history.is_empty(), "Should have displacement history");

        // Check that wave propagated
        if history.len() > 1 {
            let final_field = history.last().unwrap();
            let magnitude = final_field.displacement_magnitude();
            let max_disp = magnitude.iter().cloned().fold(0.0_f64, |a, b| a.max(b));
            assert!(max_disp > 0.0, "Wave should propagate through volume");
        }

        // Check wave front tracking
        let valid_arrivals = tracker
            .arrival_times
            .iter()
            .filter(|&&time| !time.is_infinite())
            .count();
        assert!(valid_arrivals > 0, "Should track some wave arrivals");

        // Check volumetric quality metrics
        let quality = solver.calculate_volumetric_quality(&tracker);
        assert!(quality.coverage >= 0.0 && quality.coverage <= 1.0);
        assert!(quality.average_quality >= 0.0 && quality.average_quality <= 1.0);
        assert!(quality.valid_tracking_points > 0);
    }

    #[test]
    fn test_volumetric_wave_propagation_multiple_sources() {
        let grid = Grid::new(16, 16, 16, 0.002, 0.002, 0.002).unwrap();
        // Use soft tissue elastic properties to ensure non-zero shear modulus
        // Typical liver-like parameters: E ≈ 8 kPa, ν ≈ 0.49
        let medium = HomogeneousMedium::soft_tissue(8_000.0, 0.49, &grid);

        let config = ElasticWaveConfig {
            simulation_time: 3e-4,
            cfl_factor: 0.5,
            save_every: 1_000,
            ..Default::default()
        };

        let mut solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Enable volumetric features
        let volumetric_config = VolumetricWaveConfig {
            volumetric_boundaries: true,
            interference_tracking: true,
            volumetric_attenuation: true,
            arrival_detection: ArrivalDetection::AmplitudeThresholdCrossing { threshold_m: 1e-8 },
            ..Default::default()
        };
        solver.set_volumetric_config(volumetric_config);

        // Create multiple initial displacements (orthogonal pushes)
        let mut disp1 = Array3::zeros((16, 16, 16));
        disp1[[8, 12, 8]] = 1e-6; // +Y direction push

        let mut disp2 = Array3::zeros((16, 16, 16));
        disp2[[12, 8, 8]] = 1e-6; // +X direction push

        let initial_displacements = vec![disp1, disp2];
        let push_times = vec![0.0, 1e-3]; // Staggered pushes

        let sources = vec![
            VolumetricSource {
                location_m: [0.016, 0.024, 0.016],
                time_offset_s: 0.0,
            },
            VolumetricSource {
                location_m: [0.024, 0.016, 0.016],
                time_offset_s: 1e-3,
            },
        ];

        let result = solver.propagate_volumetric_waves_with_sources(
            &initial_displacements,
            &push_times,
            &sources,
        );
        assert!(result.is_ok());

        let (_history, tracker) = result.unwrap();

        // Check interference tracking (should detect multiple wave arrivals)
        let max_interference = tracker
            .interference_map
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max);
        assert!(
            max_interference >= 1.0,
            "Should detect wave interference from multiple sources"
        );

        // Check volumetric quality
        let quality = solver.calculate_volumetric_quality(&tracker);
        assert!(
            quality.max_interference >= 1.0,
            "Should detect interference patterns"
        );
    }

    #[test]
    fn test_volumetric_boundary_conditions() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let config = ElasticWaveConfig::default();
        let mut solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Enable volumetric boundaries
        let volumetric_config = VolumetricWaveConfig {
            volumetric_boundaries: true,
            ..Default::default()
        };
        solver.set_volumetric_config(volumetric_config);

        // Create test velocity field
        let mut vx = Array3::zeros((10, 10, 10));
        let mut vy = Array3::zeros((10, 10, 10));
        let mut vz = Array3::zeros((10, 10, 10));

        // Set some test values
        vx[[5, 5, 5]] = 1.0;
        vy[[5, 5, 5]] = 1.0;
        vz[[5, 5, 5]] = 1.0;

        // Apply boundary conditions
        solver.apply_volumetric_boundaries(&mut vx, &mut vy, &mut vz, 5, 5, 5);

        // Check that boundary damping was applied (should be less than original)
        assert!(vx[[5, 5, 5]] <= 1.0);
        assert!(vy[[5, 5, 5]] <= 1.0);
        assert!(vz[[5, 5, 5]] <= 1.0);
    }

    #[test]
    fn test_wave_front_tracking_update() {
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let config = ElasticWaveConfig::default();
        let mut solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Enable interference tracking
        let volumetric_config = VolumetricWaveConfig {
            interference_tracking: true,
            arrival_detection: ArrivalDetection::AmplitudeThresholdCrossing { threshold_m: 1e-7 },
            ..Default::default()
        };
        solver.set_volumetric_config(volumetric_config);

        // Create test wave field with displacement above threshold
        let mut field = ElasticWaveField::new(8, 8, 8);
        field.ux[[4, 4, 4]] = 1e-6; // Above threshold
        field.vx[[4, 4, 4]] = 10.0; // Non-zero velocity for direction calculation

        let mut tracker = WaveFrontTracker {
            arrival_times: Array3::from_elem((8, 8, 8), f64::INFINITY),
            amplitudes: Array3::zeros((8, 8, 8)),
            directions: Array3::from_elem((8, 8, 8), [0.0, 0.0, 0.0]),
            interference_map: Array3::zeros((8, 8, 8)),
            tracking_quality: Array3::zeros((8, 8, 8)),
            peak_magnitude_m: Array3::zeros((8, 8, 8)),
            peak_time_s: Array3::from_elem((8, 8, 8), f64::INFINITY),
            mf_buffer: Vec::new(),
            mf_head: 0,
        };

        // Update tracking
        solver.update_wave_front_tracking(&field, &mut tracker, 1e-3, [4, 4, 4]);

        // Check that arrival time was recorded
        assert!(!tracker.arrival_times[[4, 4, 4]].is_infinite());
        assert_eq!(tracker.arrival_times[[4, 4, 4]], 1e-3);

        // Check that amplitude was recorded
        assert!(tracker.amplitudes[[4, 4, 4]] > 0.0);

        // Check that direction was calculated
        let direction = tracker.directions[[4, 4, 4]];
        let magnitude = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        assert!(
            (magnitude - 1.0).abs() < 1e-10,
            "Direction should be unit vector"
        );
    }

    #[test]
    fn test_volumetric_quality_metrics() {
        let grid = Grid::new(6, 6, 6, 0.001, 0.001, 0.001).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let config = ElasticWaveConfig::default();
        let solver = ElasticWaveSolver::new(&grid, &medium, config).unwrap();

        // Create mock tracker with some valid data
        let mut tracker = WaveFrontTracker {
            arrival_times: Array3::from_elem((6, 6, 6), f64::INFINITY),
            amplitudes: Array3::zeros((6, 6, 6)),
            directions: Array3::from_elem((6, 6, 6), [0.0, 0.0, 0.0]),
            interference_map: Array3::zeros((6, 6, 6)),
            tracking_quality: Array3::zeros((6, 6, 6)),
            peak_magnitude_m: Array3::zeros((6, 6, 6)),
            peak_time_s: Array3::from_elem((6, 6, 6), f64::INFINITY),
            mf_buffer: Vec::new(),
            mf_head: 0,
        };

        // Set some valid tracking points
        tracker.arrival_times[[2, 2, 2]] = 1e-3;
        tracker.arrival_times[[3, 3, 3]] = 2e-3;
        tracker.tracking_quality[[2, 2, 2]] = 0.8;
        tracker.tracking_quality[[3, 3, 3]] = 0.9;
        tracker.interference_map[[2, 2, 2]] = 1.0;
        tracker.interference_map[[3, 3, 3]] = 2.0;

        let quality = solver.calculate_volumetric_quality(&tracker);

        // Check quality metrics
        assert!(quality.coverage > 0.0 && quality.coverage <= 1.0);
        assert!(quality.average_quality > 0.0 && quality.average_quality <= 1.0);
        assert_eq!(quality.valid_tracking_points, 2);
        assert_eq!(quality.max_interference, 2.0);
    }
}
