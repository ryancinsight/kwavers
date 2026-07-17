//! Solver configuration parameters
//!
//! Consolidated configuration for all solver types.

use kwavers_boundary::cpml::CPMLConfig;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Absorbing boundary parameters for CPML-enabled solvers.
///
/// Bundles the two parameters that `enable_cpml` requires beyond the CPML
/// geometry itself: `max_sound_speed` is the domain-wide maximum of the
/// medium's sound-speed field, used to compute the Roden-Gedney σ_max:
///
/// ```text
/// σ_max = -(m+1) · ln(R₀) / (2 · d · c_max)
/// ```
///
/// `dt` is taken from `SolverConfiguration.dt` at assembly time so that the
/// Courant constraint is enforced once and consistently.
///
/// The factory cannot derive `max_sound_speed` from the abstract
/// `FactoryMediumParameters` trait without full-domain enumeration.  The
/// caller (FWI forward pass, clinical adapter, etc.) computes it directly from
/// the medium array and passes it here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbsorbingBoundaryConfig {
    /// CPML layer geometry and polynomial grading parameters.
    pub cpml: CPMLConfig,
    /// Maximum sound speed across the entire simulation domain [m/s].
    ///
    /// Must equal `model.iter().copied().fold(f64::NEG_INFINITY, f64::max)`
    /// over the sound-speed volume.  Underestimating this value produces
    /// insufficiently damped boundaries (spurious reflections).
    pub max_sound_speed: f64,
}

/// Unified solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfiguration {
    /// Solver type
    pub solver_type: SolverType,
    /// Time integration scheme
    pub time_scheme: TimeScheme,
    /// Spatial discretization order
    pub spatial_order: usize,
    /// Maximum number of time steps (was max_steps in interface config)
    pub max_steps: usize,
    /// Time step size (was dt in interface config)
    pub dt: f64,
    /// CFL number (was cfl in interface config)
    pub cfl: f64,
    /// Convergence tolerance (was tolerance)
    pub tolerance: f64,
    /// Maximum iterations (was max_iterations)
    pub max_iterations: usize,
    /// Use adaptive time stepping
    pub adaptive_dt: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Enable adaptive mesh refinement
    pub enable_amr: bool,
    /// Progress reporting interval
    pub progress_interval: Duration,
    /// Validation mode
    pub validation_mode: bool,
    /// Detailed logging
    pub detailed_logging: bool,
    /// Optional C-PML absorbing boundary.
    ///
    /// When `Some`, `SimulationSolverFactory::create_solver` applies
    /// `enable_cpml(config, dt, max_sound_speed)` on the assembled FDTD
    /// solver before returning the boxed trait object.  This hoists boundary
    /// setup into the factory so callers receive a fully configured
    /// `Box<dyn Solver>` without needing to downcast to a concrete type.
    ///
    /// Ignored for non-FDTD solver types (PSTD, Hybrid, DG) where CPML is
    /// managed through solver-specific configuration paths.
    pub absorbing_boundary: Option<AbsorbingBoundaryConfig>,
}

/// Types of wave equation solvers
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SolverType {
    /// Finite Difference Time Domain
    FDTD,
    /// Pseudo-spectral Time Domain
    PSTD,
    /// GPU-resident PSTD (requires `gpu` Cargo feature and a Hephaestus-backed provider).
    ///
    /// Grid dimensions must be powers of two with each axis ≤ 1,024. Lossless
    /// PSTD requires 24 storage buffers per compute-shader stage; the
    /// fractional-Laplacian absorption path requires 32. The
    /// `kwavers_simulation::SimulationRunner` rejects this selection unless
    /// its request-to-adapter contract is implemented; it never substitutes a
    /// CPU PSTD run. Use `GpuPstdSimulationAdapter` directly for the current
    /// GPU batch interface.
    PstdGpu,
    /// Hybrid solver combining PSTD and FDTD
    Hybrid,
    /// k-space pseudo-spectral
    KSpace,
    /// Discontinuous Galerkin
    DiscontinuousGalerkin,
    /// Finite Element Method
    FEM,
    /// Automatically selected
    Auto,
    /// Elastic-wave solver (4th-order FD with velocity-Verlet integration)
    Elastic,
    /// Pseudospectral elastic solver
    ElasticPSTD,
    /// Frequency-domain Helmholtz FEM solver
    Helmholtz,
    /// Boundary Element Method solver
    BEM,
    /// Discontinuous Galerkin / Hybrid Spectral-DG solver
    DG,
    /// Nonlinear acoustic wave solvers (Westervelt, Kuznetsov, KZK)
    Nonlinear,
    /// Biot poroelastic wave solver
    Poroelastic,
    /// Rayleigh-Sommerfeld angular-spectrum solver
    RayleighSommerfeld,
}

/// Time integration schemes
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TimeScheme {
    /// Forward Euler (1st order)
    ForwardEuler,
    /// Leapfrog (2nd order)
    Leapfrog,
    /// Runge-Kutta 4 (4th order)
    RungeKutta4,
    /// Adams-Bashforth (multi-step)
    AdamsBashforth,
}

impl Default for SolverConfiguration {
    fn default() -> Self {
        Self {
            solver_type: SolverType::FDTD,
            time_scheme: TimeScheme::Leapfrog,
            spatial_order: 4,
            max_steps: 1000,
            dt: 1e-7,
            cfl: 0.3,
            tolerance: 1e-6,
            max_iterations: 1000,
            adaptive_dt: false,
            enable_gpu: false,
            enable_amr: false,
            progress_interval: Duration::from_secs(10),
            validation_mode: false,
            detailed_logging: false,
            absorbing_boundary: None,
        }
    }
}

impl SolverConfiguration {
    /// Validate solver parameters
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn validate(&self) -> kwavers_core::error::KwaversResult<()> {
        if self.spatial_order == 0 || self.spatial_order > 16 {
            return Err(kwavers_core::error::ConfigError::InvalidValue {
                parameter: "spatial_order".to_owned(),
                value: self.spatial_order.to_string(),
                constraint: "Must be between 1 and 16".to_owned(),
            }
            .into());
        }

        if self.max_steps == 0 {
            return Err(kwavers_core::error::ConfigError::InvalidValue {
                parameter: "max_steps".to_owned(),
                value: "0".to_owned(),
                constraint: "Must be positive".to_owned(),
            }
            .into());
        }

        if self.dt <= 0.0 {
            return Err(kwavers_core::error::ConfigError::InvalidValue {
                parameter: "dt".to_owned(),
                value: self.dt.to_string(),
                constraint: "Must be positive".to_owned(),
            }
            .into());
        }

        if self.cfl <= 0.0 || self.cfl > 1.0 {
            return Err(kwavers_core::error::ConfigError::InvalidValue {
                parameter: "cfl".to_owned(),
                value: self.cfl.to_string(),
                constraint: "Must be between 0 and 1".to_owned(),
            }
            .into());
        }

        Ok(())
    }

    /// Create a configuration optimized for accuracy
    #[must_use]
    pub fn accuracy_optimized() -> Self {
        Self {
            cfl: 0.1,
            spatial_order: 6,
            ..Default::default()
        }
    }

    /// Create a configuration optimized for performance
    #[must_use]
    pub fn performance_optimized() -> Self {
        Self {
            cfl: 0.5,
            spatial_order: 2,
            enable_gpu: true,
            progress_interval: Duration::from_secs(30),
            ..Default::default()
        }
    }
}
