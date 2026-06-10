//! Seismic Imaging Parameters and Configuration
//!
//! Configuration structures for Full Waveform Inversion and Reverse Time Migration
//! Following GRASP principles: Information Expert pattern for parameter management

use crate::config::SolverType;

/// Full Waveform Inversion Parameters
#[derive(Debug, Clone)]
pub struct FwiParameters {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Step size for gradient descent
    pub step_size: f64,
    /// Time steps for simulation
    pub nt: usize,
    /// Time step size (s)
    pub dt: f64,
    /// Number of traces (for injection/recording)
    pub n_trace: usize,
    /// Depth of traces (for injection/recording)
    pub n_depth: usize,
    /// Regularization weight
    pub regularization: RegularizationParameters,
    /// Central source frequency (Hz)
    ///
    /// Used as the reference frequency for medium construction (absorption law
    /// frequency exponent) and for the Ricker wavelet.  Typical exploration
    /// seismic surveys use 10–100 Hz.
    pub frequency: f64,

    /// Radius (in voxels, L2 norm) within which the gradient is zeroed around
    /// every source position to suppress near-source body-wave artefacts.
    ///
    /// ## Theorem (near-source artefact)
    /// Close to a source voxel, `∂²p/∂t²` is dominated by the second derivative
    /// of the source wavelet rather than by propagation physics.  The
    /// cross-correlation of this large `p̈` with the adjoint wavefield produces a
    /// gradient 10–100 × larger than the physical sensitivity at the target zone.
    /// Zeroing within half a wavelength of each source removes this artefact
    /// without biasing the gradient at distant scatterers.
    ///
    /// ## Recommended value
    /// `ceil(c_min / (2 · f₀ · dx))` — half-wavelength at the minimum velocity.
    /// For 150 kHz, c_min = 1500 m/s, dx = 3 mm: ≈ 2 voxels.  Use 4 for safety.
    ///
    /// Default 0 disables the mute (backward-compatible).
    pub source_mute_radius: usize,

    /// Forward (and adjoint) solver type for the time-domain FWI passes.
    ///
    /// ## Solver selection contract
    ///
    /// Both the forward and adjoint models use the **same** solver type so that
    /// the discrete adjoint operator is the exact time-reversal of the forward
    /// operator (time-reversal theorem, Plessix 2006).  Mixing solver types
    /// between forward and adjoint breaks the gradient identity.
    ///
    /// | `SolverType` | Behaviour |
    /// |---|---|
    /// | `FDTD` | `FdtdSolver` with 2nd-order spatial stencil + CPML (default). |
    /// | `PSTD` | `PSTDSolver` with k-space spectral propagation + CPML embedded in `PSTDConfig::boundary`. |
    /// | Others | `Err(InvalidInput)` — not yet wired. |
    ///
    /// Default `SolverType::FDTD` preserves backward-compatible behaviour.
    pub solver_type: SolverType,
}

/// Regularization parameters for inversion
#[derive(Debug, Clone)]
pub struct RegularizationParameters {
    /// Tikhonov regularization weight
    pub tikhonov_weight: f64,
    /// Axis-aligned (isotropic ROF) total variation weight.
    pub tv_weight: f64,
    /// Four-direction total variation (FDTV) weight.
    ///
    /// Adds the in-plane face-diagonal difference directions to the TV operator,
    /// producing a more rotation-invariant discretization that suppresses the
    /// directional streak/staircase artefacts characteristic of sparse-aperture
    /// (few-source / few-receiver) acquisition — the acoustic analog of
    /// sparse-view CT. In any coordinate plane the operator reduces to the
    /// horizontal, vertical, and two diagonal differences of the FDTV model.
    ///
    /// Reference: Zhang et al. (2023), *adaptive four-direction TV for
    /// sparse-view CT*, PMC10745410. Default `0.0` (disabled, backward-compatible).
    pub directional_tv_weight: f64,
    /// Smoothness constraint weight
    pub smoothness_weight: f64,
}

/// Convergence criteria for iterative methods
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Relative tolerance
    pub relative_tolerance: f64,
    /// Absolute tolerance
    pub absolute_tolerance: f64,
}

/// Reverse Time Migration Settings
#[derive(Debug, Clone)]
pub struct RtmSettings {
    /// Imaging condition type
    pub imaging_condition: ImagingCondition,
    /// Source wavefield storage strategy
    pub storage_strategy: StorageStrategy,
    /// Boundary handling method
    pub boundary_type: SeismicBoundaryType,
    /// Apply Laplacian filtering
    pub apply_laplacian: bool,
}

/// Available imaging conditions for RTM
#[derive(Debug, Clone)]
pub enum ImagingCondition {
    /// Zero-lag cross-correlation
    ZeroLag,
    /// Normalized cross-correlation
    Normalized,
}

/// Wavefield storage strategies for memory management
#[derive(Debug, Clone)]
pub enum StorageStrategy {
    /// Store all timeframes (high memory)
    Full,
    /// Checkpointing strategy (balanced)
    Checkpoints(usize),
}

/// Boundary condition types for imaging
#[derive(Debug, Clone)]
pub enum SeismicBoundaryType {
    /// Absorbing boundary layers
    Absorbing,
    /// Free surface boundary
    FreeSurface,
    /// Periodic boundaries
    Periodic,
}

/// Migration aperture for focusing
#[derive(Debug, Clone)]
pub struct MigrationAperture {
    /// Half-aperture angle in radians
    pub half_angle: f64,
    /// Maximum migration distance
    pub max_distance: f64,
    /// Aperture taper function
    pub taper: TaperFunction,
}

/// Taper functions for aperture control
#[derive(Debug, Clone)]
pub enum TaperFunction {
    /// No tapering
    None,
    /// Cosine taper
    Cosine,
    /// Gaussian taper
    Gaussian { sigma: f64 },
}

impl Default for FwiParameters {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            step_size: 0.01,
            nt: 1000,
            dt: 1e-4,
            n_trace: 100,
            n_depth: 100,
            regularization: RegularizationParameters::default(),
            frequency: 20.0, // Hz — typical shallow-seismic exploration bandwidth
            source_mute_radius: 0, // disabled by default (backward-compatible)
            solver_type: SolverType::FDTD,
        }
    }
}

impl Default for RegularizationParameters {
    fn default() -> Self {
        Self {
            tikhonov_weight: 1e-3,
            tv_weight: 1e-4,
            directional_tv_weight: 0.0,
            smoothness_weight: 1e-2,
        }
    }
}

impl Default for RtmSettings {
    fn default() -> Self {
        Self {
            imaging_condition: ImagingCondition::ZeroLag,
            storage_strategy: StorageStrategy::Checkpoints(10),
            boundary_type: SeismicBoundaryType::Absorbing,
            apply_laplacian: true,
        }
    }
}

impl Default for MigrationAperture {
    fn default() -> Self {
        Self {
            half_angle: std::f64::consts::PI / 6.0, // 30 degrees
            max_distance: 1000.0,                   // 1km
            taper: TaperFunction::Cosine,
        }
    }
}
