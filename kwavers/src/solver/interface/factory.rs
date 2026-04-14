// Solver Factory trait abstraction
//
// This module defines the abstract factory interface for solver creation,
// following the Dependency Inversion Principle. The trait lives in the solver
// layer (abstract), while concrete implementations reside in the domain layer.
//
// ## Architecture
//
// Before:
// ```text
// solver/factory.rs → domain::Grid, domain::Medium, domain::Source ❌ Violates DIP
// ```
//
// After:
// ```text
// solver/interface/factory.rs (SolverFactory trait) ←── domain/factory.rs (DomainSolverFactory impl) ✅ DIP Compliant
//     ↓
// solver/factory.rs (orchestration only)
// ```
//
// ## Mathematical Specification
//
// **Theorem**: Factory Abstraction Completeness
// For any solver type T, the factory F must produce:
// F.create(T, config) → S where S: Solver and S.get_type() == T
//
// **Contract**: The factory guarantees that created solvers satisfy:
// 1. Memory budget compliance: S.memory_usage() ≤ config.memory_budget
// 2. Feature availability: S.features() ⊇ config.required_features
// 3. Performance constraints: S.benchmark() meets config.performance_target
//
// ## References
//
// - Gamma et al. (1994) "Design Patterns", Factory Pattern
// - Martin, R. C. (2017) "Clean Architecture", Chapter on DIP

use crate::solver::config::{SolverConfiguration, SolverType};
use crate::solver::interface::Solver;
use apollofft::{self, Complex64, Normalization};
use ndarray::{Array2, Array3};

/// Abstract factory for creating solver instances
///
/// This trait defines the interface for solver creation without depending
/// on domain-specific types. Implementations provide concrete solver instances
/// based on configuration parameters.
///
/// ## Type Parameters
///
/// - `Config`: Solver configuration type (solver-layer abstraction)
/// - `Error`: Error type for factory failures
///
/// ## Invariants
///
/// 1. Idempotence: create(T, cfg) called twice with same args produces
///    functionally equivalent solvers (may differ in memory address)
/// 2. Consistency: create(T, cfg).get_type() == T for all valid T
/// 3. Resource Safety: failed creation must not leak resources
///
/// THEOREM: Factory Consistency
/// ∀ T ∈ SolverType, cfg ∈ SolverConfiguration:
/// let s = factory.create(T, cfg)
/// assert!(s.is_ok() → s.unwrap().solver_type() == T)
pub trait SolverFactory {
    /// Error type for factory operations
    type Error;

    /// Create a solver of the specified type with given configuration
    ///
    /// # Arguments
    /// * `solver_type` - The type of solver to create (FDTD, PSTD, Hybrid, etc.)
    /// * `config` - Solver configuration parameters
    /// * `grid_params` - Grid parameters as abstract descriptor (avoids domain::Grid dependency)
    /// * `medium_params` - Medium parameters as abstract descriptor
    /// * `source_params` - Source parameters as abstract descriptor
    ///
    /// # Returns
    /// * `Ok(Box<dyn Solver>)` - Solver instance boxed for polymorphism
    /// * `Err(Self::Error)` - Creation failed with detailed error
    fn create_solver(
        &self,
        solver_type: SolverType,
        config: &SolverConfiguration,
        grid_params: &dyn GridParameters,
        medium_params: &dyn MediumParameters,
        source_params: &dyn SourceParameters,
    ) -> Result<Box<dyn Solver>, Self::Error>;

    /// Select best solver type based on problem characteristics
    ///
    /// Returns the optimal solver type given grid and medium properties.
    /// The selection algorithm should minimize computational cost while
    /// maintaining required accuracy.
    ///
    /// THEOREM: Solver Selection Optimality
    /// For grid G with resolution Δx and medium M with heterogeneity σ:
    /// select(G, M) = argmin_{T} [Cost(T, G, M) | Accuracy(T) ≥ A_min]
    ///
    /// # Cost Model
    /// - FDTD: O(n_timesteps × n_grid) × C_compute
    /// - PSTD: O(n_timesteps × n_grid × log(n_grid)) × C_fft
    /// - Hybrid: O(n_timesteps × (n_spectral + n_finite)) × C_mixed
    fn select_best_solver(
        &self,
        grid_params: &dyn GridParameters,
        medium_params: &dyn MediumParameters,
    ) -> SolverType;
}

/// Abstract parameters for grid creation
///
/// Dissociates solver factory from concrete Grid type while preserving
/// all necessary grid information.
pub trait GridParameters {
    /// Number of grid points in x dimension
    fn nx(&self) -> usize;

    /// Number of grid points in y dimension
    fn ny(&self) -> usize;

    /// Number of grid points in z dimension
    fn nz(&self) -> usize;

    /// Grid spacing in x dimension (meters)
    fn dx(&self) -> f64;

    /// Grid spacing in y dimension (meters)
    fn dy(&self) -> f64;

    /// Grid spacing in z dimension (meters)
    fn dz(&self) -> f64;

    /// Total number of grid points
    fn total_points(&self) -> usize {
        self.nx() * self.ny() * self.nz()
    }

    /// Characteristic grid size (max dimension)
    fn characteristic_size(&self) -> f64 {
        (self.nx() as f64 * self.dx())
            .max(self.ny() as f64 * self.dy())
            .max(self.nz() as f64 * self.dz())
    }
}

/// Abstract parameters for medium specification
///
/// Provides medium properties without depending on concrete Medium types.
pub trait MediumParameters {
    /// Sound speed at given point (m/s)
    fn sound_speed(&self, x: f64, y: f64, z: f64) -> f64;

    /// Density at given point (kg/m³)
    fn density(&self, x: f64, y: f64, z: f64) -> f64;

    /// Heterogeneity measure (coefficient of variation of sound speed)
    fn heterogeneity(&self) -> f64;

    /// Is the medium homogeneous?
    fn is_homogeneous(&self) -> bool {
        self.heterogeneity() < 1e-6
    }

    /// Absorption strength (dB/MHz²/cm)
    fn absorption(&self, frequency: f64) -> f64;
}

/// Abstract parameters for source specification
///
/// Provides source configuration without depending on concrete Source types.
pub trait SourceParameters {
    /// Source type identifier
    fn source_type(&self) -> &str;

    /// Signal frequency (Hz)
    fn frequency(&self) -> f64;

    /// Source amplitude (Pa)
    fn amplitude(&self) -> f64;

    /// Source position in grid coordinates
    fn position(&self) -> Option<(usize, usize, usize)>;

    /// Time signal duration (seconds)
    fn duration(&self) -> f64;

    /// Waveform type (sine, tone_burst, chirp, etc.)
    fn waveform(&self) -> &str;
}

/// Canonical Fourier backend contract for solver-layer consumers.
///
/// `kwavers` depends on this trait instead of directly depending on a specific
/// FFT cache or planner implementation. The canonical production owner is
/// Apollo. Consumers may rely on:
///
/// 1. FFTW-compatible normalization being stated explicitly
/// 2. backend capability inspection before dispatch
/// 3. workspace sizing being discoverable without executing the transform
pub trait FourierBackend: std::fmt::Debug + Send + Sync {
    /// Backend identifier for logging and validation reports.
    fn backend_name(&self) -> &'static str;

    /// Normalization convention implemented by this backend.
    fn normalization(&self) -> Normalization;

    /// True when the backend is usable in the current process.
    fn is_available(&self) -> bool;

    /// Required scratch size in complex values for a 3D transform.
    fn workspace_len_3d(&self, nx: usize, ny: usize, nz: usize) -> usize;

    /// Forward transform of a real-valued 3D field.
    fn forward_3d_real(&self, field: &Array3<f64>) -> Array3<Complex64>;

    /// Inverse transform of a complex-valued 3D field to a real field.
    fn inverse_3d_real(&self, spectrum: &Array3<Complex64>) -> Array3<f64>;
}

/// Canonical Apollo-backed Fourier backend adapter.
#[derive(Debug, Default, Clone, Copy)]
pub struct ApolloFourierBackend;

impl FourierBackend for ApolloFourierBackend {
    fn backend_name(&self) -> &'static str {
        "apollo-cpu"
    }

    fn normalization(&self) -> Normalization {
        Normalization::FftwCompatible
    }

    fn is_available(&self) -> bool {
        true
    }

    fn workspace_len_3d(&self, nx: usize, ny: usize, nz: usize) -> usize {
        nx * ny * nz
    }

    fn forward_3d_real(&self, field: &Array3<f64>) -> Array3<Complex64> {
        apollofft::fft_3d_array(field)
    }

    fn inverse_3d_real(&self, spectrum: &Array3<Complex64>) -> Array3<f64> {
        apollofft::ifft_3d_array(spectrum)
    }
}

/// Canonical mesh-provider contract for geometry ownership inversion.
pub trait MeshProvider: std::fmt::Debug + Send + Sync {
    /// Generate a Cartesian line-sensor arrangement.
    fn line_sensor_positions(
        &self,
        origin_m: [f64; 3],
        direction_m: [f64; 3],
        spacing_m: f64,
        count: usize,
    ) -> Vec<[f64; 3]>;

    /// Generate a Cartesian planar sensor arrangement.
    fn planar_sensor_positions(
        &self,
        origin_m: [f64; 3],
        u_axis_m: [f64; 3],
        v_axis_m: [f64; 3],
        spacing_u_m: f64,
        spacing_v_m: f64,
        count_u: usize,
        count_v: usize,
    ) -> Vec<[f64; 3]>;

    /// Map voxel centers to a detector surface representation.
    fn voxel_to_surface_map(
        &self,
        voxel_positions_m: &[[f64; 3]],
        surface_points_m: &[[f64; 3]],
    ) -> Vec<usize>;
}



/// Factory configuration with performance targets
///
/// Optional configuration for factory behavior tuning.
#[derive(Debug, Clone)]
pub struct FactoryConfiguration {
    /// Maximum memory budget (bytes)
    pub memory_budget: usize,
    /// Required solver features
    pub required_features: Vec<String>,
    /// Target performance ratio (vs reference)
    pub performance_target: f64,
    /// Enable solver auto-selection
    pub enable_auto_selection: bool,
}

impl Default for FactoryConfiguration {
    fn default() -> Self {
        Self {
            memory_budget: usize::MAX,
            required_features: Vec::new(),
            performance_target: 1.0,
            enable_auto_selection: true,
        }
    }
}

/// Factory error types with structured diagnostics
#[derive(Debug)]
pub enum FactoryError {
    /// Requested solver type not available
    SolverTypeNotSupported(SolverType),
    /// Configuration invalid or incomplete
    InvalidConfiguration(String),
    /// Resource constraints violated (memory, etc.)
    ResourceExceeded { requested: usize, available: usize },
    /// Factory not initialized properly
    NotInitialized,
    /// Internal error during solver construction
    Internal(String),
}

impl std::fmt::Display for FactoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactoryError::SolverTypeNotSupported(t) => {
                write!(f, "Solver type not supported: {:?}", t)
            }
            FactoryError::InvalidConfiguration(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            FactoryError::ResourceExceeded {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Resource exceeded: requested {} bytes, available {} bytes",
                    requested, available
                )
            }
            FactoryError::NotInitialized => write!(f, "Factory not initialized"),
            FactoryError::Internal(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for FactoryError {}

/// Abstract image registration contract.
///
/// Defines the canonical registration API used by fusion algorithms and coupled solvers.
/// Concrete implementations are provided by the ritk toolkit (classical) or ML backends.
///
/// ## Mathematical Specification
///
/// Given a fixed image $F: \Omega \to \mathbb{R}$ and a moving image $M: \Omega \to \mathbb{R}$,
/// registration finds a spatial transformation $\phi: \Omega \to \Omega$ such that
/// $M \circ \phi^{-1} \approx F$ under a chosen similarity metric $\mathcal{S}$.
///
/// - **Rigid**: $\phi(x) = Rx + t$, $R \in SO(3)$, 6 DOF
/// - **Affine**: $\phi(x) = Ax + t$, $A \in GL(3)$, 12 DOF
/// - **Deformable**: $\phi(x) = x + u(x)$, $u$ a displacement field
pub trait RegistrationEngine: std::fmt::Debug + Send + Sync {
    /// Rigid-body (6-DOF) registration via mutual information maximisation.
    ///
    /// Returns a 4×4 homogeneous transformation matrix as a row-major `Array2<f64>`.
    fn register_rigid(
        &self,
        fixed: &Array3<f64>,
        moving: &Array3<f64>,
    ) -> Result<Array2<f64>, FactoryError>;

    /// Affine (12-DOF) registration: rotation + translation + anisotropic scale.
    fn register_affine(
        &self,
        fixed: &Array3<f64>,
        moving: &Array3<f64>,
    ) -> Result<Array2<f64>, FactoryError>;

    /// Deformable registration returning a dense displacement field.
    ///
    /// Each voxel in `moving` maps to a `[f64; 3]` displacement vector.
    fn register_deformable(
        &self,
        fixed: &Array3<f64>,
        moving: &Array3<f64>,
    ) -> Result<Array3<[f64; 3]>, FactoryError>;

    /// Resample `moving` volume using a 4×4 homogeneous `transform` to `target_shape`.
    fn resample(
        &self,
        moving: &Array3<f64>,
        transform: &Array2<f64>,
        target_shape: [usize; 3],
    ) -> Result<Array3<f64>, FactoryError>;
}

/// Conversion from FactoryError to KwaversError
impl From<FactoryError> for crate::core::error::KwaversError {
    fn from(err: FactoryError) -> Self {
        crate::core::error::KwaversError::InternalError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock grid parameters for testing
    struct MockGridParams {
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
    }

    impl GridParameters for MockGridParams {
        fn nx(&self) -> usize {
            self.nx
        }
        fn ny(&self) -> usize {
            self.ny
        }
        fn nz(&self) -> usize {
            self.nz
        }
        fn dx(&self) -> f64 {
            self.dx
        }
        fn dy(&self) -> f64 {
            self.dy
        }
        fn dz(&self) -> f64 {
            self.dz
        }
    }

    #[test]
    fn grid_parameters_total_points() {
        let grid = MockGridParams {
            nx: 64,
            ny: 64,
            nz: 64,
            dx: 1e-3,
            dy: 1e-3,
            dz: 1e-3,
        };
        assert_eq!(grid.total_points(), 64_usize.pow(3));
    }

    #[test]
    fn factory_configuration_defaults() {
        let config = FactoryConfiguration::default();
        assert!(config.enable_auto_selection);
        assert_eq!(config.performance_target, 1.0);
    }

    #[test]
    fn factory_error_display() {
        let err = FactoryError::SolverTypeNotSupported(SolverType::FDTD);
        assert!(err.to_string().contains("FDTD"));
    }
}
