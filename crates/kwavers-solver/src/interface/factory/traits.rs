//! Abstract factory and domain parameter traits.

use super::types::FactoryError;
use crate::config::{SolverConfiguration, SolverType};
use crate::interface::Solver;
use kwavers_math::fft::{fft_3d_array, ifft_3d_array, Complex64, Normalization};
use leto::Array3 as LetoArray3;
use ndarray::{Array2, Array3};

/// Abstract factory for creating solver instances
pub trait SolverFactoryTrait {
    /// Error type for factory operations
    type Error;

    /// Create a solver of the specified type with given configuration
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn create_solver(
        &self,
        solver_type: SolverType,
        config: &SolverConfiguration,
        grid_params: &dyn FactoryGridParameters,
        medium_params: &dyn FactoryMediumParameters,
        source_params: &dyn FactorySourceParameters,
    ) -> Result<Box<dyn Solver>, Self::Error>;

    /// Select best solver type based on problem characteristics
    ///
    /// THEOREM: Solver Selection Optimality
    /// For grid G with resolution Δx and medium M with heterogeneity σ:
    /// select(G, M) = argmin_{T} [Cost(T, G, M) | Accuracy(T) ≥ A_min]
    fn select_best_solver(
        &self,
        grid_params: &dyn FactoryGridParameters,
        medium_params: &dyn FactoryMediumParameters,
    ) -> SolverType;
}

/// Abstract parameters for grid creation
pub trait FactoryGridParameters {
    fn nx(&self) -> usize;
    fn ny(&self) -> usize;
    fn nz(&self) -> usize;
    fn dx(&self) -> f64;
    fn dy(&self) -> f64;
    fn dz(&self) -> f64;

    fn total_points(&self) -> usize {
        self.nx() * self.ny() * self.nz()
    }

    fn characteristic_size(&self) -> f64 {
        (self.nx() as f64 * self.dx())
            .max(self.ny() as f64 * self.dy())
            .max(self.nz() as f64 * self.dz())
    }
}

/// Abstract parameters for medium specification
pub trait FactoryMediumParameters {
    fn sound_speed(&self, x: f64, y: f64, z: f64) -> f64;
    fn density(&self, x: f64, y: f64, z: f64) -> f64;
    fn heterogeneity(&self) -> f64;

    fn is_homogeneous(&self) -> bool {
        self.heterogeneity() < 1e-6
    }

    fn absorption(&self, frequency: f64) -> f64;
}

/// Abstract parameters for source specification
pub trait FactorySourceParameters {
    fn source_type(&self) -> &str;
    fn frequency(&self) -> f64;
    fn amplitude(&self) -> f64;
    fn position(&self) -> Option<(usize, usize, usize)>;
    fn duration(&self) -> f64;
    fn waveform(&self) -> &str;
}

/// Canonical Fourier backend contract for solver-layer consumers.
///
/// `kwavers` depends on this trait instead of directly depending on a specific
/// FFT cache or planner implementation.
pub trait FourierBackend: std::fmt::Debug + Send + Sync {
    fn backend_name(&self) -> &'static str;
    fn normalization(&self) -> Normalization;
    fn is_available(&self) -> bool;
    fn workspace_len_3d(&self, nx: usize, ny: usize, nz: usize) -> usize;
    fn forward_3d_real(&self, field: &LetoArray3<f64>) -> LetoArray3<Complex64>;
    fn inverse_3d_real(&self, spectrum: &LetoArray3<Complex64>) -> LetoArray3<f64>;
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

    fn forward_3d_real(&self, field: &LetoArray3<f64>) -> LetoArray3<Complex64> {
        fft_3d_array(field)
    }

    fn inverse_3d_real(&self, spectrum: &LetoArray3<Complex64>) -> LetoArray3<f64> {
        ifft_3d_array(spectrum)
    }
}

/// Canonical mesh-provider contract for geometry ownership inversion.
pub trait MeshProvider: std::fmt::Debug + Send + Sync {
    fn line_sensor_positions(
        &self,
        origin_m: [f64; 3],
        direction_m: [f64; 3],
        spacing_m: f64,
        count: usize,
    ) -> Vec<[f64; 3]>;

    #[allow(clippy::too_many_arguments)]
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

    fn voxel_to_surface_map(
        &self,
        voxel_positions_m: &[[f64; 3]],
        surface_points_m: &[[f64; 3]],
    ) -> Vec<usize>;
}

/// Abstract image registration contract.
///
/// ## Mathematical Specification
///
/// Given a fixed image $F: \Omega \to \mathbb{R}$ and a moving image $M: \Omega \to \mathbb{R}$,
/// registration finds a spatial transformation $\phi: \Omega \to \Omega$ such that
/// $M \circ \phi^{-1} \approx F$ under a chosen similarity metric $\mathcal{S}$.
pub trait RegistrationEngine: std::fmt::Debug + Send + Sync {
    /// Rigid-body (6-DOF) registration via mutual information maximisation.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn register_rigid(
        &self,
        fixed: &Array3<f64>,
        moving: &Array3<f64>,
    ) -> Result<Array2<f64>, FactoryError>;

    /// Affine (12-DOF) registration.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn register_affine(
        &self,
        fixed: &Array3<f64>,
        moving: &Array3<f64>,
    ) -> Result<Array2<f64>, FactoryError>;

    /// Deformable registration returning a dense displacement field.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn register_deformable(
        &self,
        fixed: &Array3<f64>,
        moving: &Array3<f64>,
    ) -> Result<Array3<[f64; 3]>, FactoryError>;

    /// Resample `moving` volume using a 4×4 homogeneous `transform` to `target_shape`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn resample(
        &self,
        moving: &Array3<f64>,
        transform: &Array2<f64>,
        target_shape: [usize; 3],
    ) -> Result<Array3<f64>, FactoryError>;
}
