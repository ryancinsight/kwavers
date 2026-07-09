//! FDTD solver configuration

use crate::geometry::SolverGeometry;
use kwavers_core::constants::numerical::CFL_SAFETY_FACTOR;
use kwavers_core::error::{KwaversResult, MultiError, ValidationError};
use leto::Array3;
use serde::{Deserialize, Serialize};

/// Spatial derivative mode for the FDTD solver.
///
/// Controls whether pressure/velocity gradients are computed via finite
/// differences (the classical FDTD method) or via spectral FFT operators
/// with temporal κ correction (the k-space corrected FDTD method).
///
/// ## Comparison
///
/// | Mode       | Phase-velocity error       | Cost per step     |
/// |------------|---------------------------|-------------------|
/// | `None`     | O(kΔx)² — grows to Nyquist | low (stencil ops) |
/// | `Spectral` | 0 (machine precision)      | +2 FFT pairs/step |
///
/// ## When to use `Spectral`
///
/// - Simulations where numerical dispersion affects results (e.g. long
///   propagation distances, high frequencies, parity comparisons with k-Wave)
/// - Reducing SOLVER_TOLERANCES for parity tests
///
/// ## Limitation
///
/// `Spectral` mode is incompatible with CPML boundary corrections because
/// CPML operates on finite-difference gradients. When `Spectral` is active,
/// CPML gradient corrections are silently bypassed. Use this mode without
/// CPML, or with a simpler multiplicative PML.
///
/// **Reference**: Treeby, B.E. & Cox, B.T. (2010). J. Biomed. Opt. 15(2),
/// 021314. doi:10.1117/1.3360308 (§II.A, k-space corrected FDTD)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum KSpaceCorrectionMode {
    /// Classical finite-difference stencils (2nd/4th/6th order). Default.
    #[default]
    None,
    /// Spectral FFT-based gradients + temporal κ correction (k-Wave equivalent).
    Spectral,
}

/// FDTD solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FdtdConfig {
    /// Spatial derivative order (2, 4, or 6)
    pub spatial_order: usize,
    /// Use staggered grid (Yee cell)
    pub staggered_grid: bool,
    /// CFL safety factor (typically 0.3 for 3D FDTD)
    pub cfl_factor: f64,
    /// Enable subgridding for local refinement
    pub subgridding: bool,
    /// Subgridding refinement factor
    pub subgrid_factor: usize,
    /// Enable GPU acceleration (requires "gpu" feature)
    pub enable_gpu_acceleration: bool,
    /// Spatial derivative mode: finite-difference stencil (default) or
    /// spectral k-space corrected operators.
    pub kspace_correction: KSpaceCorrectionMode,

    /// Enable Westervelt nonlinear acoustic propagation.
    ///
    /// When `true`, the nonlinear source term `(β/ρ₀c₀⁴) ∂²p²/∂t²` is added to
    /// the pressure update at each time step. Two historical pressure fields
    /// (`p^{n-1}`, `p^{n-2}`) are maintained in solver state.
    ///
    /// **Reference**: Westervelt (1963), J. Acoust. Soc. Am. 35(4), 535–537.
    pub enable_nonlinear: bool,

    // Parity fields
    /// Number of time steps
    pub nt: usize,
    /// Time step size (s)
    pub dt: f64,
    /// Data recording options
    pub sensor_mask: Option<Array3<bool>>,
    /// Spatial coordinate geometry (Cartesian 3-D or axisymmetric cylindrical).
    pub geometry: SolverGeometry,
}

impl Default for FdtdConfig {
    fn default() -> Self {
        Self {
            spatial_order: 4,
            staggered_grid: true,
            cfl_factor: CFL_SAFETY_FACTOR,
            subgridding: false,
            subgrid_factor: 2,
            enable_gpu_acceleration: false,
            kspace_correction: KSpaceCorrectionMode::None,
            enable_nonlinear: false,
            nt: 1000,
            dt: 1e-7,
            sensor_mask: None,
            geometry: SolverGeometry::Cartesian3D,
        }
    }
}

impl FdtdConfig {
    /// Validate.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        let mut multi_error = MultiError::new();

        // Validate spatial order
        if ![2, 4, 6].contains(&self.spatial_order) {
            multi_error.add(
                ValidationError::FieldValidation {
                    field: "spatial_order".to_owned(),
                    value: self.spatial_order.to_string(),
                    constraint: "Must be 2, 4, or 6".to_owned(),
                }
                .into(),
            );
        }

        // Validate CFL factor.
        //
        // The von Neumann stability limit for 3D second-order FDTD is
        //   CFL_max = 1/√3  (Courant, Friedrichs & Lewy 1928)
        // Using the exact floating-point constant avoids off-by-ε rejection of
        // values at the boundary (e.g. property-based tests that generate values
        // in (0, 1/√3]).
        const CFL_MAX_3D: f64 = 0.577_350_269_189_625_8; // 1/√3, 16 significant digits
        if self.cfl_factor <= 0.0 || self.cfl_factor > CFL_MAX_3D {
            multi_error.add(
                ValidationError::FieldValidation {
                    field: "cfl_factor".to_owned(),
                    value: self.cfl_factor.to_string(),
                    constraint: format!("Must be in (0, {CFL_MAX_3D}] for 3D stability (1/√3)"),
                }
                .into(),
            );
        }

        // Validate subgridding
        if self.subgridding && self.subgrid_factor < 2 {
            multi_error.add(
                ValidationError::FieldValidation {
                    field: "subgrid_factor".to_owned(),
                    value: self.subgrid_factor.to_string(),
                    constraint: "Must be >= 2".to_owned(),
                }
                .into(),
            );
        }

        multi_error.into_result()
    }
}
