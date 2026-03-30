//! FDTD solver configuration

use crate::core::constants::numerical::CFL_SAFETY_FACTOR;
use crate::core::error::{KwaversResult, MultiError, ValidationError};
use ndarray::Array3;
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
    /// Time step size [s]
    pub dt: f64,
    /// Data recording options
    pub sensor_mask: Option<Array3<bool>>,
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
        }
    }
}

impl FdtdConfig {
    pub fn validate(&self) -> KwaversResult<()> {
        let mut multi_error = MultiError::new();

        // Validate spatial order
        if ![2, 4, 6].contains(&self.spatial_order) {
            multi_error.add(
                ValidationError::FieldValidation {
                    field: "spatial_order".to_string(),
                    value: self.spatial_order.to_string(),
                    constraint: "Must be 2, 4, or 6".to_string(),
                }
                .into(),
            );
        }

        // Validate CFL factor (max stable for 3D is 1/sqrt(3) ≈ 0.577)
        if self.cfl_factor <= 0.0 || self.cfl_factor > 0.577 {
            multi_error.add(
                ValidationError::FieldValidation {
                    field: "cfl_factor".to_string(),
                    value: self.cfl_factor.to_string(),
                    constraint: "Must be in (0, 0.577] for 3D stability".to_string(),
                }
                .into(),
            );
        }

        // Validate subgridding
        if self.subgridding && self.subgrid_factor < 2 {
            multi_error.add(
                ValidationError::FieldValidation {
                    field: "subgrid_factor".to_string(),
                    value: self.subgrid_factor.to_string(),
                    constraint: "Must be >= 2".to_string(),
                }
                .into(),
            );
        }

        multi_error.into_result()
    }
}
