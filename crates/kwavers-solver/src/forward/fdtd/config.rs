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

/// Absorption model for the FDTD pressure update.
///
/// A two-variant enum rather than a flag plus loose parameters: the parameters
/// are meaningless without the model, and the lossless case must carry none.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum FdtdAbsorption {
    /// No absorption. The pressure update uses the lossless modulus `rho_0*c_0^2`
    /// and allocates no memory fields.
    #[default]
    Lossless,
    /// Heterogeneous power-law absorption `alpha_0(x)*(f/f_ref)^gamma(x)`,
    /// realized by relaxation memory variables fitted to the medium's own
    /// coefficient **and exponent** fields.
    ///
    /// Costs one auxiliary field per arm per voxel, so `relaxation_arms` is the
    /// memory knob: three optimized arms reproduce a decade-wide power law to
    /// about 0.2 %, and two to about 2 %.
    PowerLawRelaxation {
        /// Frequency at which the medium's `alpha_0` is quoted \[Hz].
        reference_frequency_hz: f64,
        /// Lower edge of the band the power law must hold over \[Hz].
        band_min_hz: f64,
        /// Upper edge of the band the power law must hold over \[Hz].
        band_max_hz: f64,
        /// Relaxation arms, i.e. memory fields per voxel.
        relaxation_arms: usize,
    },
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

    /// Absorption model applied in the pressure update.
    #[serde(default)]
    pub absorption: FdtdAbsorption,

    /// Time-integration scheme.
    #[serde(default)]
    pub temporal_scheme: TemporalScheme,
}

/// How the solver advances in time.
///
/// Fullwave 2.5 runs fourth order in space *and* time; kwavers matched the
/// spatial order first, and this closes the temporal half (KW-SOL-092).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TemporalScheme {
    /// Staggered leapfrog: one velocity update then one pressure update per
    /// step. Second-order accurate in time, and the cheapest step available.
    #[default]
    Leapfrog,

    /// Yoshida's symmetric triple composition, fourth-order accurate in time.
    ///
    /// # What it composes, and why not the plain step
    ///
    /// Yoshida composition reaches fourth order only when the method it
    /// composes is **self-adjoint**. The plain step is kick-then-drift, whose
    /// adjoint is drift-then-kick, so composing *that* would not gain an order.
    /// Each sub-step is therefore the symmetric `K(h/2) D(h) K(h/2)` -
    /// half-velocity, full-pressure, half-velocity - which is Stormer-Verlet
    /// and self-adjoint.
    ///
    /// # What it costs
    ///
    /// Three sub-steps per step, and the central Yoshida weight is negative with
    /// `|w0| ~ 1.7025`, so the largest sub-step is that much longer than `dt`
    /// and the stable step shrinks by the same factor. Roughly five times the
    /// work per unit simulated time, bought back by error falling as `dt^4`
    /// rather than `dt^2`.
    Yoshida4,
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
            absorption: FdtdAbsorption::Lossless,
            temporal_scheme: TemporalScheme::Leapfrog,
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

        // Absorption is rejected under composition, and the reason is *not* the
        // negative central sub-step, which was the original suspicion. Over a
        // full composition the memory exponents sum to `-(2w1 + w0)dt/tau =
        // -dt/tau` exactly, so the decay is right in aggregate; the intermediate
        // amplification is `exp(1.70 dt/tau)`, which sits just above unity for
        // any `tau` in the fit band. The actual blocker is that
        // `RelaxationAbsorption` precomputes `decay = exp(-dt/tau)` **once** from
        // the configured step, so every sub-step would silently reuse the
        // full-step decay - a quiet accuracy defect rather than a visible
        // failure. Fixing it means per-sub-step decay arrays and a memory
        // tradeoff, tracked as KW-SOL-093.
        if self.temporal_scheme == TemporalScheme::Yoshida4 {
            if self.absorption != FdtdAbsorption::Lossless {
                multi_error.add(
                    ValidationError::FieldValidation {
                        field: "temporal_scheme".to_owned(),
                        value: format!("{:?} with {:?}", self.temporal_scheme, self.absorption),
                        constraint: "fourth-order time integration is unverified with an                                      absorbing model - relaxation decay is precomputed for                                      the full step and would be reused unchanged by every                                      sub-step; use Leapfrog or run lossless (KW-SOL-093)"
                            .to_owned(),
                    }
                    .into(),
                );
            }
            if !self.staggered_grid {
                multi_error.add(
                    ValidationError::FieldValidation {
                        field: "temporal_scheme".to_owned(),
                        value: format!("{:?} with staggered_grid = false", self.temporal_scheme),
                        constraint: "fourth-order time integration is available on the                                      staggered path only"
                            .to_owned(),
                    }
                    .into(),
                );
            }
        }

        if let FdtdAbsorption::PowerLawRelaxation {
            reference_frequency_hz,
            band_min_hz,
            band_max_hz,
            relaxation_arms,
        } = self.absorption
        {
            let ok = reference_frequency_hz.is_finite()
                && reference_frequency_hz > 0.0
                && band_min_hz.is_finite()
                && band_max_hz.is_finite()
                && band_min_hz > 0.0
                && band_max_hz > band_min_hz
                && relaxation_arms >= 1;
            if !ok {
                multi_error.add(
                    ValidationError::FieldValidation {
                        field: "absorption".to_owned(),
                        value: format!("{:?}", self.absorption),
                        constraint: "requires 0 < band_min < band_max, f_ref > 0, arms >= 1"
                            .to_owned(),
                    }
                    .into(),
                );
            }
        }

        // Validate spatial order. The staggered path derives its Courant limit
        // from the stencil coefficients and so supports 8; the collocated path
        // is bounded by its tabulated limits, which stop at 6.
        let allowed: &[usize] = if self.staggered_grid {
            &[2, 4, 6, 8]
        } else {
            &[2, 4, 6]
        };
        if !allowed.contains(&self.spatial_order) {
            multi_error.add(
                ValidationError::FieldValidation {
                    field: "spatial_order".to_owned(),
                    value: self.spatial_order.to_string(),
                    constraint: if self.staggered_grid {
                        "Must be 2, 4, 6, or 8 on the staggered grid".to_owned()
                    } else {
                        "Must be 2, 4, or 6 on the collocated grid".to_owned()
                    },
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
