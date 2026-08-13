//! Core FDTD solver implementation
//!
//! # Theorem: Yee (1966) Staggered-Grid FDTD for Acoustic Waves
//!
//! The linearized Euler equations for acoustic wave propagation are:
//! ```text
//!   ρ₀ ∂u/∂t = −∇p                       (momentum conservation)
//!   ∂p/∂t    = −ρ₀c₀² ∇·u                (mass conservation + EOS)
//! ```
//!
//! Yee's staggered-grid FDTD discretizes these in a leapfrog (velocity–pressure)
//! update order. Pressure lives at integer time steps `t^n = nΔt`; velocity
//! at half-integer steps `t^{n+½} = (n+½)Δt`:
//!
//! ```text
//!   u^{n+½} = u^{n−½} − (Δt/ρ₀) · ∇p^n             [velocity update]
//!   p^{n+1} = p^n     − ρ₀c₀²Δt · ∇·u^{n+½}        [pressure update]
//! ```
//!
//! Spatial derivatives use centered finite differences on a staggered Cartesian grid:
//! pressure at cell centers `(i, j, k)`, velocity components at half-shifted faces
//! `(i+½, j, k)`, `(i, j+½, k)`, `(i, j, k+½)`.
//!
//! ## Stability: CFL Condition
//!
//! The FDTD scheme is stable only when the Courant-Friedrichs-Lewy (CFL) condition is met:
//! ```text
//!   c₀ · Δt · √(1/Δx² + 1/Δy² + 1/Δz²) ≤ 1
//! ```
//! In 3D with uniform spacing Δx = Δy = Δz:
//! ```text
//!   Δt_max = Δx / (c₀ · √3) ≈ 0.577 · Δx / c₀
//! ```
//! CFL safety factor 0.95 is applied by default.
//!
//! ## Spatial Accuracy
//!
//! | Stencil order | Accuracy | PPW required |
//! |---------------|----------|--------------|
//! | 2nd           | O(Δx²)   | ~10          |
//! | 4th (default) | O(Δx⁴)   | ~5           |
//! | 6th           | O(Δx⁶)   | ~4           |
//! | 8th           | O(Δx⁸)   | ~3           |
//!
//! Orders 2–8 are available on the staggered path through
//! `StaggeredLeapfrogOperator`; the collocated path keeps 2–6, the range its
//! CFL table covers.
//!
//! PPW = points per wavelength at the maximum frequency of interest.
//!
//! ## Boundary Conditions
//!
//! Absorbing boundaries use CPML (Convolutional PML, Roden & Gedney 2000).
//! See `domain/boundary/cpml/` for the recursive-convolution memory update.
//!
//! ## Module layout
//!
//! - `conservative_diff`: skew-symmetric collocated derivative used by the
//!   leapfrog; see its module docs for why the general central difference is
//!   not usable here.
//! - `construction`: `new` constructor — material precomputation, source
//!   scaling, k-space ops, scratch-buffer pre-allocation.
//! - `stepping`: Yee leapfrog `step_forward`, debug-only NaN scans.
//! - `sources`: dynamic pressure / velocity source dispatch and
//!   `add_source_arc` injection-mode classification.
//! - `accessors`: GPU accelerator hookup, CPML enable, CFL helpers,
//!   metrics, sensor data extraction, orchestrated run loop.
//! - `gpu_accelerator`: external GPU-backend trait surface.
//! - `interface`: `solver::interface::Solver` trait bridge.
//!
//! ## References
//! - Yee, K.S. (1966). Numerical solution of initial boundary value problems
//!   involving Maxwell's equations in isotropic media.
//!   IEEE Trans. Antennas Propag. 14(3), 302–307.
//! - Taflove, A. & Hagness, S.C. (2005). Computational Electrodynamics:
//!   The Finite-Difference Time-Domain Method, 3rd ed. Artech House.
//! - Virieux, J. (1986). P-SV wave propagation in heterogeneous media:
//!   Velocity-stress finite-difference method. Geophysics 51(4), 889–901.

mod accessors;
mod conservative_diff;
mod construction;
mod gpu_accelerator;
mod interface;
mod sources;
mod stepping;
#[cfg(test)]
mod tests;

pub(crate) use conservative_diff::ConservativeCentralDifference;
pub use gpu_accelerator::FdtdGpuAccelerator;

use kwavers_boundary::cpml::CPMLBoundary;
use kwavers_grid::Grid;
use kwavers_math::numerics::operators::StaggeredLeapfrogOperator;
use kwavers_source::{Source, SourceInjectionMode};
use leto::Array3;
use std::sync::Arc;

use super::config::FdtdConfig;
use super::kspace_correction::KSpaceFdtdOperators;
use super::metrics::FdtdMetrics;
use super::source_handler::SourceHandler;
use kwavers_receiver::recorder::simple::SensorRecorder;

/// FDTD solver for acoustic wave propagation
///
/// Supports both linear and nonlinear (Westervelt) wave propagation, CPML absorbing
/// boundaries, staggered-grid spatial operators (2nd / 4th / 6th order), and optional
/// AVX-512 SIMD pressure update kernels.
///
/// Westervelt nonlinear term `(β/ρ₀c₀⁴) ∂²p²/∂t²` is enabled via
/// [`FdtdConfig::enable_nonlinear`]. When enabled the solver allocates
/// `p_prev`, `p_prev2`, `nl_scratch`, and `nl_coeff` arrays and
/// calls `apply_westervelt_nonlinear_correction` each step.
///
/// Reference: Westervelt (1963), Hamilton & Blackstock (1998) Ch. 4.
pub struct GenericFdtdSolver<T> {
    /// Configuration
    pub config: FdtdConfig,
    /// Grid reference
    pub(crate) grid: Grid,
    /// Skew-symmetric collocated operator used by the leapfrog when the
    /// staggered grid is off; see `conservative_diff`.
    pub(crate) conservative_operator: ConservativeCentralDifference,
    /// Staggered gradient/divergence pair, of the configured order. Its two
    /// halves are negative adjoints, which is what makes the leapfrog
    /// conserve energy (KW-SOL-081).
    pub(crate) leapfrog_operator: StaggeredLeapfrogOperator,
    /// Performance metrics
    pub(crate) metrics: FdtdMetrics,
    /// C-PML boundary (if enabled)
    pub(crate) cpml_boundary: Option<CPMLBoundary>,
    /// Spatial order enum (validated at construction)
    /// Configured spatial accuracy order (2, 4, 6, or 8 on the staggered
    /// path; 2, 4, or 6 collocated).
    pub(crate) spatial_order: usize,
    pub(crate) gpu_accelerator: Option<Arc<dyn FdtdGpuAccelerator>>,

    // Shared components for source handling and sensor recording
    pub(crate) source_handler: SourceHandler,
    pub(crate) dynamic_sources: Vec<(Arc<dyn Source>, Array3<f64>)>,
    pub(crate) source_injection_modes: Vec<SourceInjectionMode>,
    pub sensor_recorder: SensorRecorder,

    // State
    pub(crate) time_step_index: usize,

    // Wave Fields (p, ux, uy, uz)
    pub fields: kwavers_field::wave::WaveFields,

    // Material Properties (rho0, c0)
    pub materials: kwavers_medium::material_fields::GenericMaterialFields<T>,

    // Precomputed fields
    pub(crate) rho_c_squared: T,

    /// Relaxation absorption state, `None` for a lossless configuration.
    ///
    /// When present its unrelaxed modulus replaces `rho_c_squared` in the
    /// pressure update and its memory fields are advanced each step.
    pub(crate) absorption: Option<super::absorption::RelaxationAbsorption>,

    // Nonlinear Westervelt fields
    pub(crate) p_prev: Option<T>,
    pub(crate) p_prev2: Option<T>,
    pub(crate) nl_scratch: Option<T>,
    pub(crate) nl_coeff: Option<T>,

    pub(crate) kspace_ops: Option<KSpaceFdtdOperators>,

    // Extracted Divergence terms
    pub(crate) dvx_scratch: T,
    pub(crate) dvy_scratch: T,
    pub(crate) divergence_scratch: T,
}

pub type FdtdSolver = GenericFdtdSolver<Array3<f64>>;

impl<T: std::fmt::Debug> std::fmt::Debug for GenericFdtdSolver<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GenericFdtdSolver")
            .field("config", &self.config)
            .field("grid", &self.grid)
            .field("conservative_operator", &self.conservative_operator)
            .field("leapfrog_operator", &self.leapfrog_operator)
            .field("metrics", &self.metrics)
            .field("cpml_boundary", &self.cpml_boundary)
            .field("spatial_order", &self.spatial_order)
            .field(
                "gpu_accelerator",
                &self.gpu_accelerator.as_ref().map(|_| "GpuAccelerator"),
            )
            .field("source_handler", &self.source_handler)
            .field("dynamic_sources_count", &(self.dynamic_sources.len()))
            .field("sensor_recorder", &self.sensor_recorder)
            .field("time_step_index", &self.time_step_index)
            .field("fields", &self.fields)
            .field("materials", &self.materials)
            .finish()
    }
}
