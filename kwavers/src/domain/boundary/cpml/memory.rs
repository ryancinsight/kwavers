//! CPML memory variables for recursive convolution
//!
//! # Background: Why Auxiliary Memory Variables Are Required
//!
//! Standard PML absorbs outgoing waves by multiplying gradients with an exponential
//! attenuation factor. For oblique incidence and evanescent waves, this causes
//! reflection errors due to the frequency-independent damping profile.
//!
//! CPML (Roden & Gedney 2000) adds an auxiliary (memory) field ψ per spatial
//! direction per field variable that accumulates the gradient history:
//! ```text
//!   ψ^{n+1} = b · ψ^n + a · ∇f^n,   b = exp(−σΔt),  a = b − 1
//! ```
//! The effective gradient used in the field update is then:
//! ```text
//!   (∇f)_eff = (∇f)/κ + ψ
//! ```
//! This frequency-dependent stretching eliminates the late-time reflection artefacts
//! present in basic split-field PML and enables correct absorption to machine precision
//! for grazing-incidence and evanescent modes.
//!
//! # Memory Layout
//!
//! ## Lemma: Minimal-footprint PML arrays
//!
//! The ψ fields are only non-zero within the PML region (σ = 0 outside).
//! Storing full-grid arrays wastes `(nx·ny·nz − 2·t·ny·nz)` elements per component.
//! Instead each memory array covers only the two PML strips:
//!
//! ```text
//!   psi_p_x: shape (2·tx, ny, nz)
//!     psi_p_x[0..tx,   j, k] ↔ left  PML strip (i = 0..tx)
//!     psi_p_x[tx..2tx, j, k] ↔ right PML strip (i = nx−tx..nx)
//!
//!   psi_p_y: shape (nx, 2·ty, nz)   — analogous for j
//!   psi_p_z: shape (nx, ny, 2·tz)   — analogous for k
//! ```
//!
//! The `psi_v_*` arrays hold memory for velocity-gradient updates and use the same layout.
//!
//! ## References
//! - Roden, J.A. & Gedney, S.D. (2000). Microwave Opt. Tech. Lett. 27(5), 334–339.
//! - Collino, F. & Tsogka, C. (2001). Geophysics 66(1), 294–307.

use super::config::CPMLConfig;
use crate::domain::grid::Grid;
use ndarray::Array3;

/// CPML memory variables for field updates
///
/// Each `psi_p_*` and `psi_v_*` field stores only the PML-strip cells
/// (2 × thickness cells per primary axis) to minimise memory allocation.
/// See module-level documentation for the index-mapping convention.
#[derive(Debug, Clone)]
pub struct CPMLMemory {
    // Memory for velocity updates (depends on pressure gradient)
    pub psi_p_x: Array3<f64>,
    pub psi_p_y: Array3<f64>,
    pub psi_p_z: Array3<f64>,

    // Memory for pressure updates (depends on velocity divergence)
    pub psi_v_x: Array3<f64>,
    pub psi_v_y: Array3<f64>,
    pub psi_v_z: Array3<f64>,
}

impl CPMLMemory {
    /// Create new memory variables with per-dimension PML thickness
    pub fn new(config: &CPMLConfig, grid: &Grid) -> Self {
        let tx = config.per_dimension.x;
        let ty = config.per_dimension.y;
        let tz = config.per_dimension.z;

        Self {
            psi_p_x: Array3::zeros((2 * tx, grid.ny, grid.nz)),
            psi_p_y: Array3::zeros((grid.nx, 2 * ty, grid.nz)),
            psi_p_z: Array3::zeros((grid.nx, grid.ny, 2 * tz)),

            psi_v_x: Array3::zeros((2 * tx, grid.ny, grid.nz)),
            psi_v_y: Array3::zeros((grid.nx, 2 * ty, grid.nz)),
            psi_v_z: Array3::zeros((grid.nx, grid.ny, 2 * tz)),
        }
    }

    /// Reset all memory variables to zero
    pub fn reset(&mut self) {
        self.psi_p_x.fill(0.0);
        self.psi_p_y.fill(0.0);
        self.psi_p_z.fill(0.0);
        self.psi_v_x.fill(0.0);
        self.psi_v_y.fill(0.0);
        self.psi_v_z.fill(0.0);
    }
}
