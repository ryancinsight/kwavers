//! BEM Solver — Boundary Element Method for Acoustic Scattering
//!
//! ## Boundary Integral Equation (Helmholtz)
//!
//! For the exterior acoustic problem at wavenumber k = ω/c, the Kirchhoff-Helmholtz
//! integral representation of the scattered pressure p at any point r outside Γ is:
//! ```text
//!   c(r) p(r) + ∫_Γ ∂G(r,r')/∂n(r') p(r') dΓ = ∫_Γ G(r,r') ∂p/∂n(r') dΓ
//! ```
//! where G(r,r') = exp(ik|r−r'|) / (4π|r−r'|) is the 3D free-space Helmholtz
//! Green's function, c(r) = 0.5 on smooth boundary, 1 in the exterior.
//!
//! ## Burton-Miller CFIE (spurious resonance suppression)
//!
//! The standard BIE has spurious interior resonances at exterior eigenvalues.
//! The Burton–Miller combined field integral equation (CFIE):
//! ```text
//!   (H + α·(0.5I + H')) p = (G + α·G') q   (q = ∂p/∂n)
//! ```
//! with α = i/k eliminates all interior eigenvalues (Amini 1990).
//!
//! ## References
//!
//! - Burton AJ, Miller GF (1971). Proc. R. Soc. Lond. A 323:201–210.
//! - Amini S (1990). Int. J. Numer. Methods Eng. 29(7):1457–1469.
//! - Colton D, Kress R (1998). *Inverse Acoustic and Electromagnetic Scattering Theory*. Springer.
//! - Wu TW (2000). *Boundary Element Acoustics*. WIT Press.

mod assembly;
mod construction;
mod solution;
#[cfg(test)]
mod tests;

use crate::domain::boundary::BemBoundaryManager;
use crate::math::linear_algebra::sparse::CompressedSparseRowMatrix;
use num_complex::Complex64;
use std::collections::HashMap;

/// Configuration for BEM solver
#[derive(Debug, Clone)]
pub struct BemConfig {
    /// Wavenumber for Helmholtz equation
    pub wavenumber: f64,
    /// Speed of sound (m/s), used to derive wavenumber from frequency
    pub sound_speed: f64,
    /// Excitation frequency (Hz)
    pub frequency: f64,
    /// Burton–Miller coupling parameter α for CFIE
    pub coupling_alpha: Complex64,
    /// Tolerance for iterative solvers
    pub tolerance: f64,
    /// Maximum iterations for iterative solvers
    pub max_iterations: usize,
    /// Use direct solver (dense matrix) instead of iterative
    pub use_direct_solver: bool,
}

impl Default for BemConfig {
    fn default() -> Self {
        use crate::core::constants::fundamental::SOUND_SPEED_TISSUE;
        use crate::core::constants::numerical::MHZ_TO_HZ;
        Self {
            wavenumber: 1.0,
            sound_speed: SOUND_SPEED_TISSUE,
            frequency: MHZ_TO_HZ,
            coupling_alpha: Complex64::new(0.0, 1.0),
            tolerance: 1e-8,
            max_iterations: 1000,
            use_direct_solver: false,
        }
    }
}

/// BEM solver for acoustic boundary element problems
#[derive(Debug)]
pub struct BemSolver {
    /// Solver configuration (public so coupled solvers can update wavenumber/frequency)
    pub config: BemConfig,
    /// Boundary mesh vertices
    pub vertices: Vec<[f64; 3]>,
    /// Boundary triangles (vertex index triples, CCW outward winding)
    pub triangles: Vec<[usize; 3]>,
    /// Map from global mesh node index to local BEM node index
    pub(self) global_to_local_node: HashMap<usize, usize>,
    /// Boundary condition manager
    pub(self) boundary_manager: BemBoundaryManager,
    /// BEM system matrices (lazy-assembled, invalidated on wavenumber change)
    pub(self) h_matrix: Option<CompressedSparseRowMatrix<Complex64>>,
    pub(self) g_matrix: Option<CompressedSparseRowMatrix<Complex64>>,
}
