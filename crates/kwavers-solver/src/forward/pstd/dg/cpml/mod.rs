//! Convolutional Perfectly Matched Layer (CPML) for the tensor acoustic DG solver.
//!
//! Implements the Roden-Gedney (2000) CPML adapted to a discontinuous-Galerkin
//! first-order acoustic system following Lazarov & Warburton (2009). The
//! formulation introduces per-node auxiliary memory variables `ψ_{q, a}` that
//! satisfy a first-order ODE (see [`memory`]) and modify the per-axis spatial
//! derivative used in the field RHS:
//!
//! ```text
//! D̃_a q = (1/κ_a) D_a q + ψ_{q, a}
//! ```
//!
//! Outside the absorbing layer σ = 0, α = 0, κ = 1; the auxiliary ODE forces
//! ψ ≡ 0 for cold starts, and the modified derivative collapses to the
//! standard DG operator. The CPML branch is therefore physics-preserving
//! everywhere outside the user-specified PML thickness.
//!
//! ## References
//! - Roden & Gedney (2000). *Microwave Opt. Tech. Lett.* 27(5), 334–339.
//! - Lazarov, R., & Warburton, T. (2009). *J. Comput. Phys.* 228, 8262–8281.
//! - Modave, A., St-Cyr, A., & Warburton, T. (2017). *Geophys. J. Int.* 211(2), 1228–1248.

pub mod config;
pub mod memory;
pub mod profiles;

pub use config::{DgCpmlAxis, DgCpmlConfig};
pub use memory::{
    pressure_memory_index, velocity_memory_index, DgCpmlMemoryWorkspace, DG_CPML_MEMORY_VARS,
};
pub use profiles::{DgCpmlAxisProfile, DgCpmlProfiles};
