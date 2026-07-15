//! Heterogeneous skull model with spatially varying properties
//!
//! ## Theorem: Bone Volume Fraction (BVF) — CT Mapping
//!
//! For a porous bone-water composite, the bone volume fraction (BVF) at a voxel
//! is linearly proportional to the Hounsfield unit (Marquet et al. 2009):
//! ```text
//!   φ(x) = clamp((HU(x) − HU_water) / (HU_cortical − HU_water), 0, 1)
//! ```
//!
//! ## Hill Averaging
//!
//! The effective bulk modulus is the Hill average of the Voigt and Reuss bounds:
//! ```text
//!   K_Hill = (K_V + K_R) / 2
//!   c_eff = sqrt(K_Hill / ρ_eff)
//! ```
//!
//! ## References
//! - Marquet F et al. (2009). Phys. Med. Biol. 54(9), 2597–2613.
//! - Aubry J-F et al. (2003). J. Acoust. Soc. Am. 113(1), 84–93.
//! - Clement GT, Hynynen K (2002). Phys. Med. Biol. 47(8), 1219–1236.
//! - Hill R (1952). Proc. Phys. Soc. A 65(5), 349–354.

mod constants;
mod ct;
mod mask;
mod model;
mod properties;
mod types;

#[cfg(test)]
mod tests;

pub use constants::{ALPHA_WATER, HU_CORTICAL, HU_WATER};
pub use model::HeterogeneousSkull;
pub use types::SkullLayer;
