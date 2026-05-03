//! WENO-based shock limiting for spectral DG methods.
//!
//! ## Algorithm: WENO3 (third-order Weighted Essentially Non-Oscillatory)
//!
//! Given a 5-point stencil `{v_{i-2}, v_{i-1}, v_i, v_{i+1}, v_{i+2}}`, three
//! candidate reconstructions are formed from sub-stencils:
//! ```text
//!   q₀ = v_{i-2}/3  − 7v_{i-1}/6 + 11v_i/6
//!   q₁ = −v_{i-1}/6 + 5v_i/6     +  v_{i+1}/3
//!   q₂ =  v_i/3     + 5v_{i+1}/6 −  v_{i+2}/6
//! ```
//! Jiang-Shu smoothness indicators (J. Comput. Phys. 126:202-228, 1996).
//!
//! ## Algorithm: WENO7 (seventh-order WENO)
//!
//! Four candidate stencils from a 9-point window; optimal weights
//! `d = [0.05, 0.45, 0.45, 0.05]` recover the 7th-order central scheme in smooth
//! regions (Balsara & Shu 2000, J. Comput. Phys. 160(2):405-452).
//!
//! ## References
//!
//! - Liu, Osher & Chan (1994). J. Comput. Phys. 115(1):200-212.
//! - Jiang & Shu (1996). J. Comput. Phys. 126(1):202-228.
//! - Balsara & Shu (2000). J. Comput. Phys. 160(2):405-452.

mod types;
mod weno3;
mod weno5;
mod weno7;

pub use types::WENOLimiter;
