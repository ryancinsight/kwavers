//! 1D radial radical diffusion solver on a logarithmic grid.
//!
//! ## Mathematical Foundation
//!
//! The Smoluchowski equation for radical species concentration in the liquid
//! surrounding a spherical bubble (Storey & Szeri 2000; Leighton 1994):
//!
//! ```text
//! ∂[Nᵢ]/∂t = Dᵢ · (1/r²) · ∂/∂r(r² ∂[Nᵢ]/∂r) + Σⱼ νᵢⱼ · rⱼ   (r > R_bubble)
//!
//! Boundary condition: [Nᵢ](R) = [Nᵢ]_bubble(t)   (Hertz-Knudsen flux at bubble wall)
//! Far field:          [Nᵢ](∞) → 0
//! ```
//!
//! ## Numerical Scheme
//!
//! **Grid**: Logarithmic radial grid with `n_points` nodes,
//! `r_j = R_bubble · exp(j · Δξ)` where `Δξ = ln(r_max / R_bubble) / (n - 1)`.
//!
//! **Time integration**: Crank-Nicolson (implicit, 2nd-order) for diffusion;
//! explicit Euler for reactions. The operator-split scheme:
//!
//! ```text
//! [Nᵢ]* = [Nᵢ]ⁿ + dt · f_reaction([Nᵢ]ⁿ)    (explicit Euler, reactions)
//! (I − dt/2 · L) [Nᵢ]ⁿ⁺¹ = (I + dt/2 · L) [Nᵢ]*  (Crank-Nicolson, diffusion)
//! ```
//!
//! where `L` is the discrete radial Laplacian operator on the log grid.
//!
//! ## References
//!
//! - Storey BD, Szeri AJ (2000). "Water vapour, sonoluminescence and sonochemistry."
//!   *Proc R Soc Lond A* **456**, 1685-1709. DOI: 10.1098/rspa.2000.0560
//! - Leighton TG (1994). *The Acoustic Bubble*. Academic Press. §4.4.
//! - Christman CL (1987). *Ultrasonics* **25**(1), 31-37.

mod grid;
mod linear;
mod step;
mod types;

pub use types::{DiffusionError, DiffusionStepResult, RadicalDiffusionSolver};

#[cfg(test)]
mod tests;
