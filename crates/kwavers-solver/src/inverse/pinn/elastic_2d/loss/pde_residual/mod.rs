//! PDE residual computation using automatic differentiation.
//!
//! Implements the complete autodiff pipeline for the 2D elastic wave equation:
//!
//! ```text
//! ρ ∂²u/∂t² = ∇·σ    where σ = λ tr(ε) I + 2μ ε,  ε = ∇_s u
//! ```
//!
//! ## Computational pipeline
//!
//! 1. `gradients` — displacement → ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y
//! 2. `strain_stress` — gradients → ε_xx, ε_yy, ε_xy → σ_xx, σ_yy, σ_xy
//! 3. `divergence` — stresses → ∂σ_xx/∂x + ∂σ_xy/∂y, ∂σ_xy/∂x + ∂σ_yy/∂y
//! 4. `time` — u → ∂u/∂t, ∂²u/∂t²
//! 5. `pipeline` — full residual R = ρ a − ∇·σ

pub mod divergence;
pub mod gradients;
pub mod pipeline;
pub mod strain_stress;
pub mod time;

#[cfg(test)]
mod tests;

#[cfg(feature = "pinn")]
pub use divergence::compute_stress_divergence;
#[cfg(feature = "pinn")]
pub use gradients::compute_displacement_gradients;
#[cfg(feature = "pinn")]
pub use pipeline::{compute_elastic_wave_pde_residual, displacement_to_stress_divergence};
#[cfg(feature = "pinn")]
pub use strain_stress::{compute_strain_from_gradients, compute_stress_from_strain};
#[cfg(feature = "pinn")]
pub use time::compute_time_derivatives;
