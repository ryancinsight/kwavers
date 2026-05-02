//! BEM boundary conditions for boundary element methods.
//!
//! The boundary manager applies Dirichlet, Neumann, Robin, and Sommerfeld
//! radiation conditions to the boundary integral formulation:
//!
//! ```text
//! c(p) + ∫_Γ G·∂p/∂n dΓ = ∫_Γ p·∂G/∂n dΓ
//! ```

mod manager;
mod types;

pub use manager::BemBoundaryManager;
pub use types::BemBoundaryCondition;

#[cfg(test)]
mod tests;
