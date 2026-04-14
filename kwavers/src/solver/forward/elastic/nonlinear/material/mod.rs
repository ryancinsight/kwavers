//! Hyperelastic material models for nonlinear elasticity
//!
//! This module implements constitutive models for finite deformation elasticity,
//! including Neo-Hookean, Mooney-Rivlin, and Ogden hyperelastic models.
//!
//! ## Theoretical Foundation
//!
//! Hyperelastic materials are characterized by strain energy density functions W(I₁,I₂,J)
//! where I₁ and I₂ are strain invariants and J is the volume ratio.
//!
//! ### Strain Invariants
//!
//! Given the right Cauchy-Green tensor C = F^T·F:
//! - I₁ = tr(C) = λ₁² + λ₂² + λ₃²
//! - I₂ = ½[(tr C)² - tr(C²)] = λ₁²λ₂² + λ₂²λ₃² + λ₃²λ₁²
//! - I₃ = det(C) = J² = (λ₁λ₂λ₃)²
//!
//! where λᵢ are the principal stretches.
//!
//! ## Literature References
//!
//! - Holzapfel, G. A. (2000). "Nonlinear Solid Mechanics", Wiley.
//! - Mooney, M. (1940). "A theory of large elastic deformation", J. Appl. Phys.
//! - Rivlin, R. S. (1948). "Large elastic deformations of isotropic materials", Phil. Trans.
//! - Ogden, R. W. (1972). "Large deformation isotropic elasticity", Proc. Roy. Soc.

pub mod energy;
pub mod invariants;
pub mod models;
pub mod stress;
#[cfg(test)]
mod tests;

pub use models::HyperelasticModel;
