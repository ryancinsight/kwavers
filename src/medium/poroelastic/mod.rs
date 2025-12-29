//! Poroelastic Tissue Modeling for Biphasic Media
//!
//! This module implements poroelastic wave propagation in fluid-saturated porous media,
//! essential for modeling wave behavior in tissues like liver, kidney, and brain.
//!
//! ## Physical Model
//!
//! Poroelastic media consist of a solid skeleton saturated with fluid. The governing equations
//! are based on Biot's theory of poroelasticity:
//!
//! ```text
//! ∇·σ = ρ ∂²u/∂t² + ρ_f ∂²w/∂t²    (solid momentum)
//! ∇·σ_f = ρ_f ∂²u/∂t² + ρ_f (1/κ) ∂w/∂t + (η/κ) ∂²w/∂t²    (fluid momentum)
//! ∂ε/∂t = ∇·(∂u/∂t) + ∇·(∂w/∂t)    (compatibility)
//! ```
//!
//! Where:
//! - σ, σ_f: solid and fluid stress tensors
//! - u, w: solid and fluid displacements
//! - ρ, ρ_f: solid and fluid densities
//! - κ: permeability
//! - η: viscosity
//!
//! ## Applications
//!
//! - **Liver Elastography**: Modeling fluid flow effects in hepatic tissue
//! - **Kidney Imaging**: Wave propagation in renal parenchyma
//! - **Brain Modeling**: Cerebrospinal fluid dynamics
//! - **Bone Characterization**: Cortical and trabecular bone
//!
//! ## References
//!
//! - Biot, M. A. (1956). "Theory of propagation of elastic waves in a fluid-saturated porous solid"
//! - Coussy, O. (2004). "Poromechanics"
//! - Fellah et al. (2003). "Application of the Biot theory to ultrasound in bone"

pub mod properties;
pub mod solver;
pub mod waves;

pub use properties::{PoroelasticProperties, FluidProperties, SolidProperties};
pub use solver::{PoroelasticSolver, PoroelasticConfig};
pub use waves::{PoroelasticWave, WaveMode, DispersionRelation};








