//! Chemistry module for sonochemical reactions and radical formation
//!
//! Design principles:
//! - Separation of Concerns: Each sub-module handles a specific aspect
//! - Open/Closed: Easy to add new reaction types without modifying existing code
//! - Interface Segregation: Traits for specific chemical behaviors
//! - Dependency Inversion: Depends on abstractions (traits) not concrete types
//! - Single Responsibility: Each component has one clear purpose
//!
//! ## Implemented (Sprint 226)
//!
//! - **Master equation kinetics**: Dormand-Prince RK45 adaptive integrator in `integrator/`
//!   evolves d[N]/dt = Σ νᵢⱼ·rⱼ(N,T,pH) forward in time with error control.
//! - **Radical diffusion**: Smoluchowski radial solver in `diffusion/` — Crank-Nicolson
//!   implicit diffusion on a logarithmic grid (64 points, 10–1000 R_bubble), operator-split
//!   with RK45 reactions.
//! - **Multi-species radicals**: OH•, H•, HO₂•, O₂⁻•, H₂O₂ with cross-reaction network
//!   (Riesz & Leighton 2012; Christman 1987).
//! - **Arrhenius temperature dependence**: k(T) = A·exp(−Eₐ/RT) in `RadicalKinetics`.
//! - **pH-dependent speciation**: pH factor applied to applicable reaction rates.

// Sub-modules
pub mod diffusion;
pub mod integrator;
pub mod model;
pub mod model_query;
pub mod model_update;
pub mod parameters;
pub mod photochemistry;
pub mod radical_initiation;
pub mod reaction_kinetics;
pub mod reactions;
pub mod ros_plasma;
pub mod trait_impls;
pub mod validation;

#[cfg(test)]
mod tests;

// Re-export commonly used types
pub use model::{ChemicalModel, ChemicalModelState};
pub use parameters::{ChemicalMetrics, ChemicalUpdateParams};
pub use reactions::{
    ChemicalReaction, ChemicalReactionConfig, LightDependence, PressureDependence, ReactionRate,
    ReactionType, Species, ThermalDependence,
};
pub use ros_plasma::{ROSConcentrations, ROSSpecies, SonochemicalYield, SonochemistryModel};
pub use validation::{ArrheniusValidator, LiteratureValue, ValidatedKinetics, ValidationResult};
