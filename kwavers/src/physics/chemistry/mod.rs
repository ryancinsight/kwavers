//! Chemistry module for sonochemical reactions and radical formation
//!
//! Design principles:
//! - Separation of Concerns: Each sub-module handles a specific aspect
//! - Open/Closed: Easy to add new reaction types without modifying existing code
//! - Interface Segregation: Traits for specific chemical behaviors
//! - Dependency Inversion: Depends on abstractions (traits) not concrete types
//! - Single Responsibility: Each component has one clear purpose
//!   TODO_AUDIT: P2 - Sonochemistry Coupling - Add comprehensive sonochemistry module with complete reaction kinetics and free radical production tracking, expanding beyond current simplified models
//!   DEPENDS ON: physics/chemistry/kinetics/master_equation.rs, physics/chemistry/radicals/diffusion.rs
//!   MISSING: Master equation for radical kinetics: d[N]/dt = ∑ kᵢⱼ [N]ⱼ with full reaction network
//!   MISSING: Smoluchowski diffusion equation for radical recombination: ∂[R]/∂t = D∇²[R] - 2k[R]²
//!   MISSING: Multiple radical species: OH•, H•, HO₂•, O₂⁻•, eₐq⁻ with cross-reactions
//!   MISSING: Temperature-dependent reaction rates: k(T) = A exp(-Eₐ/RT) with Arrhenius kinetics
//!   MISSING: pH-dependent speciation and acid-base equilibria affecting radical yields

// Sub-modules
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
