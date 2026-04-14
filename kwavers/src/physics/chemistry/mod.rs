//! Chemistry module for sonochemical reactions and radical formation
//!
//! Design principles:
//! - Separation of Concerns: Each sub-module handles a specific aspect
//! - Open/Closed: Easy to add new reaction types without modifying existing code
//! - Interface Segregation: Traits for specific chemical behaviors
//! - Dependency Inversion: Depends on abstractions (traits) not concrete types
//! - Single Responsibility: Each component has one clear purpose
//!
//! ## Not yet implemented
//!
//! - **Master equation kinetics**: Full radical reaction network d[N]/dt = Σ kᵢⱼ[N]ⱼ.
//! - **Radical diffusion**: Smoluchowski equation ∂[R]/∂t = D∇²[R] − 2k[R]² for
//!   recombination in inhomogeneous fields.
//! - **Multi-species radicals**: OH•, H•, HO₂•, O₂⁻•, eₐq⁻ with cross-reaction network.
//! - **Arrhenius temperature dependence**: k(T) = A exp(−Eₐ/RT) for all rate constants.
//! - **pH-dependent speciation**: Acid-base equilibria coupling to radical yields.

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
