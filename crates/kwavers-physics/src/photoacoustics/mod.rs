//! Canonical photoacoustic physics layer.

#[cfg(feature = "clinical-imaging")]
mod confinement;
#[cfg(feature = "clinical-imaging")]
mod governing_equations;
mod grueneisen;
mod quantitative;
mod references;
#[cfg(feature = "clinical-imaging")]
mod thermoelasticity;
#[cfg(feature = "clinical-imaging")]
mod validity;

#[cfg(feature = "clinical-imaging")]
pub use confinement::ConfinementAssessment;
#[cfg(feature = "clinical-imaging")]
pub use governing_equations::PhotoacousticGoverningEquations;
pub use grueneisen::GrueneisenModel;
pub use quantitative::{apparent_absorption, compensate_fluence, QuantitativePhotoacousticError};
pub use references::PHOTOACOUSTIC_PHYSICS_REFERENCES;
#[cfg(feature = "clinical-imaging")]
pub use thermoelasticity::ThermoelasticReport;
#[cfg(feature = "clinical-imaging")]
pub use validity::PhotoacousticValidityReport;
