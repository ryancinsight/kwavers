// Canonical photoacoustic physics layer.

mod confinement;
mod governing_equations;
mod references;
pub mod thermoelasticity;
mod validity;

pub use confinement::ConfinementAssessment;
pub use governing_equations::PhotoacousticGoverningEquations;
pub use references::PHOTOACOUSTIC_PHYSICS_REFERENCES;
pub use thermoelasticity::{GrueneisenModel, ThermoelasticReport};
pub use validity::PhotoacousticValidityReport;
