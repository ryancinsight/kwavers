//! Sonoluminescence Orchestrator
//!
//! Integrates bubble dynamics (Keller-Miksis RK4) with multi-mechanism
//! light emission calculations (blackbody, bremsstrahlung, Cherenkov).

pub mod components;
pub mod dynamics;
pub mod emission_calculator;
pub mod field_accessors;

pub use components::EmissionComponents;
pub use dynamics::IntegratedSonoluminescence;
pub use emission_calculator::SonoluminescenceEmission;
