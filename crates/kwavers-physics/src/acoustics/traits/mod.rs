//! Fundamental physics traits defining model interfaces.
//!
//! This module decouples concrete implementations from their abstract contracts,
//! ensuring that cross-cutting concerns adhere to the Dependency Inversion Principle.

pub mod cavitation;
pub mod chemical;
pub mod heterogeneity;
pub mod optical;
pub mod scattering;
pub mod streaming;
pub mod thermal;
pub mod wave;

pub use cavitation::CavitationModelBehavior;
pub use chemical::ChemicalModelTrait;
pub use heterogeneity::HeterogeneityModelTrait;
pub use optical::LightDiffusionModelTrait;
pub use scattering::AcousticScatteringModelTrait;
pub use streaming::StreamingModelTrait;
pub use thermal::ThermalModelTrait;
pub use wave::AcousticWaveModel;

/// Base bound for all physics-model traits.
///
/// Every concrete model in the simulation system must be `Debug` (for logging),
/// `Send` (movable across threads), and `Sync` (shareable by reference across
/// threads).  This supertrait is declared once here so the individual domain
/// traits (`ThermalModelTrait`, `LightDiffusionModelTrait`, etc.) do not
/// each repeat the same three bounds.
///
/// See [`kwavers_core::ComponentBase`] for the cross-crate root of this hierarchy.
pub trait PhysicsModel: kwavers_core::ComponentBase {}

/// Blanket implementation: any `Debug + Send + Sync` type satisfies `PhysicsModel`.
impl<T: kwavers_core::ComponentBase> PhysicsModel for T {}
