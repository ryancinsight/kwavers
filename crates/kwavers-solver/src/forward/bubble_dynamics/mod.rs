//! Bubble dynamics forward solver module.
//!
//! Exposes the [`BubbleDynamicsPlugin`] adapter that bridges the three
//! production bubble-equation implementations
//! ([`BubbleModel::KellerMiksis`], [`BubbleModel::RayleighPlesset`],
//! [`BubbleModel::Gilmore`]) to the unified [`Plugin`] API consumed by
//! [`PhysicsCatalog`].
//!
//! [`Plugin`]: kwavers_domain::plugin::Plugin
//! [`PhysicsCatalog`]: crate::plugin::catalog::PhysicsCatalog
//! [`BubbleModel::KellerMiksis`]: kwavers_physics::factory::models::BubbleModel::KellerMiksis
//! [`BubbleModel::RayleighPlesset`]: kwavers_physics::factory::models::BubbleModel::RayleighPlesset
//! [`BubbleModel::Gilmore`]: kwavers_physics::factory::models::BubbleModel::Gilmore

pub mod plugin;

pub use plugin::{BubbleDynamicsConfig, BubbleDynamicsPlugin};
