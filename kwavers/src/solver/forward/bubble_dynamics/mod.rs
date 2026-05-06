//! Bubble dynamics forward solver module.
//!
//! Exposes the [`BubbleDynamicsPlugin`] adapter that bridges the three
//! production bubble-equation implementations
//! ([`BubbleModel::KellerMiksis`], [`BubbleModel::RayleighPlesset`],
//! [`BubbleModel::Gilmore`]) to the unified [`Plugin`] API consumed by
//! [`PhysicsCatalog`].
//!
//! [`Plugin`]: crate::domain::plugin::Plugin
//! [`PhysicsCatalog`]: crate::physics::factory::catalog::PhysicsCatalog
//! [`BubbleModel::KellerMiksis`]: crate::physics::factory::models::BubbleModel::KellerMiksis
//! [`BubbleModel::RayleighPlesset`]: crate::physics::factory::models::BubbleModel::RayleighPlesset
//! [`BubbleModel::Gilmore`]: crate::physics::factory::models::BubbleModel::Gilmore

pub mod plugin;

pub use plugin::{BubbleDynamicsConfig, BubbleDynamicsPlugin};
