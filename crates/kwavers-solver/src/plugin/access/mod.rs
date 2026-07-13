//! Field-access utilities for plugins
//!
//! Domain-layer helpers that scope plugin access to fields they have declared
//! via [`Plugin::required_fields`] and [`Plugin::provided_fields`]. These types
//! enforce the read/write capability the plugin manager validates at registration.
//!
//! Variants:
//! - [`DirectPluginFieldAccess`]: operates directly on the simulation `Array4<f64>`
//!   buffer; no physics-state dependency. Lives in `domain` because it depends only
//!   on `domain::field::mapping::UnifiedFieldType`.
//!
//! State-based accessors (`PluginFieldAccess`, `PluginFieldAccessMut`) operate on
//! `kwavers_physics::acoustics::state::PhysicsState` and live alongside that type
//! at `kwavers_physics::acoustics::state::access` to preserve the `domain → physics`
//! dependency direction (DIP).
//!
//! [`Plugin::required_fields`]: super::Plugin::required_fields
//! [`Plugin::provided_fields`]: super::Plugin::provided_fields

pub mod direct;

pub use direct::DirectPluginFieldAccess;
