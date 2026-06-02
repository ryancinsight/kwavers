//! State-based field accessors for plugins
//!
//! Plugin access wrappers over [`super::PhysicsState`] that enforce the
//! read/write capability declared by the plugin's `required_fields` and
//! `provided_fields`. These accessors live alongside `PhysicsState` (rather
//! than under `domain::plugin::access`) because they depend on it; co-locating
//! them preserves the `domain → physics` dependency direction.
//!
//! For an accessor that operates directly on the simulation `Array4<f64>`
//! buffer without a `PhysicsState`, see [`kwavers_domain::plugin::access::DirectPluginFieldAccess`].

pub mod mutable;
pub mod readonly;

pub use mutable::PluginFieldAccessMut;
pub use readonly::PluginFieldAccess;

#[cfg(test)]
mod tests;
