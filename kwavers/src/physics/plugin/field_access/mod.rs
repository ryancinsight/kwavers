//! Safe field access for plugins
//!
//! This module provides a safe API for plugins to access only the fields
//! they have declared as required or provided, preventing accidental access
//! to unrelated fields and improving encapsulation.

pub mod direct;
pub mod mutable;
pub mod readonly;

pub use direct::DirectPluginFieldAccess;
pub use mutable::PluginFieldAccessMut;
pub use readonly::PluginFieldAccess;

pub type FieldAccessor<'a> = PluginFieldAccess<'a>;

#[cfg(test)]
mod tests;
