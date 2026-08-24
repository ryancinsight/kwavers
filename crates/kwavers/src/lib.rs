#![doc = include_str!("../README.md")]
// Strict warning configuration for code quality
#![warn(
    unused_imports,
    unused_mut,
    unreachable_code,
    unreachable_patterns,
    unused_must_use,
    unused_unsafe,
    path_statements,
    unused_attributes,
    unused_macros
)]
#![warn(missing_debug_implementations)]
#![warn(trivial_casts, trivial_numeric_casts)]
#![warn(unsafe_code)]
#![allow(
    clippy::type_complexity,
    clippy::assertions_on_constants,
    clippy::field_reassign_with_default
)]
#![allow(unexpected_cfgs)]

use std::collections::HashMap;

mod parallel;
pub mod theranostic;

/// Selectable visualization transfer providers.
#[cfg(feature = "gpu-visualization")]
pub use kwavers_gpu::visualization;

/// Initialize logging for the kwavers application.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn init_logging() -> kwavers_core::error::KwaversResult<()> {
    env_logger::init();
    Ok(())
}

/// Get application version and build information.
#[must_use]
pub fn get_version_info() -> HashMap<String, String> {
    let mut info = HashMap::new();
    info.insert("version".to_owned(), env!("CARGO_PKG_VERSION").to_owned());
    info.insert("name".to_owned(), env!("CARGO_PKG_NAME").to_owned());
    info.insert(
        "description".to_owned(),
        env!("CARGO_PKG_DESCRIPTION").to_owned(),
    );
    info.insert("authors".to_owned(), env!("CARGO_PKG_AUTHORS").to_owned());
    info.insert(
        "repository".to_owned(),
        env!("CARGO_PKG_REPOSITORY").to_owned(),
    );
    info.insert("license".to_owned(), env!("CARGO_PKG_LICENSE").to_owned());
    info
}

#[cfg(test)]
mod tests;
