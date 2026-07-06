//! # Kwavers — top-level application / integration crate
//!
//! `kwavers` is the thin top crate of the workspace. It does **not** re-export the
//! layer crates — the former facade was removed (ADR 011 amendment). Depend on the
//! layer crates directly:
//!
//! - `kwavers_core` — error types, logging, time, arena allocator
//! - `kwavers_math` — FFT, geometry, linear algebra, SIMD, numerics, tensor
//! - `kwavers_grid` / `kwavers_field` / `kwavers_signal` — discretization, field indices, signals
//! - `kwavers_medium` / `kwavers_phantom` / `kwavers_optics` — materials, phantoms, optical data
//! - `kwavers_source` / `kwavers_receiver` / `kwavers_transducer` — excitation, recording, devices
//! - `kwavers_boundary` / `kwavers_mesh` / `kwavers_imaging` — boundaries, meshes, medical imaging
//! - `kwavers_physics` — acoustics, optics, thermal, chemistry, electromagnetic, therapy
//! - `kwavers_solver` — forward (FDTD/PSTD/elastic), inverse (FWI/PINN), analytical
//! - `kwavers_simulation` — orchestration, backends, modalities, result I/O (`io`)
//! - `kwavers_analysis` — signal processing, beamforming, validation, ML, performance
//! - `kwavers_diagnostics` / `kwavers_therapy` — clinical diagnostic imaging / therapy
//! - `kwavers_gpu` (feature `"gpu"`) - Hephaestus-backed provider-generic GPU backend
//!
//! This crate carries only the binary (`main.rs`), the cross-cutting integration
//! tests / examples / benches, and the small application utilities below.

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

pub mod theranostic;

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
