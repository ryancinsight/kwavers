//! Compatibility facade for core infrastructure (DEPRECATED)
//!
//! **This module has been moved to `crate::core::`**
//!
//! # Migration Path
//!
//! Old code:
//! ```rust,ignore
//! use crate::domain::core::error::{KwaversError, KwaversResult};
//! use crate::domain::core::constants::WATER_SOUND_SPEED;
//! use crate::domain::core::time::Time;
//! ```
//!
//! New code:
//! ```rust,ignore
//! use crate::core::error::{KwaversError, KwaversResult};
//! use crate::core::constants::WATER_SOUND_SPEED;
//! use crate::core::time::Time;
//! ```
//!
//! # Deprecation Timeline
//!
//! - **v2.15.0**: Deprecation warnings added (this version)
//! - **v3.0.0**: Compatibility facade removed (breaking change)
//!
//! # Rationale
//!
//! Core infrastructure (error types, constants, utilities, logging, time)
//! should not be nested under the domain layer. These are foundational
//! primitives used across all layers (domain, physics, analysis, solver, etc.).
//!
//! The correct architecture follows deep vertical hierarchy principles:
//!
//! ```text
//! src/
//! ├── core/           ← Foundational primitives (moved here)
//! │   ├── error/
//! │   ├── constants/
//! │   ├── utils/
//! │   ├── log/
//! │   └── time/
//! ├── domain/         ← Domain logic only
//! ├── physics/        ← Physics models
//! ├── analysis/       ← Signal processing & analysis
//! ├── solver/         ← Numerical solvers
//! └── simulation/     ← Simulation orchestration
//! ```
//!
//! # Architectural Decision Record
//!
//! See ADR-007: Core Infrastructure Extraction for full rationale.

#![deprecated(
    since = "2.15.0",
    note = "Use `crate::core::` instead - `domain::core` violates layer hierarchy"
)]

// Re-export from canonical location
pub use crate::core::constants;
pub use crate::core::error;
pub use crate::core::log;
pub use crate::core::time;
pub use crate::core::utils;
