//! Idiomatic re-export surface for downstream `kwavers-*` consumers.
//!
//! # SSOT role
//!
//! This module is the **single idiomatic entry point** [`kwavers_driver::prelude`] that
//! downstream crates can `use` to bring the entire public surface into scope. Phase 0 only
//! carries the SSOT marker; Phase 1+ will populate it with the precise re-exports that the
//! `src/lib.rs::pub use` block already mirrors. Keeping the prelude in lock-step with
//! `lib.rs` enforces SRP (the re-export decision lives in exactly one place per alias) and
//! SOC (downstream consumers depend on a single, small surface, not the internal hierarchy).
//!
//! # Plan
//!
//! Phase 1 will populate this file by **copying the `pub use {...}` blocks from
//! `src/lib.rs` verbatim**, turning it into:
//!
//! ```ignore
//! pub use crate::{
//!     acoustic::{...},
//!     ampacity::{...},
//!     // ...full SSOT re-export tree from lib.rs...
//! };
//! ```
//!
//! Until then the prelude is a stub. The empty namespace does not yet declare itself in
//! `src/lib.rs::pub mod prelude;` — that lands in Phase 1 once we cut over `lib.rs` to use
//! the prelude as its source of truth.
//!
//! See `docs/ARCHITECTURE.md` for the planned SSOT/prelude interaction.
