#![doc = include_str!("../README.md")]

pub mod arena;
pub mod constants;
pub mod error;
pub mod log;
pub mod path;
/// Rejection assertions for the workspace's test modules.
///
/// Gated behind `test-util` so the helpers never enter a default build.
#[cfg(feature = "test-util")]
pub mod test_support;
pub mod time;
pub mod traversal;
pub mod units;
pub mod utils;
pub mod validation;

/// Base bound for all kwavers component types.
///
/// Any simulation component — physics model, solver, transducer, boundary,
/// or signal primitive — must be thread-safe and diagnosable.  This trait
/// (with its blanket implementation) is the lowest-level common base for the
/// domain-specific supertrait families:
///
/// - `kwavers_physics::acoustics::traits::PhysicsModel` — physics models
/// - `kwavers_signal::SignalPrimitive` — signal components
/// - `kwavers_solver::interface::factory::SolverComponent` — solver infrastructure
///
/// Use the domain-specific supertrait where possible; fall back to
/// `ComponentBase` only when a trait spans multiple domains or lives in a
/// crate that does not depend on those higher-level crates.
pub trait ComponentBase: std::fmt::Debug + Send + Sync {}

/// Blanket implementation: every `Debug + Send + Sync` type is a `ComponentBase`.
impl<T: std::fmt::Debug + Send + Sync> ComponentBase for T {}
