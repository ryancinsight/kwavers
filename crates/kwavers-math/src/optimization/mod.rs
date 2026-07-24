//! Numerical optimisation utilities — SSOT: `leto_ops::application::optimization`.
//!
//! Re-exported here as the kwavers vocabulary.
pub mod lbfgs {
    pub use leto_ops::application::optimization::lbfgs::{
        minimize, LbfgsConfig, LbfgsMemory, LbfgsResult,
    };
}

pub use lbfgs::{minimize, LbfgsConfig, LbfgsMemory, LbfgsResult};
