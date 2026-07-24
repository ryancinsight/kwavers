//! Vector norms - SSOT: leto_ops application linalg norms.
//!
//! Re-exported here as the kwavers vocabulary so higher layers depend on one
//! import path while the implementation lives in leto-ops.

pub use leto_ops::{
    norm, norm_l1, norm_l2, norm_max, MatrixNorm, NormKind, NormL1, NormL2, NormMax,
};

/// Back-compat marker for older kwavers imports.
///
/// Use the free norm functions re-exported from this module instead.
pub trait VectorOperations {}

impl<T> VectorOperations for T {}
