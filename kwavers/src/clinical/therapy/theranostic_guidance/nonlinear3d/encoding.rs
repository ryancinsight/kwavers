//! Deterministic source encoding for nonlinear 3-D FWI.
//!
//! # Theorem
//!
//! Let `A_j(m)` be the nonlinear observation operator for source element `j`.
//! A coded shot with weights `w_kj` observes `sum_j w_kj A_j(m)` through one
//! simultaneous transmission. If the code matrix rows are linearly independent,
//! the stacked coded residual preserves independent source sensitivity in the
//! least-squares normal equations through `W^T W`.
//!
//! # Proof sketch
//!
//! Linearizing the nonlinear map at `m` gives coded Jacobian `J_k = W_k J`.
//! The stacked Gauss-Newton Hessian is `J^T W^T W J`. Positive diagonal entries
//! of `W^T W` retain each element's sensitivity; additional off-diagonal terms
//! are deterministic cross-talk rather than lost information.

use crate::core::constants::numerical::TWO_PI;
#[derive(Clone, Copy, Debug)]
pub(super) struct SourceEncoding {
    pub index: usize,
    pub count: usize,
}

#[derive(Clone, Debug)]
pub(super) struct EncodedTrace {
    pub encoding: SourceEncoding,
    pub traces: Vec<f64>,
}

impl SourceEncoding {
    pub(super) fn all(count: usize) -> Vec<Self> {
        (0..count).map(|index| Self { index, count }).collect()
    }

    pub(super) fn source_weight(self, source_index: usize, source_count: usize) -> f64 {
        if self.count == 1 || self.index == 0 {
            return 1.0;
        }
        let angle =
            TWO_PI * (self.index as f64) * (source_index as f64 + 0.5) / source_count.max(1) as f64;
        if angle.sin() >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }
}
