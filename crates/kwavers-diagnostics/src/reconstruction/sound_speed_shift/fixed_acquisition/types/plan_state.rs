//! Fixed-acquisition plan state.

use super::super::super::operator::SoundSpeedShiftOperator;
use super::super::super::solver::SoundSpeedShiftSolverMetrics;
use super::super::super::types::{SoundSpeedShiftConfig, SoundSpeedShiftSample};

/// Cached geometry and operator for repeated speed-shift frames.
///
/// A plan fixes the acquisition geometry, active mask, sampling policy,
/// propagation model, and sensitivity model. Per-frame measured shifts are
/// supplied as a plain slice in the original acquisition row order.
#[derive(Clone, Debug)]
pub struct SoundSpeedShiftPlan {
    pub(in super::super) samples: Vec<SoundSpeedShiftSample>,
    pub(in super::super) operator: SoundSpeedShiftOperator,
    pub(in super::super) metrics: SoundSpeedShiftSolverMetrics,
    pub(in super::super) config: SoundSpeedShiftConfig,
    pub(in super::super) shape: (usize, usize),
}
