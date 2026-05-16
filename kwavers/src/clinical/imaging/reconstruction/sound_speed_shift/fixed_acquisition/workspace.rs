//! Workspace lifecycle for fixed-acquisition frame solves.

use std::mem::size_of;

use super::types::SoundSpeedShiftPlanWorkspace;

impl SoundSpeedShiftPlanWorkspace {
    /// Construct an empty fixed-plan workspace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Retained sampled-RHS capacity in selected measurement rows.
    #[must_use]
    pub fn sampled_rhs_capacity(&self) -> usize {
        self.sampled_rhs.capacity()
    }

    /// Total retained `f64` capacity across sampled RHS and solver buffers.
    #[must_use]
    pub fn allocated_slots(&self) -> usize {
        self.sampled_rhs.capacity() + self.solver.allocated_slots()
    }

    /// Total retained workspace memory in bytes.
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.allocated_slots() * size_of::<f64>()
    }

    /// Zero active buffers while preserving allocations.
    pub fn clear(&mut self) {
        self.sampled_rhs.fill(0.0);
        self.solver.clear();
    }
}
