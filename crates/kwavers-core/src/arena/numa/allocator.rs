use themis::{CpuTopology, PlacementHint};

use crate::arena::numa::detect_topology;

/// NUMA-aware memory allocator wrapper.
///
/// Holds a placement preference ([`PlacementHint`]) and a topology snapshot
/// ([`CpuTopology`]). Actual allocation and binding is performed by the
/// execution primitives in [`super::memory`]; themis owns the placement
/// vocabulary only.
#[derive(Debug, Clone)]
pub struct NumaAllocator {
    hint: PlacementHint,
    topology: CpuTopology,
}

impl NumaAllocator {
    #[must_use]
    pub fn new() -> Self {
        let topology = detect_topology();
        Self {
            hint: PlacementHint::Current,
            topology,
        }
    }

    #[must_use]
    pub fn with_hint(hint: PlacementHint) -> Self {
        let topology = detect_topology();
        Self { hint, topology }
    }

    #[must_use]
    pub fn hint(&self) -> PlacementHint {
        self.hint
    }

    #[must_use]
    pub fn topology(&self) -> &CpuTopology {
        &self.topology
    }
}

impl Default for NumaAllocator {
    fn default() -> Self {
        Self::new()
    }
}
