use super::policy::NumaAllocPolicy;
use super::topology::NumaTopology;

/// NUMA-aware memory allocator wrapper.
#[derive(Debug, Clone)]
pub struct NumaAllocator {
    policy: NumaAllocPolicy,
    topology: NumaTopology,
}

impl NumaAllocator {
    #[must_use]
    pub fn new() -> Self {
        let topology = NumaTopology::detect();
        Self {
            policy: NumaAllocPolicy::FirstTouch,
            topology,
        }
    }

    #[must_use]
    pub fn with_policy(policy: NumaAllocPolicy) -> Self {
        let topology = NumaTopology::detect();
        Self { policy, topology }
    }

    #[must_use]
    pub fn policy(&self) -> NumaAllocPolicy {
        self.policy
    }

    #[must_use]
    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }
}

impl Default for NumaAllocator {
    fn default() -> Self {
        Self::new()
    }
}
