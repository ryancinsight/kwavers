use super::policy::NumaPolicy;
use super::topology::NumaTopology;

/// NUMA-aware memory allocator wrapper.
#[derive(Debug, Clone)]
pub struct NumaAllocator {
    policy: NumaPolicy,
    topology: NumaTopology,
}

impl NumaAllocator {
    pub fn new() -> Self {
        let topology = NumaTopology::detect();
        Self {
            policy: NumaPolicy::FirstTouch,
            topology,
        }
    }

    pub fn with_policy(policy: NumaPolicy) -> Self {
        let topology = NumaTopology::detect();
        Self { policy, topology }
    }

    pub fn policy(&self) -> NumaPolicy {
        self.policy
    }

    pub fn topology(&self) -> &NumaTopology {
        &self.topology
    }
}

impl Default for NumaAllocator {
    fn default() -> Self {
        Self::new()
    }
}
