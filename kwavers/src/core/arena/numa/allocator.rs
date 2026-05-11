use super::policy::NumaPolicy;
use super::topology::NumaTopology;

/// NUMA-aware memory allocator wrapper.
#[derive(Debug, Clone)]
pub struct NumaAllocator {
    policy: NumaPolicy,
    topology: NumaTopology,
}

impl NumaAllocator {
    #[must_use] 
    pub fn new() -> Self {
        let topology = NumaTopology::detect();
        Self {
            policy: NumaPolicy::FirstTouch,
            topology,
        }
    }

    #[must_use] 
    pub fn with_policy(policy: NumaPolicy) -> Self {
        let topology = NumaTopology::detect();
        Self { policy, topology }
    }

    #[must_use] 
    pub fn policy(&self) -> NumaPolicy {
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
