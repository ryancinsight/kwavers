use themis::{CpuTopology, NumaNodeId};

use crate::arena::numa::detect_topology;
use crate::error::{KwaversError, KwaversResult};

/// Thread affinity configuration.
#[derive(Debug, Clone)]
pub struct ThreadAffinity {
    pub node: Option<NumaNodeId>,
    pub cpus: Option<Vec<usize>>,
    pub respect_existing: bool,
}

impl ThreadAffinity {
    #[must_use]
    pub fn for_node(node: NumaNodeId) -> Self {
        Self {
            node: Some(node),
            cpus: None,
            respect_existing: false,
        }
    }

    #[must_use]
    pub fn for_cpus(cpus: Vec<usize>) -> Self {
        Self {
            node: None,
            cpus: Some(cpus),
            respect_existing: false,
        }
    }

    /// Unrestricted.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn unrestricted() -> Self {
        Self {
            node: None,
            cpus: None,
            respect_existing: true,
        }
    }
}

/// Logical processors assigned to a NUMA node, from the themis topology table.
///
/// Returns an empty set when the node is not present in the snapshot.
fn node_processors(topology: &CpuTopology, node: NumaNodeId) -> Vec<u32> {
    topology
        .node_index(node)
        .and_then(|index| topology.numa_nodes().get(index))
        .map(|n| n.processors.to_vec())
        .unwrap_or_default()
}

/// Set thread affinity.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn set_thread_affinity(affinity: &ThreadAffinity) -> KwaversResult<()> {
    set_current_thread_affinity(affinity)
}

#[cfg(target_os = "linux")]
fn set_current_thread_affinity(affinity: &ThreadAffinity) -> KwaversResult<()> {
    use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};

    unsafe {
        let mut set: cpu_set_t = std::mem::zeroed();
        let topology = detect_topology();

        if let Some(node) = affinity.node {
            CPU_ZERO(&mut set);
            for cpu in node_processors(&topology, node) {
                if (cpu as usize) < topology.logical_processors() {
                    CPU_SET(cpu as usize, &mut set);
                }
            }
        } else if let Some(ref cpus) = affinity.cpus {
            CPU_ZERO(&mut set);
            for &cpu in cpus {
                CPU_SET(cpu, &mut set);
            }
        } else {
            CPU_ZERO(&mut set);
            for cpu in 0..topology.logical_processors() {
                CPU_SET(cpu, &mut set);
            }
        }

        let result = sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &raw const set);
        if result != 0 {
            return Err(KwaversError::System(
                crate::error::SystemError::ResourceUnavailable {
                    resource: format!(
                        "CPU affinity binding failed: errno {}",
                        std::io::Error::last_os_error().raw_os_error().unwrap_or(-1)
                    ),
                },
            ));
        }
    }

    Ok(())
}

#[cfg(target_os = "windows")]
fn set_current_thread_affinity(affinity: &ThreadAffinity) -> KwaversResult<()> {
    extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn SetThreadAffinityMask(
            hThread: *mut std::ffi::c_void,
            dwThreadAffinityMask: usize,
        ) -> usize;
    }

    unsafe {
        let topology = detect_topology();
        let mask = if let Some(node) = affinity.node {
            node_processors(&topology, node)
                .into_iter()
                .fold(0usize, |acc, cpu| {
                    if cpu < usize::BITS {
                        acc | (1usize << cpu)
                    } else {
                        acc
                    }
                })
        } else if let Some(ref cpus) = affinity.cpus {
            cpus.iter().fold(0usize, |acc, &cpu| {
                if cpu < usize::BITS as usize {
                    acc | (1usize << cpu)
                } else {
                    acc
                }
            })
        } else {
            !0usize
        };

        let handle = GetCurrentThread();
        let old_mask = SetThreadAffinityMask(handle, mask);

        if old_mask == 0 {
            return Err(KwaversError::System(
                crate::error::SystemError::ResourceUnavailable {
                    resource: "Failed to set thread affinity mask".to_owned(),
                },
            ));
        }
    }

    Ok(())
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn set_current_thread_affinity(_affinity: &ThreadAffinity) -> KwaversResult<()> {
    Ok(())
}
