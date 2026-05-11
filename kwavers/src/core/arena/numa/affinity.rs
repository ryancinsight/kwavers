use super::topology::NumaTopology;
use crate::core::error::{KwaversError, KwaversResult};

/// Thread affinity configuration.
#[derive(Debug, Clone)]
pub struct ThreadAffinity {
    pub node: Option<usize>,
    pub cpus: Option<Vec<usize>>,
    pub respect_existing: bool,
}

impl ThreadAffinity {
    #[must_use] 
    pub fn for_node(node: usize) -> Self {
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
/// Set thread affinity.
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn set_thread_affinity(affinity: &ThreadAffinity) -> KwaversResult<()> {
    set_current_thread_affinity(affinity)
}

#[must_use] 
pub fn current_numa_node() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let cpu = unsafe { libc::sched_getcpu() };
        if cpu < 0 {
            return None;
        }
        fs::read_to_string(format!("/sys/devices/system/cpu/cpu{}/cpulist", cpu))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn set_current_thread_affinity(affinity: &ThreadAffinity) -> KwaversResult<()> {
    use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};

    unsafe {
        let mut set: cpu_set_t = std::mem::zeroed();
        let topology = NumaTopology::detect();

        if let Some(node) = affinity.node {
            let cpus_per_node = topology.cpus_per_node;
            CPU_ZERO(&mut set);
            for cpu in (node * cpus_per_node)..((node + 1) * cpus_per_node) {
                if cpu < topology.total_cpus {
                    CPU_SET(cpu, &mut set);
                }
            }
        } else if let Some(ref cpus) = affinity.cpus {
            CPU_ZERO(&mut set);
            for &cpu in cpus {
                CPU_SET(cpu, &set);
            }
        } else {
            CPU_ZERO(&mut set);
            for cpu in 0..topology.total_cpus {
                CPU_SET(cpu, &mut set);
            }
        }

        let result = sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &set);
        if result != 0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
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
        let topology = NumaTopology::detect();
        let mask = if let Some(node) = affinity.node {
            let cpus_per_node = topology.cpus_per_node;
            let start_cpu = node * cpus_per_node;
            ((1usize << cpus_per_node) - 1) << start_cpu
        } else {
            !0usize
        };

        let handle = GetCurrentThread();
        let old_mask = SetThreadAffinityMask(handle, mask);

        if old_mask == 0 {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
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
