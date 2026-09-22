//! CPU topology detection, delegated to themis.
//!
//! kwavers previously hand-rolled a `NumaTopology` here (sysfs node scanning
//! and distance-matrix parsing). That duplication is deleted: themis owns the
//! topology snapshot SSOT (`themis::CpuTopology`). This module keeps only the
//! kwavers-specific fallback convenience, mirroring the removed
//! `NumaTopology::detect` contract (which never returned `None`).

use std::sync::OnceLock;

use themis::CpuTopology;

/// Detects the CPU topology, falling back to a single-node topology when the
/// platform does not report one.
///
/// `themis::CpuTopology::detect` returns [`Option`] (a failed platform probe
/// is typed absence). The removed `NumaTopology::detect` always produced a
/// usable snapshot, so this wrapper preserves that contract for the execution
/// primitives (`mbind`, thread affinity, interleaved allocation) that need
/// *some* node/processor table to operate against.
#[must_use]
pub fn detect_topology() -> CpuTopology {
    CpuTopology::detect().unwrap_or_else(|| {
        let cpus = std::thread::available_parallelism().map_or(1, usize::from);
        CpuTopology::single_node(cpus)
    })
}

/// Bytes of the largest cache level the platform reports, or `None` when it
/// reports none.
///
/// A kernel whose working set crosses this boundary changes cost regime:
/// below it a field written by one pass is still resident when the next
/// reads it, above it the re-read comes from DRAM. Kernels that choose
/// between traversal shapes on that basis compare their live set against
/// this.
///
/// Detected once. The topology does not change during a run, and the probe
/// reads the operating system on every call.
#[must_use]
pub fn last_level_cache_bytes() -> Option<usize> {
    static BYTES: OnceLock<Option<usize>> = OnceLock::new();
    *BYTES.get_or_init(|| {
        let topology = detect_topology();
        let levels = topology.cache_levels()?;
        levels
            .iter()
            .max_by_key(|level| level.level)
            .map(|level| level.size_bytes)
    })
}
