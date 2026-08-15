//! CPU topology detection, delegated to themis.
//!
//! kwavers previously hand-rolled a `NumaTopology` here (sysfs node scanning
//! and distance-matrix parsing). That duplication is deleted: themis owns the
//! topology snapshot SSOT (`themis::CpuTopology`). This module keeps only the
//! kwavers-specific fallback convenience, mirroring the removed
//! `NumaTopology::detect` contract (which never returned `None`).

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
