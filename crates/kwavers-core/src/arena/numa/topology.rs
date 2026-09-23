//! CPU topology detection, delegated to themis.
//!
//! kwavers previously hand-rolled a `NumaTopology` here (sysfs node scanning
//! and distance-matrix parsing). That duplication is deleted: themis owns the
//! topology snapshot SSOT (`themis::CpuTopology`). This module keeps only the
//! kwavers-specific fallback convenience, mirroring the removed
//! `NumaTopology::detect` contract (which never returned `None`).

use std::collections::BTreeSet;
use std::sync::OnceLock;

use themis::{CacheLevel, CpuTopology};

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

/// Bytes of cache a working set split evenly across every processor keeps
/// resident, or `None` when the platform reports no caches.
///
/// Each processor holds its share in its private caches past the first
/// level -- a cache shared by a cluster counts its size over the processors
/// that share it -- and all of them hold the rest in the shared last level.
/// An even split overflows first where the private share is smallest, so
/// that share times the processor count, plus the last level, is the
/// capacity: on a hybrid part whose efficiency cores share one second-level
/// cache per cluster of four, the efficiency cores' quarter bounds it.
///
/// On a hierarchy whose last level duplicates the private levels this
/// overstates the capacity by the private total.
///
/// Detected once, like [`last_level_cache_bytes`].
#[must_use]
pub fn cache_capacity_bytes() -> Option<usize> {
    static BYTES: OnceLock<Option<usize>> = OnceLock::new();
    *BYTES.get_or_init(|| even_split_capacity(detect_topology().cache_levels()?))
}

/// [`cache_capacity_bytes`] over a reported cache hierarchy.
fn even_split_capacity(levels: &[CacheLevel]) -> Option<usize> {
    let last = levels.iter().map(|cache| cache.level).max()?;
    let shared: Vec<&CacheLevel> = levels.iter().filter(|cache| cache.level == last).collect();
    let processors: BTreeSet<u32> = shared
        .iter()
        .flat_map(|cache| cache.shared_processors.iter().copied())
        .collect();
    let smallest_share = processors
        .iter()
        .map(|processor| {
            levels
                .iter()
                .filter(|cache| cache.level > 1 && cache.level < last)
                .filter(|cache| cache.shared_processors.contains(processor))
                .map(|cache| cache.size_bytes / cache.shared_processors.len())
                .sum::<usize>()
        })
        .min()
        .unwrap_or(0);
    let last_level: usize = shared.iter().map(|cache| cache.size_bytes).sum();
    Some(smallest_share * processors.len() + last_level)
}

#[cfg(test)]
mod tests {
    use super::even_split_capacity;
    use themis::CacheLevel;

    fn cache(level: u32, mebibytes: usize, processors: core::ops::Range<u32>) -> CacheLevel {
        CacheLevel {
            level,
            size_bytes: mebibytes << 20,
            line_bytes: Some(64),
            shared_processors: processors.collect(),
        }
    }

    /// A 285K-shaped part: eight performance cores with 3 MiB of L2 each,
    /// four clusters of four efficiency cores sharing 4 MiB each, and a
    /// 36 MiB L3 over all 24. The efficiency cores' 1 MiB share bounds the
    /// split: 24 MiB of L2 plus the L3, where the L2 instances sum to 40.
    /// First-level caches do not count.
    #[test]
    fn the_smallest_private_share_bounds_an_even_split() {
        let mut levels: Vec<CacheLevel> = (0..8).map(|p| cache(2, 3, p..p + 1)).collect();
        levels.extend((0..4).map(|c| cache(2, 4, 8 + 4 * c..12 + 4 * c)));
        levels.extend((0..24).map(|p| cache(1, 1, p..p + 1)));
        levels.push(cache(3, 36, 0..24));
        assert_eq!(even_split_capacity(&levels), Some(60 << 20));
    }

    /// With only a shared level reported the capacity is that level, and
    /// with nothing reported there is none.
    #[test]
    fn a_lone_shared_level_is_the_capacity() {
        assert_eq!(even_split_capacity(&[cache(3, 36, 0..24)]), Some(36 << 20));
        assert_eq!(even_split_capacity(&[]), None);
    }
}
