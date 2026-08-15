#![cfg_attr(test, expect(clippy::unwrap_used, reason = "ratchet KWAVERS-UNWRAP-1"))]

use themis::NumaNodeId;

use super::affinity::ThreadAffinity;
use super::allocator::NumaAllocator;
use super::memory::first_touch_memory;
use super::policy::PAGE_SIZE;
use super::topology::detect_topology;

#[test]
fn test_numa_topology_detection_sanity() {
    let topo = detect_topology();
    assert!(!topo.numa_nodes().is_empty());
    assert!(topo.logical_processors() >= 1);

    for node in topo.numa_nodes() {
        let local = topo.distance(node.id, node.id);
        assert!(
            local <= 20,
            "Local access distance should be ≤20, got {local}"
        );
    }
}

#[test]
fn test_adjacent_nodes_sorted() {
    let topo = detect_topology();
    for node in topo.numa_nodes() {
        let ordered = topo.adjacent_nodes(node.id);
        let mut prev: Option<NumaNodeId> = None;
        for &other in ordered {
            if let Some(previous) = prev {
                assert!(
                    topo.distance(node.id, other) >= topo.distance(node.id, previous),
                    "adjacent nodes must be sorted by non-decreasing distance"
                );
            }
            prev = Some(other);
        }
    }
}

#[test]
fn test_thread_affinity_construction() {
    let unres = ThreadAffinity::unrestricted();
    assert!(unres.node.is_none());
    assert!(unres.cpus.is_none());

    let node = ThreadAffinity::for_node(NumaNodeId::ZERO);
    assert_eq!(node.node, Some(NumaNodeId::ZERO));

    let cpus = ThreadAffinity::for_cpus(vec![0, 2, 4]);
    assert_eq!(cpus.cpus, Some(vec![0, 2, 4]));
}

#[test]
fn test_first_touch_memory() {
    let layout = std::alloc::Layout::from_size_align(PAGE_SIZE, PAGE_SIZE).unwrap();
    let ptr = unsafe { std::alloc::alloc(layout) };
    if !ptr.is_null() {
        unsafe { first_touch_memory(ptr, PAGE_SIZE) };
        unsafe { std::alloc::dealloc(ptr, layout) };
    }
}

#[test]
fn test_numa_allocator_default() {
    let alloc = NumaAllocator::new();
    assert!(!alloc.topology().numa_nodes().is_empty());
}
