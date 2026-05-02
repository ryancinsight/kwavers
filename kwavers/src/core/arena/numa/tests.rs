use super::affinity::ThreadAffinity;
use super::allocator::NumaAllocator;
use super::memory::first_touch_memory;
use super::policy::PAGE_SIZE;
use super::topology::NumaTopology;

#[test]
fn test_numa_topology_detection_sanity() {
    let topo = NumaTopology::detect();
    assert!(topo.node_count >= 1);
    assert!(topo.total_cpus >= 1);
    assert_eq!(topo.distance_matrix.len(), topo.node_count);

    for i in 0..topo.node_count {
        let local = topo.distance(i, i);
        assert!(
            local <= 20,
            "Local access distance should be ≤20, got {}",
            local
        );
    }
}

#[test]
fn test_nodes_by_distance_sorted() {
    let topo = NumaTopology::detect();
    for node in 0..topo.node_count {
        let ordered = topo.nodes_by_distance(node);
        assert_eq!(ordered.len(), topo.node_count);
        assert_eq!(ordered[0].0, node, "self should be closest");
        for i in 1..ordered.len() {
            assert!(
                ordered[i].1 >= ordered[i - 1].1,
                "distances must be non-decreasing"
            );
        }
    }
}

#[test]
fn test_thread_affinity_construction() {
    let unres = ThreadAffinity::unrestricted();
    assert!(unres.node.is_none());
    assert!(unres.cpus.is_none());

    let node = ThreadAffinity::for_node(0);
    assert_eq!(node.node, Some(0));

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
    assert!(alloc.topology().node_count >= 1);
}
