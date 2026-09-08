use super::tracker::GpuAllocationTracker;
use std::sync::Arc;

#[test]
fn tracker_init() {
    let tracker = GpuAllocationTracker::with_capacity(1024 * 1024 * 1024); // 1 GB
    assert_eq!(tracker.current_memory(), 0);
    assert_eq!(tracker.peak_memory(), 0);

    let stats = tracker.stats();
    assert_eq!(stats.device_capacity, 1024 * 1024 * 1024);
    assert_eq!(
        stats.budget_bytes,
        (1024u64 * 1024 * 1024 * 9 / 10) as usize
    );
}

#[test]
fn allocation_tracks_memory() {
    let tracker = GpuAllocationTracker::with_capacity(1024 * 1024); // 1 MB
    let _guard = tracker.allocate(100, "test_allocation").unwrap();

    assert_eq!(tracker.current_memory(), 100);
    assert!(tracker.peak_memory() >= 100);
}

#[test]
fn guard_releases_memory() {
    let tracker = GpuAllocationTracker::with_capacity(1024 * 1024);
    {
        let _guard = tracker.allocate(100, "test").unwrap();
        assert_eq!(tracker.current_memory(), 100);
    } // Guard dropped

    assert_eq!(tracker.current_memory(), 0);
}

#[test]
fn budget_enforcement() {
    let tracker = GpuAllocationTracker::with_capacity(1000); // 900-byte budget
    let _g1 = tracker.allocate(300, "a").unwrap();
    let _g2 = tracker.allocate(300, "b").unwrap();

    // 600 + 300 = 900 == budget, strict `>` check → should succeed
    let guard = tracker
        .allocate(300, "c")
        .expect("allocation at exactly the budget should succeed");
    assert_eq!(tracker.current_bytes(), 900, "the third block is counted");
    drop(guard);

    // 900 + 1 = 901 > 900 → must fail
    let _g3 = tracker
        .allocate(300, "c")
        .expect("the released block is available again");
    let rejection = tracker
        .allocate(1, "d")
        .expect_err("one byte past the budget must fail")
        .to_string();
    assert!(
        rejection.contains("GPU OOM"),
        "the rejection must name the exhausted budget, got {rejection:?}"
    );
}

#[test]
fn stats_utilization() {
    let tracker = GpuAllocationTracker::with_capacity(1000);
    let _guard = tracker.allocate(500, "half").unwrap();

    let stats = tracker.stats();
    assert_eq!(stats.utilization(), 0.5); // 500/1000
    assert_eq!(stats.current_bytes, 500);
}

#[test]
fn peak_tracking() {
    let tracker = GpuAllocationTracker::with_capacity(1024 * 1024);

    let _g1 = tracker.allocate(100, "a").unwrap();
    assert_eq!(tracker.peak_memory(), 100);

    let _g2 = tracker.allocate(200, "b").unwrap();
    assert_eq!(tracker.peak_memory(), 300); // 100 + 200

    drop(_g1);
    assert_eq!(tracker.peak_memory(), 300); // unchanged
}

#[test]
fn available_bytes_calculation() {
    let tracker = GpuAllocationTracker::with_capacity(1000);
    let _guard = tracker.allocate(300, "test").unwrap();

    // 1000 * 0.9 = 900 budget, 300 used → 600 available
    assert_eq!(tracker.available_bytes(), 600);
}

#[test]
fn would_exceed_budget_prediction() {
    let tracker = GpuAllocationTracker::with_capacity(1000);
    let _guard = tracker.allocate(300, "test").unwrap();

    // 300 + 500 = 800 < 900 → should not exceed
    assert!(!tracker.would_exceed_budget(500));
    // 300 + 700 = 1000 > 900 → should exceed
    assert!(tracker.would_exceed_budget(700));
}

#[test]
fn multi_thread_safety() {
    use std::thread;

    let tracker = GpuAllocationTracker::with_capacity(1024 * 1024 * 10);
    let tracker_arc = Arc::new(tracker);

    let handles: Vec<_> = (0..10)
        .map(|i| {
            let t = Arc::clone(&tracker_arc);
            thread::spawn(move || {
                for j in 0..100 {
                    let name = format!("t{}_alloc_{}", i, j);
                    let _g = t.allocate(100, &name);
                    // Immediately drop to simulate churn
                }
                i
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let stats = tracker_arc.stats();
    assert_eq!(stats.total_allocations, 1000); // 10 threads × 100 each
    assert_eq!(stats.total_deallocations, 1000);
    assert_eq!(stats.current_bytes, 0); // All dropped
}
