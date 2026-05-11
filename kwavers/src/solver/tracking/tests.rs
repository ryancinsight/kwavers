use std::sync::Arc;

use crate::solver::validation::contract::MemoryBudget;

use super::{AllocationGuard, GlobalAllocationTracker, ThreadAllocationTracker};

fn create_test_budget() -> Arc<MemoryBudget> {
    Arc::new(MemoryBudget::new(1024 * 1024 * 100)) // 100 MB
}

#[test]
fn thread_tracker_records_allocations() {
    let budget = create_test_budget();
    let tracker = ThreadAllocationTracker::new(budget);

    assert_eq!(tracker.current_bytes(), 0);
    assert_eq!(tracker.peak_bytes(), 0);

    tracker.allocate(1000);
    assert_eq!(tracker.current_bytes(), 1000);
    assert_eq!(tracker.peak_bytes(), 1000);

    tracker.allocate(500);
    assert_eq!(tracker.current_bytes(), 1500);
    assert_eq!(tracker.peak_bytes(), 1500);

    tracker.deallocate(500);
    assert_eq!(tracker.current_bytes(), 1000);
    assert_eq!(tracker.peak_bytes(), 1500); // Peak preserved
}

#[test]
fn thread_tracker_tracks_allocations_count() {
    let budget = create_test_budget();
    let tracker = ThreadAllocationTracker::new(budget);

    assert_eq!(tracker.total_allocations(), 0);

    tracker.allocate(100);
    tracker.allocate(200);
    tracker.allocate(300);

    assert_eq!(tracker.total_allocations(), 3);
}

#[test]
fn thread_tracker_reset_clears_current() {
    let budget = create_test_budget();
    let tracker = ThreadAllocationTracker::new(budget);

    tracker.allocate(1000);
    assert_eq!(tracker.current_bytes(), 1000);

    tracker.reset();
    assert_eq!(tracker.current_bytes(), 0);
    assert_eq!(tracker.peak_bytes(), 1000); // Peak preserved
}

#[test]
fn allocation_guard_tracks_on_drop() {
    let budget = create_test_budget();
    let tracker = ThreadAllocationTracker::new(budget);

    {
        let _guard = AllocationGuard::new(&tracker, 500);
        assert_eq!(tracker.current_bytes(), 500);
    }

    // Should be deallocated on drop
    assert_eq!(tracker.current_bytes(), 0);
    assert_eq!(tracker.peak_bytes(), 500);
}

#[test]
fn allocation_guard_release_prevents_double_dealloc() {
    let budget = create_test_budget();
    let tracker = ThreadAllocationTracker::new(budget);

    let guard = AllocationGuard::new(&tracker, 500);
    guard.release();

    // After explicit release, drop should not deallocate again
    assert_eq!(tracker.current_bytes(), 0);
}

#[test]
fn global_tracker_aggregates_threads() {
    let budget = create_test_budget();
    let global = GlobalAllocationTracker::new(2, budget, None);

    // Simulate allocations on thread 0
    global.thread_trackers[0].allocate(1000);
    global.thread_trackers[0].allocate(500);

    // Simulate allocations on thread 1
    global.thread_trackers[1].allocate(200);

    assert_eq!(global.total_current_bytes(), 1700);
    assert_eq!(global.total_peak_bytes(), 1500); // Max of threads
}
