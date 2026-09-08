#![cfg(feature = "gpu")]

use kwavers_gpu::profiling::{GpuAllocationConfig, GpuAllocationTracker};

#[test]
fn test_gpu_budget_enforcement() {
    // Budget of 1MB = 1048576 bytes
    let config = GpuAllocationConfig::with_safety_factor(1.0);
    let tracker = GpuAllocationTracker::new(1024 * 1024, config);

    // Allocate 0.5 MB -> success
    let _buf1 = tracker.allocate(500_000, "buf1").unwrap();
    assert_eq!(tracker.current_bytes(), 500_000);

    // Allocate 0.6 MB -> should fail due to OOM budget
    let error = tracker
        .allocate(600_000, "buf2")
        .expect_err("600 kB on top of 500 kB exceeds the 1 MB budget");
    let rendered = error.to_string();
    assert!(
        rendered.contains("budget") || rendered.contains("OutOfMemory") || rendered.contains("OOM"),
        "the rejection must name the exhausted budget, got {rendered:?}"
    );

    // The current bytes strictly does not increase on failure
    assert_eq!(tracker.current_bytes(), 500_000);
}

#[test]
fn test_gpu_allocation_guard_raii_release() {
    let config = GpuAllocationConfig::with_safety_factor(1.0);
    let tracker = GpuAllocationTracker::new(1024 * 1024, config);

    {
        let _buf1 = tracker.allocate(200_000, "buf1").unwrap();
        assert_eq!(tracker.current_bytes(), 200_000);
        assert_eq!(tracker.peak_bytes(), 200_000);

        {
            let _buf2 = tracker.allocate(300_000, "buf2").unwrap();
            assert_eq!(tracker.current_bytes(), 500_000);
            assert_eq!(tracker.peak_bytes(), 500_000);
        } // _buf2 drops here

        assert_eq!(tracker.current_bytes(), 200_000); // released
        assert_eq!(tracker.peak_bytes(), 500_000); // peak unchanged
    } // _buf1 drops here

    assert_eq!(tracker.current_bytes(), 0);
    assert_eq!(tracker.peak_bytes(), 500_000);
}
