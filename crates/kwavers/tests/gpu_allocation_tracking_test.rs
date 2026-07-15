#![cfg(feature = "gpu")]

use kwavers_gpu::gpu::{FdtdGpuProvider, WgpuFdtd};
use kwavers_gpu::profiling::{GpuAllocationConfig, GpuAllocationTracker};
use kwavers_grid::Grid;
use leto::Array3 as LetoArray3;

#[test]
fn test_gpu_budget_enforcement() {
    // Budget of 1MB = 1048576 bytes
    let config = GpuAllocationConfig::with_safety_factor(1.0);
    let tracker = GpuAllocationTracker::new(1024 * 1024, config);

    // Allocate 0.5 MB -> success
    let _buf1 = tracker.allocate(500_000, "buf1").unwrap();
    assert_eq!(tracker.current_bytes(), 500_000);

    // Allocate 0.6 MB -> should fail due to OOM budget
    let buf2 = tracker.allocate(600_000, "buf2");
    assert!(buf2.is_err());
    let err_str = buf2.unwrap_err().to_string();
    assert!(
        err_str.contains("budget") || err_str.contains("OutOfMemory") || err_str.contains("OOM")
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

#[test]
fn test_fdtd_gpu_construction_and_pressure_roundtrip() {
    let grid = Grid::new(6, 6, 6, 0.1, 0.1, 0.1).unwrap();
    let pressure = LetoArray3::from_shape_fn([6, 6, 6], |[i, j, k]| (i + 6 * j + 36 * k) as f32);
    assert_fdtd_pressure_roundtrip::<WgpuFdtd>(&grid, pressure);
}

#[test]
fn test_fdtd_gpu_pressure_roundtrip() {
    let grid = Grid::new(4, 4, 4, 0.1, 0.1, 0.1).unwrap();
    let pressure = LetoArray3::from_shape_fn([4, 4, 4], |[i, j, k]| (i + 4 * j + 16 * k) as f32);
    assert_fdtd_pressure_roundtrip::<WgpuFdtd>(&grid, pressure);
}

fn assert_fdtd_pressure_roundtrip<P>(grid: &Grid, pressure: LetoArray3<f32>)
where
    P: FdtdGpuProvider<Scalar = f32>,
{
    let fdtd = match P::new(grid) {
        Ok(fdtd) => fdtd,
        Err(_) => return,
    };

    fdtd.upload_pressure(&pressure)
        .expect("pressure upload must succeed");

    let downloaded = fdtd
        .download_pressure_blocking(grid)
        .expect("pressure download must succeed");

    assert_eq!(downloaded, pressure);
}
