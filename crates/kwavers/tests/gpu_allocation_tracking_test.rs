#![cfg(feature = "gpu")]

use kwavers_grid::Grid;
use kwavers_gpu::gpu::FdtdGpu;
use kwavers_gpu::profiling::{GpuAllocationConfig, GpuAllocationTracker};
use ndarray::Array3;

async fn create_test_device() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .map_err(|_| "No GPU adapter found".to_string())?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::PUSH_CONSTANTS,
            required_limits: wgpu::Limits {
                max_push_constant_size: 128,
                max_compute_invocations_per_workgroup: 512,
                ..wgpu::Limits::default()
            },
            ..Default::default()
        })
        .await
        .map_err(|_| "No device found".to_string())?;

    Ok((device, queue))
}

#[tokio::test]
async fn test_gpu_budget_enforcement() {
    let (_device, _queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => return,
    };

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

#[tokio::test]
async fn test_gpu_allocation_guard_raii_release() {
    let (_device, _queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => return,
    };

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

#[tokio::test]
async fn test_fdtd_gpu_construction_and_pressure_roundtrip() {
    let (device, queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => return,
    };

    let grid = Grid::new(6, 6, 6, 0.1, 0.1, 0.1).unwrap();
    let fdtd = FdtdGpu::new(&device, &grid).expect("Failed to create FdtdGpu");

    let pressure = Array3::from_shape_fn((6, 6, 6), |(i, j, k)| (i + 6 * j + 36 * k) as f64);
    fdtd.upload_pressure(&queue, &pressure);

    let downloaded = fdtd
        .download_pressure(&device, &queue, &grid)
        .await
        .expect("pressure download must succeed");

    assert_eq!(downloaded, pressure);
}

#[tokio::test]
async fn test_fdtd_gpu_pressure_roundtrip() {
    let (device, queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => return,
    };

    let grid = Grid::new(4, 4, 4, 0.1, 0.1, 0.1).unwrap();
    let fdtd = FdtdGpu::new(&device, &grid).expect("Failed to create FdtdGpu");

    let pressure = Array3::from_shape_fn((4, 4, 4), |(i, j, k)| (i + 4 * j + 16 * k) as f64);
    fdtd.upload_pressure(&queue, &pressure);

    let downloaded = fdtd
        .download_pressure(&device, &queue, &grid)
        .await
        .expect("pressure download must succeed");

    assert_eq!(downloaded, pressure);
}
