#![cfg(feature = "gpu")]

use kwavers::domain::grid::Grid;
use kwavers::gpu::buffer::{BufferUsage, GpuBuffer};
use kwavers::gpu::FdtdGpu;
use kwavers::profiling::gpu_allocator::{GpuAllocationConfig, GpuAllocationTracker};
use std::sync::Arc;

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
    let (device, _queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => return,
    };

    // Budget of 1MB = 1048576 bytes
    let config = GpuAllocationConfig::with_safety_factor(1.0);
    let tracker = GpuAllocationTracker::new(1024 * 1024, config);

    // Allocate 0.5 MB -> success
    let buf1 = GpuBuffer::create(
        &device,
        500_000,
        BufferUsage::STORAGE | BufferUsage::COPY_DST,
        Some(tracker.clone()),
    );
    assert!(buf1.is_ok());
    assert_eq!(tracker.current_bytes(), 500_000);

    // Allocate 0.6 MB -> should fail due to OOM budget
    let buf2 = GpuBuffer::create(
        &device,
        600_000,
        BufferUsage::STORAGE | BufferUsage::COPY_DST,
        Some(tracker.clone()),
    );
    assert!(buf2.is_err());
    let err_str = buf2.unwrap_err().to_string();
    assert!(
        err_str.contains("budget") || err_str.contains("OutOfMemory") || err_str.contains("OOM")
    );

    // The current bytes strictly does not increase on failure
    assert_eq!(tracker.current_bytes(), 500_000);
}

#[tokio::test]
async fn test_gpu_buffer_raii_release() {
    let (device, _queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => return,
    };

    let config = GpuAllocationConfig::with_safety_factor(1.0);
    let tracker = GpuAllocationTracker::new(1024 * 1024, config);

    {
        let _buf1 = GpuBuffer::create(
            &device,
            200_000,
            BufferUsage::STORAGE | BufferUsage::COPY_DST,
            Some(tracker.clone()),
        )
        .unwrap();
        assert_eq!(tracker.current_bytes(), 200_000);
        assert_eq!(tracker.peak_bytes(), 200_000);

        {
            let _buf2 = GpuBuffer::create(
                &device,
                300_000,
                BufferUsage::STORAGE | BufferUsage::COPY_DST,
                Some(tracker.clone()),
            )
            .unwrap();
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
async fn test_fdtd_gpu_tracker_integration() {
    let (device, _queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => return,
    };

    // 10x10x10 grid = 1000 nodes
    let nx = 10;
    let ny = 10;
    let nz = 10;

    let grid = Grid::new(nx, ny, nz, 0.1, 0.1, 0.1).unwrap();

    let buffer_size = (nx * ny * nz) as usize * std::mem::size_of::<f32>();
    // FdtdGpu creates 3 buffers:
    // pressure: 1 * buffer_size
    // velocity: 3 * buffer_size
    // medium:   2 * buffer_size
    // total = 6 * buffer_size
    let expected_total_bytes = 6 * buffer_size;

    let config = GpuAllocationConfig::with_safety_factor(1.0);
    let tracker = GpuAllocationTracker::new(1024 * 1024 * 100, config);

    {
        let _fdtd =
            FdtdGpu::new(&device, &grid, Some(tracker.clone())).expect("Failed to create FdtdGpu");
        assert_eq!(tracker.current_bytes(), expected_total_bytes);
        assert_eq!(tracker.peak_bytes(), expected_total_bytes);
    }

    // Check RAII properly released FdtdGpu buffers
    assert_eq!(tracker.current_bytes(), 0);
}
