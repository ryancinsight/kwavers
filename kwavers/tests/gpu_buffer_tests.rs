//! GPU Buffer Management Tests
//!
//! Comprehensive tests for GPU buffer creation, data transfer, and memory management.
//! Tests cover both synchronous and asynchronous patterns with proper error handling.

#![cfg(feature = "gpu")]

use kwavers::gpu::buffer::{BufferUsage, GpuBuffer};

/// Helper to create a test GPU device
async fn create_test_device() -> Result<(wgpu::Device, wgpu::Queue), Box<dyn std::error::Error>> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .ok_or("No GPU adapter found")?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await?;

    Ok((device, queue))
}

#[tokio::test]
async fn test_buffer_creation() {
    let (device, _queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    // Test creating an empty buffer
    let buffer = GpuBuffer::create(&device, 1024, BufferUsage::STORAGE | BufferUsage::COPY_DST);

    assert!(buffer.is_ok());
    let buffer = buffer.unwrap();
    assert_eq!(buffer.size(), 1024);
}

#[tokio::test]
async fn test_buffer_with_data() {
    let (device, _queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let buffer =
        GpuBuffer::create_with_data(&device, &data, BufferUsage::STORAGE | BufferUsage::COPY_SRC);

    assert!(buffer.is_ok());
    let buffer = buffer.unwrap();
    assert_eq!(buffer.size(), data.len() * std::mem::size_of::<f32>());
}

#[tokio::test]
async fn test_buffer_write() {
    let (device, queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    let initial_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let buffer = GpuBuffer::create_with_data(
        &device,
        &initial_data,
        BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
    )
    .unwrap();

    // Write new data
    let new_data: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];
    buffer.write(&queue, &new_data);

    // Verify write succeeded (buffer still valid)
    assert_eq!(buffer.size(), new_data.len() * std::mem::size_of::<f32>());
}

#[tokio::test]
async fn test_buffer_read_write_roundtrip() {
    let (device, queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    let original_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let buffer = GpuBuffer::create_with_data(
        &device,
        &original_data,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
    )
    .unwrap();

    // Read back data
    let read_data: Vec<f32> = buffer.read_to_vec::<f32>(&device, &queue).await.unwrap();

    // Verify data matches
    assert_eq!(read_data.len(), original_data.len());
    for (i, (&original, &read)) in original_data.iter().zip(read_data.iter()).enumerate() {
        assert!(
            (original - read).abs() < 1e-6,
            "Mismatch at index {}: expected {}, got {}",
            i,
            original,
            read
        );
    }
}

#[tokio::test]
async fn test_buffer_different_types() {
    let (device, queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    // Test with u32
    let u32_data: Vec<u32> = vec![1, 2, 3, 4, 5];
    let u32_buffer = GpuBuffer::create_with_data(
        &device,
        &u32_data,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
    )
    .unwrap();

    let read_u32: Vec<u32> = u32_buffer.read_to_vec(&device, &queue).await.unwrap();
    assert_eq!(read_u32, u32_data);

    // Test with i32
    let i32_data: Vec<i32> = vec![-1, 2, -3, 4, -5];
    let i32_buffer = GpuBuffer::create_with_data(
        &device,
        &i32_data,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
    )
    .unwrap();

    let read_i32: Vec<i32> = i32_buffer.read_to_vec(&device, &queue).await.unwrap();
    assert_eq!(read_i32, i32_data);
}

#[tokio::test]
async fn test_buffer_large_data() {
    let (device, queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    // Test with larger dataset (1MB)
    let size = 256 * 1024; // 256K floats = 1MB
    let large_data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    let buffer = GpuBuffer::create_with_data(
        &device,
        &large_data,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
    )
    .unwrap();

    let read_data: Vec<f32> = buffer.read_to_vec::<f32>(&device, &queue).await.unwrap();

    // Verify first, middle, and last elements
    assert_eq!(read_data[0], 0.0);
    assert_eq!(read_data[size / 2], (size / 2) as f32);
    assert_eq!(read_data[size - 1], (size - 1) as f32);
}

#[test]
fn test_buffer_usage_flags() {
    // Test that BufferUsage constants are defined correctly
    assert_eq!(BufferUsage::STORAGE, wgpu::BufferUsages::STORAGE);
    assert_eq!(BufferUsage::UNIFORM, wgpu::BufferUsages::UNIFORM);
    assert_eq!(BufferUsage::COPY_SRC, wgpu::BufferUsages::COPY_SRC);
    assert_eq!(BufferUsage::COPY_DST, wgpu::BufferUsages::COPY_DST);

    // Test combining flags
    let combined = BufferUsage::STORAGE | BufferUsage::COPY_DST;
    assert!(combined.contains(wgpu::BufferUsages::STORAGE));
    assert!(combined.contains(wgpu::BufferUsages::COPY_DST));
}

#[tokio::test]
async fn test_buffer_zero_initialization() {
    let (device, queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    let zero_data: Vec<f32> = vec![0.0; 100];
    let buffer = GpuBuffer::create_with_data(
        &device,
        &zero_data,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
    )
    .unwrap();

    let read_data: Vec<f32> = buffer.read_to_vec(&device, &queue).await.unwrap();

    for (i, &val) in read_data.iter().enumerate() {
        assert_eq!(val, 0.0, "Non-zero value at index {}: {}", i, val);
    }
}

#[tokio::test]
async fn test_buffer_sequential_writes() {
    let (device, queue) = match create_test_device().await {
        Ok(d) => d,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    let buffer = GpuBuffer::create(
        &device,
        16 * std::mem::size_of::<f32>(),
        BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
    )
    .unwrap();

    // First write
    let data1: Vec<f32> = vec![1.0; 16];
    buffer.write(&queue, &data1);
    device.poll(wgpu::Maintain::Wait);

    let read1: Vec<f32> = buffer.read_to_vec::<f32>(&device, &queue).await.unwrap();
    assert!(read1.iter().all(|&x| (x - 1.0).abs() < 1e-6));

    // Second write
    let data2: Vec<f32> = vec![2.0; 16];
    buffer.write(&queue, &data2);
    device.poll(wgpu::Maintain::Wait);

    let read2: Vec<f32> = buffer.read_to_vec::<f32>(&device, &queue).await.unwrap();
    assert!(read2.iter().all(|&x| (x - 2.0).abs() < 1e-6));
}
