//! GPU Device Management Tests
//!
//! Tests for GPU device initialization, capability detection, and resource management.

#![cfg(feature = "gpu")]

use kwavers_core::error::KwaversResult;
use kwavers_gpu::gpu::device::{DeviceFeature, DevicePreference, GpuDevice, GpuDeviceInfo};
use kwavers_gpu::gpu::{BufferUsage, GpuBufferData};

mod common;

fn create_device(preference: DevicePreference) -> KwaversResult<GpuDevice> {
    GpuDevice::try_create(preference)
}

#[test]
fn test_device_creation_high_performance() {
    if let Some(device) = common::or_skip(
        "high-performance device",
        create_device(DevicePreference::HighPerformance),
    ) {
        assert!(device.limits().max_buffer_size > 0);
        assert!(device.wgpu_queue().get_timestamp_period() > 0.0);
    }
}

#[test]
fn test_device_creation_low_power() {
    if let Some(device) = common::or_skip(
        "low-power device",
        create_device(DevicePreference::LowPower),
    ) {
        assert!(device.limits().max_buffer_size > 0);
    }
}

#[test]
fn test_device_info() {
    let Some(device) = common::or_skip(
        "high-performance device",
        create_device(DevicePreference::HighPerformance),
    ) else {
        return;
    };

    let info = device.info();

    // Verify device info is populated
    assert!(!info.name.is_empty(), "Device name should not be empty");
    assert!(info.vendor > 0, "Vendor ID should be non-zero");
    assert!(
        !info.device_type.is_empty(),
        "Device type should not be empty"
    );
    assert!(!info.backend.is_empty(), "Backend should not be empty");

    // Print info for debugging
    println!("Device: {}", info.name);
    println!("Vendor: {}", info.vendor);
    println!("Type: {}", info.device_type);
    println!("Backend: {}", info.backend);
}

#[test]
fn test_device_limits() {
    let Some(device) = common::or_skip(
        "high-performance device",
        create_device(DevicePreference::HighPerformance),
    ) else {
        return;
    };

    let limits = device.limits();

    // Verify limits are reasonable
    assert!(
        limits.max_buffer_size >= 128 * 1024 * 1024,
        "Max buffer size should be at least 128MB"
    );
    assert!(
        limits.max_compute_workgroup_size_x >= 64,
        "Workgroup size X should be at least 64"
    );
    assert!(
        limits.max_compute_workgroup_size_y >= 64,
        "Workgroup size Y should be at least 64"
    );
    assert!(
        limits.max_compute_workgroup_size_z >= 64,
        "Workgroup size Z should be at least 64"
    );
    assert!(
        limits.max_compute_invocations_per_workgroup >= 128,
        "Invocations per workgroup should be at least 128"
    );

    println!(
        "Max buffer size: {} MB",
        limits.max_buffer_size / (1024 * 1024)
    );
    println!(
        "Max workgroup size: ({}, {}, {})",
        limits.max_compute_workgroup_size_x,
        limits.max_compute_workgroup_size_y,
        limits.max_compute_workgroup_size_z
    );
    println!(
        "Max invocations per workgroup: {}",
        limits.max_compute_invocations_per_workgroup
    );
}

#[test]
fn test_device_features() {
    let Some(device) = common::or_skip(
        "high-performance device",
        create_device(DevicePreference::HighPerformance),
    ) else {
        return;
    };

    // Test feature checking
    let has_timestamp = device.supports_feature(DeviceFeature::TimestampQuery);
    let has_f64 = device.supports_feature(DeviceFeature::ShaderF64);
    let has_f16 = device.supports_feature(DeviceFeature::ShaderF16);

    println!("Timestamp query support: {}", has_timestamp);
    println!("F64 shader support: {}", has_f64);
    println!("F16 shader support: {}", has_f16);

    // These are informational, not assertions since feature support varies
}

#[test]
fn test_device_multiple_instances() {
    let Some(first) = common::or_skip(
        "first of two devices",
        create_device(DevicePreference::HighPerformance),
    ) else {
        return;
    };

    // The host has an adapter, so the second acquisition has no absence excuse:
    // a device that can be opened once and not twice is the defect this test
    // exists to catch.
    let second = create_device(DevicePreference::HighPerformance)
        .expect("a second device must open on a host whose first one did");

    assert!(first.limits().max_buffer_size > 0);
    assert!(second.limits().max_buffer_size > 0);
    assert_eq!(
        first.limits().max_buffer_size,
        second.limits().max_buffer_size,
        "two devices on one adapter must report the same limits"
    );
}

#[test]
fn test_device_queue_operations() {
    let Some(device) = common::or_skip(
        "high-performance device",
        create_device(DevicePreference::HighPerformance),
    ) else {
        return;
    };

    let buffer = GpuBufferData::create_on_device(
        &device,
        256,
        BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
    )
    .expect("buffer creation must succeed");

    let data: Vec<u8> = vec![42; 256];
    buffer.write_on_device(&device, &data);

    device.synchronize().unwrap();

    let read_back: Vec<u8> = buffer
        .read_to_vec_on_device(&device)
        .expect("buffer readback must succeed");
    assert_eq!(read_back, data);
}

#[test]
fn test_device_info_clone() {
    let info = GpuDeviceInfo {
        name: "Test Device".to_string(),
        vendor: 0x1234,
        device_type: "DiscreteGpu".to_string(),
        backend: "Vulkan".to_string(),
    };

    let cloned = info.clone();

    assert_eq!(info.name, cloned.name);
    assert_eq!(info.vendor, cloned.vendor);
    assert_eq!(info.device_type, cloned.device_type);
    assert_eq!(info.backend, cloned.backend);
}

#[test]
fn test_device_info_debug() {
    let info = GpuDeviceInfo {
        name: "Test Device".to_string(),
        vendor: 0x1234,
        device_type: "DiscreteGpu".to_string(),
        backend: "Vulkan".to_string(),
    };

    let debug_str = format!("{:?}", info);
    assert!(debug_str.contains("Test Device"));
    assert!(debug_str.contains("4660")); // 0x1234 in decimal
}
