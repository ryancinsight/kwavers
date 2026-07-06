//! Provider-owned GPU buffer tests.
//!
//! These tests validate the canonical `GpuBufferData` wrapper through the
//! Kwavers/Hephaestus GPU context. They intentionally avoid direct WGPU adapter
//! acquisition and Tokio runtime requirements.

#![cfg(feature = "gpu")]

use kwavers_gpu::gpu::{BufferUsage, CoreGpuContext, GpuBufferData};

fn test_context() -> Option<CoreGpuContext> {
    CoreGpuContext::try_new()
        .map_err(|error| {
            eprintln!("GPU not available, skipping test: {error}");
            error
        })
        .ok()
}

#[test]
fn buffer_creation_records_size_when_gpu_available() {
    let Some(context) = test_context() else {
        return;
    };

    let buffer = GpuBufferData::create_in_context(
        &context,
        1024,
        BufferUsage::STORAGE | BufferUsage::COPY_DST,
    )
    .expect("buffer creation must succeed");

    assert_eq!(buffer.size(), 1024);
}

#[test]
fn buffer_with_data_records_byte_size_when_gpu_available() {
    let Some(context) = test_context() else {
        return;
    };

    let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let buffer = GpuBufferData::create_with_data_in_context(
        &context,
        &data,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
    )
    .expect("buffer creation with data must succeed");

    assert_eq!(buffer.size(), std::mem::size_of_val(&data));
}

#[test]
fn buffer_write_read_roundtrip_preserves_values_when_gpu_available() {
    let Some(context) = test_context() else {
        return;
    };

    let original = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let buffer = GpuBufferData::create_with_data_in_context(
        &context,
        &original,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
    )
    .expect("buffer creation with data must succeed");

    let read_back: Vec<f32> = buffer
        .read_to_vec_in_context(&context)
        .expect("readback must succeed");

    assert_eq!(read_back, original);
}

#[test]
fn buffer_write_updates_contents_when_gpu_available() {
    let Some(context) = test_context() else {
        return;
    };

    let initial = [1.0_f32, 2.0, 3.0, 4.0];
    let updated = [5.0_f32, 6.0, 7.0, 8.0];
    let buffer = GpuBufferData::create_with_data_in_context(
        &context,
        &initial,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
    )
    .expect("buffer creation with data must succeed");

    buffer.write_in_context(&context, &updated);

    let read_back: Vec<f32> = buffer
        .read_to_vec_in_context(&context)
        .expect("readback must succeed");

    assert_eq!(read_back, updated);
}

#[test]
fn buffer_readback_preserves_integer_types_when_gpu_available() {
    let Some(context) = test_context() else {
        return;
    };

    let unsigned = [1_u32, 2, 3, 4, 5];
    let unsigned_buffer = GpuBufferData::create_with_data_in_context(
        &context,
        &unsigned,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
    )
    .expect("u32 buffer creation must succeed");
    let read_unsigned: Vec<u32> = unsigned_buffer
        .read_to_vec_in_context(&context)
        .expect("u32 readback must succeed");
    assert_eq!(read_unsigned, unsigned);

    let signed = [-1_i32, 2, -3, 4, -5];
    let signed_buffer = GpuBufferData::create_with_data_in_context(
        &context,
        &signed,
        BufferUsage::STORAGE | BufferUsage::COPY_SRC,
    )
    .expect("i32 buffer creation must succeed");
    let read_signed: Vec<i32> = signed_buffer
        .read_to_vec_in_context(&context)
        .expect("i32 readback must succeed");
    assert_eq!(read_signed, signed);
}

#[test]
fn buffer_usage_flags_match_wgpu_contract() {
    let combined = BufferUsage::STORAGE | BufferUsage::COPY_DST;
    assert!(combined.contains(BufferUsage::STORAGE));
    assert!(combined.contains(BufferUsage::COPY_DST));
    assert!(!combined.contains(BufferUsage::UNIFORM));
    assert!(!combined.contains(BufferUsage::COPY_SRC));
}
