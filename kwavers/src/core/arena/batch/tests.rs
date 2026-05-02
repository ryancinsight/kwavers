use super::*;

#[test]
fn test_batch_config_validation() {
    let valid = BatchFieldConfig::for_3d_fields(10, 10, 10, 4);
    assert!(valid.validate().is_ok());

    let invalid_zero_elems = BatchFieldConfig {
        field_elements: 0,
        num_fields: 4,
        alignment: 64,
        numa_node: None,
    };
    assert!(invalid_zero_elems.validate().is_err());

    let invalid_alignment = BatchFieldConfig {
        field_elements: 100,
        num_fields: 4,
        alignment: 63,
        numa_node: None,
    };
    assert!(invalid_alignment.validate().is_err());
}

#[test]
fn test_soa_buffer_allocation() {
    let config = BatchFieldConfig::for_3d_fields(10, 10, 10, 4);
    let mut buffer = SoAFieldBuffer::new(config).unwrap();

    assert_eq!(buffer.field_size(), 1000);
    assert_eq!(buffer.num_fields(), 4);
    assert!(buffer.is_cache_aligned());

    let field0 = buffer.field_mut(0);
    assert_eq!(field0.len(), 1000);
    field0[0] = 1.0;

    // Other fields must be independent.
    assert_eq!(buffer.field(1)[0], 0.0);
}

#[test]
fn test_temp_buffer_pool() {
    let mut pool = TempBufferPool::new();
    pool.preallocate(2, 2, 2);

    let buffer = pool.acquire(100);
    assert_eq!(buffer.len(), 100);
    assert!(buffer.capacity() >= 256);

    pool.release(buffer);
    assert_eq!(pool.total_allocated(), 0);

    let _ = pool.acquire(65537); // Custom size → new allocation
    assert_eq!(pool.total_allocated(), 1);
}

#[test]
fn test_batch_field_handle_wave_sim() {
    let mut handle = BatchFieldHandle::for_wave_simulation(8, 8, 8).unwrap();

    assert_eq!(handle.pressure().len(), 512);
    let (vx, vy, vz) = handle.velocity();
    assert_eq!(vx.len(), 512);
    assert_eq!(vy.len(), 512);
    assert_eq!(vz.len(), 512);

    handle.clear();
    assert_eq!(handle.pressure()[0], 0.0);
    assert!(handle.is_cache_efficient());
}

#[test]
fn test_buffer_size_classification() {
    assert_eq!(BufferSize::classify(100), BufferSize::Small);
    assert_eq!(BufferSize::classify(1000), BufferSize::Medium);
    assert_eq!(BufferSize::classify(10000), BufferSize::Large);
    assert_eq!(BufferSize::classify(100000), BufferSize::Custom(100000));
}
