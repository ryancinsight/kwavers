use super::*;

#[test]
fn test_pool_config_buffer_size_alignment() {
    // 100 × 8 = 800 bytes → rounds up to 832 (= 13 × 64)
    let config = PoolConfig::for_f64_field(100, 16, -1);
    let raw_size = 100 * 8;
    let padded = (raw_size + 63) & !63;
    assert_eq!(config.buffer_size(), padded);
    assert_eq!(config.buffer_size() % 64, 0);
}

#[test]
fn test_pool_acquire_release_cycle() {
    let config = PoolConfig::for_f64_field(128, 4, -1);
    let pool = BufferPool::new(config).expect("pool must create");

    assert_eq!(pool.stats().allocated, 0);

    let buf = pool.acquire().expect("first acquire must succeed");
    assert_eq!(pool.stats().allocated, 1);

    drop(buf);
    assert_eq!(pool.stats().allocated, 0);
}

#[test]
fn test_pool_exhaustion_then_recycle() {
    let config = PoolConfig::for_f64_field(64, 2, -1);
    let pool = BufferPool::new(config).expect("pool must create");

    let b0 = pool.acquire().expect("first acquire");
    let b1 = pool.acquire().expect("second acquire");

    assert!(pool.acquire().is_err());

    drop(b0);
    assert_eq!(pool.stats().allocated, 1);

    let _b2 = pool.acquire().expect("acquire after release must succeed");
    drop(b1);
    drop(_b2);
    assert_eq!(pool.stats().allocated, 0);
}

#[test]
fn test_buffer_batch_all_or_nothing() {
    let config = PoolConfig::for_f64_field(64, 3, -1);
    let pool = BufferPool::new(config).expect("pool must create");

    assert!(BufferBatch::acquire(&pool, 4).is_err());
    assert_eq!(pool.stats().allocated, 0);
}

#[test]
fn test_pool_stats_peak_tracking() {
    let config = PoolConfig::for_f64_field(64, 4, -1);
    let pool = BufferPool::new(config).expect("pool must create");

    let b0 = pool.acquire().unwrap();
    let b1 = pool.acquire().unwrap();
    let b2 = pool.acquire().unwrap();

    assert_eq!(pool.stats().peak_allocated, 3);

    drop(b0);
    drop(b1);
    drop(b2);

    assert_eq!(pool.stats().peak_allocated, 3);
    assert_eq!(pool.stats().allocated, 0);
}

#[test]
fn test_pooled_buffer_typed_access() {
    let config = PoolConfig::for_f64_field(8, 2, -1);
    let pool = BufferPool::new(config).expect("pool must create");
    let mut buf = pool.acquire().expect("acquire");

    let data: &mut [f64] = buf.as_typed_mut();
    assert_eq!(data.len(), 8);
    data[0] = std::f64::consts::PI;
    assert_eq!(buf.as_typed::<f64>()[0], std::f64::consts::PI);
}
