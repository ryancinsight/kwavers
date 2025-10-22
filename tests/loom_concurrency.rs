//! Concurrency safety tests using loom
//!
//! These tests validate Arc/RwLock patterns using loom's model checker
//! to detect data races and deadlocks.
//!
//! ## Sprint 138 - Persona Requirements
//!
//! Per senior Rust engineer persona: "testing with loom for concurrency"
//!
//! ## Running Loom Tests
//!
//! ```bash
//! RUSTFLAGS="--cfg loom" cargo test --test loom_concurrency --release
//! ```

#[cfg(loom)]
mod loom_tests {
    use loom::sync::{Arc, RwLock};
    use loom::thread;

    #[test]
    fn test_arc_rwlock_concurrent_reads() {
        loom::model(|| {
            let data = Arc::new(RwLock::new(42));

            let data1 = Arc::clone(&data);
            let data2 = Arc::clone(&data);

            let h1 = thread::spawn(move || {
                let value = data1.read().unwrap();
                *value
            });

            let h2 = thread::spawn(move || {
                let value = data2.read().unwrap();
                *value
            });

            let v1 = h1.join().unwrap();
            let v2 = h2.join().unwrap();

            assert_eq!(v1, 42);
            assert_eq!(v2, 42);
        });
    }

    #[test]
    fn test_arc_rwlock_write_then_read() {
        loom::model(|| {
            let data = Arc::new(RwLock::new(0));

            let data1 = Arc::clone(&data);
            let data2 = Arc::clone(&data);

            let h1 = thread::spawn(move || {
                let mut value = data1.write().unwrap();
                *value = 100;
            });

            h1.join().unwrap();

            let h2 = thread::spawn(move || {
                let value = data2.read().unwrap();
                *value
            });

            let result = h2.join().unwrap();
            assert_eq!(result, 100);
        });
    }

    #[test]
    fn test_arc_rwlock_concurrent_writes() {
        loom::model(|| {
            let data = Arc::new(RwLock::new(0));

            let data1 = Arc::clone(&data);
            let data2 = Arc::clone(&data);

            let h1 = thread::spawn(move || {
                let mut value = data1.write().unwrap();
                *value += 1;
            });

            let h2 = thread::spawn(move || {
                let mut value = data2.write().unwrap();
                *value += 1;
            });

            h1.join().unwrap();
            h2.join().unwrap();

            let final_value = *data.read().unwrap();
            assert_eq!(final_value, 2);
        });
    }

    #[test]
    fn test_fft_cache_pattern() {
        loom::model(|| {
            // Simulates FFT cache Arc<RwLock<HashMap>> pattern
            use loom::sync::Arc;
            use std::collections::HashMap;

            let cache = Arc::new(RwLock::new(HashMap::<usize, Vec<f64>>::new()));

            let cache1 = Arc::clone(&cache);
            let cache2 = Arc::clone(&cache);

            // Thread 1: Insert data
            let h1 = thread::spawn(move || {
                let mut c = cache1.write().unwrap();
                c.insert(256, vec![1.0, 2.0, 3.0]);
            });

            // Thread 2: Read data (may or may not see thread 1's insert)
            let h2 = thread::spawn(move || {
                let c = cache2.read().unwrap();
                c.get(&256).map(|v| v.len())
            });

            h1.join().unwrap();
            let result = h2.join().unwrap();

            // Result is either None (read before write) or Some(3) (read after write)
            assert!(result.is_none() || result == Some(3));
        });
    }
}

#[cfg(not(loom))]
#[cfg(test)]
mod regular_tests {
    use std::sync::{Arc, RwLock};
    use std::thread;

    #[test]
    fn test_arc_rwlock_basic() {
        let data = Arc::new(RwLock::new(42));
        let data_clone = Arc::clone(&data);

        let handle = thread::spawn(move || {
            let value = data_clone.read().unwrap();
            *value
        });

        let result = handle.join().unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_arc_rwlock_write() {
        let data = Arc::new(RwLock::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let data_clone = Arc::clone(&data);
            let handle = thread::spawn(move || {
                let mut value = data_clone.write().unwrap();
                *value += 1;
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let final_value = *data.read().unwrap();
        assert_eq!(final_value, 10);
    }
}
