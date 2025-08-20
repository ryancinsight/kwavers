//! Thread-safe cache for absorption coefficients

use super::float_key::FloatKey;
use std::collections::HashMap;
use std::sync::Mutex;

/// A thread-safe cache for storing acoustic absorption coefficients
#[derive(Debug)]
pub struct AbsorptionCache {
    cache: Mutex<HashMap<FloatKey, f64>>,
}

impl AbsorptionCache {
    /// Creates a new, empty AbsorptionCache
    pub fn new() -> Self {
        AbsorptionCache {
            cache: Mutex::new(HashMap::new()),
        }
    }
    
    /// Gets or computes a value using the provided closure
    pub fn get_or_insert_with<F>(&self, key: FloatKey, f: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        let mut guard = self.cache.lock().unwrap();
        *guard.entry(key).or_insert_with(f)
    }
}

impl Clone for AbsorptionCache {
    fn clone(&self) -> Self {
        let guard = self.cache.lock().unwrap();
        let cloned_map = guard.clone();
        AbsorptionCache {
            cache: Mutex::new(cloned_map),
        }
    }
}