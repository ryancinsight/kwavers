//! Thread-safe absorption coefficient cache

use std::collections::HashMap;
use std::sync::Mutex;
use super::float_key::FloatKey;

/// Thread-safe cache for absorption coefficients
pub struct AbsorptionCache {
    cache: Mutex<HashMap<FloatKey, f64>>,
}

impl AbsorptionCache {
    /// Create a new empty cache
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Get or compute absorption coefficient
    pub fn get_or_compute<F>(&self, frequency: f64, compute: F) -> f64
    where
        F: FnOnce() -> f64,
    {
        let key = FloatKey(frequency);
        
        // Try to get from cache first
        if let Ok(cache) = self.cache.lock() {
            if let Some(&value) = cache.get(&key) {
                return value;
            }
        }
        
        // Compute if not in cache
        let value = compute();
        
        // Store in cache
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key, value);
        }
        
        value
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }
}

impl Default for AbsorptionCache {
    fn default() -> Self {
        Self::new()
    }
}