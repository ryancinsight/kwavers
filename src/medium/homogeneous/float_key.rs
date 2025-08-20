//! Float key for hashmap caching
//!
//! Provides a wrapper for f64 that implements Hash and Eq for use in HashMap keys

use std::hash::{Hash, Hasher};

/// Quantization factor for float comparison and hashing
const FLOAT_QUANTIZATION_FACTOR: f64 = 1e6;

/// A wrapper for f64 that implements Hash and Eq for use in HashMap keys
///
/// Standard f64 values do not implement Eq and Hash in a way suitable
/// for direct use as hash map keys due to floating-point precision issues.
#[derive(Debug, Clone, Copy)]
pub struct FloatKey(pub f64);

impl FloatKey {
    /// Create a new FloatKey
    pub fn new(value: f64) -> Self {
        Self(value)
    }
    
    /// Get the inner value
    pub fn value(&self) -> f64 {
        self.0
    }
    
    /// Get the quantized value used for comparison
    fn quantized(&self) -> i64 {
        (self.0 * FLOAT_QUANTIZATION_FACTOR).round() as i64
    }
}

impl PartialEq for FloatKey {
    fn eq(&self, other: &Self) -> bool {
        // Equality based on quantized values for consistency with Hash
        self.quantized() == other.quantized()
    }
}

impl Eq for FloatKey {}

impl Hash for FloatKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the quantized value to ensure close floats hash the same
        self.quantized().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_float_key_equality() {
        let key1 = FloatKey::new(1.0000001);
        let key2 = FloatKey::new(1.0000002);
        assert_eq!(key1, key2); // Should be equal due to quantization
        
        let key3 = FloatKey::new(1.1);
        assert_ne!(key1, key3);
    }
    
    #[test]
    fn test_float_key_in_hashmap() {
        let mut map = HashMap::new();
        map.insert(FloatKey::new(1.0), "one");
        map.insert(FloatKey::new(2.0), "two");
        
        assert_eq!(map.get(&FloatKey::new(1.0000001)), Some(&"one"));
        assert_eq!(map.get(&FloatKey::new(2.0)), Some(&"two"));
    }
}