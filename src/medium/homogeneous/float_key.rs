//! Float key for hashmap caching

use std::hash::{Hash, Hasher};
use super::constants::FLOAT_QUANTIZATION_FACTOR;

/// A wrapper for f64 that implements Hash and Eq for use in HashMap keys
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
        self.quantized() == other.quantized()
    }
}

impl Eq for FloatKey {}

impl Hash for FloatKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.quantized().hash(state);
    }
}