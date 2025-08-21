//! Float key wrapper for HashMap usage

use std::hash::{Hash, Hasher};

/// A wrapper for `f64` to allow its use as a key in `HashMap`.
///
/// Standard `f64` values do not implement `Eq` and `Hash` in a way that is suitable
/// for direct use as hash map keys due to floating-point precision issues.
#[derive(Debug, Clone, Copy)]
pub struct FloatKey(pub f64);

impl PartialEq for FloatKey {
    fn eq(&self, other: &Self) -> bool {
        (self.0 - other.0).abs() < 1e-10
    }
}

impl Eq for FloatKey {}

impl Hash for FloatKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Quantize the float to ensure close values hash the same
        let quantized = (self.0 * 1e6).round() as i64;
        quantized.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_float_key_equality() {
        let key1 = FloatKey(1.0);
        let key2 = FloatKey(1.0 + 1e-11);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_float_key_hash() {
        let mut map = HashMap::new();
        map.insert(FloatKey(1.0), "value");
        assert_eq!(map.get(&FloatKey(1.0 + 1e-11)), Some(&"value"));
    }
}
