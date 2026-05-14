//! Wave scattering physics module

#[derive(Debug)]
pub struct ScatteringCalculator;

impl ScatteringCalculator {
    #[must_use]
    pub fn new(_frequency: f64, _wave_speed: f64) -> Self {
        Self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ScatteringCalculator::new constructs without panic for physical parameters.
    #[test]
    fn new_constructs_for_physical_parameters() {
        let _sc = ScatteringCalculator::new(1e6, 1500.0);
    }

    /// Debug output is non-empty.
    #[test]
    fn debug_non_empty() {
        let sc = ScatteringCalculator::new(500e3, 3400.0);
        let s = format!("{sc:?}");
        assert!(
            !s.is_empty(),
            "ScatteringCalculator debug must not be empty"
        );
    }
}
